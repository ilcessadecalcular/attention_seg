# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2



from model.twoD_flow.backbone_utils import flow_warp, ResidualBlockNoBN, make_layer
from model.twoD_flow.HRNet import HRNetSeg
from model.twoD_flow.bn_helper import BatchNorm2d,  BatchNorm2d_class, relu_inplace

BN_MOMENTUM = 0.1
ALIGN_CORNERS = None

class cnnflowNet(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, config, mid_channels=64, num_blocks=20, out_channels=1):

        super().__init__()

        self.config = config()
        self.mid_channels = mid_channels

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained)

        # feature network
        self.hrnet = HRNetSeg(config)

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels*3, mid_channels*2, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels*3, mid_channels*2, num_blocks)

        # upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            BatchNorm2d(mid_channels,momentum=BN_MOMENTUM),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # last
        self.last = nn.Sequential(
            nn.Conv2d(mid_channels * 2 , mid_channels, 3, 1, 1),
            # BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            # BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
        )

        self.fusion = nn.Conv2d(
            mid_channels * 4, mid_channels * 2, 1, 1, 0, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    # def compute_flow(self, lrs):
    #     """Compute optical flow using SPyNet for feature warping.
    #
    #     Note that if the input is an mirror-extended sequence, 'flows_forward'
    #     is not needed, since it is equal to 'flows_backward.flip(1)'.
    #
    #     Args:
    #         lrs (tensor): Input LR images with shape (n, t, c, h, w)
    #
    #     Return:
    #         tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
    #             flows used for forward-time propagation (current to previous).
    #             'flows_backward' corresponds to the flows used for
    #             backward-time propagation (current to next).
    #     """
    #
    #     n, t, c, h, w = lrs.size()
    #     lrs_input =lrs[0,:,0,:,:]
    #
    #     dis = cv2.DISOpticalFlow_create(2)  # PRESET_ULTRAFAST, PRESET_FAST and PRESET_MEDIUM
    #
    #     flows_forward = []
    #     flows_backward = []
    #     for i in range(t):
    #         if i == 0:
    #             flow_backward = None
    #         else:
    #             flow_backward = dis.calc(lrs_input[i],lrs_input[i-1], None)
    #             flow_backward = flow_backward.unsqueeze(0)
    #         if i == t-1:
    #             flow_forward = None
    #         else:
    #             flow_forward = dis.calc(lrs_input[i], lrs_input[i+1], None)
    #             flow_forward = flow_forward.unsqueeze(0)
    #
    #         flows_forward.append(flow_forward)
    #         flows_backward.append(flow_backward)
    #
    #     return flows_forward, flows_backward

    def compute_feature(self, lrs_array):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            feature(Tensor): with shape (n, t, out_channel, h, w)
        """


        lrs_array_input =lrs_array[0,:,:,:,:]

        feature = self.hrnet(lrs_array_input)
        up_feature = self.upsample(feature)
        # feature = feature.unsqueeze(0)
        up_feature = up_feature.unsqueeze(0)
        return up_feature


    def forward(self, lrs_array, flows_forward, flows_backward):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
            lrs_array(Tensor): Input LR_array sequence with shape (n, t, c, h, w),the value through znorm.
        Returns:
            Tensor: Output segmentation sequence with shape (n, t, c, h, w).
        """

        n, t, c, h, w = lrs_array.size()

        # compute optical flow
        # flows_forward, flows_backward = self.compute_flow(lrs_array)
        up_feature = self.compute_feature(lrs_array)
        # backward-time propgation
        outputs = []
        feat_prop = lrs_array.new_zeros(n, self.mid_channels*2, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow)

            feat_prop = torch.cat([up_feature[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            up_feature_curr = up_feature[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow)

            feat_prop = torch.cat([up_feature_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.last(out)
            outputs[i] = out

        return torch.stack(outputs, dim=1)

    def init_weights(self):
        #logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

def get_seg_model(cfg, **kwargs):
    model = cnnflowNet(cfg, **kwargs)
    model.init_weights()

    return model


