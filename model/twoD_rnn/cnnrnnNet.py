# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2



from model.twoD_rnn.backbone_utils import ResidualBlockNoBN, make_layer
from model.twoD_rnn.unet.unet_model import OnlyUnet



class cnnrnnNet(nn.Module):
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

    def __init__(self, n_channels, n_classes, bilinear=False,num_blocks=10):

        super().__init__()

        self.n_channel = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # feature network
        self.unet = OnlyUnet(self.n_channel,self.n_classes,self.bilinear)

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            self.n_classes*3, self.n_classes*2, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            self.n_classes*3, self.n_classes*2, num_blocks)


        # last
        self.last = nn.Sequential(
            nn.Conv2d(self.n_classes * 2 , self.n_classes * 2, 3, 1, 1),
            # BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.BatchNorm2d(self.n_classes * 2),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_classes * 2 , self.n_classes , 3, 1, 1),
            # BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.BatchNorm2d(self.n_classes),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_classes, 1, 3, 1, 1),
        )

        self.fusion = nn.Conv2d(
            self.n_classes * 4, self.n_classes * 2, 1, 1, 0, bias=True)

        # activation function
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.lrelu = nn.ReLU(inplace=True)


    def compute_feature(self, lrs_array):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            feature(Tensor): with shape (n, t, out_channel, h, w)
        """


        feature = self.unet(lrs_array)

        return feature


    def forward(self, lrs_array):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
            lrs_array(Tensor): Input LR_array sequence with shape (n, t, c, h, w),the value through znorm.
        Returns:
            Tensor: Output segmentation sequence with shape (n, t, c, h, w).
        """

        n, t, c, h, w = lrs_array.size()

        feature = self.compute_feature(lrs_array)
        # backward-time propgation
        outputs = []
        feat_prop = feature.new_zeros(n, self.n_classes*2, h, w)
        for i in range(t - 1, -1, -1):

            feat_prop = torch.cat([feature[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            feature_curr = feature[:, i, :, :, :]

            feat_prop = torch.cat([feature_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            # out = self.last(out)
            # outputs[i] = out

            outputs[i] = out
            midput = torch.stack(outputs, dim =1)

        x = midput[0,:,:,:,:]
        real_output = self.last(x)
        return real_output.unsqueeze(0)

        # return torch.stack(outputs, dim=1)

    def init_weights(self):
        #logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
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



