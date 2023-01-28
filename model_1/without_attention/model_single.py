import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torchstat import stat
from model_part import *
from clstm import ConvLSTMCell
import argparse

"model：目前使用模型，每一层都有skip_feature，但大小固定，由hidden_size决定,还是不行啊，内存太大了"
"model2：为初始模型，每一层都有skip_feature，并且skip_feature大小随模型大小变化，内存太大"
"model3：改变了一下，最上面一层没有skip_feature，并且skip_feature大小随模型大小变化，内存太大"

class FeatureExtractor_resnet(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''

    def __init__(self, args):
        super(FeatureExtractor_resnet, self).__init__()
        self.layer_num = args.down_num + 1
        self.layer_list = nn.ModuleList()
        self.skip_convs_list = nn.ModuleList()
        self.skip_bns_list = nn.ModuleList()
        self.feature_dim = args.init_feature
        self.hidden_size = args.hidden_size
        for i in range(self.layer_num):
            if i == 0:
                encoder_i = nn.Sequential(
                    Unet_backbone(1, self.feature_dim),
                    First_resnet(in_channels = self.feature_dim,
                                 out_channels = self.feature_dim,
                                 baseWidth = args.baseWidth,
                                 init_feature = args.init_feature,
                                 scale = args.scale,
                                 blocks = args.block_num)
                )
                skip_conv_i = nn.Conv2d(self.feature_dim, self.hidden_size, kernel_size = 3, padding = 1)
                skip_bn_i = nn.BatchNorm2d(self.hidden_size)
            else:
                encoder_i = Down_resnet(in_channels = self.feature_dim,
                                        out_channels = self.feature_dim * 2,
                                        baseWidth = args.baseWidth,
                                        init_feature = args.init_feature,
                                        scale = args.scale,
                                        blocks = args.block_num,
                                        )
                self.feature_dim = self.feature_dim * 2
                skip_conv_i = nn.Conv2d(self.feature_dim, self.hidden_size*2, kernel_size = 3, padding = 1)
                skip_bn_i = nn.BatchNorm2d(self.hidden_size*2)
            self.layer_list.append(encoder_i)
            self.skip_convs_list.append(skip_conv_i)
            self.skip_bns_list.append(skip_bn_i)

    def forward(self, x):
        feature_list = []
        input_feature = x

        for i in range(self.layer_num):
            feature = self.layer_list[i](input_feature)
            feature_list.append(feature)
            input_feature = feature
        for i in range(self.layer_num):
            feature_list[i] = self.skip_bns_list[i](self.skip_convs_list[i](feature_list[i]))

        return feature_list[::-1]


class RSIS(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, args):

        super(RSIS, self).__init__()


        self.kernel_size = args.kernel_size
        padding = 0 if self.kernel_size == 1 else 1
        self.dropout = args.dropout
        self.layer_num = args.down_num + 1
        # convlstms have decreasing dimension as width and height increase
        # skip_dims_out = [self.hidden_size, int(self.hidden_size / 2),
        #                  int(self.hidden_size / 4), int(self.hidden_size / 8)]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        self.hidden_size = args.hidden_size

        # skip_dims_list = []
        # for i in range(self.layer_num):
        #     if i == self.layer_num - 1:
        #         skip_dims = self.hidden_size
        #     else:
        #         skip_dims = self.hidden_size *2
        #     skip_dims_list.append(skip_dims)


        # self.baseWidth = args.baseWidth
        # clstm_in_dim = self.baseWidth * 2
        # clstm_out_dim = self.baseWidth
        # upsample_dim = self.baseWidth
        # 4 is the number of deconv steps that we need to reach image size in the output
        # for i in range(self.layer_num):
        #     if i == self.layer_num - 1:
        #         clstm_in_dim //= 2
        #         clstm_i = ConvLSTMCell(args, clstm_in_dim, clstm_out_dim, self.kernel_size, padding = padding)
        #     else:
        #         clstm_i = ConvLSTMCell(args, clstm_in_dim, clstm_out_dim, self.kernel_size, padding = padding)
        #         clstm_in_dim *= 2
        #         clstm_out_dim *= 2
        #     self.clstm_list.append(clstm_i)

        for i in range(self.layer_num):
            if i == 0:
                clstm_i = ConvLSTMCell(args, self.hidden_size*2, self.hidden_size*2, self.kernel_size, padding = padding)
            else:
                if i == self.layer_num-1:
                    clstm_i= ConvLSTMCell(args, self.hidden_size*2, self.hidden_size, self.kernel_size, padding = padding)
                else:
                    if i == self.layer_num-2:
                        clstm_i= ConvLSTMCell(args, self.hidden_size*4, self.hidden_size, self.kernel_size, padding = padding)
                    else:
                        clstm_i = ConvLSTMCell(args, self.hidden_size*4, self.hidden_size*2, self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)

        # for i in range(self.layer_num-1):
        #     upsample_i = nn.ConvTranspose2d(upsample_dim * 2, upsample_dim, kernel_size = 2, stride = 2)
        #     upsample_dim *= 2
        #     self.upsample_list.append(upsample_i)

        for i in range(self.layer_num-1):
            if i == self.layer_num-2:
                upsample_i = nn.ConvTranspose2d(self.hidden_size, self.hidden_size, kernel_size = 2, stride = 2)
            else:
                upsample_i = nn.ConvTranspose2d(self.hidden_size*2, self.hidden_size*2, kernel_size = 2, stride = 2)
            self.upsample_list.append(upsample_i)

        # self.clstm_list = self.clstm_list[::-1]
        # self.upsample_list = self.upsample_list[::-1]
        # for i in range(len(skip_dims_out)):
        #     if i == 0:
        #         clstm_in_dim = self.hidden_size
        #     else:
        #         clstm_in_dim = skip_dims_out[i - 1]
        #         if self.skip_mode == 'concat':
        #             clstm_in_dim *= 2
        #
        #     clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i], self.kernel_size, padding = padding)
        #     self.clstm_list.append(clstm_i)

        self.conv_out = nn.Conv2d(self.hidden_size, 1, self.kernel_size, padding = padding)

        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        # fc_dim = 0
        # for sk in skip_dims_out:
        #     fc_dim += sk

    def forward(self, feature_list, prev_hidden_temporal):

        clstm_in = feature_list[0]
        skip_feats = feature_list[1:]
        hidden_list = []

        for i in range(self.layer_num):
            if prev_hidden_temporal is None:
                state = self.clstm_list[i](clstm_in, None)

            else:
                state = self.clstm_list[i](clstm_in, prev_hidden_temporal[i])

            # hidden states will be initialized the first time forward is called
            # if prev_state_spatial is None:
            #     if prev_hidden_temporal is None:
            #         state = self.clstm_list[i](clstm_in,None, None)
            #     else:
            #         state = self.clstm_list[i](clstm_in,None, prev_hidden_temporal[i])
            # else:
            #     # else we take the ones from the previous step for the forward pass
            #     if prev_hidden_temporal is None:
            #         state = self.clstm_list[i](clstm_in, prev_state_spatial[i], None)
            #
            #     else:
            #         state = self.clstm_list[i](clstm_in, prev_state_spatial[i], prev_hidden_temporal[i])

            # if prev_hidden_temporal is None:
            #     state = self.clstm_list[i](clstm_in, None)
            #
            # else:
            #     state = self.clstm_list[i](clstm_in, prev_hidden_temporal[i])
            # # state = self.clstm_list[i](clstm_in, prev_hidden_temporal[i])

            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            # skip_vec = skip_feats[i]
            # upsample = nn.ConvTranspose2d(hidden.size()[-3], skip_vec.size()[-3], kernel_size = 2, stride = 2)
            # # upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2], skip_vec.size()[-1]))
            # hidden = upsample(hidden)

            # apply skip connection
            if i < self.layer_num - 1:
                skip_vec = skip_feats[i]
                upsample = self.upsample_list[i]
                # upsample = nn.ConvTranspose2d(hidden.size()[-3], skip_vec.size()[-3], kernel_size = 2, stride = 2)
                # upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2], skip_vec.size()[-1]))
                hidden = upsample(hidden)
                clstm_in = torch.cat([hidden, skip_vec], 1)
                # # skip connection
                # if self.skip_mode == 'concat':
                #     clstm_in = torch.cat([hidden, skip_vec], 1)
                # elif self.skip_mode == 'sum':
                #     clstm_in = hidden + skip_vec
                # elif self.skip_mode == 'mul':
                #     clstm_in = hidden * skip_vec
                # elif self.skip_mode == 'none':
                #     clstm_in = hidden
                # else:
                #     raise Exception('Skip connection mode not supported !')
            else:
                # self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2] * 2, hidden.size()[-1] * 2))
                # hidden = self.upsample(hidden)
                clstm_in = hidden

        out_mask = self.conv_out(clstm_in)
        # classification branch

        return out_mask, hidden_list


class Rvosnet(nn.Module):
    def __init__(self, args):
        super(Rvosnet, self).__init__()

        self.encoder = FeatureExtractor_resnet(args)
        self.decoder = RSIS(args)

    def forward(self, x):
        n, t, c, h, w = x.size()
        prev_hidden_temporal = None
        out_mask_list = []
        x = x[0, :, :, :, :]
        print(x.shape)
        feats = self.encoder(x)
        feats = feats.unsqueeze(0)
        for i in range(t):
            print(i)
            input = feats[:, i, :, :, :]
            # feats = self.encoder(input)
            hidden_temporal = prev_hidden_temporal

            out_mask, hidden = self.decoder(input, hidden_temporal)
            out_mask_list.append(out_mask)
            prev_hidden_temporal = hidden

        real_out_mask = torch.stack(out_mask_list, 1)
        return real_out_mask


# if __name__ == "__main__":
#     t = torch.ones((10, 1, 128, 128))
#     model_in = FeatureExtractor_resnet(init_feature = 64,
#                                        baseWidth = 26,
#                                        scale = 3,
#                                        block_num = 2,
#                                        down_num = 3)
#     model_out = RSIS(kernel_size = 3, dropout = 0, down_num = 3, baseWidth = 26)
#     n_parameters1 = sum(p.numel() for p in model_in.parameters() if p.requires_grad)
#     stat(model_in, input_size = (1, 128, 128))
#     print('number of params (M): %.2f' % (n_parameters1 / 1.e6))
#     n_parameters2 = sum(p.numel() for p in model_out.parameters() if p.requires_grad)
#     print('number of params (M): %.2f' % (n_parameters2 / 1.e6))
#     mid = model_in(t)
#     out, _ = model_out(mid, None)
#     for i in range(4):
#         print(mid[i].shape)
#     print(out.shape)
#
#     model_all = Rvosnet()
#     al = torch.ones((10, 5, 1, 128, 128))
#     n_parameters1 = sum(p.numel() for p in model_all.parameters() if p.requires_grad)
#     # stat(model_all, input_size = (5,1, 256, 256))
#     print('number of params (M): %.2f' % (n_parameters1 / 1.e6))
#     out_all = model_all(al)
#     print(out_all.shape)


def get_args_parser():
    parser = argparse.ArgumentParser('Medical segmentation ', add_help = False)
    parser.add_argument('--cpu', dest = 'use_gpu', action = 'store_false')
    parser.set_defaults(use_gpu = False)
    # parser.add_argument('-base_model', dest = 'base_model', default = 'resnet34',
    #                     choices = ['resnet101', 'resnet50', 'resnet34', 'vgg16'])
    # parser.add_argument('-skip_mode', dest = 'skip_mode', default = 'concat',
    #                     choices = ['sum', 'concat', 'mul', 'none'])
    # parser.add_argument('-hidden_size', dest = 'hidden_size', default = 128, type = int)

    parser.add_argument('--init_feature', dest = 'init_feature', default = 64, type = int)
    parser.add_argument('--scale', dest = 'scale', default = 3, type = int)
    parser.add_argument('--baseWidth', dest = 'baseWidth', default = 18, type = int)
    parser.add_argument('--block_num', dest = 'block_num', default = 3, type = int)
    parser.add_argument('--down_num', dest = 'down_num', default = 3, type = int)
    parser.add_argument('--kernel_size', dest = 'kernel_size', default = 3, type = int)
    parser.add_argument('--dropout', dest = 'dropout', default = 0.0, type = float)
    parser.add_argument('--hidden_size', dest = 'hidden_size', default = 16, type = int)

    return parser

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    t = torch.ones((10, 1, 128, 128))
    model_in = FeatureExtractor_resnet(args)
    model_out = RSIS(args)
    n_parameters1 = sum(p.numel() for p in model_in.parameters() if p.requires_grad)
    stat(model_in, input_size = (1, 128, 128))
    print('number of params (M): %.2f' % (n_parameters1 / 1.e6))
    n_parameters2 = sum(p.numel() for p in model_out.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters2 / 1.e6))
    mid = model_in(t)

    for i in range(4):
        print(mid[i].shape)
    out, _ = model_out(mid, None)
    print(out.shape)

    model_all = Rvosnet(args)
    al = torch.ones((1, 20, 1, 384, 256))
    n_parameters1 = sum(p.numel() for p in model_all.parameters() if p.requires_grad)
    # stat(model_all, input_size = (5,1, 256, 256))
    print('number of params (M): %.2f' % (n_parameters1 / 1.e6))
    out_all = model_all(al)
    print(out_all.shape)