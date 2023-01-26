""" Full assembly of the parts to form the complete network """

from unet_parts import *
from torchinfo import summary
from torchstat import stat
from thop import profile
from clstm import ConvLSTMCell


#
# args.base_dims
# args.backbone
# args.block_num
# args.down_num


#
#
# class FeatureExtractor(nn.Module):
#     '''
#     Returns base network to extract visual features from image
#     '''
#
#     def __init__(self, args):
#         super(FeatureExtractor, self).__init__()
#         self.feature_dims = [args.base_dims,
#                              args.base_dims * 2,
#                              args.base_dims * 4,
#                              args.base_dims * 8,
#                              args.base_dims * 16,
#                              ]
#         self.inc = backbone(args.backbone,
#                             in_channels = 1,
#                             out_channels = self.feature_dims[0],
#                             blocks = args.block_num)
#         self.down_num = args.down_num
#         self.down_list = nn.ModuleList()
#         for i in range(self.down_num):
#             if i == 0:
#                 encoder_i = backbone(args.backbone,
#                                      in_channels = 1,
#                                      out_channels = args.base_dims * (i + 1),
#                                      blocks = args.block_num)
#             else:
#                 encoder_i = Down(args.backbone,
#                                  in_channels = args.base_dims * i,
#                                  out_channels = args.base_dims * (i + 1),
#                                  blocks = args.block_num)
#             self.down_list.append(encoder_i)
#     def forward(self, x):
#         feature_list = []
#         input_feature = x
#
#         for i in range(self.down_num):
#             feature = self.down_list[i](input_feature)
#             feature_list.append(feature)
#             input_feature = feature
#
#         return feature_list

class FeatureExtractor(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''

    def __init__(self, base_dims,
                 args_backbone,
                 block_num,
                 down_num):
        super(FeatureExtractor, self).__init__()
        # self.feature_dims = [base_dims,
        #                      base_dims * 2,
        #                      base_dims * 4,
        #                      base_dims * 8,
        #                      base_dims * 16,
        #                      ]
        # self.inc = backbone(backbone,
        #                     in_channels = 1,
        #                     out_channels = self.feature_dims[0],
        #                     blocks = block_num)
        self.layer_num = down_num + 1
        self.layer_list = nn.ModuleList()
        self.feature_dim = base_dims
        for i in range(self.layer_num):
            if i == 0:
                encoder_i = layer_block(args_backbone,
                                        1,
                                        base_dims,
                                        blocks = block_num)
            else:
                encoder_i = Down(args_backbone,
                                 self.feature_dim,
                                 self.feature_dim * 2,
                                 blocks = block_num)
                self.feature_dim = self.feature_dim * 2
            self.layer_list.append(encoder_i)

    def forward(self, x):
        feature_list = []
        input_feature = x

        for i in range(self.layer_num):
            feature = self.layer_list[i](input_feature)
            feature_list.append(feature)
            input_feature = feature

        return feature_list[::-1]


# args.down_num

#
# class RSIS(nn.Module):
#     """
#     The recurrent decoder
#     """
#
#     def __init__(self, args):
#
#         super(RSIS, self).__init__()
#         self.hidden_size = args.hidden_size
#         self.kernel_size = args.kernel_size
#         padding = 0 if self.kernel_size == 1 else 1
#         self.dropout = args.dropout
#         self.layer_num = args.down_num + 1
#         # convlstms have decreasing dimension as width and height increase
#         # skip_dims_out = [self.hidden_size, int(self.hidden_size / 2),
#         #                  int(self.hidden_size / 4), int(self.hidden_size / 8)]
#
#         # initialize layers for each deconv stage
#         self.clstm_list = nn.ModuleList()
#         self.base_dim = args.base_dim
#         clstm_in_dim = self.base_dim * 2
#         clstm_out_dim = self.base_dim
#         # 4 is the number of deconv steps that we need to reach image size in the output
#         for i in range(self.layer_num):
#             if i == self.layer_num - 1:
#                 clstm_in_dim //= 2
#                 clstm_i = ConvLSTMCell(args, clstm_in_dim, clstm_out_dim, self.kernel_size, padding = padding)
#             else:
#                 clstm_i = ConvLSTMCell(args, clstm_in_dim, clstm_out_dim, self.kernel_size, padding = padding)
#                 clstm_in_dim *= 2
#                 clstm_out_dim *= 2
#             self.clstm_list.append(clstm_i)
#         self.clstm_list = self.clstm_list[::-1]
#         # for i in range(len(skip_dims_out)):
#         #     if i == 0:
#         #         clstm_in_dim = self.hidden_size
#         #     else:
#         #         clstm_in_dim = skip_dims_out[i - 1]
#         #         if self.skip_mode == 'concat':
#         #             clstm_in_dim *= 2
#         #
#         #     clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i], self.kernel_size, padding = padding)
#         #     self.clstm_list.append(clstm_i)
#
#         self.conv_out = nn.Conv2d(self.base_dim, 1, self.kernel_size, padding = padding)
#
#         # calculate the dimensionality of classification vector
#         # side class activations are taken from the output of the convlstm
#         # therefore we need to compute the sum of the dimensionality of outputs
#         # from all convlstm layers
#         # fc_dim = 0
#         # for sk in skip_dims_out:
#         #     fc_dim += sk
#
#     def forward(self, feature_list, prev_hidden_temporal):
#
#         clstm_in = feature_list[0]
#         skip_feats = feature_list[1:]
#         hidden_list = []
#
#         for i in range(self.layer_num):
#             if prev_hidden_temporal is None:
#                 state = self.clstm_list[i](clstm_in, None)
#
#             else:
#                 state = self.clstm_list[i](clstm_in, prev_hidden_temporal[i])
#
#             # hidden states will be initialized the first time forward is called
#             # if prev_state_spatial is None:
#             #     if prev_hidden_temporal is None:
#             #         state = self.clstm_list[i](clstm_in,None, None)
#             #     else:
#             #         state = self.clstm_list[i](clstm_in,None, prev_hidden_temporal[i])
#             # else:
#             #     # else we take the ones from the previous step for the forward pass
#             #     if prev_hidden_temporal is None:
#             #         state = self.clstm_list[i](clstm_in, prev_state_spatial[i], None)
#             #
#             #     else:
#             #         state = self.clstm_list[i](clstm_in, prev_state_spatial[i], prev_hidden_temporal[i])
#
#             # if prev_hidden_temporal is None:
#             #     state = self.clstm_list[i](clstm_in, None)
#             #
#             # else:
#             #     state = self.clstm_list[i](clstm_in, prev_hidden_temporal[i])
#             # # state = self.clstm_list[i](clstm_in, prev_hidden_temporal[i])
#
#             hidden_list.append(state)
#             hidden = state[0]
#
#             if self.dropout > 0:
#                 hidden = nn.Dropout2d(self.dropout)(hidden)
#
#             skip_vec = skip_feats[i]
#             upsample = nn.ConvTranspose2d(hidden.size()[-3], skip_vec.size()[-3], kernel_size = 2, stride = 2)
#             # upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2], skip_vec.size()[-1]))
#             hidden = upsample(hidden)
#
#             # apply skip connection
#             if i < self.layer_num - 1:
#
#                 # skip_vec = skip_feats[i]
#                 # upsample = nn.ConvTranspose2d(hidden.size()[-3], skip_vec.size()[-3], kernel_size=2, stride=2)
#                 # # upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2], skip_vec.size()[-1]))
#                 # hidden = upsample(hidden)
#                 clstm_in = torch.cat([hidden, skip_vec], 1)
#                 # # skip connection
#                 # if self.skip_mode == 'concat':
#                 #     clstm_in = torch.cat([hidden, skip_vec], 1)
#                 # elif self.skip_mode == 'sum':
#                 #     clstm_in = hidden + skip_vec
#                 # elif self.skip_mode == 'mul':
#                 #     clstm_in = hidden * skip_vec
#                 # elif self.skip_mode == 'none':
#                 #     clstm_in = hidden
#                 # else:
#                 #     raise Exception('Skip connection mode not supported !')
#             else:
#                 # self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2] * 2, hidden.size()[-1] * 2))
#                 # hidden = self.upsample(hidden)
#                 clstm_in = hidden
#
#         out_mask = self.conv_out(clstm_in)
#         # classification branch
#
#         return out_mask, hidden_list


class RSIS(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, kernel_size=3,dropout=0,down_num=3,base_dim=26):

        super(RSIS, self).__init__()
        self.kernel_size = kernel_size
        padding = 0 if self.kernel_size == 1 else 1
        self.dropout = dropout
        self.layer_num = down_num + 1
        # convlstms have decreasing dimension as width and height increase
        # skip_dims_out = [self.hidden_size, int(self.hidden_size / 2),
        #                  int(self.hidden_size / 4), int(self.hidden_size / 8)]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        self.base_dim = base_dim
        clstm_in_dim = self.base_dim * 2
        clstm_out_dim = self.base_dim
        # 4 is the number of deconv steps that we need to reach image size in the output
        for i in range(self.layer_num):
            if i == self.layer_num - 1:
                clstm_in_dim //= 2
                clstm_i = ConvLSTMCell(False, clstm_in_dim, clstm_out_dim, self.kernel_size, padding = padding)
            else:
                clstm_i = ConvLSTMCell(False, clstm_in_dim, clstm_out_dim, self.kernel_size, padding = padding)
                clstm_in_dim *= 2
                clstm_out_dim *= 2
            self.clstm_list.append(clstm_i)
        self.clstm_list = self.clstm_list[::-1]
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

        self.conv_out = nn.Conv2d(self.base_dim, 1, self.kernel_size, padding = padding)

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
                upsample = nn.ConvTranspose2d(hidden.size()[-3], skip_vec.size()[-3], kernel_size=2, stride=2)
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




if __name__ == "__main__":
    t = torch.ones((32, 1, 256, 256))
    model = FeatureExtractor(base_dims = 64,
                             args_backbone = Unet_backbone,
                             block_num = 2,
                             down_num = 3)
    model2= RSIS()
    # model3= RSIS(FeatureExtractor(),None)

    # out = model(t)
    # mask,hidden_list = model3(t)
    # out2 = model2(t)
    # flops, params = profile(model2,inputs = (out,None))
    # print(flops,params)
    stat(model, input_size = (1, 256, 256))
    # stat(model2, input_size = (1, 256, 256))
    # model_structure(model)
    # for i in range(5):
        # print(out[i].shape)
    #     # print(out2[i].shape)
    # print(mask.shape)
    n_parameters1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters1 / 1.e6))
    print('number of params (M): %.2f' % (n_parameters2 / 1.e6))