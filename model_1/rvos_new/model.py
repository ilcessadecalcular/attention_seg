import torch
import torch.nn as nn
from clstm import ConvLSTMCell, ConvLSTMCellMask
import argparse
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn
import math
from vision import VGG16, ResNet34, ResNet50, ResNet101
import sys
sys.path.append("..")
# from tools.utils import get_skip_dims
from torchstat import stat


def get_skip_dims(model_name):
    if model_name == 'resnet50' or model_name == 'resnet101':
        skip_dims_in = [2048,1024,512,256,64]
    elif model_name == 'resnet34':
        skip_dims_in = [512,256,128,64,64]
    elif model_name =='vgg16':
        skip_dims_in = [512,512,256,128,64]

    return skip_dims_in

class FeatureExtractor(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self,args):
        super(FeatureExtractor,self).__init__()
        skip_dims_in = get_skip_dims(args.base_model)

        if args.base_model == 'resnet34':
            self.base = ResNet34()
            # self.base.load_state_dict(models.resnet34(pretrained=False).state_dict())
        elif args.base_model == 'resnet50':
            self.base = ResNet50()
            # self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())
        elif args.base_model == 'resnet101':
            self.base = ResNet101()
            # self.base.load_state_dict(models.resnet101(pretrained=True).state_dict())
        elif args.base_model == 'vgg16':
            self.base = VGG16()
            # self.base.load_state_dict(models.vgg16(pretrained=True).state_dict())

        else:
            raise Exception("The base model you chose is not supported !")

        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        self.padding = 0 if self.kernel_size == 1 else 1

        self.sk5 = nn.Conv2d(skip_dims_in[0],int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk4 = nn.Conv2d(skip_dims_in[1],int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk3 = nn.Conv2d(skip_dims_in[2],int(self.hidden_size/2),self.kernel_size,padding=self.padding)
        self.sk2 = nn.Conv2d(skip_dims_in[3],int(self.hidden_size/4),self.kernel_size,padding=self.padding)

        self.bn5 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn4 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn3 = nn.BatchNorm2d(int(self.hidden_size/2))
        self.bn2 = nn.BatchNorm2d(int(self.hidden_size/4))

    def forward(self,x,semseg=False, raw = False):
        x5,x4,x3,x2,x1 = self.base(x)

        x5_skip = self.bn5(self.sk5(x5))
        x4_skip = self.bn4(self.sk4(x4))
        x3_skip = self.bn3(self.sk3(x3))
        x2_skip = self.bn2(self.sk2(x2))

        if semseg:
            return x5
        elif raw:
            return x5, x4, x3, x2, x1
        else:
            #return total_feats
            del x5, x4, x3, x2, x1, x
            return x5_skip, x4_skip, x3_skip, x2_skip

class RSIS(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, args):

        super(RSIS,self).__init__()
        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        padding = 0 if self.kernel_size == 1 else 1

        self.dropout = args.dropout
        self.skip_mode = args.skip_mode

        # convlstms have decreasing dimension as width and height increase
        skip_dims_out = [self.hidden_size, int(self.hidden_size/2),
                         int(self.hidden_size/4),int(self.hidden_size/8)]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 4 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim*=2

            clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)

        self.conv_out = nn.Conv2d(skip_dims_out[-1], 1,self.kernel_size, padding = padding)

        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim+=sk


   
    def forward(self, skip_feats, prev_hidden_temporal):
                  
        clstm_in = skip_feats[0]
        skip_feats = skip_feats[1:]
        hidden_list = []

        for i in range(len(skip_feats)+1):

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

            if prev_hidden_temporal is None:
                state = self.clstm_list[i](clstm_in, None)

            else:
                state = self.clstm_list[i](clstm_in, prev_hidden_temporal[i])
            # state = self.clstm_list[i](clstm_in, prev_hidden_temporal[i])

            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            # apply skip connection
            if i < len(skip_feats):

                skip_vec = skip_feats[i]
                upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2],skip_vec.size()[-1]))
                hidden = upsample(hidden)
                # skip connection
                if self.skip_mode == 'concat':
                    clstm_in = torch.cat([hidden,skip_vec],1)
                elif self.skip_mode == 'sum':
                    clstm_in = hidden + skip_vec
                elif self.skip_mode == 'mul':
                    clstm_in = hidden*skip_vec
                elif self.skip_mode == 'none':
                    clstm_in = hidden
                else:
                    raise Exception('Skip connection mode not supported !')
            else:
                self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2]*2,hidden.size()[-1]*2))
                hidden = self.upsample(hidden)
                clstm_in = hidden

        out_mask = self.conv_out(clstm_in)
        # classification branch

        return out_mask, hidden_list
        
class Rvosnet(nn.Module):
    def __init__(self,args):
        super(Rvosnet, self).__init__()

        self.encoder = FeatureExtractor(args)
        self.decoder = RSIS(args)
        self.only_temporal = True

    def forward(self,x):
        n, t, c, h, w = x.size()
        prev_hidden_temporal = None
        out_mask_list = []
        for i in range(t):
            input = x[:,i,:,:,:]
            # hidden_spatial = None
            # hidden_temporal = None

            feats = self.encoder(input)
            # if prev_hidden_temporal is not None:
            #     hidden_temporal = prev_hidden_temporal
            #     if self.only_temporal:
            #         hidden_spatial = None
            # else:
            #     hidden_temporal = None

            hidden_temporal = prev_hidden_temporal

            out_mask, hidden = self.decoder(feats, hidden_temporal)
            upsample_match = nn.UpsamplingBilinear2d(size = (x.size()[-2], x.size()[-1]))
            out_mask = upsample_match(out_mask)
            out_mask_list.append(out_mask)
            # hidden_tmp = []
            # for ss in range(len(hidden)):
            #     hidden_tmp.append(hidden[ss][0])
            # hidden_spatial = hidden
            prev_hidden_temporal = hidden

        real_out_mask = torch.stack(out_mask_list,1)
        return real_out_mask


def get_args_parser():
    parser = argparse.ArgumentParser('Medical segmentation ', add_help = False)
    parser.add_argument('--cpu', dest = 'use_gpu', action = 'store_false')
    parser.set_defaults(use_gpu = False)
    parser.add_argument('-base_model', dest = 'base_model', default = 'resnet34',
                        choices = ['resnet101', 'resnet50', 'resnet34', 'vgg16'])
    parser.add_argument('-skip_mode', dest = 'skip_mode', default = 'concat',
                        choices = ['sum', 'concat', 'mul', 'none'])
    parser.add_argument('-hidden_size', dest = 'hidden_size', default = 128, type = int)
    parser.add_argument('-kernel_size', dest = 'kernel_size', default = 3, type = int)
    parser.add_argument('-dropout', dest = 'dropout', default = 0.0, type = float)

    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    t = torch.ones((10, 1, 384, 256))
    model_in = FeatureExtractor(args)
    model_out = RSIS(args)
    n_parameters1 = sum(p.numel() for p in model_in.parameters() if p.requires_grad)
    stat(model_in, input_size = (1, 384, 256))
    print('number of params (M): %.2f' % (n_parameters1 / 1.e6))
    n_parameters2 = sum(p.numel() for p in model_out.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters2 / 1.e6))
    mid = model_in(t)
    out, _ = model_out(mid, None)
    for i in range(4):
        print(mid[i].shape)
    print(out.shape)

    model_all = Rvosnet(args)
    al = torch.ones((10, 60, 1, 256, 256))
    n_parameters1 = sum(p.numel() for p in model_all.parameters() if p.requires_grad)
    # stat(model_all, input_size = (5,1, 256, 256))
    print('number of params (M): %.2f' % (n_parameters1 / 1.e6))
    out_all = model_all(al)
    print(out_all.shape)
