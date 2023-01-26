import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torchstat import stat
import math
from model_part import *


class FeatureExtractor_unet(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''

    def __init__(self, init_feature,
                 block_num,
                 down_num):
        super(FeatureExtractor_unet, self).__init__()
        self.layer_num = down_num + 1
        self.layer_list = nn.ModuleList()
        self.feature_dim = init_feature
        for i in range(self.layer_num):
            if i == 0:
                encoder_i = layer_block_unet(1,
                                             self.feature_dim,
                                             blocks = block_num)
            else:
                encoder_i = Down_unet(self.feature_dim,
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


if __name__ == "__main__":
    t = torch.ones((32, 1, 256, 256))
    model = FeatureExtractor_unet(init_feature = 64,
                                  block_num = 2,
                                  down_num = 3)
    n_parameters1 = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stat(model, input_size = (1, 256, 256))

    print('number of params (M): %.2f' % (n_parameters1 / 1.e6))

