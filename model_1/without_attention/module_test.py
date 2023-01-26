import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torchstat import stat
import math


class Down1(nn.Module):
    """先下采样，再改变维度（再去多尺度）"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down2(nn.Module):
    """直接通过卷积完成下采样（再去多尺度）"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2,padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

if __name__ == "__main__":
    t = torch.ones((32, 1, 256, 256))
    model = Down1(in_channels = 64,
                  out_channels = 128)
    model2 = Down2(in_channels = 64,
                  out_channels = 128)
    n_parameters1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    # print("Model = %s" % str(model_without_ddp))
    stat(model, input_size = (64, 256, 256))
    stat(model2, input_size = (64, 256, 256))

    print('number of params (M): %.2f' % (n_parameters1 / 1.e6))
    print('number of params (M): %.2f' % (n_parameters2 / 1.e6))
    "down1比down2 花销大"