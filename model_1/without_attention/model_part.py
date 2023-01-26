import torch
import torch.nn as nn
import math

class Unet_backbone(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Res2net_backbone(nn.Module):

    def __init__(self, in_channels, out_channels, baseWidth, init_feature, scale = 3):
        super().__init__()

        width = int(math.floor(in_channels * (baseWidth / init_feature)))
        self.conv1 = nn.Conv2d(in_channels, width * scale, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size = 3, stride = 1, padding = 1, bias = False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, out_channels, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace = True)
        # self.downsample = downsample
        # self.stype = stype
        self.scale = scale
        self.width = width

        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace = True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


def layer_block_resnet(in_channels: int,
                       out_channels: int,
                       baseWidth: int,
                       init_feature: int,
                       scale: int,
                       blocks: int,
                       ) -> nn.Sequential:
    layers = []
    layers.append(
        Res2net_backbone(
            in_channels = in_channels,
            out_channels = out_channels,
            baseWidth = baseWidth,
            init_feature = init_feature,
            scale = scale
        )
    )
    for _ in range(1, blocks):
        layers.append(
            Res2net_backbone(
                in_channels = out_channels,
                out_channels = out_channels,
                baseWidth = baseWidth,
                init_feature = init_feature,
                scale = scale
            )
        )
    return nn.Sequential(*layers)


class Down_resnet(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 baseWidth: int,
                 init_feature: int,
                 scale: int,
                 blocks: int,
                 ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            layer_block_resnet(in_channels = out_channels,
                               out_channels = out_channels,
                               baseWidth = baseWidth,
                               init_feature = init_feature,
                               scale = scale,
                               blocks = blocks)
        )

    def forward(self, x):
        return self.maxpool_conv(x)