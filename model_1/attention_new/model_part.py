import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import Softmax, Parameter


class Unet_backbone(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Res2net_backbone_f(nn.Module):

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


class Res2net_backbone(nn.Module):

    def __init__(self, in_channels, out_channels, baseWidth, init_feature, in_dim, scale = 3, key_dim = 32, ):
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

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim = -1)
        self.key_dim = key_dim
        self.transform = nn.Linear(in_dim, self.key_dim)

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

        m_batchsizeq, Cq, height, width = x.size()
        m_batchsizek, Ck, _, _ = out.size()
        proj_query = self.transform(x.view(m_batchsizeq * Cq, -1)).view(m_batchsizeq, Cq, -1)
        proj_key = self.transform(out.view(m_batchsizek * Ck, -1)).view(m_batchsizek, Ck, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim = True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = out.view(m_batchsizek, Ck, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsizeq, Cq, height, width)

        out = self.gamma * out + x

        # out = self.conv3(out)
        # out = self.bn3(out)
        #
        # out += residual
        # out = self.relu(out)

        return out


class Down_resnet(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 baseWidth: int,
                 init_feature: int,
                 scale: int,
                 blocks: int,
                 in_dim: int,
                 ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            self.layer_block_resnet(in_channels = out_channels,
                                    out_channels = out_channels,
                                    baseWidth = baseWidth,
                                    init_feature = init_feature,
                                    scale = scale,
                                    blocks = blocks,
                                    in_dim = in_dim)
        )

    def layer_block_resnet(self, in_channels: int,
                           out_channels: int,
                           baseWidth: int,
                           init_feature: int,
                           scale: int,
                           blocks: int,
                           in_dim: int
                           ):
        layers = []
        layers.append(
            Res2net_backbone(
                in_channels = in_channels,
                out_channels = out_channels,
                baseWidth = baseWidth,
                init_feature = init_feature,
                scale = scale,
                in_dim = in_dim
            )
        )
        for _ in range(1, blocks):
            layers.append(
                Res2net_backbone(
                    in_channels = out_channels,
                    out_channels = out_channels,
                    baseWidth = baseWidth,
                    init_feature = init_feature,
                    scale = scale,
                    in_dim = in_dim
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_resnet(nn.Module):
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
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
        self.channel_conv = Unet_backbone(in_channels, out_channels)
        self.conv = nn.Sequential(self.layer_block_resnet(in_channels = out_channels,
                                                          out_channels = out_channels,
                                                          baseWidth = baseWidth,
                                                          init_feature = init_feature,
                                                          scale = scale,
                                                          blocks = blocks)
                                  )

    def layer_block_resnet(self, in_channels: int,
                           out_channels: int,
                           baseWidth: int,
                           init_feature: int,
                           scale: int,
                           blocks: int,
                           ):
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

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(self.channel_conv(x))


class First_resnet(nn.Module):
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
            self.layer_block_resnet(in_channels = out_channels,
                                    out_channels = out_channels,
                                    baseWidth = baseWidth,
                                    init_feature = init_feature,
                                    scale = scale,
                                    blocks = blocks)
        )

    def layer_block_resnet(self, in_channels: int,
                           out_channels: int,
                           baseWidth: int,
                           init_feature: int,
                           scale: int,
                           blocks: int,
                           ):
        layers = []
        layers.append(
            Res2net_backbone_f(
                in_channels = in_channels,
                out_channels = out_channels,
                baseWidth = baseWidth,
                init_feature = init_feature,
                scale = scale
            )
        )
        for _ in range(1, blocks):
            layers.append(
                Res2net_backbone_f(
                    in_channels = out_channels,
                    out_channels = out_channels,
                    baseWidth = baseWidth,
                    init_feature = init_feature,
                    scale = scale
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.maxpool_conv(x)
