import torch
import torch.nn as nn


class Unet_backbone(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def layer_block_unet(in_channels: int,
                out_channels: int,
                blocks: int,
                ) -> nn.Sequential:
    layers = []
    layers.append(
        Unet_backbone(
            in_channels, out_channels,
        )
    )
    for _ in range(1, blocks):
        layers.append(
            Unet_backbone(
                out_channels,
                out_channels,
            )
        )
    return nn.Sequential(*layers)

class Down_unet(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int,
                 out_channels: int,
                 blocks: int,
                 ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            layer_block_unet(in_channels, out_channels, blocks)
        )

    def forward(self, x):
        return self.maxpool_conv(x)