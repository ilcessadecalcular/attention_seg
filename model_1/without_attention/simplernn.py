
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    "加个bn"
    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # self.ln1 = nn.LayerNorm([mid_channels,256,256],elementwise_affine=False)
        # self.ln2 = nn.LayerNorm([mid_channels, 256, 256], elementwise_affine=False)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.


    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x

        out = self.conv2(self.bn1(self.relu(self.conv1(x))))

        # out = self.conv1(x)
        # out = self.ln1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.ln2(out)


        return identity + out * self.res_scale

def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, args,in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        self.use_gpu = args.use_gpu
        main = []
        self.out_channels = out_channels
        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.ReLU(inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, input_,temporal):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if temporal is None:
            state_size = [batch_size, self.out_channels] + list(spatial_size)
            if self.use_gpu:
                temporal = (
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                temporal = (
                    Variable(torch.zeros(state_size))
                )

        stacked_inputs = torch.cat([input_, temporal], 1)

        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(stacked_inputs)


