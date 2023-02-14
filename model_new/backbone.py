import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F


class Basicblock(nn.Module):

    def __init__(self, in_channels, out_channels, baseWidth, scale = 4):
        super().__init__()

        width = int(math.floor(in_channels * (baseWidth / 64)))
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


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth = 26, scale = 4):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            # nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1])
        self.layer3 = self._make_layer(block, 512, layers[2])
        self.layer4 = self._make_layer(block, 1024, layers[3])
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks):
        # downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.AvgPool2d(kernel_size=stride, stride=stride,
        #                      ceil_mode=True, count_include_pad=False),
        #         nn.Conv2d(self.inplanes, planes * block.expansion,
        #                   kernel_size=1, stride=1, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )
        #
        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample=downsample,
        #                     stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        # self.inplanes = planes * block.expansion
        layers.append(nn.Conv2d(planes // 2, planes, kernel_size = 2, stride = 2, padding = 0, bias = False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace = True))

        for i in range(0, blocks):
            layers.append(block(planes, planes, baseWidth = self.baseWidth, scale = self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        f1 = self.conv1(x)  # 1, 32

        f2 = self.layer1(f1)  # 1/2, 64
        f4 = self.layer2(f2)  # 1/4, 128
        f8 = self.layer3(f4)    # 1/8, 256
        f16 = self.layer4(f8)   # 1/16, 512

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return f1,f2,f4,f8,f16


if __name__ == '__main__':
    images = torch.rand(10, 1, 256, 256).cuda(0)
    model = Res2Net(Basicblock, [2, 2, 2, 2], baseWidth=26, scale=4)
    model = model.cuda(0)
    out  = model(images)
    for i in range(5):
        print(out[i].shape)
    # print(model(images).size())
