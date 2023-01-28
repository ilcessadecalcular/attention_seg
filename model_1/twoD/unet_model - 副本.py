""" Full assembly of the parts to form the complete network """

from unet_parts import *
from clstm import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,init_feature, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, init_feature)
        self.down1 = Down(init_feature, init_feature*2)
        self.down2 = Down(init_feature*2, init_feature*4)
        factor = 2 if bilinear else 1
        self.down3 = Down(init_feature*4, init_feature*8// factor)
        self.up1 = Up(init_feature*8, init_feature*4 // factor, bilinear)
        self.up2 = Up(init_feature*4, init_feature*2 // factor, bilinear)
        self.up3 = Up(init_feature*2, init_feature // factor, bilinear)
        self.outc = OutConv(init_feature, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        return x

class RSIS_unet(nn.Module):
    def __init__(self, n_channels, n_classes,init_feature, bilinear=False):
        super(RSIS_unet, self).__init__()
        self.unet = UNet(n_channels, n_classes,init_feature, bilinear=False)
        self.clstm = ConvLSTMCell(init_feature, init_feature, kernel_size = 3, padding = 1)
        self.outc = OutConv(init_feature, n_classes)
    def forward(self,x):
        n, t, c, h, w = x.size()
        prev_hidden_temporal = None
        out_mask_list = []
        for i in range(t):
            input = x[:, i, :, :, :]
            feats = self.unet(input)
            hidden_temporal = prev_hidden_temporal
            hidden_state = self.clstm(feats, hidden_temporal)
            hidden = hidden_state[0]
            out_mask = self.outc(hidden)
            out_mask_list.append(out_mask)
            prev_hidden_temporal = hidden_state

        real_out_mask = torch.stack(out_mask_list, 1)
        return real_out_mask


class OnlyUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(OnlyUnet, self).__init__()
        # self.down=nn.Conv2d(in_feat,mid_feat,3,stride=4,padding=1)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.unet_seg = UNet(self.n_channels, self.n_classes, self.bilinear)
    def forward(self, x):
        #x:b,d,c,h,w
        input = x[0, :, :, :, :]
        #input:d,c,h,w
        out = self.unet_seg(input)
        real_out = out.unsqueeze(0)

        return real_out

    def init_weights(self):
        #logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std = 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def get_seg_model(n_channels, n_classes, bilinear=False, **kwargs):
    model = OnlyUnet(n_channels, n_classes, bilinear=False, **kwargs)
    model.init_weights()

    return model


if __name__ == "__main__":
    model = UNet(1,1,64)
    n_parameters1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # n_parameters2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters1 / 1.e6))
    # t = torch.ones((100, 1, 256, 256))
    # out = model(t)
    # print(out.shape)
    model2 =RSIS_unet(1,1,64)
    n_parameters2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    # n_parameters2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters2 / 1.e6))
    t = torch.ones((10,10, 1, 384, 256))
    out = model2(t)
    print(out.shape)