import torch.nn as nn
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# class PPM(nn.Module):
#     def __init__(self, pooling_sizes = (1, 3, 5)):
#         super().__init__()
#         self.layer = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size = (size, size)) for size in pooling_sizes])
#
#     def forward(self, feat):
#         b, c, h, w = feat.shape
#         output = [layer(feat).view(b, c, -1) for layer in self.layer]
#         output = torch.cat(output, dim = -1)
#         return output


# Efficient self attention
class ESA_layer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size = 1, stride = 1, padding = 0, bias = False)

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)

        # self.ppm = PPM(pooling_sizes = (1, 3, 5))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, cat_x):
        # input x (b, c, h, w)
        b, c1, h, w = x.shape
        b, c2, h, w = cat_x.shape
        x = rearrange(x, 'b c1 h w -> b c1 (h w)')
        cat_x = rearrange(cat_x, 'b c2 h w -> b c2 (h w)')
        q = self.to_q(x)
        k, v = self.to_kv(cat_x).chunk(2, dim = -1)
        q = rearrange(q, 'b c1 (head d)  -> b head c1 d', head = self.heads)
        k = rearrange(k, 'b c2 (head d)  -> b head c2 d', head = self.heads)
        v = rearrange(v, 'b c2 (head d)  -> b head c2 d', head = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b head n d -> b n (head d)')

        return self.to_out(out)

        # q, k, v = self.to_qkv(x).chunk(3, dim = 1)  # q/k/v shape: (b, inner_dim, h, w)
        # q = rearrange(q, 'b (head d) h w -> b head (h w) d', head = self.heads)  # q shape: (b, head, n_q, d)
        #
        # k, v = self.ppm(k), self.ppm(v)  # k/v shape: (b, inner_dim, n_kv)
        # k = rearrange(k, 'b (head d) n -> b head n d', head = self.heads)  # k shape: (b, head, n_kv, d)
        # v = rearrange(v, 'b (head d) n -> b head n d', head = self.heads)  # v shape: (b, head, n_kv, d)
        #
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # shape: (b, head, n_q, n_kv)
        #
        # attn = self.attend(dots)
        #
        # out = torch.matmul(attn, v)  # shape: (b, head, n_q, d)
        # out = rearrange(out, 'b head n d -> b n (head d)')
        # return self.to_out(out)


class ESA_blcok(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 16, mlp_dim = 64, dropout = 0.):
        super().__init__()
        self.ESAlayer = ESA_layer(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))

    def forward(self, x, cat_x):
        b, c, h, w = x.shape
        # out = rearrange(x, 'b c h w -> b (h w) c')
        out = rearrange(x, 'b c h w -> b c (h w)')
        out = self.ESAlayer(x, cat_x) + out
        out = self.ff(out) + out
        out = rearrange(out, 'b c (h w) -> b c h w', h = h)

        return out

if __name__ == "__main__":
    x  = torch.ones((10,10,24,24))
    cat_x= torch.ones((10,20,24,24))
    net = ESA_blcok(24*24)
    out = net(x,cat_x)
    print(out.shape)