import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT
from einops import rearrange
from abc import ABC
from torch import einsum

def norm(x):
    return (1 - torch.exp(-x)) / (1 + torch.exp(-x))


def norm_(x):
    import numpy as np
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


class PreNorm(nn.Module, ABC):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module, ABC):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


def Seg():
    dict = {0: 0, 1: 1, 2: 8, 3: 16, 4: 9, 5: 2, 6: 3, 7: 10, 8: 17,
            9: 24, 10: 32, 11: 25, 12: 18, 13: 11, 14: 4, 15: 5, 16: 12,
            17: 19, 18: 26, 19: 33, 20: 40, 21: 48, 22: 41, 23: 34, 24: 27,
            25: 20, 26: 13, 27: 6, 28: 7, 29: 14, 30: 21, 31: 28, 32: 35,
            33: 42, 34: 49, 35: 56, 36: 57, 37: 50, 38: 43, 39: 36, 40: 29,
            41: 22, 42: 15, 43: 23, 44: 30, 45: 37, 46: 44, 47: 51, 48: 58,
            49: 59, 50: 52, 51: 45, 52: 38, 53: 31, 54: 39, 55: 46, 56: 53,
            57: 60, 58: 61, 59: 54, 60: 47, 61: 55, 62: 62, 63: 63}
    a = torch.zeros(1, 64, 1, 1)

    for i in range(0, 32):
        a[0, dict[i + 32], 0, 0] = 1

    return a


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """

    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp - 1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def _init_weights(self, init_linear='normal'):
        init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class FDA(nn.Module):

    def __init__(self):
        super(FDA, self).__init__()

        self.seg = Seg()

        self.high_band1 = Transformer(dim=256, depth=2, heads=4, dim_head=256 * 2, mlp_dim=128 * 2, dropout=0)
        self.low_band1 = Transformer(dim=256, depth=2, heads=4, dim_head=256 * 2, mlp_dim=128 * 2, dropout=0)

        self.band1 = Transformer(dim=256, depth=2, heads=4, dim_head=256 * 2, mlp_dim=128 * 2, dropout=0)
        self.spatial1 = Transformer(dim=192, depth=2, heads=4, dim_head=128 * 2, mlp_dim=64 * 2, dropout=0)

        self.mlp = MLP1D(192, 192 * 2, 1)

        self.pool1D = torch.nn.AdaptiveMaxPool1d(1)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, x):

        # Convert to YCBCR
        mean = torch.tensor(self.mean).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(self.std).view(1, 3, 1, 1).to(x.device)
        x = x * std + mean
        x = x * 255.0

        transform_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ]).T.to(x.device)
        offset = torch.tensor([0, 128.0, 128.0]).to(x.device)

        x = x.permute(0, 2, 3, 1)

        x_ycbcr = torch.matmul(x, transform_matrix) + offset
        x_ycbcr = x_ycbcr.permute(0, 3, 2, 1)
        x_ycbcr = torch.clamp(x_ycbcr, 0, 255)

        # DCT
        B, C, H, W = x_ycbcr.shape[0], x_ycbcr.shape[1], x_ycbcr.shape[2], x_ycbcr.shape[3]

        x_ycbcr = x_ycbcr.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 2, 4, 1, 3, 5)
        DCT_x = DCT.dct_2d(x_ycbcr, norm='ortho')
        DCT_x = DCT_x.reshape(B, H // 8, W // 8, -1).permute(0, 3, 1, 2)

        self.seg = self.seg.to(DCT_x.device)
        feat_y = DCT_x[:, 0:64, :, :] * self.seg
        feat_Cb = DCT_x[:, 64:128, :, :] * self.seg
        feat_Cr = DCT_x[:, 128:192, :, :] * self.seg

        high = torch.cat([feat_y[:, 32:, :, :], feat_Cb[:, 32:, :, :], feat_Cr[:, 32:, :, :]], 1)
        low = torch.cat([feat_y[:, :32, :, :], feat_Cb[:, :32, :, :], feat_Cr[:, :32, :, :]], 1)

        # Feature
        b, n, h, w = high.shape
        high = torch.nn.functional.interpolate(high, size=(16, 16))
        low = torch.nn.functional.interpolate(low, size=(16, 16))

        low = rearrange(low, 'b n h w -> b n (h w)')
        high = rearrange(high, 'b n h w -> b n (h w)')

        high = self.high_band1(high)
        low = self.low_band1(low)

        y_h, b_h, r_h = torch.split(high, 32, 1)
        y_l, b_l, r_l = torch.split(low, 32, 1)
        feat_y = torch.cat([y_l, y_h], 1)
        feat_Cb = torch.cat([b_l, b_h], 1)
        feat_Cr = torch.cat([r_l, r_h], 1)

        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)

        feat_DCT = self.band1(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = self.spatial1(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)

        # Weight
        weight_feat = feat_DCT
        feat_DCT = self.mlp(feat_DCT)
        feat_DCT = self.pool1D(feat_DCT).contiguous().view(feat_DCT.shape[0], -1)

        return feat_DCT, weight_feat



