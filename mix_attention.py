import torch.nn as nn
import torch

import torch.nn.functional as F

from einops.layers.torch import Rearrange

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class Mix_Attention(nn.Module):
    def __init__(self, dim, num_heads=3, v_head=1, e_v=1, e_c=1, k_size=31):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.e_v = e_v
        self.e_c = e_c
        self.v_head = int(v_head / 3 * num_heads)
        self.c_head = num_heads - self.v_head

        dim_v = int(dim * self.v_head / self.num_heads)
        dim_c = int(dim * self.c_head / self.num_heads)

        # VIT head
        head_dim = dim * e_v // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim_v, int(dim_v * e_v) * 3, bias=True)
        self.proj = nn.Linear(int(dim_v * e_v), dim_v)

        # Conv head
        padding = calc_same_padding(k_size)
        inner_dim = dim_c * e_c
        self.net = nn.Sequential(

            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim_c, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=k_size, padding=padding),
            nn.BatchNorm1d(inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim_c, 1),
            Rearrange('b c n -> b n c')
        )


    def forward(self, x):
        B, N, C = x.shape
        C_v = int(C * self.v_head / self.num_heads)
        C_c = int(C * self.c_head / self.num_heads)
        X_v = x[:, :, :C_v]
        X_c = x[:, :, -C_c:]

        qkv = self.qkv(X_v).reshape(B, N, 3, self.v_head, int(C // self.num_heads * self.e_v)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_v = (attn @ v).transpose(1, 2).reshape(B, N, int(C_v * self.e_v))
        x_v = self.proj(x_v)

        x_c = self.net(X_c)

        x = torch.cat([x_v, x_c], dim=2)

        return x

class Vit_Attention(nn.Module):
    def __init__(self, dim, num_heads=3, e_v=1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.e_v = e_v
        head_dim = dim * e_v // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, int(dim * e_v) * 3, bias=True)
        self.proj = nn.Linear(int(dim * e_v), dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, int(C // self.num_heads * self.e_v)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_v = (attn @ v).transpose(1, 2).reshape(B, N, int(C * self.e_v))
        x_v = self.proj(x_v)

        return x_v

class Conv_Attention(nn.Module):
    def __init__(self, dim, num_heads=3, v_head=1, e_v=1, e_c=1, k_size=31):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # Conv head
        padding = calc_same_padding(k_size)
        inner_dim = dim * e_c
        self.net = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=k_size, padding=padding),
            nn.BatchNorm1d(inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c')
        )

    def forward(self, x):
        x = self.net(x)
        return x

