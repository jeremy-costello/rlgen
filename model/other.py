from typing import Optional

from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor


# should normalization be affine?
class ConvNeXtBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        assert isinstance(num_channels, int)

        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels,
                               kernel_size=7,
                               padding=3,
                               groups=num_channels,
                               bias=False)
        
        # instance norm?
        self.norm = nn.GroupNorm(num_groups=num_channels,
                                 num_channels=num_channels,
                                 affine=False)
        
        self.conv2 = nn.Conv2d(in_channels=num_channels,
                               out_channels=4*num_channels,
                               kernel_size=1,
                               bias=False)
        
        # GeGLU?
        self.activation = nn.GELU()

        self.conv3 = nn.Conv2d(in_channels=4*num_channels,
                               out_channels=num_channels,
                               kernel_size=1,
                               bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)

        # 1x1 convolution for different input and output channels?
        x += identity
        return x


# make this more similar to a transformer stem?
class InputStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # should this be depthwise?
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=7,
                              padding=3,
                              bias=False)

        # is norm needed since no spatial resolution change?
        self.norm = nn.GroupNorm(num_groups=out_channels,
                                 num_channels=out_channels,
                                 affine=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


# make this more similar to a transformer stem?
class OutputStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        
        # is norm needed since no spatial resolution change?
        self.norm = nn.GroupNorm(num_groups=in_channels,
                                 num_channels=in_channels,
                                 affine=False)
        
        # should this be depthwise?
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=7,
                              padding=3,
                              bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        self.norm = nn.GroupNorm(num_groups=in_channels,
                                 num_channels=in_channels,
                                 affine=False)
        
        # should this be depthwise?
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        # upsample vs. interpolate?
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='nearest')
        
        # should norm be before or after upsample?
        self.norm = nn.GroupNorm(num_groups=in_channels,
                                 num_channels=in_channels,
                                 affine=False)
        
        # should this be depthwise?
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=7,
                              padding=3,
                              bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.norm(x)
        x = self.conv(x)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, num_channels, num_heads, num_layers, cond_dim):
        super().__init__()
        
        self.norm = nn.GroupNorm(num_groups=num_channels,
                                 num_channels=num_channels,
                                 affine=False)
        
        self.proj_in = nn.Conv2d(in_channels=num_channels,
                                 out_channels=num_channels,
                                 kernel_size=1,
                                 bias=False)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(num_channels=num_channels,
                                   num_heads=num_heads,
                                   head_dim=num_channels // num_heads,
                                   cond_dim=cond_dim)
            for _ in range(num_layers)]
        )

        self.proj_out = nn.Conv2d(in_channels=num_channels,
                                  out_channels=num_channels,
                                  kernel_size=1,
                                  bias=False)
    
    def forward(self, x: Tensor, cond: Tensor):
        _, _, height, width = x.shape
        identity = x

        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=height, w=width)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, cond)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)
        x = self.proj_out(x)

        x += identity
        return x
        

class BasicTransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, head_dim, cond_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dim)
        self.attn1 = CrossAttention(model_dim, model_dim, num_heads, head_dim)

        self.norm2 = nn.LayerNorm(model_dim)
        self.attn2 = CrossAttention(model_dim, cond_dim, num_heads, head_dim)

        self.norm3 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim)
    
    def forward(self, x: Tensor, cond: Tensor):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), cond=cond) + x
        x = self.ff(self.norm3(x)) + x
        return x


class CrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads, head_dim, cond_dim):
        super().__init__()

        self.scale = head_dim ** -0.5
        attn_dim = num_heads * head_dim

        # tensor shape: (batch, height * width, channels)
        self.q_linear = nn.Linear(model_dim, attn_dim, bias=False)
        self.q_rearrange = Rearrange('b i (h d) -> b i h d', h=num_heads, d=head_dim)

        self.k_linear = nn.Linear(cond_dim, attn_dim, bias=False)
        self.k_rearrange = Rearrange('b j (h d) -> b j h d', h=num_heads, d=head_dim)

        self.v_linear = nn.Linear(cond_dim, attn_dim, bias=False)
        self.v_rearrange = Rearrange('b j (h d) -> b j h d', h=num_heads, d=head_dim)

        self.to_out = nn.Linear(attn_dim, model_dim)
    
    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        if cond is None:
            cond = x
        
        q = self.q_rearrange(self.q_linear(x))
        k = self.k_rearrange(self.k_linear(x))
        v = self.v_rearrange(self.v_linear(x))

        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        out = rearrange(out, 'b i h d -> b i (h d)')
        out = self.to_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, model_dim, dim_mult, dropout=0.0):
        self.activation = GeGLU(model_dim, model_dim * dim_mult)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(model_dim * dim_mult, model_dim)
    
    def forward(self, x: Tensor):
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class GeGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        self.proj = nn.Linear(d_in, d_out * 2)
    
    def forward(self, x: torch.Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
