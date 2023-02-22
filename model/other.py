import functools

import torch.nn as nn
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

        out = self.conv1(x)
        out = self.norm(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.conv3(out)

        out += identity
        return out


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
