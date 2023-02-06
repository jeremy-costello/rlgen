import torch
import torch.nn as nn
import torch.nn.functional as F


# https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html
class GeGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        self.proj = nn.Linear(d_in, d_out * 2)
    
    def forward(self, x: torch.Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def get_activation(activation_name):
    if activation_name.lower() == 'geglu':
        return GeGLU()
    elif activation_name.lower() == 'gelu':
        return nn.GELU()
    else:
        raise ValueError("Invalid activation name!")
