import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation_name):
    if activation_name.lower() == 'geglu':
        return GeGLU()
    elif activation_name.lower() == 'gelu':
        return nn.GELU()
    else:
        raise ValueError("Invalid activation name!")


def configure_optimizers():
    pass
