# imports
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


# https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html


class ModelConfig:
    def __init__(self):
        self.latent_size = 20
        self.input_size = 28 * 28
        self.dropout = 0.5
        self.initialization = 'normal'
        self.normalization = 'group'
        self.encoder_layers = [1024, 1024]
        self.decoder_layers = [1024, 1024]
        self.activation = 'geglu'


