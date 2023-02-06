# imports
import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from encoder import Encoder
from decoder import Decoder


class ModelConfig:
    def __init__(self):
        self.conditional = True
        self.num_labels = 10
        self.latent_size = 20
        self.input_size = 28 * 28
        self.dropout = 0.5
        self.initialization = 'normal'
        self.normalization = 'group'
        self.encoder_layers = [1024, 1024]
        self.decoder_layers = [1024, 1024]
        self.activation = 'gelu'


class Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        if model_config.conditional:
            assert isinstance(model_config.num_labels, int)
            assert model_config.num_labels > 0
        
        self.input_size = model_config.input_size
        self.sqrt2pi = math.sqrt(2 * math.pi)

        assert isinstance(model_config.latent_size, int)
        assert isinstance(model_config.encoder_layers, list)
        assert isinstance(model_config.decoder_layers, list)

        self.actor = Encoder(model_config)
        self.target_actor = Encoder(model_config)
        self.environment = Decoder(model_config)

        self.apply(self._init_weights)
        self._update_param()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _update_param(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
    
    def forward(self, x, c=None):
        mean, log_var = self.encoder(x, c)

        latent_distribution = Normal(mean, torch.exp(0.5 * log_var))
        action = latent_distribution.rsample()
        z = (1 / (torch.exp(0.5 * log_var) * self.sqrt2pi)) \
            * torch.exp(-0.5 * (action - mean) ** 2 / torch.exp(log_var))
        
        recon_x = self.decoder(z, c)
    
    def kl_divergence(self, p_mean, p_sigma, q_mean, q_sigma):
        pass
