import math
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from artist import NormalArtist, MultiVariateNormalArtist
from critic import Critic


class Model(object):
    def __init__(self, model_config):
        if model_config.conditional:
            assert isinstance(model_config.num_labels, int)
            assert model_config.num_labels > 0
        
        self.input_size = model_config.input_size
        self.latent_size = model_config.latent_size
        self.sqrt2pi = math.sqrt(2 * math.pi)

        assert isinstance(model_config.latent_size, int)
        assert isinstance(model_config.encoder_layers, list)
        assert isinstance(model_config.decoder_layers, list)

        self.artist = NormalArtist(model_config)
        self.target_artist = NormalArtist(model_config)
        self.critic = Critic(model_config)
        self.target_critic = Critic(model_config)

        self.artist.apply(self._init_weights)
        self.target_artist.apply(self._init_weights)
        self.critic.apply(self._init_weights)
        self.target_critic.apply(self._init_weights)

        self._update_param()

        # MPO parameters
        self.eta = np.random.rand()
        self.alpha_mu = 0.0
        self.alpha_sigma = 0.0
    
    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _update_param(self):
        for target_param, param in zip(self.target_artist.parameters(), self.artist.parameters()):
            target_param.data.copy_(param.data)
    
    def training_step(self, x, c=None):
        with torch.no_grad():
            baseline_mean, baseline_log_var = self.target_artist.forward(x, c)
            baseline_distribution = Normal(baseline_mean, torch.exp(0.5 * baseline_log_var))
        
        mean, log_var = self.artist(x)

        distribution = Normal(mean, torch.exp(0.5 * log_var))
        action = distribution.rsample()
        z = (1 / (torch.exp(0.5 * log_var) * self.sqrt2pi)) \
            * torch.exp(-0.5 * (action - mean) ** 2 / torch.exp(log_var))
        
        # c is kind of like a state. could be text embedding for guidance
        recon_x = self.critic.forward(z, c)

        reward = self.loss_fn(x, recon_x, baseline_mean, baseline_log_var, mean, log_var)
        return reward
    
    def loss_fn(self, x, recon_x, baseline_mean, baseline_log_var, mean, log_var, epsilon=1e-8):
        baseline_sigma = torch.exp(0.5 * baseline_log_var)
        sigma = torch.exp(0.5 * log_var)

        recon_loss = -1.0 * torch.sum(x * torch.log(recon_x + epsilon), dim=-1)
        kl_loss = self.kl_divergence(baseline_mean, baseline_sigma, mean, sigma)

        reward = recon_loss + kl_loss
        return reward.mean()

    def kl_divergence(self, p_mean, p_sigma, q_mean, q_sigma):
        var_division = torch.sum(p_sigma ** 2 / q_sigma ** 2, dim=-1)
        diff_term = torch.sum((q_mean - p_mean) ** 2 / q_sigma ** 2, dim=-1)
        logvar_det_division = torch.sum(torch.log(q_sigma ** 2) - torch.log(p_sigma ** 2), dim=-1)
        return 0.5 * (var_division + diff_term - self.latent_size + logvar_det_division)
