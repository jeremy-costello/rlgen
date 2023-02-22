import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from .model_utils import get_activation


class NormalArtist(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.activation = get_activation(model_config.activation)

        self.layers = nn.ModuleList([])
        for (in_feat, out_feat) in zip([model_config.input_size] + model_config.encoder_layers[:-1],
                                       model_config.encoder_layers):
            self.layers.append(nn.ModuleList([nn.Linear(in_feat, out_feat),
                                              self.activation,
                                              nn.Dropout(p=model_config.dropout)]))
    
    def forward(self, state):
        pass


class MultiVariateNormalArtist(nn.Module):
    def __init__(self, model_config, env):
        super().__init__()

        self.activation = get_activation(model_config.activation)

        self.layers = nn.ModuleList([])
        for (in_feat, out_feat) in zip([model_config.input_size] + model_config.encoder_layers[:-1],
                                       model_config.encoder_layers):
            self.layers.append(nn.ModuleList([nn.Linear(in_feat, out_feat),
                                              self.activation,
                                              nn.Dropout(p=model_config.dropout)]))

        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.observation_space.shape[0]
        self.mean_layer = nn.Linear(model_config.encoder_layers[-1], self.da)
        self.cholesky_layer = nn.Linear(model_config.encoder_layers[-1], (self.da * (self.da + 1)) // 2)

    def forward(self, state):
        device = state.device
        B = state.size(0)
        ds = self.ds
        da = self.da
        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)

        x = state
        for (linear, activation, dropout) in self.layers:
            x = linear(x)
            x = activation(x)
            x = dropout(x)
        
        # is sigmoid here needed?
        mean = F.sigmoid(self.mean_layer(x))
        mean = action_low + (action_high - action_low) * mean
        
        cholesky_vector = self.cholesky_layer(x)
        cholesky_diag_index = torch.arange(da, dtype=torch.long) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        # softplus or gelu below?
        cholesky_vector[:, cholesky_diag_index] = F.gelu(cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=da, col=da, offset=0)
        cholesky = torch.zeros(size=(B, da, da), dtype=torch.float32).to(device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector

        return mean, cholesky
    
    def action(self, state):
        mean, cholesky = self.forward(state[None, ...])
        action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
        action = action_distribution.sample()
        return action[0]
