import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import get_activation


class Critic(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.conditional = model_config.conditional
        self.num_labels = model_config.num_labels

        self.activation = get_activation(model_config.activation)

        latent_size = model_config.latent_size + int(self.conditional) * self.num_labels

        self.layers = nn.ModuleList([])
        for (in_feat, out_feat) in zip([latent_size] + model_config.encoder_layers,
                                       model_config.encoder_layers + [model_config.input_size]):
            self.layers.append(nn.ModuleList([nn.Linear(in_feat, out_feat),
                                              self.activation,
                                              nn.Dropout(p=model_config.dropout)]))

    def forward(self, x, c=None):
        if self.conditional:
            c = F.one_hot(torch.LongTensor(c), num_classes=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        
        for (linear, activation, dropout) in self.layers:
            x = linear(x)
            x = activation(x)
            x = dropout(x)

        return x
