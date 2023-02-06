import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import get_activation


class Encoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.conditional = model_config.conditional
        self.num_labels = model_config.num_labels

        self.activation = get_activation(model_config.activation)

        self.layers = nn.ModuleList([])
        for (in_feat, out_feat) in zip([model_config.input_size] + model_config.encoder_layers[:-1],
                                       model_config.encoder_layers):
            self.layers.append(nn.ModuleList([nn.Linear(in_feat, out_feat),
                                              self.activation,
                                              nn.Dropout(p=model_config.dropout)]))

        self.linear_mean = nn.Linear(model_config.encoder_layers[-1], model_config.latent_size)
        self.linear_log_var = nn.Linear(model_config.encoder_layers[-1], model_config.latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            c = F.one_hot(torch.LongTensor(c), num_classes=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        
        for (linear, activation, dropout) in self.layers:
            x = linear(x)
            x = activation(x)
            x = dropout(x)
        
        mean = self.linear_mean(x)
        log_var = self.linear_log_var(x)

        return mean, log_var
