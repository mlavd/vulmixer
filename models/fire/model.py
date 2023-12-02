"""FIRE Model Module.

This module provides the base model for mixer-based files. All options are
selected with the configuration files.
"""
import torch
import torch.nn as nn
from .classifier import classifiers
from .bottleneck import bottlenecks
from .mixer import mixers
from .util import DropPath

ACTIVATIONS = {
    None: nn.Identity,
    'gelu': nn.GELU,
}

class FireMixerLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()

        # Add the mode and index to each of the configs
        config.token['mode'] = 'token'
        config.token['index'] = index
        config.feature['mode'] = 'feature'
        config.feature['index'] = index

        self.token_norm = nn.LayerNorm(config.hidden_dim)
        self.token_mixer = mixers.get(config.token)
        self.token_activation = ACTIVATIONS[config.token.activation]()

        self.feature_norm = nn.LayerNorm(config.hidden_dim)
        self.feature_dropout = DropPath(p=config.feature.dropout)
        self.feature_mixer = mixers.get(config.feature)
        self.feature_activation = ACTIVATIONS[config.feature.activation]()
    
    def forward(self, x):
        residual = x
        x = self.token_norm(x)
        x = self.token_mixer(x)
        x = self.token_activation(x + residual)

        residual = x
        x = self.feature_norm(x)
        x = self.feature_dropout(x)
        x = self.feature_mixer(x)
        x = self.feature_activation(x + residual)
        return x

class FireMixer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(*[
            FireMixerLayer(config, i)
            for i in range(config.depth)
        ])
        # config.token
        # config.feature
        # config.depth
        # Token Mixing
        # Channel Mixin
    
    def forward(self, x):
        return self.layers(x)


class FireSVD(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Bottleneck
        self.bottleneck = bottlenecks.get(config.bottleneck)

        # Mixing   
        self.mixers = nn.Sequential(*[
            FireMixer(mixer)
            for mixer in config.mixers
        ])

        # Classifier
        self.classifier = classifiers.get(config.classifier)
        print('classifier:', self.classifier)
    
    def get_bottleneck(self, config):
        if config.bottleneck.type == 'linear':
            return nn.Linear(
                in_features=config.input.embedding_size,
                out_features=config.bottleneck.hidden_dim)

        elif config.bottleneck.type == 'identity':
            return nn.Identity()
    
        else:
            raise NotImplementedError(f'Invalid bottleneck: {config.bottleneck.type}')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck(x)
        x = self.mixers(x)
        return self.classifier(x)

            # inputs = inputs[:, :, self.in_features]