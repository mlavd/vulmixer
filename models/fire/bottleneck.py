"""FIRE Bottlneck Module.

This module provides different bottlenecks to use. If you
would like to add additional options, import the registry
from this file and add your own. The option may then be used
in a configuration file.
"""
from .util import ModuleRegistry
import torch
import torch.nn as nn

class LinearBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bottleneck = nn.Linear(
            in_features=config.embedding_size,
            out_features=config.hidden_dim,
        )
    
    def forward(self, x):
        return self.bottleneck(x)
    
class TruncateBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
    
    def forward(self, x):
        return x[:, :, :self.hidden_dim]


bottlenecks = ModuleRegistry('bottleneck')
bottlenecks.register('linear', LinearBottleneck)
bottlenecks.register('truncate', TruncateBottleneck)

