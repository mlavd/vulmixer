"""FIRE Classifier Module.

This module provides different classifiers to use. If you
would like to add additional options, import the registry
from this file and add your own. The option may then be used
in a configuration file.
"""
from .util import ModuleRegistry
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(
                in_features=config.hidden_dim,
                out_features=config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FIXME Move pooling into it's own layer
        if self.config.pooling == 'max':
            x = torch.max(x, dim=1)[0]
        
        return self.classifier(x)
    

class TCAMClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attention_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.classifier_projection = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_projection(inputs)
        attention = self.attention_projection(inputs)
        attention = F.softmax(attention, dim=-2)
        seq_repr = torch.sum(attention * features, dim=-2)
        logits = self.classifier_projection(seq_repr)
        return logits

classifiers = ModuleRegistry('classifier')
classifiers.register('linear', LinearClassifier)
classifiers.register('tcam', TCAMClassifier)

