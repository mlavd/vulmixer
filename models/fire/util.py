"""FIRE Utils Module.

This module provides utilities that don't readily fit in other places.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModuleRegistry:
    def __init__(self, name):
        self.modules = {
            'identity': nn.Identity
        }
        self.name = name

    def register(self, name, callable):
        self.modules[name] = callable

    def get(self, config):
        if config.type not in self.modules:
            raise NotImplementedError(f'Invalid {self.name}: {config.type}')

        return self.modules[config.type](config)


class MlpLayer(nn.Module):
    # FIXME Document
    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.Sequential(*[
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(), # FIXME Make this configurable
            nn.Linear(intermediate_dim, hidden_dim)
        ])

    def forward(self, x):
        return self.layers(x)


class SparseMLP(nn.Module):
    def __init__(self, W, H, channels):
        super().__init__()
        # FIXME Figure out why this was asserted
        # assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(1)
        self.proj_h = nn.Conv2d(H, H, (1, 1))
        self.proj_w = nn.Conv2d(W, W, (1, 1))
        self.fuse = nn.Conv2d(channels*3, channels, (1,1), (1,1), bias=False)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.activation(self.BN(x))
        x_h = self.proj_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proj_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.fuse(torch.cat([x, x_h, x_w], dim=1))
        x = x.squeeze()
        return x

class BernoulliSamplingNLP:
    def __init__(self, max_len=100, prob=1):
        self.max_len = max_len
        self.prob = prob

    def __call__(self, inputs):
        n_samples, max_seq_len, embedding_dim = inputs.size(0), inputs.size(1), inputs.size(-1)
        Benoulli = torch.distributions.bernoulli.Bernoulli(self.prob)
        masks = Benoulli.sample((n_samples, max_seq_len)).unsqueeze(-1).repeat(1, 1, embedding_dim).bool().cuda()
        inputs = F.softmax(inputs.masked_fill(~masks, -np.inf), dim=-2)
        return inputs

class MHTCA(nn.Module):
    def __init__(self, n_head, max_seq_len, embedding_dim, prob, kernel_size, dilation, padding):
        # FIXME Document
        super().__init__()
        assert max_seq_len % n_head == 0, 'max_seq_len must be divisible by the n_head.'

        # FIXME clean this all up
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.input_dim = int(max_seq_len // n_head)
        self.local_information = nn.Conv1d(embedding_dim, embedding_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=self.input_dim,
        )
        self.activate = nn.GELU()
        self.norm = nn.LayerNorm(self.input_dim)
        self.global_information = nn.Linear(self.input_dim, self.input_dim)
        self.bernolli_sampling = BernoulliSamplingNLP(prob=prob)
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs):
        # The use of view here makes use of the batch dimension to simulate multiple heads. The
        # first dimension of each concept will be batch_size * n_heads.
        q = inputs.view(-1, self.embedding_dim, self.input_dim)
        k = inputs.view(-1, self.embedding_dim, self.input_dim)
        v = inputs.view(-1, self.embedding_dim, self.input_dim)

        q = self.norm(self.activate(self.local_information(q)+q))
        k = self.activate(self.bernolli_sampling(k))
        v = self.activate(self.global_information(v))

        # The math used by the original authors is identical to the scaled_dot_product_attention,
        # so I'm going to go ahead and use this verison.
        attention = self.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.embedding_dim))
        output = torch.bmm(attention, v)
        return output.reshape(-1, self.max_seq_len, self.embedding_dim)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, p: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = p
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return (x * random_tensor)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
