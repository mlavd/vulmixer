"""FIRE Mixer Module.

This module provides different mixers to use. If you
would like to add additional options, import the registry
from this file and add your own. The option may then be used
in a configuration file.
"""
from .util import ModuleRegistry, MlpLayer, SparseMLP, MHTCA
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNetMixer(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class MHAMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mode == "token", f"TCA does not support feature mixing."

        self.mha = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.drop_prob,
        )

    def forward(self, x):
        return self.mha(query=x, key=x, value=x)[0]


class MLPMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp = MlpLayer(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.inner_dim,
        )

        if config.mode == "token":
            self.mlp = nn.Sequential(
                Rearrange("b t f -> b f t"),
                mlp,
                Rearrange("b f t -> b t f"),
            )

        elif config.mode == "feature":
            self.mlp = mlp

        else:
            raise NotImplementedError(f"Invalid mode: {config.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PoolMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mode == "token", f"Pooling does not support feature mixing."

        layer = {
            "max": nn.MaxPool1d,
            "avg": nn.AvgPool1d,
        }[config.pool]

        if isinstance(config.kernels, list):
            kernel = config.kernels[config.index]
        else:
            kernel = config.kernels

        self.pooler = nn.Sequential(
            Rearrange("b t f -> b f t"),
            layer(
                kernel_size=kernel,
                stride=config.get("stride", None),
                padding=config.get("padding", 0),
                # dilation=config.get('dilation', 1),
            ),
            Rearrange("b f t -> b t f"),
        )

    def forward(self, x):
        size = x.size(1)
        x = self.pooler(x)

        diff = size - x.size(1)
        pad = diff // 2

        x = F.pad(x, (0, 0, pad, pad + diff % 2), "constant", 0)
        return x


class RollMixer(nn.Module):
    def __init__(self, config):
        super().__init__()

        if isinstance(config.shifts, list):
            self.shifts = config.shifts[config.index]
        else:
            self.shifts = config.shifts

        if config.mode == "token":
            self.dim = 1
        elif config.mode == "feature":
            self.dim = 2
        else:
            raise NotImplementedError(f"Invalid mode: {config.mode}")

    def forward(self, x):
        return x + torch.roll(x, shifts=self.shifts, dims=self.dim) / 2


class SparseMLPMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.smlp = SparseMLP(
            W=config.token_size,
            H=config.feature_size,
            channels=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.smlp(x)


class TCAMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mode == "token", f"TCA does not support feature mixing."

        self.mhtca = MHTCA(
            n_head=config.num_heads,
            max_seq_len=config.max_seq_length,
            embedding_dim=config.hidden_dim,
            prob=config.keep_prob,
            kernel_size=config.kernel_size[config.index],
            dilation=config.dilation[config.index],
            padding=config.padding[config.index],
        )

    def forward(self, x):
        return self.mhtca(x)


class BernoulliSamplingNLP(nn.Module):
    # FIXME Document
    def __init__(self, max_len=100, prob=1):
        super().__init__()
        self.max_len = max_len
        self.prob = prob

    def __call__(self, inputs):
        inputs = F.dropout1d(inputs, p=1 - self.prob, training=self.training)
        return F.softmax(inputs, dim=-2)
    
class VulMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mode == "token", f"VulMixer does not support feature mixing."

        self.embedding_dim = config.hidden_dim
        self.n_head = config.num_heads
        self.max_seq_len = config.max_seq_length
        self.input_dim = int(config.max_seq_length // config.num_heads)
        self.statement_info = nn.Conv1d(config.hidden_dim, config.hidden_dim,
            kernel_size=config.kernel_size[config.index],
            stride=1,
            padding=config.padding[config.index],
            dilation=config.dilation[config.index],
            groups=self.input_dim,
        )
        self.activate = nn.GELU()
        self.norm = nn.LayerNorm(self.input_dim)
        self.function_info = nn.Linear(self.input_dim, self.input_dim)

        self.block_info = nn.Linear(64, 1)
        self.drop_prob = 1 - config.keep_prob

        import torchvision
        self.blur = torchvision.transforms.GaussianBlur(config.blur_kernel, sigma=(0.1, 2.0))

    def forward(self, inputs):
        # The use of view here makes use of the batch dimension to simulate multiple heads. The
        # first dimension of each concept will be batch_size * n_heads.
        q = inputs.view(-1, self.embedding_dim, self.input_dim)
        k = inputs.view(-1, self.embedding_dim, self.input_dim)
        v = inputs.view(-1, self.embedding_dim, self.input_dim)
        m = inputs.view(-1, self.embedding_dim, self.input_dim)

        # Statement path
        q = self.norm(self.activate(self.statement_info(q) + q))

        # Function path
        v = self.activate(self.function_info(v))

        # Token Path
        k = F.dropout1d(k, p=self.drop_prob, training=self.training)
        k = F.softmax(k, dim=-2)
        k = self.activate(k)
        
        # Block path
        m = self.blur(m)
        m = self.block_info(m)
        m = m.expand(-1, -1, self.embedding_dim)

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return output.reshape(-1, self.max_seq_len, self.embedding_dim)

mixers = ModuleRegistry("mixer")
mixers.register("fnet", FNetMixer)
mixers.register("mha", MHAMixer)
mixers.register("mlp", MLPMixer)
mixers.register("pool", PoolMixer)
mixers.register("roll", RollMixer)
mixers.register("smlp", SparseMLPMixer)
mixers.register("tca", TCAMixer)
mixers.register("vul", VulMixer)