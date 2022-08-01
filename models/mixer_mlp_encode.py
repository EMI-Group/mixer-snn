import torch.nn as nn

from spikingjelly.activation_based import neuron, layer, surrogate
from einops.layers.torch import Rearrange, Reduce
from layers import BatchNorm1d


class MlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim, bn_dim):
        super(MlpBlock, self).__init__()
        self.skip_bn = BatchNorm1d(bn_dim)

        self.mlp = nn.Sequential(
            layer.Linear(dim, hidden_dim, bias=False),
            BatchNorm1d(bn_dim),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True),
            layer.Linear(hidden_dim, dim, bias=False),
            BatchNorm1d(bn_dim)
        )

        self.lif = neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

    def forward(self, x):
        return self.lif(self.skip_bn(x) + self.mlp(x))


class MixerBlock(nn.Module):
    def __init__(self, config):
        super(MixerBlock, self).__init__()
        self.model = nn.Sequential(
            Rearrange('t b n c -> t b c n'),
            MlpBlock(config.n_patches, config.token_hidden_dim, config.encode_dim),
            Rearrange('t b c n -> t b n c'),
            MlpBlock(config.encode_dim, config.channel_hidden_dim, config.n_patches)
        )

    def forward(self, x):
        return self.model(x)


class MixerNet(nn.Module):
    def __init__(self, config):
        super(MixerNet, self).__init__()
        config.n_patches = (config.img_size // config.patch_size) ** 2

        self.model = nn.Sequential(
            Rearrange('t b c (h p1) (w p2) -> t b (h w) (p1 p2 c)', p1=config.patch_size, p2=config.patch_size),
            layer.Linear((config.patch_size ** 2) * 3, config.encode_dim, bias=False),
            BatchNorm1d(config.n_patches),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True),
            *[MixerBlock(config) for _ in range(config.num_blocks)],
            BatchNorm1d(config.n_patches),
            Reduce('t b n c -> t b c', 'mean'),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True),
            layer.Linear(config.encode_dim, config.num_classes)
        )

    def forward(self, x):
        return self.model(x)
