import torch.nn as nn

from spikingjelly.activation_based import neuron, layer, surrogate
from einops.layers.torch import Rearrange, Reduce
from layers import BatchNorm1d


class MlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim, bn_dim):
        super(MlpBlock, self).__init__()
        self.skip_bn = BatchNorm1d(bn_dim)

        self.mlp = nn.Sequential(
            layer.Linear(dim, hidden_dim),
            BatchNorm1d(bn_dim),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True),
            layer.Linear(hidden_dim, dim),
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
            MlpBlock(config.n_patches, config.token_hidden_dim, config.hidden_dim),
            Rearrange('t b c n -> t b n c'),
            MlpBlock(config.hidden_dim, config.channel_hidden_dim, config.n_patches)
        )

    def forward(self, x):
        return self.model(x)


class MixerNet(nn.Module):
    def __init__(self, config):
        super(MixerNet, self).__init__()
        config.n_patches = (config.img_size // config.patch_size) ** 2

        self.model = nn.Sequential(
            layer.Conv2d(3, config.encode_dim, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm2d(config.encode_dim),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True),
            layer.Conv2d(config.encode_dim, config.hidden_dim, kernel_size=config.patch_size, stride=config.patch_size),
            layer.BatchNorm2d(config.hidden_dim),
            Rearrange('t b c h w -> t b (h w) c'),
            *[MixerBlock(config) for _ in range(config.num_blocks)],
            BatchNorm1d(config.n_patches),
            Reduce('t b n c -> t b c', 'mean'),
            layer.Linear(config.hidden_dim, config.num_classes * config.voting_num),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(spiking=False), detach_reset=True),
            layer.VotingLayer(config.voting_num)
        )

    def forward(self, x):
        return self.model(x)
