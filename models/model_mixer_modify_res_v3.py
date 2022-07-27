import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, layer, surrogate, functional
from einops.layers.torch import Rearrange, Reduce


class BatchNorm1d(nn.BatchNorm1d, layer.base.StepModule):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, step_mode='s'):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, super().forward)


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
            layer.Linear((config.patch_size ** 2) * 3, config.encode_dim),
            BatchNorm1d(config.n_patches),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True),
            *[MixerBlock(config) for _ in range(config.num_blocks)],
            BatchNorm1d(config.n_patches),
            Reduce('t b n c ->t b c', 'mean'),
            layer.Linear(config.encode_dim, config.num_classes * config.voting_num),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True),
            layer.VotingLayer(config.voting_num)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    from configs import get_model_mixer_modify_res_v1_config
    config = get_model_mixer_modify_res_v1_config()
    config.num_classes = 10
    config.img_size = 32
    model = MixerNet(config)
    functional.set_step_mode(model, 'm')
    print(model)

    img = torch.rand([4, 1, 3, 32, 32])
    print(img)

    out = model(img).mean(0)
    print(out)
