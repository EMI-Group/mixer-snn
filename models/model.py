import copy

import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, layer, surrogate, functional


class LayerNorm(nn.LayerNorm, layer.base.StepModule):
    def __init__(self, normalized_shape, eps=1e-5, step_mode='s'):
        super(LayerNorm, self).__init__(normalized_shape, eps)
        self.step_mode = step_mode


class BatchNorm1d(nn.BatchNorm1d, layer.base.StepModule):
    def __init__(self, num_features, eps=1e-5, step_mode='s'):
        super(BatchNorm1d, self).__init__(num_features, eps)
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, super().forward)


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, bn_dim):
        super(MlpBlock, self).__init__()
        self.fc0 = layer.Linear(hidden_dim, hidden_dim, bias=False)
        self.lif0 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.bn0 = BatchNorm1d(bn_dim)
        self.fc1 = layer.Linear(hidden_dim, hidden_dim, bias=False)
        # NOTE: LIF Node here
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.bn1 = BatchNorm1d(bn_dim)

    def forward(self, x):
        x = self.fc0(x)
        x = self.bn0(x)
        h = x
        x = self.lif0(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = x + h
        x = self.lif1(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, config):
        super(MixerBlock, self).__init__()
        self.token_mlp_block = MlpBlock(config.n_patches, config.hidden_dim)
        self.channel_mlp_block = MlpBlock(config.hidden_dim, config.n_patches)
        self.pre_norm = LayerNorm(config.hidden_dim, eps=1e-6)
        self.post_norm = LayerNorm(config.hidden_dim, eps=1e-6)

    def forward(self, x):
        h = x
        x = self.pre_norm(x)
        x = x.transpose(-1, -2)
        x = self.token_mlp_block(x)
        x = x.transpose(-1, -2)
        x = x + h

        h = x
        x = self.post_norm(x)
        x = self.channel_mlp_block(x)
        x = x + h
        return x


class MlpMixer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, patch_size=16):
        super(MlpMixer, self).__init__()
        self.num_classes = num_classes
        n_patches = (img_size // patch_size) * (img_size // patch_size)
        config.n_patches = n_patches

        self.stem = layer.Conv2d(in_channels=3,
                                 out_channels=config.hidden_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size)

        self.pre_head_ln = LayerNorm(config.hidden_dim, eps=1e-6)

        self.head = nn.Sequential(
            layer.Linear(config.hidden_dim, num_classes, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

        self.layer = nn.ModuleList()
        for _ in range(config.num_blocks):
            block = MixerBlock(config)
            self.layer.append(copy.deepcopy(block))

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(-2)
        x = x.transpose(-1, -2)

        for block in self.layer:
            x = block(x)
        x = self.pre_head_ln(x)
        x = torch.mean(x, dim=-2)    # [T, N, P, C] => [T, N, C]
        out_fr = self.head(x)   # [T, N, num_classes]

        return out_fr


if __name__ == '__main__':
    from configs import get_mixer_b16_config
    config = get_mixer_b16_config()
    model = MlpMixer(config, img_size=224)
    functional.set_step_mode(model, 'm')
    print(model)

    encode_img = (torch.rand([4, 64, 3, 224, 224]) * 2).int().float()
    print(encode_img)

    out = model(encode_img).mean(0)
    print(out.shape)
