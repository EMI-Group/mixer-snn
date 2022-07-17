import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, layer, surrogate, functional


class LayerNorm(nn.LayerNorm, layer.base.StepModule):
    def __init__(self, normalized_shape, eps=1e-5, step_mode='s'):
        super(LayerNorm, self).__init__(normalized_shape, eps)
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, super().forward)


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(MlpBlock, self).__init__()
        self.fc1 = layer.Linear(hidden_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())

        self.fc2 = layer.Linear(hidden_dim, hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())

    def forward(self, x):
        x = self.norm1(self.fc1(x))
        h = x
        x = self.lif1(x)
        x = self.norm2(self.fc2(x))
        x = x + h
        x = self.lif2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, config):
        super(MixerBlock, self).__init__()
        self.token_mlp_block = MlpBlock(config.n_patches)
        self.channel_mlp_block = MlpBlock(config.hidden_dim)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.token_mlp_block(x)
        x = x.transpose(-1, -2)
        x = self.channel_mlp_block(x)
        return x


class VLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def supported_backends(self):
        return ('torch',)

    def single_step_forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        y = self.v
        spike = self.neuronal_fire()
        y = y * spike
        self.neuronal_reset(spike)
        return y


class MixerNet(nn.Module):
    def __init__(self, config):
        super(MixerNet, self).__init__()
        config.n_patches = (config.img_size // config.patch_size) ** 2

        patch_h = patch_w = config.img_size // config.patch_size

        self.encoder = nn.Sequential(
            layer.Conv2d(3, config.hidden_dim, kernel_size=3, stride=1, padding=1),
            LayerNorm([config.img_size, config.img_size]),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid()),
        )

        self.patcher = nn.Sequential(
            layer.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=config.patch_size, stride=config.patch_size),
            LayerNorm([patch_h, patch_w])
        )

        self.classifier_norm = LayerNorm(config.hidden_dim)

        self.classifier = nn.Sequential(
            layer.Linear(config.hidden_dim, config.num_classes),
            VLIFNode(surrogate_function=surrogate.Sigmoid())
        )
        self.blocks = nn.ModuleList()
        for _ in range(config.num_blocks):
            self.blocks.append(MixerBlock(config))

    def forward(self, x):
        x = self.encoder(x)
        x = self.patcher(x)
        x = x.flatten(-2).transpose(-1, -2)

        for block in self.blocks:
            x = block(x)

        x = self.classifier_norm(x)
        x = torch.mean(x, dim=-2)
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    from configs import get_mixer_v1_config

    config = get_mixer_v1_config()
    config.num_classes = 10
    config.img_size = 32
    model = MixerNet(config)
    functional.set_step_mode(model, 'm')

    img = torch.rand([4, 1, 3, 32, 32])
    out = model(img).mean(0)
    print(out)
