import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, layer, surrogate, functional


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
        self.fc1 = layer.Linear(dim, hidden_dim)
        self.norm1 = BatchNorm1d(bn_dim)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.fc2 = layer.Linear(hidden_dim, hidden_dim)
        self.norm2 = BatchNorm1d(bn_dim)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.fc3 = layer.Linear(hidden_dim, dim)
        self.norm3 = BatchNorm1d(bn_dim)
        self.lif3 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

    def forward(self, x):
        x = self.norm1(self.fc1(x))
        h = x
        x = self.lif1(x)
        x = self.norm2(self.fc2(x))
        x = x + h
        x = self.lif2(x)
        x = self.norm3(self.fc3(x))
        x = self.lif3(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, config):
        super(MixerBlock, self).__init__()
        self.token_mlp_block = MlpBlock(config.n_patches, config.token_hidden_dim, config.encode_dim)
        self.channel_mlp_block = MlpBlock(config.encode_dim, config.channel_hidden_dim, config.n_patches)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.token_mlp_block(x)
        x = x.transpose(-1, -2)
        x = self.channel_mlp_block(x)
        return x


class MixerNet(nn.Module):
    def __init__(self, config):
        super(MixerNet, self).__init__()
        config.n_patches = (config.img_size // config.patch_size) ** 2

        self.encoder = nn.Sequential(
            layer.Conv2d(3, config.encode_dim, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm2d(config.encode_dim),
            neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True),
        )

        self.patcher = nn.Sequential(
            layer.Conv2d(config.encode_dim, config.encode_dim, kernel_size=config.patch_size, stride=config.patch_size),
            layer.BatchNorm2d(config.encode_dim)
        )

        self.classifier_norm = BatchNorm1d(config.n_patches)

        self.classifier = nn.Sequential(
            layer.Linear(config.encode_dim, config.num_classes * config.voting_num),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(spiking=False), detach_reset=True),
            layer.VotingLayer(config.voting_num)
        )

        self.blocks = nn.ModuleList()
        for _ in range(config.num_blocks):
            self.blocks.append(MixerBlock(config))

        self._initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.patcher(x)
        x = x.flatten(-2).transpose(-1, -2)

        for block in self.blocks:
            x = block(x)

        x = self.classifier_norm(x)
        x = torch.mean(x, dim=-2)
        out_fr = self.classifier(x)

        return out_fr

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d) or isinstance(m, BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from configs import get_mixer_var_dim_config
    config = get_mixer_var_dim_config()
    config.num_classes = 10
    config.img_size = 32
    model = MixerNet(config)
    functional.set_step_mode(model, 'm')
    print(model)

    img = torch.rand([4, 1, 3, 32, 32])
    print(img)

    out = model(img).mean(0)
    print(out)
