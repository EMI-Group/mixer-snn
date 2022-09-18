import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from spikingjelly.activation_based import neuron, layer


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, bn_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer.Linear(dim, hidden_dim),
            nn.BatchNorm1d(bn_dim),
            neuron.LIFNode(detach_reset=True),
            layer.Linear(hidden_dim, dim),
            nn.BatchNorm1d(bn_dim),
        )

    def forward(self, x):
        return self.net(x)


class sMLPBlock(nn.Module):
    def __init__(self, H, W, in_channels, out_channels):
        super().__init__()
        self.activation = neuron.LIFNode(detach_reset=True)
        self.BN = nn.BatchNorm2d(in_channels)
        self.proj_w = nn.Sequential(
            Rearrange('t b c h w -> t b w c h'),
            layer.Conv2d(W, W, (1, 1)),
            nn.BatchNorm2d(W),
            neuron.LIFNode(detach_reset=True),
            Rearrange('t b w c h -> t b c h w')
        )
        self.proj_h = nn.Sequential(
            Rearrange('t b c h w -> t b h c w'),
            layer.Conv2d(H, H, (1, 1)),
            nn.BatchNorm2d(H),
            neuron.LIFNode(detach_reset=True),
            Rearrange('t b h c w -> t b c h w'),
        )
        self.fuse = nn.Sequential(
            layer.Conv2d(in_channels * 3, out_channels, (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x_identity = self.activation(self.BN(x))
        x_h = self.proj_h(x)
        x_w = self.proj_w(x)
        x = self.fuse(torch.cat([x_identity, x_h, x_w], dim=2))
        return x


class sMLPNet(nn.Module):
    def __init__(self, in_channels=3, dim=80, alpha=3, num_classes=1000, patch_size=4, image_size=224,
                 depths=[2, 8, 14, 2]):
        super(sMLPNet, self).__init__()
        self.num_patch = image_size // patch_size
        self.depths = depths

        self.to_patch_embedding = nn.ModuleList([])
        self.to_patch_embedding_lif = nn.ModuleList([])
        self.token_mix = nn.ModuleList([])
        self.token_mix_lif = nn.ModuleList([])
        self.channel_mix = nn.ModuleList([])
        self.channel_mix_lif = nn.ModuleList([])

        for i in range(len(self.depths)):
            ratio = 2 ** i
            if i == 0:
                self.to_patch_embedding.append(
                    nn.Sequential(layer.Conv2d(in_channels, dim, patch_size, patch_size, bias=False),
                                  nn.BatchNorm2d(dim)))
            else:
                self.to_patch_embedding.append(
                    nn.Sequential(layer.Conv2d(dim * ratio // 2, dim * ratio, 2, 2, bias=False),
                                  nn.BatchNorm2d(dim * ratio)))
            self.to_patch_embedding_lif.append(neuron.LIFNode(detach_reset=True))

            for j in range(self.depths[i]):
                self.channel_mix.append(nn.Sequential(
                    Rearrange('t b c h w -> t b (h w) c'),
                    FeedForward(dim * ratio, dim * ratio * alpha, (self.num_patch // ratio) ** 2),
                    Rearrange('t b (h w) c -> t b c h w', h=self.num_patch // ratio, w=self.num_patch // ratio)
                ))
                self.channel_mix_lif.append(neuron.LIFNode(detach_reset=True))

                self.token_mix.append(sMLPBlock(self.num_patch // ratio, self.num_patch // ratio,
                                                dim * ratio,
                                                dim * ratio))
                self.token_mix_lif.append(neuron.LIFNode(detach_reset=True))

        self.batch_norm = nn.BatchNorm2d(dim * 2 ** (len(self.depths) - 1))

        self.mlp_head = nn.Sequential(
            Reduce('t b c h w -> t b c', 'mean'),
            layer.Linear(dim * 2 ** (len(self.depths) - 1), num_classes)
        )

    def forward(self, x):
        shift = 0
        for i in range(len(self.depths)):
            h = self.to_patch_embedding[i](x)
            x = self.to_patch_embedding_lif[i](h)
            for j in range(self.depths[i]):
                token_h = self.token_mix[j + shift](x)
                x = self.token_mix_lif[j + shift](h + token_h)
                x = self.channel_mix_lif[j + shift](h + token_h + self.channel_mix[j + shift](x))
            shift += self.depths[i]

        x = self.batch_norm(x)

        return self.mlp_head(x)
