import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional


class PatchMerging(nn.Module):
    def __init__(self, height, width, channels):
        super(PatchMerging, self).__init__()
        self.reduction = nn.Sequential(
            layer.Linear(4 * channels, 2 * channels, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )
        self.h = height
        self.w = width

    def forward(self, feature_map):
        feature_map = feature_map.permute(0, 1, 3, 4, 2)
        x0 = feature_map[:, :, 0::2, 0::2, :]
        x1 = feature_map[:, :, 1::2, 0::2, :]
        x2 = feature_map[:, :, 0::2, 1::2, :]
        x3 = feature_map[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.reduction(x)
        out = x.permute(0, 1, 4, 2, 3)
        return out


class SNNMlpBlock(nn.Module):
    def __init__(self, hidden_dim, channels):
        super(SNNMlpBlock, self).__init__()
        self.fc1 = layer.Linear(hidden_dim, hidden_dim)
        self.bn1 = layer.BatchNorm2d(channels)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(hidden_dim, hidden_dim)
        self.bn2 = layer.BatchNorm2d(channels)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())

    def forward(self, feature_map):
        x = self.fc1(feature_map)
        x = self.bn1(x)
        h = x
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = x + h
        x = self.lif2(x)
        return x


class SNNTokenMixingBlock(nn.Module):
    def __init__(self, height, weight, channels):
        super(SNNTokenMixingBlock, self).__init__()
        self.pre_bn = layer.BatchNorm2d(channels)
        self.dwConv = nn.Sequential(
            layer.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            layer.BatchNorm2d(channels)
        )
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.mlp_h = SNNMlpBlock(height, channels)
        self.mlp_w = SNNMlpBlock(weight, channels)
        self.out_conv = layer.Conv2d(3 * channels, channels, kernel_size=1)

    def forward(self, feature_map):
        x = self.pre_bn(feature_map)
        h = x
        x = self.dwConv(x) + h
        x = self.lif(x)
        x_id = x
        x_h = self.mlp_h(x.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        x_w = self.mlp_w(x)
        x = torch.cat([x_h, x_w, x_id], dim=2)
        out = self.out_conv(x)
        return out


class SNNChannelMixingBlock(nn.Module):
    def __init__(self, channels):
        super(SNNChannelMixingBlock, self).__init__()
        self.fc1 = layer.Linear(channels, channels)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(channels, channels)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())

    def forward(self, feature_map):
        x = feature_map.permute(0, 1, 3, 4, 2)
        x = self.fc1(x)
        h = x
        x = self.lif1(x)
        x = self.fc2(x)
        x = x + h
        x = self.lif2(x)
        out = x.permute(0, 1, 4, 2, 3)
        return out


class SNNStageBlock(nn.Module):
    def __init__(self, height, width, channels, depth):
        super(SNNStageBlock, self).__init__()
        self.h = height
        self.w = width
        self.c = channels
        self.d = depth
        self.patch_merge = PatchMerging(height, width, channels)
        self.model = nn.Sequential(
            *[
                nn.Sequential(
                    SNNTokenMixingBlock(height // 2, width // 2, channels * 2),
                    SNNChannelMixingBlock(channels * 2)
                ) for _ in range(depth)
            ]
        )

    def forward(self, feature_map):
        x = self.patch_merge(feature_map)
        x = self.model(x)
        return x


class SNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm2d(out_channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        return self.encoder(x)


class SNNPatchPartition(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size):
        super(SNNPatchPartition, self).__init__()
        self.model = nn.Sequential(
            layer.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, x):
        return self.model(x)


class SNNNet(nn.Module):
    def __init__(self, height, width, in_channels, hidden_dim, patch_size, depth, num_classes):
        super(SNNNet, self).__init__()
        self.partition = SNNPatchPartition(in_channels, hidden_dim, patch_size)
        height = height // patch_size
        width = width // patch_size
        self.stages = nn.ModuleList()
        model_c = hidden_dim
        for i in range(len(depth)):
            i_depth = depth[i]
            i_stage = SNNStageBlock(height // (2 ** i), width // (2 ** i), model_c, i_depth)
            self.stages.append(i_stage)
            model_c = model_c * 2

        self.avg = layer.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            layer.Linear(model_c, num_classes),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        x = self.partition(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avg(x).squeeze()
        out_fr = self.head(x)
        return out_fr


if __name__ == '__main__':
    T, N, C, H, W = 2, 2, 3, 224, 224
    patch_size = 4
    hidden_dim = 128
    x = torch.rand([T, N, C, H, W])
    net = SNNNet(H, W, C, hidden_dim, patch_size, [2, 3, 3], 10)
    functional.set_step_mode(net, 'm')
    out = net(x)
    print(out.shape)
    print(out.mean(0))

