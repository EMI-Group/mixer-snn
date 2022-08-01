import torch.nn as nn

from spikingjelly.activation_based import layer, functional


class BatchNorm1d(nn.BatchNorm1d, layer.base.StepModule):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, step_mode='s'):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, super().forward)