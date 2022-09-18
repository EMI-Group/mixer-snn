import torch
import torch.nn as nn

from spikingjelly.activation_based import layer


def convert_bn_to_sync_bn(module, process_group=None):
    module_output = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module_output = nn.SyncBatchNorm(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
            process_group,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
        module_output = layer.SeqToANNContainer(module_output)
    for name, child in module.named_children():
        module_output.add_module(
            name, convert_bn_to_sync_bn(child, process_group)
        )
    del module
    return module_output
