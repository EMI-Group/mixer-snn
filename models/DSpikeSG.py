import math

import torch
from spikingjelly.activation_based.surrogate import SurrogateFunctionBase, heaviside

@torch.jit.script
def dSpike_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    mask = (x.abs() > 0.5)
    const = alpha / (2. * math.tanh(alpha / 2.))
    grad_x = (grad_output * const / ((alpha * x).cosh_()).square_()).masked_fill_(mask, 0)
    return grad_x, None


class dSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return dSpike_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class DSpike(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        super().__init__(alpha, spiking)
        assert alpha > 0, 'alpha must be lager than 0'

    @staticmethod
    def spiking_function(x, alpha):
        return dSpike.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha):
        return (alpha * x).tanh_() / (2. * math.tanh(alpha / 2.)) + 0.5
