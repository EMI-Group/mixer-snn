import math

import torch
from spikingjelly.activation_based.surrogate import SurrogateFunctionBase, heaviside,\
    tab4_str, curly_bracket_l, curly_bracket_r


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

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_x_abs = fabsf({x});
            float {y};
            if ({sg_name}_x_abs > 0.5f)
            {curly_bracket_l}
                {y} = 0.0f;
            {curly_bracket_r}
            else
            {curly_bracket_l}
                {tab4_str}const float {sg_name}_prefix = {alpha} / (2.0f * tanhf({alpha} / 2.0f));
                {tab4_str}const float {sg_name}_cosh_alpha_x = coshf({alpha} * {x});
                {tab4_str}const float {sg_name}_square = {sg_name}_cosh_alpha_x * {sg_name}_cosh_alpha_x;
                {tab4_str}{y} = {sg_name}_prefix / {sg_name}_square;
            {curly_bracket_r}
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_half2_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_half2_alpha_2 = __h2div({sg_name}_half2_alpha, __float2half2_rn(2.0f));
            {tab4_str}const half2 {sg_name}_x_abs = __habs2({x});
            {tab4_str}const half2 {sg_name}_exp_alpha_2 = h2exp({sg_name}_half2_alpha_2);
            {tab4_str}const half2 {sg_name}_exp_neg_alpha_2 = h2exp(-{sg_name}_half2_alpha_2);
            {tab4_str}const half2 {sg_name}_exp_sum = __hadd2({sg_name}_exp_alpha_2, {sg_name}_exp_neg_alpha_2);
            {tab4_str}const half2 {sg_name}_exp_sub = __hsub2({sg_name}_exp_alpha_2, {sg_name}_exp_neg_alpha_2);
            {tab4_str}const half2 {sg_name}_tanh_alpha_2 = __h2div({sg_name}_exp_sub, {sg_name}_exp_sum);
            {tab4_str}const half2 {sg_name}_x_abs_gt = __hgt2({sg_name}_x_abs, __float2half2_rn(0.5f));
            {tab4_str}const half2 {sg_name}_prefix = __h2div({sg_name}_half2_alpha, __hmul2(__float2half2_rn(2.0f), {sg_name}_tanh_alpha_2));
            {tab4_str}const half2 {sg_name}_half_alpha_x = __hmul2({sg_name}_half2_alpha, {x});
            {tab4_str}const half2 {sg_name}_exp_alpha_x = h2exp({sg_name}_half_alpha_x);
            {tab4_str}const half2 {sg_name}_exp_neg_alpha_x = h2exp(-{sg_name}_half_alpha_x);
            {tab4_str}const half2 {sg_name}_cosh_alpha_x = __h2div(__hadd2({sg_name}_exp_alpha_x, {sg_name}_exp_neg_alpha_x), __float2half2_rn(2.0f));
            {tab4_str}const half2 {sg_name}_temp_y = __h2div({sg_name}_prefix, __hmul2({sg_name}_cosh_alpha_x, {sg_name}_cosh_alpha_x));
            {tab4_str}half2 {y} = __hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_x_abs_gt), {sg_name}_temp_y);
            '''
        else:
            raise NotImplementedError

        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code
