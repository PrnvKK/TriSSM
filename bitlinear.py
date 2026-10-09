"""BitLinear: linear layer with ternary weights and int8 activations (BitNet b1.58 style)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightQuantSTE(torch.autograd.Function):
    """Absmean ternary quantization of weights to {-1, 0, 1}; straight-through gradient."""

    @staticmethod
    def forward(ctx, weight):
        scale = weight.abs().mean().clamp(min=1e-5)
        weight_q = (weight / scale).round().clamp(-1, 1)
        return weight_q, scale

    @staticmethod
    def backward(ctx, grad_weight_q, grad_scale):
        return grad_weight_q.clone()


class ActivationQuantSTE(torch.autograd.Function):
    """Per-token absmax quantization of activations to [-128, 127]; straight-through gradient."""

    @staticmethod
    def forward(ctx, x):
        scale = (x.abs().max(dim=-1, keepdim=True)[0] / 127.0).clamp(min=1e-5)
        x_q = (x / scale).round().clamp(-128, 127)
        return x_q, scale

    @staticmethod
    def backward(ctx, grad_x_q, grad_scale):
        return grad_x_q.clone()


class BitLinear(nn.Module):
    """Latent full-precision weights are quantized on the fly in every forward pass.
    The matmul runs on the quantized values in floating point (simulated quantization)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / in_features) ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight_q, w_scale = WeightQuantSTE.apply(self.weight)
        x_q, x_scale = ActivationQuantSTE.apply(x)
        out = F.linear(x_q, weight_q) * (w_scale * x_scale)
        if self.bias is not None:
            out = out + self.bias
        return out
