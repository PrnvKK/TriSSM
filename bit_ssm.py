"""Selective SSM block with optionally ternary (BitLinear) input-dependent projections."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlinear import BitLinear


def _linear(in_features, out_features, bias=True, quantized=True):
    if quantized:
        return BitLinear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


def _scan_fwd(a, b):
    """Inclusive prefix scan for h_t = a_t * h_{t-1} + b_t with h_{-1} = 0.

    Hillis-Steele scan over (a, b) pairs combined as
    (a2, b2) o (a1, b1) = (a2 * a1, a2 * b1 + b2), in log2(T) parallel rounds.
    A closed form via cumulative products is avoided: the products underflow
    to 0 in fp32 and the division then overflows.
    """
    a = a.clone()
    b = b.clone()
    T = a.shape[1]
    step = 1
    while step < T:
        a_shift = F.pad(a[:, :-step], (0, 0, 0, 0, step, 0), value=1.0)
        b_shift = F.pad(b[:, :-step], (0, 0, 0, 0, step, 0), value=0.0)
        b = a * b_shift + b
        a = a * a_shift
        step *= 2
    return b


class _ParallelScanLinear(torch.autograd.Function):
    """Parallel scan for h_t = a_t * h_{t-1} + b_t, h_{-1} = 0.

    a, b and h have shape (batch, T, d_model, d_state). The backward pass is
    analytic, so only `a` and `h` are saved instead of the scan intermediates:
        g_t = a_{t+1} * g_{t+1} + grad_h_t,  grad_b_t = g_t,  grad_a_t = g_t * h_{t-1}
    The reverse recurrence for g is evaluated with the same scan on time-reversed inputs.
    """

    @staticmethod
    def forward(ctx, a, b):
        with torch.no_grad():
            h = _scan_fwd(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, gout):
        a, h = ctx.saved_tensors
        ones = torch.ones_like(a[:, :1])
        a_rev = torch.cat([ones, a[:, 1:].flip(dims=[1])], dim=1)
        gout_rev = gout.flip(dims=[1])

        with torch.no_grad():
            g_rev = _scan_fwd(a_rev, gout_rev)
        g = g_rev.flip(dims=[1])

        h_prev = F.pad(h[:, :-1], (0, 0, 0, 0, 1, 0), value=0.0)
        return g * h_prev, g


def parallel_scan_linear(a, b):
    return _ParallelScanLinear.apply(a, b)


class BitSSM(nn.Module):
    """Selective SSM block: causal depthwise Conv1d (k=4), input-dependent
    dt, B, C projections, and a diagonal state-space recurrence.

    With quantized=True the dt, B and C projections are BitLinear (ternary
    weights, int8 activations); with quantized=False they are nn.Linear. The
    conv, initialization and scan are identical in both cases.

    After each forward pass, `stats` holds the hidden-state variance
    (`var_hidden_state`), which the variance probe in train.py reads.
    """

    def __init__(self, d_model, d_state=16, quantized=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.quantized = quantized

        # A = -exp(A_log) keeps the continuous-time poles in the left half plane.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            padding=3,  # causal padding; the trailing outputs are cropped in forward
            groups=d_model,
        )

        # An RMSNorm precedes each projection.
        self.norm_dt = nn.RMSNorm(d_model)
        self.dt_proj = _linear(d_model, d_model, bias=True, quantized=quantized)

        self.norm_B = nn.RMSNorm(d_model)
        self.B_proj = _linear(d_model, d_state, bias=True, quantized=quantized)

        self.norm_C = nn.RMSNorm(d_model)
        self.C_proj = _linear(d_model, d_state, bias=True, quantized=quantized)

        # dt is initialized log-uniform in [0.001, 0.1] (as in Mamba) through the bias.
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            if not quantized:
                nn.init.constant_(self.dt_proj.weight, 0.0)

        self.stats = {}

    def forward(self, x):
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)."""
        seq_len = x.shape[1]

        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :seq_len]
        x = F.silu(x_conv).transpose(1, 2)

        A = -torch.exp(self.A_log)  # (d_model, d_state)

        dt = F.softplus(self.dt_proj(self.norm_dt(x)))  # (batch, T, d_model)
        B = self.B_proj(self.norm_B(x))                 # (batch, T, d_state)
        C = self.C_proj(self.norm_C(x))                 # (batch, T, d_state)

        # Zero-order-hold discretization:
        #   h_t = A_bar_t * h_{t-1} + b_t,  A_bar_t = exp(dt_t * A),
        #   b_t = ((A_bar_t - 1) / A) * B_t * x_t
        A_bar = torch.exp(dt.unsqueeze(-1) * A)         # (batch, T, d_model, d_state)
        B_bar = (A_bar - 1.0) / A * B.unsqueeze(2)
        b_term = B_bar * x.unsqueeze(-1)

        h = parallel_scan_linear(A_bar, b_term)

        self.stats['var_hidden_state'] = h.var().detach()

        return (h * C.unsqueeze(2)).sum(dim=-1)         # (batch, T, d_model)
