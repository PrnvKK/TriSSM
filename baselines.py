"""Transformer baselines.

* TransformerLM with quantized=True: decoder-only Transformer whose attention
  (QKVO) and MLP linear layers are BitLinear (ternary weights, int8 activations).
* TransformerLM with quantized=False: the same architecture with nn.Linear.
* qkvo_transformer: ternary Q/K/V/O projections with a full-precision MLP.

The embedding and LM head are full precision in every model. The architecture
is the same for all variants, so they differ only in quantization.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlinear import BitLinear


class _Attention(nn.Module):
    """Causal multi-head self-attention.

    After each forward pass, `stats` holds the variance of the pre-softmax
    attention scores over the causal triangle (`var_attn_score`), which the
    variance probe in train.py reads.
    """

    def __init__(self, d_model, n_heads, linear):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = linear(d_model, 3 * d_model, bias=False)
        self.proj = linear(d_model, d_model, bias=False)
        self.stats = {}

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))

        n_valid = causal.sum().float()
        mean = scores.masked_fill(~causal, 0.0).sum() / n_valid
        sq = ((scores - mean) ** 2).masked_fill(~causal, 0.0)
        self.stats['var_attn_score'] = (sq.sum() / n_valid).detach()

        att = scores.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class _MLP(nn.Module):
    def __init__(self, d_model, d_ff, linear):
        super().__init__()
        self.fc = linear(d_model, d_ff, bias=False)
        self.proj = linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class _Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, linear_attn, linear_mlp):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = _Attention(d_model, n_heads, linear_attn)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = _MLP(d_model, d_ff, linear_mlp)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer. `quantized` controls the attention linear
    layers and `quantized_mlp` (default: same as `quantized`) the MLP ones."""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff,
                 quantized=False, quantized_mlp=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        if quantized_mlp is None:
            quantized_mlp = quantized

        def linear_attn(in_f, out_f, bias=True):
            if quantized:
                return BitLinear(in_f, out_f, bias=bias)
            return nn.Linear(in_f, out_f, bias=bias)

        def linear_mlp(in_f, out_f, bias=True):
            if quantized_mlp:
                return BitLinear(in_f, out_f, bias=bias)
            return nn.Linear(in_f, out_f, bias=bias)

        self.blocks = nn.ModuleList(
            [_Block(d_model, n_heads, d_ff, linear_attn, linear_mlp)
             for _ in range(n_layers)]
        )
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, last_only=False):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        if last_only:
            x = x[:, -1:]
        return self.lm_head(self.norm_f(x))


def bitnet_transformer(vocab_size=8000, d_model=192, n_layers=4,
                       n_heads=8, d_ff=768):
    """Ternary Transformer: all attention and MLP linear layers are BitLinear."""
    return TransformerLM(vocab_size, d_model, n_layers, n_heads, d_ff,
                         quantized=True)


def fp_transformer(vocab_size=8000, d_model=192, n_layers=4,
                   n_heads=8, d_ff=768):
    """Full-precision Transformer with the same architecture."""
    return TransformerLM(vocab_size, d_model, n_layers, n_heads, d_ff,
                         quantized=False)


def qkvo_transformer(vocab_size=8000, d_model=192, n_layers=4,
                     n_heads=8, d_ff=768):
    """Control: ternary Q/K/V/O projections, full-precision MLP."""
    return TransformerLM(vocab_size, d_model, n_layers, n_heads, d_ff,
                         quantized=True, quantized_mlp=False)
