"""TriSSM: a stack of selective SSM blocks with ternary sequence-mixing projections."""
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from bit_ssm import BitSSM


class TriSSMBlock(nn.Module):
    """Pre-norm residual block around a BitSSM.

    quantized=True gives the ternary TriSSM block; quantized=False gives the
    full-precision SSM baseline with identical structure.
    """

    def __init__(self, d_model, d_state=16, quantized=True):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.ssm = BitSSM(d_model, d_state, quantized=quantized)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class TriSSM(nn.Module):
    """Language model: embedding, TriSSM blocks, final RMSNorm, linear LM head.
    The embedding and LM head stay full precision."""

    def __init__(self, vocab_size, d_model, n_layers, d_state=16, quantized=True,
                 use_checkpoint=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TriSSMBlock(d_model, d_state, quantized) for _ in range(n_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, last_only=False):
        """x: (batch, seq_len) token ids -> logits (batch, seq_len, vocab_size).

        With last_only=True only the final position is projected, giving
        (batch, 1, vocab_size). The recurrence is causal, so this equals
        slicing the full logits.
        """
        x = self.embedding(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        if last_only:
            x = x[:, -1:]
        return self.lm_head(self.norm_f(x))
