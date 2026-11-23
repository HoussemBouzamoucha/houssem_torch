from __future__ import annotations
import numpy as np
from houssem_torch.nn import init
from houssem_torch.nn.dropout import Dropout
from houssem_torch.nn.linear import Linear
from houssem_torch.nn.module import Module

# Added to the scores of masked positions before softmax. A large finite
# number rather than -inf, so a row whose keys are ALL masked (an all-padding
# sentence) gives a uniform distribution instead of NaN. Fine for float32;
# fp16 kernels will need a smaller magnitude (about -1e4).
MASK_VALUE = -1e9


def build_attention_mask(key_padding_mask=None, causal: bool = False,
                         q_len: int | None = None, k_len: int | None = None):
    """Combine the two kinds of masks into one boolean array (True = hide).

    key_padding_mask: (B, k_len) bool, True where the KEY is padding. Every
                      query must ignore those keys.
    causal:           hide future keys: query i may only see keys 0..i.

    Returns None (nothing to hide) or an array that broadcasts against the
    scores (B, h, q_len, k_len): shape (B, 1, 1, k_len), (1, 1, q_len, k_len)
    or (B, 1, q_len, k_len).
    """
    mask = None
    if key_padding_mask is not None:
        mask = np.asarray(key_padding_mask, dtype=bool)[:, None, None, :]
    if causal:
        future = np.triu(np.ones((q_len, k_len), dtype=bool), k=1)[None, None]
        mask = future if mask is None else (mask | future)
    return mask


def _fused_projection(d_model: int, parts: int, bias: bool) -> Linear:
    """One Linear (d, parts·d) holding `parts` separate (d, d) projections
    side by side (Q|K|V or K|V), each Xavier-initialized as its own (d, d)
    matrix. One multiply instead of `parts`, same starting scale as separate
    layers: Xavier over the whole (d, parts·d) block would start them
    sqrt((parts + 1) / 2) times smaller."""
    layer = Linear(d_model, parts * d_model, bias=bias)
    layer.weight.numpy()[...] = np.concatenate(
        [init.xavier_uniform((d_model, d_model)) for _ in range(parts)], axis=1)
    return layer


class MultiHeadAttention(Module):
    """Multi-head scaled dot-product attention, in one of two layouts.

    Self-attention (cross=False): queries, keys and values all come from the
    same sequence x, so one fused projection w_qkv (d, 3d) produces all three.
    Used by the encoder (no causal mask) and the decoder (causal=True).

    Cross-attention (cross=True): queries come from x (the target side), keys
    and values from `memory` (the encoder's output for the source): w_q (d, d) and
    a fused w_kv (d, 2d). Used by the decoder.

    Computation, for h heads of size d_k = d / h:
      1. project and split into heads:    (B, L, d) -> (B, h, L, d_k)
      2. scores = Q @ Kᵀ / sqrt(d_k):      (B, h, Tq, Tk)
      3. hide masked keys (padding / future) with MASK_VALUE
      4. weights = softmax(scores) over keys, then dropout
      5. out = weights @ V:               (B, h, Tq, d_k)
      6. merge heads and project with w_o: (B, Tq, d)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 cross: bool = False, bias: bool = True):
        """bias=False drops the bias of all four projections (Q, K, V, output),
        as in the paper's reference code and many PyTorch models."""
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by "
                             f"n_heads ({n_heads}).")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.cross = cross
        if cross:
            self.w_q = Linear(d_model, d_model, bias=bias)
            self.w_kv = _fused_projection(d_model, 2, bias)
        else:
            self.w_qkv = _fused_projection(d_model, 3, bias)
        self.w_o = Linear(d_model, d_model, bias=bias)
        self.attn_dropout = Dropout(dropout)

    def _split_heads(self, t):
        """(B, L, d) -> (B, h, L, d_k)"""
        B, L, _ = t.shape
        return t.reshape(B, L, self.n_heads, self.d_k).swapaxes(1, 2)

    def _merge_heads(self, t):
        """(B, h, L, d_k) -> (B, L, d)"""
        B, _, L, _ = t.shape
        return t.swapaxes(1, 2).reshape(B, L, self.d_model)

    def forward(self, x, memory=None, key_padding_mask=None, causal: bool = False):
        """
        x:                (B, Tq, d) the sequence asking the questions (queries).
        memory:           (B, Tk, d) the sequence being looked at; required
                          for cross-attention, must be None for self-attention.
        key_padding_mask: (B, Tk) bool, True at padding keys (or None).
        causal:           if True, query i only sees keys 0..i.
        Returns (B, Tq, d).
        """
        if self.cross:
            if memory is None:
                raise ValueError("cross-attention needs `memory`.")
            q = self.w_q(x)
            k, v = self.w_kv(memory).split(self.d_model, axis=-1)
        else:
            if memory is not None:
                raise ValueError("self-attention takes no `memory`.")
            q, k, v = self.w_qkv(x).split(self.d_model, axis=-1)

        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        scores = (q @ k.mT) * (1.0 / np.sqrt(self.d_k))

        mask = build_attention_mask(key_padding_mask, causal,
                                    q_len=scores.shape[-2], k_len=scores.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(mask, MASK_VALUE)

        weights = self.attn_dropout(scores.softmax(axis=-1))
        return self.w_o(self._merge_heads(weights @ v))
