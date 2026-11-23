"""A ready-made encoder-decoder (sequence-to-sequence) transformer.

Use it as is (e.g. for translation), or as a reference for building your
own model from the blocks in `houssem_torch.nn`.

Layout ("pre-norm": each block normalizes its input first):

    source ids ──► embed ──► [EncoderLayer × N] ──► LayerNorm ──► memory
                                                                     │
    target ids ──► embed ──► [DecoderLayer × N] ◄────────────────────┘
    (shifted right)                 │
                                    ▼
                    LayerNorm ──► @ embeddingᵀ ──► logits (B, T, V)

One embedding table is shared by the encoder input, the decoder input and
the output projection (one joint source+target vocabulary).

With norm_position="post" (the original paper) each block normalizes after
its residual add instead, and the two final LayerNorms are dropped.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
import json
import numpy as np

from houssem_torch.autograd import no_grad
from houssem_torch.nn import (Dropout, Embedding, FeedForward, LayerNorm, Module,
                              ModuleList, MultiHeadAttention)
from houssem_torch.nn.embedding import EMBEDDING_INITS
from houssem_torch.ops.loss import SMOOTHING_MODES, cross_entropy
from houssem_torch.tensor import Tensor

NORM_POSITIONS = ("pre", "post")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerConfig:
    """Every size and setting of the model, decided once (the config file).

    vocab_size:       shared source+target vocabulary, e.g. 8192 or 16384.
    d_model:          width of every hidden vector.
    n_heads:          attention heads; d_model must divide evenly by it.
    d_ff:             hidden width of the feed-forward blocks (usually 4·d_model).
    n_encoder_layers, n_decoder_layers: stack depths (N).
    dropout:          dropout rate used throughout (0 disables it).
    max_len:          longest sequence the position encoding supports.
    activation:       "relu" (original paper) or "gelu".
    label_smoothing:  ε for the loss (paper: 0.1).
    pad_id, bos_id, eos_id: ids of the special tokens.

    Architecture switches (defaults keep earlier configs and checkpoints
    rebuilding exactly):
    norm_position:    "pre"  x + Sublayer(LayerNorm(x)), plus one final
                             LayerNorm per stack (stable without careful warmup);
                      "post" LayerNorm(x + Sublayer(x)), the original paper,
                             no final LayerNorm.
    attention_bias:   bias on the Q/K/V/output projections (paper code: False).
    ffn_dropout:      dropout between the feed-forward block's two layers;
                      None = same as `dropout` (paper code: 0.0).
    label_smoothing_mode: "uniform" (ε/V on every class, PyTorch's) or
                      "non_target" (ε over the classes other than the target
                      and padding). See `houssem_torch.cross_entropy`.
    embedding_init:   "normal" (Normal(0, 1/d)) or "xavier".
    embedding_padding_idx: if True, pad_id is the embedding's padding_idx:
                      its row starts at zero and lookups send it no gradient.
    """
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    dropout: float = 0.1
    max_len: int = 256
    activation: str = "relu"
    label_smoothing: float = 0.1
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    norm_position: str = "pre"
    attention_bias: bool = True
    ffn_dropout: float | None = None
    label_smoothing_mode: str = "uniform"
    embedding_init: str = "normal"
    embedding_padding_idx: bool = False

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by "
                             f"n_heads ({self.n_heads}).")
        if self.activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', "
                             f"got '{self.activation}'.")
        choices = {"norm_position": NORM_POSITIONS,
                   "label_smoothing_mode": SMOOTHING_MODES,
                   "embedding_init": EMBEDDING_INITS}
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key} must be one of {allowed}, "
                                 f"got {getattr(self, key)!r}.")
        if self.ffn_dropout is not None and not 0.0 <= self.ffn_dropout < 1.0:
            raise ValueError(f"ffn_dropout must be in [0, 1) or None, "
                             f"got {self.ffn_dropout}.")
        specials = (self.pad_id, self.bos_id, self.eos_id)
        if len(set(specials)) != 3 or not all(0 <= s < self.vocab_size
                                              for s in specials):
            raise ValueError(f"pad/bos/eos ids must be distinct and inside the "
                             f"vocabulary, got {specials}.")

    @property
    def ffn_dropout_rate(self) -> float:
        """The feed-forward block's inner dropout (ffn_dropout, else dropout)."""
        return self.dropout if self.ffn_dropout is None else self.ffn_dropout

    @classmethod
    def from_dict(cls, values: dict) -> "TransformerConfig":
        """Build from a dict. Unknown keys are an error (catches typos)."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str) -> "TransformerConfig":
        """Load the config file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Position encoding
# ---------------------------------------------------------------------------

def sinusoidal_position_encoding(max_len: int, d_model: int) -> np.ndarray:
    """The paper's fixed (not learned) position signal, shape (max_len, d).

        PE[pos, 2i]   = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

    Attention by itself has no notion of word order; adding PE gives every
    position a unique pattern of waves, with wavelengths from 2π to 10000·2π,
    so the model can tell positions (and distances between them) apart.
    """
    positions = np.arange(max_len, dtype=np.float64)[:, None]
    rates = 1.0 / (10000.0 ** (np.arange(0, d_model, 2, dtype=np.float64) / d_model))
    pe = np.zeros((max_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(positions * rates)
    pe[:, 1::2] = np.cos(positions * rates[: d_model // 2])
    return pe.astype(np.float32)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

def residual(x, sublayer, norm, dropout, pre: bool):
    """One residual block around `sublayer`, dropout on its output.

        pre:  x + dropout(sublayer(norm(x)))
        post: norm(x + dropout(sublayer(x)))     (the original paper)

    The `x + ...` connection lets information (and gradients) flow straight
    through the stack, so each layer only learns a refinement.
    """
    if pre:
        return x + dropout(sublayer(norm(x)))
    return norm(x + dropout(sublayer(x)))


def _attention(cfg: TransformerConfig, cross: bool = False) -> MultiHeadAttention:
    return MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cross=cross,
                              bias=cfg.attention_bias)


def _feed_forward(cfg: TransformerConfig) -> FeedForward:
    return FeedForward(cfg.d_model, cfg.d_ff, cfg.ffn_dropout_rate, cfg.activation)


class EncoderLayer(Module):
    """One encoder layer: self-attention block + feed-forward block.

    pre-norm (default):
        x = x + dropout(SelfAttention(LayerNorm(x)))   every word sees every word
        x = x + dropout(FeedForward(LayerNorm(x)))
    post-norm: x = LayerNorm(x + dropout(Sublayer(x))) for each block.

    Shapes: (B, S, d) -> (B, S, d).
    """

    def __init__(self, cfg: TransformerConfig):
        self.pre = cfg.norm_position == "pre"
        self.norm_attn = LayerNorm(cfg.d_model)
        self.self_attn = _attention(cfg)
        self.norm_ffn = LayerNorm(cfg.d_model)
        self.ffn = _feed_forward(cfg)
        self.dropout = Dropout(cfg.dropout)

    def forward(self, x, src_pad_mask=None):
        """src_pad_mask: (B, S) bool, True at source padding positions."""
        x = residual(x, lambda h: self.self_attn(h, key_padding_mask=src_pad_mask),
                     self.norm_attn, self.dropout, self.pre)
        return residual(x, self.ffn, self.norm_ffn, self.dropout, self.pre)


class DecoderLayer(Module):
    """One decoder layer: masked self-attention + cross-attention + feed-forward.

    pre-norm (default):
        y = y + dropout(MaskedSelfAttention(LayerNorm(y)))  only earlier target tokens
        y = y + dropout(CrossAttention(LayerNorm(y), memory))  looks at the source
        y = y + dropout(FeedForward(LayerNorm(y)))
    post-norm: y = LayerNorm(y + dropout(Sublayer(y))) for each block.

    Shapes: y (B, T, d), memory (B, S, d) -> (B, T, d).
    """

    def __init__(self, cfg: TransformerConfig):
        self.pre = cfg.norm_position == "pre"
        self.norm_self = LayerNorm(cfg.d_model)
        self.self_attn = _attention(cfg)
        self.norm_cross = LayerNorm(cfg.d_model)
        self.cross_attn = _attention(cfg, cross=True)
        self.norm_ffn = LayerNorm(cfg.d_model)
        self.ffn = _feed_forward(cfg)
        self.dropout = Dropout(cfg.dropout)

    def forward(self, y, memory, tgt_pad_mask=None, src_pad_mask=None):
        """tgt_pad_mask: (B, T) bool, True at target padding positions.
        src_pad_mask: (B, S) bool, True at source padding positions."""
        y = residual(y, lambda h: self.self_attn(h, key_padding_mask=tgt_pad_mask,
                                                 causal=True),
                     self.norm_self, self.dropout, self.pre)
        y = residual(y, lambda h: self.cross_attn(h, memory=memory,
                                                  key_padding_mask=src_pad_mask),
                     self.norm_cross, self.dropout, self.pre)
        return residual(y, self.ffn, self.norm_ffn, self.dropout, self.pre)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class Transformer(Module):
    """Encoder-decoder transformer: source sequence in, target sequence out.

    Main entry points:
        model.loss(src, tgt_in, tgt_out)  -> scalar loss tensor (training)
        model(src, tgt_in)                -> logits (B, T, V)
        model.translate_greedy(src)       -> target ids (inference)

    All token inputs are integer NumPy arrays (or int64 Tensors):
        src     (B, S): source ids, padded with pad_id
        tgt_in  (B, T): target ids shifted right: <bos> w1 w2 ...
        tgt_out (B, T): target ids, the labels:   w1 w2 ... <eos>
    """

    def __init__(self, cfg: TransformerConfig):
        self.cfg = cfg
        self.embedding = Embedding(  # one table shared by encoder, decoder, output
            cfg.vocab_size, cfg.d_model, init=cfg.embedding_init,
            padding_idx=cfg.pad_id if cfg.embedding_padding_idx else None)
        self._pe = sinusoidal_position_encoding(cfg.max_len, cfg.d_model)  # constant
        self.embed_dropout = Dropout(cfg.dropout)
        self.encoder = ModuleList(EncoderLayer(cfg)
                                  for _ in range(cfg.n_encoder_layers))
        self.decoder = ModuleList(DecoderLayer(cfg)
                                  for _ in range(cfg.n_decoder_layers))
        # Pre-norm leaves the residual stream unnormalized, so each stack ends
        # with one LayerNorm. Post-norm already ends on one: no extra layer
        # (and no extra parameters in the state dict).
        if cfg.norm_position == "pre":
            self.encoder_norm = LayerNorm(cfg.d_model)
            self.decoder_norm = LayerNorm(cfg.d_model)
        else:
            self.encoder_norm = self.decoder_norm = None

    def _final_norm(self, norm, x):
        return x if norm is None else norm(x)

    # --- building blocks -------------------------------------------------

    def pad_mask(self, ids) -> np.ndarray:
        """(B, L) bool, True where the token is padding."""
        arr = ids.numpy() if hasattr(ids, "_storage") else np.asarray(ids)
        return arr == self.cfg.pad_id

    def embed(self, ids):
        """ids (B, L) -> embedding × sqrt(d) + position encoding, dropout: (B, L, d)."""
        length = np.shape(ids.numpy() if hasattr(ids, "_storage") else ids)[1]
        if length > self.cfg.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len "
                             f"{self.cfg.max_len}.")
        x = self.embedding(ids) * float(np.sqrt(self.cfg.d_model))
        x = x + Tensor(self._pe[:length])
        return self.embed_dropout(x)

    def encode(self, src):
        """Source ids (B, S) -> memory (B, S, d), plus the padding mask."""
        src_pad = self.pad_mask(src)
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, src_pad)
        return self._final_norm(self.encoder_norm, x), src_pad

    def decode(self, tgt_in, memory, src_pad):
        """Target ids (B, T) + memory -> logits (B, T, V).

        The output projection reuses the embedding table: logits = h @ Eᵀ.
        """
        tgt_pad = self.pad_mask(tgt_in)
        y = self.embed(tgt_in)
        for layer in self.decoder:
            y = layer(y, memory, tgt_pad, src_pad)
        h = self._final_norm(self.decoder_norm, y)
        B, T, d = h.shape
        logits = h.reshape(B * T, d) @ self.embedding.weight.transpose()
        return logits.reshape(B, T, self.cfg.vocab_size)

    # --- training --------------------------------------------------------

    def forward(self, src, tgt_in):
        """Teacher-forced pass over the whole target sequence: logits (B, T, V)."""
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)

    def loss(self, src, tgt_in, tgt_out):
        """Mean label-smoothed cross-entropy over non-padding target tokens."""
        return cross_entropy(self(src, tgt_in), tgt_out,
                             label_smoothing=self.cfg.label_smoothing,
                             ignore_index=self.cfg.pad_id,
                             label_smoothing_mode=self.cfg.label_smoothing_mode)

    # --- inference -------------------------------------------------------

    def translate_greedy(self, src, max_new_tokens: int | None = None) -> np.ndarray:
        """Translate by always picking the most likely next token.

        Runs in eval mode with no graph. Returns int ids (B, ≤ max_new_tokens),
        without the leading <bos>; positions after a sentence's <eos> are
        pad_id. Simple, not fast: the decoder is re-run over the whole prefix
        at every step. (The fast engine will cache keys/values and add beam
        search.)
        """
        cfg = self.cfg
        max_new = max_new_tokens or cfg.max_len - 1
        was_training = self.training
        self.eval()
        try:
            with no_grad():
                memory, src_pad = self.encode(src)
                B = memory.shape[0]
                ys = np.full((B, 1), cfg.bos_id, dtype=np.int64)
                finished = np.zeros(B, dtype=bool)
                for _ in range(max_new):
                    logits = self.decode(ys, memory, src_pad).numpy()[:, -1]
                    nxt = np.where(finished, cfg.pad_id, logits.argmax(axis=-1))
                    ys = np.concatenate([ys, nxt[:, None]], axis=1)
                    finished |= nxt == cfg.eos_id
                    if finished.all():
                        break
        finally:
            self.train(was_training)
        return ys[:, 1:]
