from __future__ import annotations
from houssem_torch.nn import init as _init
from houssem_torch.nn.module import Module
from houssem_torch.ops.embedding import embedding
from houssem_torch.parameter import Parameter

EMBEDDING_INITS = ("normal", "xavier")


class Embedding(Module):
    """A lookup table: token id -> learned vector of size d.

    weight: (vocab_size, d). init picks how it starts:
      "normal" (default): Normal(0, 1/d), so that after the transformer's
                ×sqrt(d) scaling the embeddings have unit scale.
      "xavier": uniform in ±sqrt(6 / (vocab_size + d)), what PyTorch models
                get from `nn.init.xavier_uniform_` over every matrix.

    padding_idx: that row starts at zero and the lookup never sends it a
    gradient, so padding tokens embed to a fixed vector (PyTorch's
    `padding_idx`). A tied output projection still updates the row.

    Input: integer ids (...) -> output (..., d). The same table can be reused
    as the output projection (`x @ weight.T`), which is how the translation
    model shares one vocabulary between encoder, decoder and output.
    """

    def __init__(self, vocab_size: int, d: int, padding_idx: int | None = None,
                 init: str = "normal"):
        if init not in EMBEDDING_INITS:
            raise ValueError(f"init must be one of {EMBEDDING_INITS}, got {init!r}.")
        if padding_idx is not None and not 0 <= padding_idx < vocab_size:
            raise ValueError(f"padding_idx {padding_idx} out of range "
                             f"[0, {vocab_size}).")
        self.vocab_size = vocab_size
        self.d = d
        self.padding_idx = padding_idx
        if init == "xavier":
            table = _init.xavier_uniform((vocab_size, d))
        else:
            table = _init.normal((vocab_size, d), std=d ** -0.5)
        if padding_idx is not None:
            table[padding_idx] = 0.0
        self.weight = Parameter(table)

    def forward(self, ids):
        return embedding(ids, self.weight, self.padding_idx)
