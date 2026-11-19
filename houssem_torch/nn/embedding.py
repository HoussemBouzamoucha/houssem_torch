from __future__ import annotations
from houssem_torch.nn import init
from houssem_torch.nn.module import Module
from houssem_torch.ops.embedding import embedding
from houssem_torch.parameter import Parameter


class Embedding(Module):
    """A lookup table: token id -> learned vector of size d.

    weight: (vocab_size, d), initialized Normal(0, 1/d) so that after the
    transformer's ×sqrt(d) scaling the embeddings have unit scale.

    Input: integer ids (...) -> output (..., d). The same table can be reused
    as the output projection (`x @ weight.T`), which is how the translation
    model shares one vocabulary between encoder, decoder and output.
    """

    def __init__(self, vocab_size: int, d: int):
        self.vocab_size = vocab_size
        self.d = d
        self.weight = Parameter(init.normal((vocab_size, d), std=d ** -0.5))

    def forward(self, ids):
        return embedding(ids, self.weight)
