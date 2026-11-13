from houssem_torch.tensor import Tensor


class Parameter(Tensor):
    """A leaf tensor that is always trainable."""

    def __init__(self, data, device="cpu", dtype=None):
        super().__init__(data, device=device, dtype=dtype, requires_grad=True)