from houssem_torch.autograd.grad_mode import is_grad_enabled


class Function:
    """A node in the graph. Stores its inputs; subclasses implement backward."""

    def __init__(self, *inputs):
        self.inputs = inputs
        self.out = None  # detached copy of the output, set by record()

    def backward(self, grad):
        """Return one gradient (Tensor or None) per input, in order."""
        raise NotImplementedError


def needs_grad(*tensors) -> bool:
    return is_grad_enabled() and any(
        getattr(t, "requires_grad", False) for t in tensors
    )


def record(out, fn_cls, *inputs, **kwargs):
    """If any input requires grad, attach a backward node to `out`."""
    if needs_grad(*inputs):
        fn = fn_cls(*inputs, **kwargs)
        fn.out = out.detach()
        out.requires_grad = True
        out._grad_fn = fn
    return out