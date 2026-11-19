from __future__ import annotations
from houssem_torch.autograd.graph import result
from houssem_torch.ops import grads


def layer_norm(x, weight, bias, eps: float = 1e-5):
    """LayerNorm over the last axis: weight * (x - mean) / sqrt(var + eps) + bias.

    x: (..., d)   weight, bias: (d,)   ->   (..., d)
    Variance is the biased (divide-by-d) one, as in PyTorch.
    """
    from houssem_torch.backends.cpu import kernels as cpu

    d = x.shape[-1]
    if weight.shape != (d,) or bias.shape != (d,):
        raise ValueError(
            f"layer_norm weight/bias must have shape ({d},), "
            f"got {weight.shape} and {bias.shape}."
        )
    out, x_hat, rstd = cpu.layer_norm(x._storage, weight._storage,
                                      bias._storage, eps)
    # saved for backward: x_hat (..., d) and rstd (..., 1)

    def _backward(g):
        dx, dweight, dbias = grads.layer_norm_backward(g, x_hat, rstd,
                                                       weight.numpy())
        x._accumulate_grad(dx)
        weight._accumulate_grad(dweight)
        bias._accumulate_grad(dbias)

    return result(out, x, (x, weight, bias), "layer_norm", _backward)
