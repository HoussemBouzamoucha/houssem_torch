from __future__ import annotations
from houssem_torch.autograd.graph import result
from houssem_torch.backends.cpu import fast
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
    xv, wv, bv = x.numpy(), weight.numpy(), bias.numpy()
    if fast.use(xv, wv, bv) and xv.dtype == wv.dtype == bv.dtype:
        out2d, x_hat, rstd = fast.layer_norm_forward(
            fast.rows(xv), fast.flat(wv), fast.flat(bv), eps)
        out = cpu.CPUStorage(out2d.reshape(xv.shape))
        backward_fn = fast.layer_norm_backward
        rows = fast.rows
    else:
        out, x_hat, rstd = cpu.layer_norm(x._storage, weight._storage,
                                          bias._storage, eps)
        backward_fn = grads.layer_norm_backward
        rows = _same
    # saved for backward: x_hat (..., d) and rstd (..., 1)

    def _backward(g):
        dx, dweight, dbias = backward_fn(rows(g), x_hat, rstd, wv)
        x._accumulate_grad(dx.reshape(xv.shape))
        weight._accumulate_grad(dweight)
        bias._accumulate_grad(dbias)

    return result(out, x, (x, weight, bias), "layer_norm", _backward)


def _same(a):
    return a
