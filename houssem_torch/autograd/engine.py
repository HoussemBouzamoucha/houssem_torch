from __future__ import annotations
import numpy as np
from houssem_torch.autograd.grad_mode import no_grad
from houssem_torch.fusion import fusion_backward_mode


def backward(root, grad=None, retain_grads: bool = False):
    """micrograd's backward: topo-sort, seed the root, run each `_backward`.

    Leaf gradients accumulate across calls (call `zero_grad` / set
    `.grad = None` between steps). Intermediate gradients are recomputed on
    every call and, unless `retain_grads=True`, freed as soon as they have
    been propagated to keep activation memory down.
    """
    from houssem_torch.tensor import Tensor

    if not root.requires_grad:
        raise RuntimeError(
            "backward() called on a tensor that does not require grad.")

    if grad is None:
        if root.numel != 1:
            raise RuntimeError("grad can only be implicitly created for scalar outputs.")
        seed = np.ones(root.shape, dtype=root.dtype.numpy_name)
    else:
        seed = grad.numpy() if isinstance(grad, Tensor) else np.asarray(
            grad, dtype=root.dtype.numpy_name)
    if seed.shape != root.shape:
        raise ValueError(f"grad shape {seed.shape} != tensor shape {root.shape}")

    # Iterative post-order DFS -> topological order (no recursion limit).
    topo, visited = [], set()
    stack = [(root, False)]
    while stack:
        t, expanded = stack.pop()
        if expanded:
            topo.append(t)
            continue
        if id(t) in visited:
            continue
        visited.add(id(t))
        stack.append((t, True))
        for parent in t._prev:
            if id(parent) not in visited:
                stack.append((parent, False))

    for t in topo:  # stale intermediate grads from a previous backward()
        if not t.is_leaf:
            t._grad = None

    root._accumulate_grad(seed)
    with no_grad(), fusion_backward_mode():
        for t in reversed(topo):
            if t.is_leaf or t._grad is None:
                continue
            try:
                t._backward(t._grad)
            except Exception as exc:
                if hasattr(exc, "add_note"):  # Python 3.11+
                    exc.add_note(f"(raised in the backward pass of op '{t._op}')")
                raise
            if not retain_grads:
                t._grad = None
