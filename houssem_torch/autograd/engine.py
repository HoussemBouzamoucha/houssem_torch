from __future__ import annotations
from houssem_torch.autograd.grad_mode import no_grad
from houssem_torch.fusion import fusion_backward_mode


def backward(root, grad=None):
    from houssem_torch.tensor import Tensor
    from houssem_torch.ops.creation import ones

    if not root.requires_grad:
        raise RuntimeError("backward() called on a tensor that does not require grad.")

    if grad is None:
        if root.numel != 1:
            raise RuntimeError("grad can only be implicitly created for scalar outputs.")
        grad = ones(root.shape, dtype=root.dtype, device=root.device)
    elif not isinstance(grad, Tensor):
        grad = Tensor(grad, dtype=root.dtype, device=root.device)
    if grad.shape != root.shape:
        raise ValueError(f"grad shape {grad.shape} != tensor shape {root.shape}")

    # Iterative post-order DFS -> topological order (no recursion limit).
    topo, seen = [], set()
    stack = [(root, False)]
    while stack:
        t, done = stack.pop()
        if done:
            topo.append(t)
            continue
        if id(t) in seen:
            continue
        seen.add(id(t))
        stack.append((t, True))
        if t._grad_fn is not None:
            for inp in t._grad_fn.inputs:
                if isinstance(inp, Tensor) and inp.requires_grad and id(inp) not in seen:
                    stack.append((inp, False))

    grads = {id(root): grad}
    with no_grad(), fusion_backward_mode():
        for t in reversed(topo):
            g = grads.pop(id(t), None)
            if g is None:
                continue
            fn = t._grad_fn
            if fn is None:  # leaf: accumulate into .grad
                t.grad = g if t.grad is None else t.grad + g
                continue
            for inp, ig in zip(fn.inputs, fn.backward(g)):
                if ig is None or not isinstance(inp, Tensor) or not inp.requires_grad:
                    continue
                prev = grads.get(id(inp))
                grads[id(inp)] = ig if prev is None else prev + ig