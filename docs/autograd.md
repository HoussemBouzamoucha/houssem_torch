# Autograd (micrograd-style)

Houssem Torch uses the same design as
[micrograd](https://github.com/karpathy/micrograd), but each node holds a whole
NumPy array instead of a single scalar. Every op writes its own gradient by
hand; `backward()` only orders the graph and calls those functions.

## Graph state on a `Tensor`

| Field | Meaning |
|---|---|
| `_prev` | parents that require grad (empty for leaves) |
| `_op` | op name, for debugging |
| `_backward(g)` | your function: `g` is dLoss/d(this tensor) as a NumPy array |
| `_grad` / `.grad` | accumulated gradient (NumPy array / `Tensor` view of it) |

## Writing an op

```python
from houssem_torch.autograd import result, unbroadcast
from houssem_torch.backends.cpu.kernels import CPUStorage

def mul(a, b):
    def _backward(g):
        a._accumulate_grad(unbroadcast(g * b.numpy(), a.shape))  # d/da = b
        b._accumulate_grad(unbroadcast(g * a.numpy(), b.shape))  # d/db = a
    out = CPUStorage(a.numpy() * b.numpy())
    return result(out, a, (a, b), "mul", _backward)
```

Rules:

- **Accumulate, never assign.** Call `parent._accumulate_grad(...)`; it adds to
  the existing gradient and ignores parents that don't require grad.
- **Return the parent's exact shape.** If the forward broadcast an input, sum
  the gradient back with `unbroadcast(g, parent.shape)`. A shape mismatch
  raises an error that names the op.
- **Take `g` as an argument; don't capture the output tensor.** Capturing
  `out` in its own `_backward` creates a reference cycle, which keeps
  activations alive until Python's cycle collector runs. If you need the
  forward value (as `exp` does), capture the output *storage* instead.
- **Read inputs with `.numpy()` inside `_backward`.** The closure runs under
  `no_grad`, so NumPy math there builds no graph.
- **Verify every new op against PyTorch** by adding a case to `CASES` in
  `tests/test_autograd.py`.

## `backward()`

`loss.backward()` topologically sorts the graph from `loss`, seeds
`loss.grad = 1`, and calls each node's `_backward` in reverse order.

- Leaf gradients **accumulate across calls**. Set `p.grad = None` between
  optimizer steps.
- Intermediate gradients are recomputed on each call, so calling
  `backward()` twice on the same graph is safe. They are freed after use to
  save memory; pass `retain_grads=True` to keep them for inspection.
