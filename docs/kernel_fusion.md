# CPU pointwise kernel fusion

Fusion is opt-in and requires the `fusion` extra:

```powershell
pip install -e ".[fusion]"
```

```python
import houssem_torch as ht

with ht.fusion_mode():
    y = ht.relu(x * 2.0 + 3.0)

result = y.numpy()
```

Within the context, supported pointwise operations build a lazy expression
region. Compatible regions are lowered to a Numba-compiled CPU loop when the
result is first needed. Accessing `.numpy()`, starting a reduction, or using an
unsupported operation materializes the current region. Use
`with ht.fusion_mode(enabled=False):` to force eager execution for comparison
and debugging. Fusion is disabled by default.

The expression region is separate from the autograd graph. Each ordinary
pointwise operation still records its existing backward node and inputs, so
gradient propagation, broadcasting reductions, `requires_grad`, and gradient
accumulation continue to use the normal autograd engine. This first version
fuses forward execution only; backward remains operation-by-operation.

The initial CPU implementation supports `add`, `subtract`, `multiply`,
`divide`, `relu`, `exp`, and `log` for contiguous `float32` and `int64` inputs
when the operation preserves the eager output dtype. Tensor broadcasting and
scalar operands are supported. Non-contiguous inputs, incompatible shapes,
mixed dtypes, unsupported dtypes, and integer `divide`/`exp`/`log` use the
existing eager NumPy backend. Reductions and matrix multiplication are
boundaries and are not fused.

Input values are snapshotted when an expression leaf is created, matching the
eager forward result even if the source array is subsequently changed before
materialization. This costs one copy per leaf; performance depends on workload
and tensor size. Run `benchmarks/kernel_fusion.py` to compare eager Houssem
Torch, fused Houssem Torch, PyTorch eager, and `torch.compile` where available.
