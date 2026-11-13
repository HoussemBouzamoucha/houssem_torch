from __future__ import annotations


def _wrap(storage, ref_tensor):
    from houssem_torch.tensor import Tensor
    return Tensor(_storage=storage, device=ref_tensor.device,
                  dtype=ref_tensor.dtype)


def matmul(a, b):
    from houssem_torch.backends.cpu import kernels as cpu

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"matmul requires 2D tensors, got {a.ndim}D and {b.ndim}D."
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul shape mismatch: {a.shape} @ {b.shape}."
        )
    return _wrap(cpu.matmul(a._storage, b._storage), a)