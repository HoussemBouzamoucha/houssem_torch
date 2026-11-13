from __future__ import annotations
import numpy as np
from typing import Union, Tuple, Optional, Sequence
from houssem_torch.device import Device
from houssem_torch.dtypes import DType, float32

ShapeLike = Union[Tuple[int, ...], Sequence[int]]


class Tensor:
    """
    Core tensor object for Houssem Torch.
    Storage is owned by the backend (CPUStorage or CUDAStorage).
    The Tensor object holds metadata and delegates all computation
    to the appropriate backend.
    """

    def __init__(
        self,
        data=None,
        device: Union[Device, str] = "cpu",
        dtype: Optional[DType] = None,
        requires_grad: bool = False,
        _storage=None,  # internal: pass a CPUStorage directly
    ):
        if isinstance(device, str):
            device = Device(device)

        self.device = device
        self.dtype = dtype or float32
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._grad_fn = None

        if _storage is not None:
            # created internally by an op: storage already prepared
            self._storage = _storage
        elif data is None:
            raise ValueError("Tensor requires either data or _storage.")
        else:
            # created by user: wrap the data
            self._storage = self._init_storage(data)

    def _init_storage(self, data):
        from houssem_torch.backends.cpu import kernels as cpu

        if isinstance(data, np.ndarray):
            arr = data.astype(self.dtype.numpy_name)
        else:
            arr = np.array(data, dtype=self.dtype.numpy_name)

        return cpu.CPUStorage(arr)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._storage.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def strides(self) -> Tuple[int, ...]:
        return tuple(self._storage.strides)

    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array. CPU tensors only."""
        if not self.device.is_cpu:
            raise RuntimeError("Cannot call .numpy() on a non-CPU tensor.")
        return self._storage.array

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------

    def reshape(self, *shape) -> Tensor:
        from houssem_torch.backends.cpu import kernels as cpu
        new_shape = shape[0] if len(shape) == 1 and hasattr(shape[0], '__iter__') else shape
        return Tensor(_storage=cpu.reshape(self._storage, new_shape),
                      device=self.device, dtype=self.dtype)

    def transpose(self, *axes) -> Tensor:
        from houssem_torch.backends.cpu import kernels as cpu
        ax = axes[0] if len(axes) == 1 and hasattr(axes[0], '__iter__') else (axes or None)
        return Tensor(_storage=cpu.transpose(self._storage, ax),
                      device=self.device, dtype=self.dtype)

    def squeeze(self, axis=None) -> Tensor:
        from houssem_torch.backends.cpu import kernels as cpu
        return Tensor(_storage=cpu.squeeze(self._storage, axis),
                      device=self.device, dtype=self.dtype)

    def unsqueeze(self, axis: int) -> Tensor:
        from houssem_torch.backends.cpu import kernels as cpu
        return Tensor(_storage=cpu.unsqueeze(self._storage, axis),
                      device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other) -> Tensor:
        from houssem_torch.ops.pointwise import add
        return add(self, other)

    def __radd__(self, other) -> Tensor:
        from houssem_torch.ops.pointwise import add
        return add(self, other)

    def __sub__(self, other) -> Tensor:
        from houssem_torch.ops.pointwise import subtract
        return subtract(self, other)

    def __rsub__(self, other) -> Tensor:
        from houssem_torch.ops.pointwise import subtract
        return subtract(other, self)

    def __mul__(self, other) -> Tensor:
        from houssem_torch.ops.pointwise import multiply
        return multiply(self, other)

    def __rmul__(self, other) -> Tensor:
        from houssem_torch.ops.pointwise import multiply
        return multiply(self, other)

    def __truediv__(self, other) -> Tensor:
        from houssem_torch.ops.pointwise import divide
        return divide(self, other)
    
    def __rtruediv__(self, other) -> Tensor:
        from houssem_torch.ops.pointwise import divide
        return divide(other, self)
        

    def __neg__(self) -> Tensor:
        from houssem_torch.ops.pointwise import negate
        return negate(self)

    @property
    def is_leaf(self) -> bool:
        return self._grad_fn is None

    def detach(self) -> Tensor:
        """Same storage, no graph history."""
        return Tensor(_storage=self._storage, device=self.device, dtype=self.dtype)

    def backward(self, grad=None) -> None:
        from houssem_torch.autograd.engine import backward
        backward(self, grad)
        

    def __matmul__(self, other) -> Tensor:
        from houssem_torch.ops.linalg import matmul
        return matmul(self, other)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False) -> Tensor:
        from houssem_torch.ops.reductions import sum
        return sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False) -> Tensor:
        from houssem_torch.ops.reductions import mean
        return mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False) -> Tensor:
        from houssem_torch.ops.reductions import max
        return max(self, axis=axis, keepdims=keepdims)

    
    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"device={self.device})\n"
            f"{self._storage.array}"
        )
    def __rtruediv__(self, other) -> Tensor:
        from houssem_torch.ops.pointwise import divide
        return divide(other, self)