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
        # micrograd-style graph state (set by autograd.graph.result):
        self._grad: Optional[np.ndarray] = None  # dLoss/dself, NumPy array
        self._prev: tuple = ()                   # parents that require grad
        self._op: str = ""                       # op name, for debugging
        self._backward = None                    # fn(g) -> pushes grads to _prev

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
        from houssem_torch.ops.shape import reshape
        new_shape = shape[0] if len(shape) == 1 and hasattr(shape[0], '__iter__') else shape
        return reshape(self, new_shape)

    def transpose(self, *axes) -> Tensor:
        """NumPy semantics: no args reverses all dims, else a full permutation.

        Use ``swapaxes(d0, d1)`` to swap just two dims (PyTorch ``transpose``).
        """
        from houssem_torch.ops.shape import permute
        ax = axes[0] if len(axes) == 1 and hasattr(axes[0], '__iter__') else (axes or None)
        return permute(self, ax)

    def permute(self, *axes) -> Tensor:
        from houssem_torch.ops.shape import permute
        ax = axes[0] if len(axes) == 1 and hasattr(axes[0], '__iter__') else axes
        return permute(self, ax)

    def swapaxes(self, axis0: int, axis1: int) -> Tensor:
        from houssem_torch.ops.shape import swapaxes
        return swapaxes(self, axis0, axis1)

    @property
    def mT(self) -> Tensor:
        """Swap the last two dimensions (batched matrix transpose)."""
        return self.swapaxes(-2, -1)

    def squeeze(self, axis=None) -> Tensor:
        from houssem_torch.ops.shape import squeeze
        return squeeze(self, axis)

    def unsqueeze(self, axis: int) -> Tensor:
        from houssem_torch.ops.shape import unsqueeze
        return unsqueeze(self, axis)

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

    def __pow__(self, exponent) -> Tensor:
        from houssem_torch.ops.pointwise import pow
        return pow(self, exponent)

    def sqrt(self) -> Tensor:
        from houssem_torch.ops.pointwise import sqrt
        return sqrt(self)

    def masked_fill(self, mask, value: float) -> Tensor:
        from houssem_torch.ops.pointwise import masked_fill
        return masked_fill(self, mask, value)

    def softmax(self, axis: int = -1) -> Tensor:
        from houssem_torch.ops.activation import softmax
        return softmax(self, axis)

    def log_softmax(self, axis: int = -1) -> Tensor:
        from houssem_torch.ops.activation import log_softmax
        return log_softmax(self, axis)

    def __getitem__(self, index) -> Tensor:
        from houssem_torch.ops.shape import getitem
        return getitem(self, index)

    def split(self, split_size_or_sections, axis: int = -1):
        from houssem_torch.ops.shape import split
        return split(self, split_size_or_sections, axis)

    # ------------------------------------------------------------------
    # Autograd
    # ------------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return not self._prev

    @property
    def grad(self) -> Optional[Tensor]:
        if self._grad is None:
            return None
        return Tensor(_storage=self._grad_storage(), device=self.device,
                      dtype=self.dtype)

    @grad.setter
    def grad(self, value) -> None:
        if value is None:
            self._grad = None
        else:
            # always a private copy, so accumulation can add into it in place
            self._grad = np.array(value.numpy() if isinstance(value, Tensor)
                                  else value, dtype=self.dtype.numpy_name,
                                  copy=True)

    def _grad_storage(self):
        from houssem_torch.backends.cpu import kernels as cpu
        return cpu.CPUStorage(self._grad)

    def _accumulate_grad(self, g: np.ndarray) -> None:
        """`self.grad += g` — call this from your `_backward` closures."""
        if not self.requires_grad:
            return
        if isinstance(g, np.generic):  # NumPy returns scalars for 0-d math
            g = np.asarray(g)
        if not isinstance(g, np.ndarray):
            raise TypeError(
                f"gradient must be a NumPy array, got {type(g).__name__}. "
                "Did a backward function forget to `return`?"
            )
        if g.shape != self.shape:
            raise RuntimeError(
                f"gradient shape {g.shape} != tensor shape {self.shape}"
            )
        if self._grad is None:
            # Leaves own a private, writable copy (closures may pass views or
            # the same array to several parents); intermediates are transient.
            self._grad = np.array(g, dtype=self.dtype.numpy_name, copy=True) \
                if self.is_leaf else g
        elif self.is_leaf:
            self._grad += g  # the leaf owns its array: add in place
        else:
            self._grad = self._grad + g  # may be shared with another node

    def detach(self) -> Tensor:
        """Same storage, no graph history."""
        return Tensor(_storage=self._storage, device=self.device, dtype=self.dtype)

    def backward(self, grad=None, retain_grads: bool = False) -> None:
        from houssem_torch.autograd.engine import backward
        backward(self, grad, retain_grads=retain_grads)
        

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