from __future__ import annotations
import numpy as np
from typing import Union, Tuple, Optional
from houssem_torch.device import Device
from houssem_torch.dtypes import DType, float32

ShapeLike = Union[Tuple[int, ...], list]


class Tensor:
    """
    Core tensor object for Houssem Torch.
    Backed by a NumPy array on CPU for now.
    CUDA backend will be added in Phase 2.
    """

    def __init__(
        self,
        data: np.ndarray,
        device: Union[Device, str] = "cpu",
        dtype: Optional[DType] = None,
        requires_grad: bool = False,
    ):
        if isinstance(device, str):
            device = Device(device)

        self.device = device
        self.dtype = dtype or float32
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._grad_fn = None  # set by autograd

        # Storage: NumPy array on CPU for now
        self._data = np.array(data, dtype=self.dtype.numpy_name)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def numel(self) -> int:
        return int(self._data.size)

    @property
    def strides(self) -> Tuple[int, ...]:
        # strides in number of elements, not bytes
        itemsize = self._data.itemsize
        return tuple(s // itemsize for s in self._data.strides)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Tensor({self._data}, "
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"device={self.device})"
        )