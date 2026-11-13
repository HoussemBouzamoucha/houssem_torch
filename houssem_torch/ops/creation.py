from __future__ import annotations
from typing import Union, Tuple, Sequence
from houssem_torch.dtypes import DType, float32, int64
from houssem_torch.device import Device

ShapeLike = Union[Tuple[int, ...], Sequence[int]]


def _make_tensor(storage, device, dtype):
    from houssem_torch.tensor import Tensor
    return Tensor(_storage=storage, device=device, dtype=dtype)


def zeros(shape: ShapeLike, dtype: DType = float32,
          device: Union[Device, str] = "cpu"):
    from houssem_torch.backends.cpu import kernels as cpu
    device = Device(device) if isinstance(device, str) else device
    return _make_tensor(cpu.zeros(shape, dtype), device, dtype)


def ones(shape: ShapeLike, dtype: DType = float32,
         device: Union[Device, str] = "cpu"):
    from houssem_torch.backends.cpu import kernels as cpu
    device = Device(device) if isinstance(device, str) else device
    return _make_tensor(cpu.ones(shape, dtype), device, dtype)


def full(shape: ShapeLike, fill_value: float, dtype: DType = float32,
         device: Union[Device, str] = "cpu"):
    from houssem_torch.backends.cpu import kernels as cpu
    device = Device(device) if isinstance(device, str) else device
    return _make_tensor(cpu.full(shape, fill_value, dtype), device, dtype)


def arange(start: float, stop: float, step: float = 1.0,
           dtype: DType = float32, device: Union[Device, str] = "cpu"):
    from houssem_torch.backends.cpu import kernels as cpu
    device = Device(device) if isinstance(device, str) else device
    return _make_tensor(cpu.arange(start, stop, step, dtype), device, dtype)


def randn(shape: ShapeLike, dtype: DType = float32,
          device: Union[Device, str] = "cpu"):
    from houssem_torch.backends.cpu import kernels as cpu
    device = Device(device) if isinstance(device, str) else device
    return _make_tensor(cpu.randn(shape, dtype), device, dtype)


def tensor(data, dtype: DType = float32,
           device: Union[Device, str] = "cpu", requires_grad: bool = False):
    from houssem_torch.tensor import Tensor
    return Tensor(data, device=device, dtype=dtype, requires_grad=requires_grad)