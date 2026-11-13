import pytest
import numpy as np
import houssem_torch as ht
from houssem_torch.tensor import Tensor
from houssem_torch.device import Device
from houssem_torch.dtypes import float32, int64


def test_import():
    assert ht.__version__ == "0.1.0"


def test_tensor_creation():
    t = Tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.ndim == 1
    assert t.numel == 3
    assert t.dtype == float32


def test_tensor_device_default():
    t = Tensor([1.0, 2.0])
    assert t.device.is_cpu


def test_tensor_device_explicit():
    t = Tensor([1.0], device="cpu")
    assert t.device.type == "cpu"


def test_invalid_device():
    with pytest.raises(ValueError):
        Tensor([1.0], device="tpu")


def test_tensor_shape_2d():
    data = np.zeros((3, 4), dtype="float32")
    t = Tensor(data)
    assert t.shape == (3, 4)
    assert t.ndim == 2
    assert t.numel == 12


def test_tensor_strides_contiguous():
    data = np.ones((2, 3), dtype="float32")
    t = Tensor(data)
    # row-major: stride for axis 0 = 3, axis 1 = 1
    assert t.strides == (3, 1)


def test_requires_grad_default():
    t = Tensor([1.0, 2.0])
    assert t.requires_grad is False


def test_requires_grad_set():
    t = Tensor([1.0, 2.0], requires_grad=True)
    assert t.requires_grad is True