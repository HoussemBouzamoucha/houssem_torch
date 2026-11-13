import pytest
import numpy as np
import houssem_torch as ht
from houssem_torch.tensor import Tensor


# ------------------------------------------------------------------
# Creation
# ------------------------------------------------------------------

def test_zeros():
    t = ht.zeros((2, 3))
    assert t.shape == (2, 3)
    assert np.all(t.numpy() == 0)

def test_ones():
    t = ht.ones((3, 3))
    assert t.shape == (3, 3)
    assert np.all(t.numpy() == 1)

def test_full():
    t = ht.full((2, 2), 7.0)
    assert np.all(t.numpy() == 7.0)

def test_arange():
    t = ht.arange(0, 5, 1)
    assert t.shape == (5,)
    assert np.allclose(t.numpy(), [0, 1, 2, 3, 4])

def test_randn_shape():
    t = ht.randn((4, 5))
    assert t.shape == (4, 5)

def test_tensor_from_list():
    t = ht.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)


# ------------------------------------------------------------------
# Shape operations
# ------------------------------------------------------------------

def test_reshape():
    t = ht.arange(0, 6, 1)
    t2 = t.reshape(2, 3)
    assert t2.shape == (2, 3)

def test_transpose_2d():
    t = ht.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
    t2 = t.transpose()
    assert t2.shape == (3, 2)

def test_squeeze():
    t = ht.zeros((1, 3, 1))
    t2 = t.squeeze()
    assert t2.shape == (3,)

def test_unsqueeze():
    t = ht.zeros((3,))
    t2 = t.unsqueeze(0)
    assert t2.shape == (1, 3)


# ------------------------------------------------------------------
# Pointwise arithmetic
# ------------------------------------------------------------------

def test_add_tensors():
    a = ht.tensor([1.0, 2.0, 3.0])
    b = ht.tensor([4.0, 5.0, 6.0])
    c = a + b
    assert np.allclose(c.numpy(), [5.0, 7.0, 9.0])

def test_add_scalar():
    a = ht.tensor([1.0, 2.0, 3.0])
    c = a + 10.0
    assert np.allclose(c.numpy(), [11.0, 12.0, 13.0])

def test_subtract():
    a = ht.tensor([5.0, 6.0])
    b = ht.tensor([1.0, 2.0])
    assert np.allclose((a - b).numpy(), [4.0, 4.0])

def test_multiply():
    a = ht.tensor([2.0, 3.0])
    b = ht.tensor([4.0, 5.0])
    assert np.allclose((a * b).numpy(), [8.0, 15.0])

def test_divide():
    a = ht.tensor([6.0, 9.0])
    b = ht.tensor([2.0, 3.0])
    assert np.allclose((a / b).numpy(), [3.0, 3.0])

def test_negate():
    a = ht.tensor([1.0, -2.0, 3.0])
    assert np.allclose((-a).numpy(), [-1.0, 2.0, -3.0])

def test_relu():
    a = ht.tensor([-1.0, 0.0, 2.0, -0.5])
    assert np.allclose(ht.relu(a).numpy(), [0.0, 0.0, 2.0, 0.0])

def test_exp():
    a = ht.tensor([0.0, 1.0])
    result = ht.exp(a).numpy()
    assert np.allclose(result, [1.0, np.e], rtol=1e-5)

def test_log():
    a = ht.tensor([1.0, np.e])
    result = ht.log(a).numpy()
    assert np.allclose(result, [0.0, 1.0], rtol=1e-5)


# ------------------------------------------------------------------
# Reductions
# ------------------------------------------------------------------

def test_sum_all():
    a = ht.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(a.sum().numpy(), 10.0)

def test_sum_axis():
    a = ht.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(a.sum(axis=0).numpy(), [4.0, 6.0])

def test_mean():
    a = ht.tensor([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(a.mean().numpy(), 2.5)

def test_max():
    a = ht.tensor([[1.0, 5.0], [3.0, 2.0]])
    assert np.allclose(a.max().numpy(), 5.0)

def test_max_axis():
    a = ht.tensor([[1.0, 5.0], [3.0, 2.0]])
    assert np.allclose(a.max(axis=1).numpy(), [5.0, 3.0])

def test_sum_keepdims():
    a = ht.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = a.sum(axis=1, keepdims=True)
    assert result.shape == (2, 1)


# ------------------------------------------------------------------
# Linear algebra
# ------------------------------------------------------------------

def test_matmul_2d():
    a = ht.tensor([[1.0, 2.0],
                   [3.0, 4.0]])
    b = ht.tensor([[5.0, 6.0],
                   [7.0, 8.0]])
    c = a @ b
    expected = np.array([[19.0, 22.0],
                         [43.0, 50.0]])
    assert np.allclose(c.numpy(), expected)

def test_matmul_shape_mismatch():
    a = ht.zeros((2, 3))
    b = ht.zeros((4, 2))
    with pytest.raises(ValueError):
        a @ b

def test_matmul_non_2d_raises():
    a = ht.zeros((3,))
    b = ht.zeros((3,))
    with pytest.raises(ValueError):
        a @ b


def test_rtruediv():
    a = ht.tensor([2.0, 4.0])
    assert np.allclose((8.0 / a).numpy(), [4.0, 2.0])


# ------------------------------------------------------------------
# Numerical edge cases
# ------------------------------------------------------------------

def test_add_large_values():
    a = ht.tensor([1e38, -1e38])
    b = ht.tensor([1e38, -1e38])
    result = (a + b).numpy()
    assert np.isfinite(result[1])

def test_log_near_zero():
    a = ht.tensor([1e-7])
    result = ht.log(a).numpy()
    assert np.isfinite(result[0])

def test_relu_all_negative():
    a = ht.tensor([-5.0, -3.0, -0.001])
    assert np.all(ht.relu(a).numpy() == 0.0)