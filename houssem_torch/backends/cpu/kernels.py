"""
CPU backend — all operations implemented via NumPy.
NumPy is used as a correctness reference only; the public API is always
through the Tensor object, never through NumPy directly.
"""
from __future__ import annotations
import numpy as np


# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------

class CPUStorage:
    """Owns the raw NumPy array for a CPU tensor."""

    def __init__(self, data: np.ndarray):
        self._array = np.asarray(data)

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def shape(self):
        return self._array.shape

    @property
    def strides(self):
        itemsize = self._array.itemsize
        return tuple(s // itemsize for s in self._array.strides)

    @property
    def dtype(self):
        return self._array.dtype

    def __repr__(self):
        return f"CPUStorage(shape={self.shape}, dtype={self.dtype})"


# ------------------------------------------------------------------
# Creation
# ------------------------------------------------------------------

def zeros(shape, dtype) -> CPUStorage:
    return CPUStorage(np.zeros(shape, dtype=dtype.numpy_name))

def ones(shape, dtype) -> CPUStorage:
    return CPUStorage(np.ones(shape, dtype=dtype.numpy_name))

def full(shape, fill_value, dtype) -> CPUStorage:
    return CPUStorage(np.full(shape, fill_value, dtype=dtype.numpy_name))

def arange(start, stop, step, dtype) -> CPUStorage:
    return CPUStorage(np.arange(start, stop, step, dtype=dtype.numpy_name))

def randn(shape, dtype) -> CPUStorage:
    arr = np.random.randn(*shape).astype(dtype.numpy_name)
    return CPUStorage(arr)

def from_numpy(array: np.ndarray) -> CPUStorage:
    return CPUStorage(array.copy())


# ------------------------------------------------------------------
# Shape operations
# ------------------------------------------------------------------

def reshape(storage: CPUStorage, new_shape) -> CPUStorage:
    return CPUStorage(storage.array.reshape(new_shape))

def transpose(storage: CPUStorage, axes=None) -> CPUStorage:
    return CPUStorage(storage.array.transpose(axes))

def squeeze(storage: CPUStorage, axis=None) -> CPUStorage:
    return CPUStorage(np.squeeze(storage.array, axis=axis))

def unsqueeze(storage: CPUStorage, axis: int) -> CPUStorage:
    return CPUStorage(np.expand_dims(storage.array, axis=axis))


# ------------------------------------------------------------------
# Pointwise operations
# ------------------------------------------------------------------

def add(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(a.array + b.array)

def subtract(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(a.array - b.array)

def multiply(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(a.array * b.array)

def divide(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(a.array / b.array)

def scalar_add(a: CPUStorage, scalar: float) -> CPUStorage:
    return CPUStorage(a.array + scalar)

def scalar_multiply(a: CPUStorage, scalar: float) -> CPUStorage:
    return CPUStorage(a.array * scalar)

def negate(a: CPUStorage) -> CPUStorage:
    return CPUStorage(-a.array)

def relu(a: CPUStorage) -> CPUStorage:
    return CPUStorage(np.maximum(0, a.array))

def exp(a: CPUStorage) -> CPUStorage:
    return CPUStorage(np.exp(a.array))

def log(a: CPUStorage) -> CPUStorage:
    return CPUStorage(np.log(a.array))

def pow(a: CPUStorage, exponent: float) -> CPUStorage:
    return CPUStorage(np.power(a.array, a.array.dtype.type(exponent)))

def sqrt(a: CPUStorage) -> CPUStorage:
    return CPUStorage(np.sqrt(a.array))

def gelu(a: CPUStorage) -> CPUStorage:
    """GELU, tanh approximation (matches torch gelu(approximate='tanh'))."""
    x = a.array
    c = x.dtype.type(np.sqrt(2.0 / np.pi))
    u = x * x                      # x*x*x is ~5x faster than x**3
    u *= 0.044715 * x
    u += x
    u *= c
    np.tanh(u, out=u)              # in place: one temporary for the chain
    u += 1.0
    u *= x
    u *= 0.5
    return CPUStorage(u)

def masked_fill(a: CPUStorage, mask: np.ndarray, value: float) -> CPUStorage:
    return CPUStorage(np.where(mask, a.array.dtype.type(value), a.array))


# ------------------------------------------------------------------
# Indexing
# ------------------------------------------------------------------

def getitem(a: CPUStorage, index) -> CPUStorage:
    return CPUStorage(np.asarray(a.array[index]))

def embedding(weight: CPUStorage, ids: np.ndarray) -> CPUStorage:
    return CPUStorage(weight.array[ids])


# ------------------------------------------------------------------
# Reductions
# ------------------------------------------------------------------

def sum(a, axis=None, keepdims=False):
    return CPUStorage(np.asarray(np.sum(a.array, axis=axis, keepdims=keepdims)))

def mean(a, axis=None, keepdims=False):
    return CPUStorage(np.asarray(np.mean(a.array, axis=axis, keepdims=keepdims)))

def max(a, axis=None, keepdims=False):
    return CPUStorage(np.asarray(np.max(a.array, axis=axis, keepdims=keepdims)))

def min(a, axis=None, keepdims=False):
    return CPUStorage(np.asarray(np.min(a.array, axis=axis, keepdims=keepdims)))



# ------------------------------------------------------------------
# Linear algebra
# ------------------------------------------------------------------

def matmul(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(np.matmul(a.array, b.array))


# ------------------------------------------------------------------
# Normalization primitives
# ------------------------------------------------------------------

def softmax(a: CPUStorage, axis: int = -1) -> CPUStorage:
    # numerically stable: subtract row max before exp
    x = a.array
    e_x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(e_x, out=e_x)           # in place: one full-size array in total
    e_x /= np.sum(e_x, axis=axis, keepdims=True)
    return CPUStorage(e_x)

def log_softmax(a: CPUStorage, axis: int = -1) -> CPUStorage:
    # log(softmax(x)) = (x - max) - log(sum(exp(x - max))), never takes log(0)
    shifted = a.array - np.max(a.array, axis=axis, keepdims=True)
    shifted -= np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return CPUStorage(shifted)

def layer_norm(a: CPUStorage, weight: CPUStorage, bias: CPUStorage,
               eps: float = 1e-5):
    """Normalize over the last axis. Returns (out, x_hat, rstd).

    x_hat = (x - mean) * rstd and rstd = 1 / sqrt(var + eps) are returned
    because the backward pass needs them.
    """
    x = a.array
    x_hat = x - np.mean(x, axis=-1, keepdims=True)    # centered, computed once
    var = np.mean(np.square(x_hat), axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + x.dtype.type(eps))
    x_hat *= rstd
    out = x_hat * weight.array
    out += bias.array
    return CPUStorage(out), x_hat, rstd
