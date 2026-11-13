from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import threading
from typing import Any

import numpy as np

from houssem_torch.fusion import is_fusion_backward, is_fusion_enabled


@dataclass(frozen=True)
class _Leaf:
    source: Any
    values: np.ndarray
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _Expr:
    op: str
    left: _Expr | _Leaf
    right: _Expr | _Leaf | None = None


class FusedCPUStorage:
    """Lazy CPU storage for a pointwise expression compiled as one Numba loop."""

    def __init__(self, expr: _Expr, shape: tuple[int, ...], dtype: np.dtype):
        self.expr = expr
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._array = None
        self._lock = threading.RLock()

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            with self._lock:
                if self._array is None:
                    if is_fusion_backward():
                        self._array = _execute_numpy(self.expr)
                    else:
                        self._array = _execute(self.expr, self.shape, self._dtype)
        return self._array

    @property
    def materialized(self) -> bool:
        return self._array is not None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        stride = 1
        result = []
        for size in reversed(self.shape):
            result.append(stride)
            stride *= size
        return tuple(reversed(result))

    @property
    def dtype(self) -> np.dtype:
        return self._dtype


_SUPPORTED_OPS = {"add", "subtract", "multiply", "divide", "relu", "exp", "log"}
_MAX_FUSED_OPS = 16


def try_fuse(op: str, *tensors) -> FusedCPUStorage | None:
    """Build or extend a lazy region; return None for eager backend fallback."""
    if (not is_fusion_enabled() or is_fusion_backward()
            or op not in _SUPPORTED_OPS):
        return None
    if not tensors or any(not tensor.device.is_cpu for tensor in tensors):
        return None

    dtype = tensors[0].dtype
    if any(tensor.dtype != dtype for tensor in tensors):
        return None
    if dtype.name not in {"float32", "int64"}:
        return None
    if dtype.name == "int64" and op in {"divide", "exp", "log"}:
        return None
    expected_dtype = np.dtype(dtype.numpy_name)
    if any(np.dtype(tensor._storage.dtype) != expected_dtype for tensor in tensors):
        return None

    try:
        output_shape = tuple(np.broadcast_shapes(*(tensor.shape for tensor in tensors)))
    except ValueError:
        return None

    leaf_map: dict[int, _Leaf] = {}
    operands = []
    for tensor in tensors:
        storage = tensor._storage
        if isinstance(storage, FusedCPUStorage) and not storage.materialized:
            _collect_leaves(storage.expr, leaf_map)
            operands.append(storage.expr)
            continue

        source = storage
        leaf = leaf_map.get(id(source))
        if leaf is None:
            values = source.array
            if not values.flags.c_contiguous:
                return None
            snapshot = np.array(values, copy=True, order="C")
            leaf = _Leaf(source, snapshot, tuple(tensor.shape))
            leaf_map[id(source)] = leaf
        operands.append(leaf)

    expr = _Expr(op, operands[0], operands[1] if len(operands) == 2 else None)
    if _operation_count(expr) > _MAX_FUSED_OPS:
        return None
    return FusedCPUStorage(expr, output_shape, expected_dtype)


def _operation_count(expr: _Expr | _Leaf) -> int:
    count = 0
    pending = [expr]
    while pending:
        node = pending.pop()
        if isinstance(node, _Leaf):
            continue
        count += 1
        pending.append(node.left)
        if node.right is not None:
            pending.append(node.right)
    return count


def _collect_leaves(expr: _Expr | _Leaf, leaf_map: dict[int, _Leaf]) -> None:
    if isinstance(expr, _Leaf):
        leaf_map.setdefault(id(expr.source), expr)
        return
    _collect_leaves(expr.left, leaf_map)
    if expr.right is not None:
        _collect_leaves(expr.right, leaf_map)


def _walk_leaves(expr: _Expr | _Leaf, leaves: list[_Leaf]) -> None:
    if isinstance(expr, _Leaf):
        if all(existing is not expr for existing in leaves):
            leaves.append(expr)
        return
    _walk_leaves(expr.left, leaves)
    if expr.right is not None:
        _walk_leaves(expr.right, leaves)


def _element_index(output_shape: tuple[int, ...], input_shape: tuple[int, ...],
                   output_index: str) -> str:
    if not output_shape:
        return "0"

    padded_shape = (1,) * (len(output_shape) - len(input_shape)) + input_shape
    terms = []
    for axis, (output_size, input_size) in enumerate(zip(output_shape, padded_shape)):
        if input_size == 1:
            continue
        output_stride = int(np.prod(output_shape[axis + 1:], dtype=np.int64))
        input_stride = int(np.prod(padded_shape[axis + 1:], dtype=np.int64))
        coordinate = f"(({output_index} // {output_stride}) % {output_size})"
        terms.append(f"({coordinate} * {input_stride})")
    return " + ".join(terms) if terms else "0"


def _emit(expr: _Expr | _Leaf, leaf_ids: dict[int, int],
          output_shape: tuple[int, ...], lines: list[str],
          indices: dict[int, str], next_value: list[int]) -> str:
    if isinstance(expr, _Leaf):
        index = leaf_ids[id(expr)]
        if index not in indices:
            index_variable = f"index_{index}"
            lines.append(
                f"        {index_variable} = "
                f"{_element_index(output_shape, expr.shape, 'i')}"
            )
            indices[index] = index_variable
        return f"input_{index}[{indices[index]}]"

    left = _emit(expr.left, leaf_ids, output_shape, lines, indices, next_value)
    if expr.op == "relu":
        name = f"value_{next_value[0]}"
        next_value[0] += 1
        lines.append(f"        {name} = np.maximum(0, {left})")
    elif expr.op == "exp":
        name = f"value_{next_value[0]}"
        next_value[0] += 1
        lines.append(f"        {name} = np.exp({left})")
    elif expr.op == "log":
        name = f"value_{next_value[0]}"
        next_value[0] += 1
        lines.append(f"        {name} = np.log({left})")
    else:
        if expr.right is None:
            raise ValueError(f"Operation {expr.op} requires two operands.")
        right = _emit(
            expr.right, leaf_ids, output_shape, lines, indices, next_value
        )
        name = f"value_{next_value[0]}"
        next_value[0] += 1
        symbol = {
            "add": "+",
            "subtract": "-",
            "multiply": "*",
            "divide": "/",
        }[expr.op]
        lines.append(f"        {name} = {left} {symbol} {right}")
    return name


def _kernel_source(expr: _Expr, output_shape: tuple[int, ...],
                   leaves: list[_Leaf]) -> str:
    leaf_ids = {id(leaf): index for index, leaf in enumerate(leaves)}
    lines = ["def fused_kernel(output, " +
             ", ".join(f"input_{i}" for i in range(len(leaves))) + "):",
             "    for i in range(output.size):"]
    value = _emit(expr, leaf_ids, output_shape, lines, {}, [0])
    lines.append(f"        output[i] = {value}")
    return "\n".join(lines)


@lru_cache(maxsize=128)
def _compile(source: str):
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError(
            "CPU kernel fusion requires the optional Numba dependency. "
            "Install it with `pip install houssem-torch[fusion]`."
        ) from exc

    namespace = {"np": np}
    exec(source, namespace)
    return njit(error_model="numpy", nogil=True)(namespace["fused_kernel"])


def _execute(expr: _Expr, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    leaves: list[_Leaf] = []
    _walk_leaves(expr, leaves)
    if any(size == 0 for size in shape):
        return np.empty(shape, dtype=dtype)
    source = _kernel_source(expr, shape, leaves)
    output = np.empty(shape, dtype=dtype)
    flat_inputs = [leaf.values.reshape(-1) for leaf in leaves]
    _compile(source)(output.reshape(-1), *flat_inputs)
    return output


def _execute_numpy(expr: _Expr | _Leaf) -> np.ndarray:
    if isinstance(expr, _Leaf):
        return expr.values

    left = _execute_numpy(expr.left)
    if expr.op == "relu":
        return np.maximum(0, left)
    if expr.op == "exp":
        return np.exp(left)
    if expr.op == "log":
        return np.log(left)

    right = _execute_numpy(expr.right)
    if expr.op == "add":
        return left + right
    if expr.op == "subtract":
        return left - right
    if expr.op == "multiply":
        return left * right
    if expr.op == "divide":
        return left / right
    raise ValueError(f"Unsupported fused operation: {expr.op}")
