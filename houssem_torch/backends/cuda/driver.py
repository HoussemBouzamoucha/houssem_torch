"""Thin GPU driver: the one module that talks to the GPU.

Everything the GPU engine needs from the hardware goes through here:

    Buffer          a block of device memory with a shape and a dtype
    Stream          an ordered queue of GPU work
    compile_kernel  our CUDA C++ kernels, compiled at startup (NVRTC) with the
                    run's sizes baked in as compile-time constants
    gemm            cuBLAS matrix multiply (row-major, fp32 or fp16)
    gemm_batched    cuBLAS strided batched multiply (one per attention head)
    gemm_tuning     pick the fastest cuBLAS algorithm per multiply shape

CuPy is used only as a set of bindings (memory pool, streams, NVRTC,
cuBLAS handle). No CuPy type is returned from this module, so the rest of
houssem_torch never depends on it and this file can later be swapped for
cuda-python or a compiled extension without touching the engine.

All matrices are row-major, like NumPy. cuBLAS is column-major; `gemm`
handles the conversion (a row-major matrix is its own transpose in
column-major, so C = A·B is computed as Cᵀ = B^T·Aᵀ with no copies).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

try:  # CuPy is an optional dependency (pip install houssem_torch[cuda]).
    import cupy as _cp
    from cupy.cuda import cublas as _cublas
    from cupy.cuda import runtime as _rt
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # ImportError, or a broken CUDA install
    _cp = _cublas = _rt = None
    _IMPORT_ERROR = exc

# Dtypes the GPU engine stores. fp16 for weights/activations in mixed
# precision, fp32 for master weights, optimizer state and reductions,
# int32 for token ids and masks.
DTYPES = (np.float16, np.float32, np.int32)

# cudaDataType / cublasComputeType values (stable across CUDA 11-13).
_CUDA_R_32F = 0
_CUDA_R_16F = 2
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
_CUDA_TYPE = {np.dtype(np.float32): _CUDA_R_32F, np.dtype(np.float16): _CUDA_R_16F}


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """True if CuPy imports and at least one CUDA GPU is visible."""
    if _cp is None:
        return False
    try:
        return _rt.getDeviceCount() > 0
    except Exception:
        return False


def _require() -> None:
    if _cp is None:
        raise RuntimeError(
            "the CUDA backend needs CuPy: pip install cupy-cuda12x "
            f"(import failed: {_IMPORT_ERROR})")


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory: int          # bytes
    multiprocessors: int
    shared_memory_per_block: int


def device_info(index: int = 0) -> DeviceInfo:
    """Name, compute capability and limits of GPU `index`."""
    _require()
    p = _rt.getDeviceProperties(index)
    name = p["name"]
    return DeviceInfo(
        index=index,
        name=name.decode() if isinstance(name, bytes) else name,
        compute_capability=(p["major"], p["minor"]),
        total_memory=p["totalGlobalMem"],
        multiprocessors=p["multiProcessorCount"],
        shared_memory_per_block=p["sharedMemPerBlock"],
    )


def set_device(index: int) -> None:
    """Make GPU `index` current for this thread."""
    _require()
    _cp.cuda.Device(index).use()


def synchronize() -> None:
    """Wait for all queued GPU work on the current device."""
    _require()
    _cp.cuda.Device().synchronize()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class Buffer:
    """A contiguous block of device memory with a shape and a dtype.

    Plain storage: no autograd, no operators. The compiled training step
    allocates all of its buffers once at startup and reuses them every step.
    """

    __slots__ = ("_arr",)

    def __init__(self, _arr):  # internal: use empty / zeros / from_numpy
        self._arr = _arr

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._arr.dtype

    @property
    def size(self) -> int:
        return int(self._arr.size)

    @property
    def nbytes(self) -> int:
        return int(self._arr.nbytes)

    @property
    def ptr(self) -> int:
        """Raw device address (for kernel arguments and library calls)."""
        return int(self._arr.data.ptr)

    def numpy(self) -> np.ndarray:
        """Copy to a new host array (waits for the copy to finish)."""
        return _cp.asnumpy(self._arr)

    def copy_from(self, host: np.ndarray) -> None:
        """Overwrite with a host array of the same shape (cast to this dtype)."""
        host = np.ascontiguousarray(host, dtype=self.dtype)
        if host.shape != self.shape:
            raise ValueError(f"shape mismatch: buffer {self.shape}, "
                             f"host {host.shape}")
        self._arr.set(host)

    def copy_from_async(self, host: np.ndarray) -> None:
        """Queue an upload of a pinned host array (see `pinned_empty`) of the
        same shape and dtype on the current stream, without waiting for
        earlier GPU work. The host array must not change until the copy has
        run (record an `Event` after it and wait on that before reuse)."""
        if host.shape != self.shape or host.dtype != self.dtype:
            raise ValueError(f"copy_from_async mismatch: {self.shape}/"
                             f"{self.dtype} vs {host.shape}/{host.dtype}")
        self._arr.set(host, stream=_cp.cuda.get_current_stream())

    def copy_(self, other: "Buffer") -> None:
        """Device-to-device copy from a buffer of the same shape and dtype."""
        if other.shape != self.shape or other.dtype != self.dtype:
            raise ValueError(f"copy_ mismatch: {self.shape}/{self.dtype} vs "
                             f"{other.shape}/{other.dtype}")
        _cp.copyto(self._arr, other._arr)

    def fill_(self, value: float) -> None:
        with _Timed("fill"):
            self._arr.fill(value)

    def view(self, *shape: int) -> "Buffer":
        """Same memory, new shape (sizes must match)."""
        return Buffer(self._arr.reshape(shape))

    def __getitem__(self, index) -> "Buffer":
        """A contiguous sub-buffer sharing this memory: buf[i] (one entry of
        the leading axis) or buf[a:b] (a range of it, step 1)."""
        if isinstance(index, slice) and index.step not in (None, 1):
            raise ValueError("Buffer slices must have step 1")
        if not isinstance(index, (int, np.integer, slice)):
            raise TypeError("Buffer index must be an int or a slice")
        return Buffer(self._arr[index])

    def __repr__(self) -> str:
        return f"Buffer(shape={self.shape}, dtype={self.dtype})"


def _check_dtype(dtype) -> np.dtype:
    dt = np.dtype(dtype)
    if dt not in [np.dtype(d) for d in DTYPES]:
        raise TypeError(f"unsupported GPU dtype {dt}; use one of "
                        f"{[np.dtype(d).name for d in DTYPES]}")
    return dt


def empty(shape, dtype=np.float32) -> Buffer:
    """Uninitialized device buffer."""
    _require()
    return Buffer(_cp.empty(shape, dtype=_check_dtype(dtype)))


def zeros(shape, dtype=np.float32) -> Buffer:
    _require()
    return Buffer(_cp.zeros(shape, dtype=_check_dtype(dtype)))


class Arena:
    """One block of device memory that buffers are carved from, front to back.

        meter = Arena()                 # measuring: real buffers, bytes counted
        ... build with meter.take ...   # -> meter.used
        arena = Arena(meter.used)       # one block of that size
        a, b = arena.cursor(), arena.cursor()

    Each `cursor()` carves the same block again from its start, so the
    buffers of different cursors overlap: for owners that never run at the
    same time (e.g. the activations of training steps queued on one
    stream). The block starts zeroed; later contents are whatever the last
    owner left.
    """

    _ALIGN = 256

    def __init__(self, nbytes: int | None = None):
        _require()
        self._mem = None if nbytes is None else _cp.zeros(max(int(nbytes), 1),
                                                          np.uint8)
        self.nbytes = None if nbytes is None else int(nbytes)
        self.used = 0

    def cursor(self) -> "Arena":
        """A new carver over this block, starting at its front."""
        if self._mem is None:
            raise ValueError("a measuring Arena has no block to share")
        c = Arena.__new__(Arena)
        c._mem, c.nbytes, c.used = self._mem, self.nbytes, 0
        return c

    def take(self, shape, dtype=np.float32) -> Buffer:
        """The next buffer of this shape and dtype."""
        dtype = _check_dtype(dtype)
        shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        n = int(np.prod(shape)) * dtype.itemsize
        off = -(-self.used // self._ALIGN) * self._ALIGN
        self.used = off + n
        if self._mem is None:
            return Buffer(_cp.zeros(shape, dtype))
        if self.used > self.nbytes:
            raise MemoryError(f"arena of {self.nbytes} bytes is full "
                              f"(needs {self.used})")
        return Buffer(self._mem[off:off + n].view(dtype).reshape(shape))


def from_numpy(host: np.ndarray, dtype=None) -> Buffer:
    """Upload a host array (optionally casting it first)."""
    _require()
    dt = _check_dtype(host.dtype if dtype is None else dtype)
    return Buffer(_cp.asarray(np.ascontiguousarray(host, dtype=dt)))


def pinned_empty(shape, dtype=np.float32) -> np.ndarray:
    """A NumPy array in page-locked host memory: uploads from it can run
    asynchronously (from ordinary pageable memory, CUDA first waits for all
    earlier work on the stream)."""
    _require()
    dtype = _check_dtype(dtype)
    shape = tuple(int(n) for n in (shape if isinstance(shape, tuple) else (shape,)))
    count = int(np.prod(shape, dtype=np.int64))
    mem = _cp.cuda.alloc_pinned_memory(max(count, 1) * dtype.itemsize)
    return np.frombuffer(mem, dtype, count).reshape(shape)  # keeps mem alive


class Event:
    """A marker in the current stream's queue of work."""

    def __init__(self):
        _require()
        self._e = _cp.cuda.Event(disable_timing=True)

    def record(self) -> None:
        self._e.record()

    def synchronize(self) -> None:
        """Wait (on the host) until the GPU has passed the marker."""
        self._e.synchronize()


def memory_in_use() -> int:
    """Bytes currently held by live buffers."""
    _require()
    return int(_cp.get_default_memory_pool().used_bytes())


# ---------------------------------------------------------------------------
# Streams and timing
# ---------------------------------------------------------------------------

class Stream:
    """An ordered queue of GPU work. Work on different streams may overlap."""

    def __init__(self, non_blocking: bool = True):
        _require()
        self._s = _cp.cuda.Stream(non_blocking=non_blocking)

    @property
    def ptr(self) -> int:
        return int(self._s.ptr)

    def synchronize(self) -> None:
        self._s.synchronize()

    def __enter__(self) -> "Stream":
        self._s.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._s.__exit__(*exc)


def _current_stream_ptr() -> int:
    return int(_cp.cuda.get_current_stream().ptr)


class Timer:
    """GPU-side elapsed time between start() and stop(), in milliseconds."""

    def __init__(self):
        _require()
        self._a = _cp.cuda.Event()
        self._b = _cp.cuda.Event()

    def start(self) -> None:
        self._a.record()

    def stop(self) -> float:
        self._b.record()
        self._b.synchronize()
        return float(_cp.cuda.get_elapsed_time(self._a, self._b))


# ---------------------------------------------------------------------------
# Profiling: GPU time of every launch, tagged with a region
# ---------------------------------------------------------------------------

_PROFILER: "Profiler | None" = None
_REGIONS: list[str] = []


class region:
    """Tag the launches inside this block: `with region("fwd"): ...`.
    Nested regions join with "/" ("fwd/enc/ffn"). Costs almost nothing
    when no Profiler is active."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        _REGIONS.append(self.name)

    def __exit__(self, *exc):
        _REGIONS.pop()


@dataclass
class LaunchRecord:
    name: str           # kernel name, "gemm", "gemm_batched" or "fill"
    region: str
    flops: float        # matrix multiplies only, else 0
    shape: str          # matrix multiplies: "m×n×k[×batch]"
    ms: float = 0.0


class Profiler:
    """Times every launch on the GPU (CUDA events around each one).

        with driver.Profiler() as prof:
            step(...)
        for r in prof.records(): ...

    The events add a little time between launches, so per-launch times are
    reliable while the profiled total runs slightly above an unprofiled step.
    """

    def __init__(self):
        _require()
        self._open: list = []

    def __enter__(self) -> "Profiler":
        global _PROFILER
        if _PROFILER is not None:
            raise RuntimeError("a Profiler is already active")
        _PROFILER = self
        return self

    def __exit__(self, *exc) -> None:
        global _PROFILER
        _PROFILER = None

    def _begin(self, name, flops=0.0, shape=""):
        a, b = _cp.cuda.Event(), _cp.cuda.Event()
        a.record()
        rec = LaunchRecord(name, "/".join(_REGIONS), float(flops), shape)
        self._open.append((rec, a, b))
        return b

    def records(self) -> list[LaunchRecord]:
        """All launches so far, with their GPU time (waits for the GPU)."""
        if self._open:
            self._open[-1][2].synchronize()
        for rec, a, b in self._open:
            rec.ms = float(_cp.cuda.get_elapsed_time(a, b))
        return [rec for rec, _, _ in self._open]


class _Timed:
    """Context for one launch: no-op unless a Profiler is active."""

    __slots__ = ("_end",)

    def __init__(self, name, flops=0.0, shape=""):
        self._end = None if _PROFILER is None else _PROFILER._begin(name, flops,
                                                                   shape)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._end is not None:
            self._end.record()


# ---------------------------------------------------------------------------
# Kernels (NVRTC, compiled at startup)
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict[str, "Kernel"] = {}


_NULL = np.uint64(0)


def _arg(x):
    if isinstance(x, Buffer):
        return x._arr
    if x is None:  # optional pointer argument
        return _NULL
    if isinstance(x, (bool, np.bool_)):
        return np.int32(x)
    if isinstance(x, (int, np.integer)) and not isinstance(x, np.generic):
        return np.int32(x)
    if isinstance(x, float):
        return np.float32(x)
    return x  # already a NumPy scalar (np.int64, np.float16, ...)


class Kernel:
    """One compiled __global__ function."""

    def __init__(self, fn, name: str):
        self._fn = fn
        self.name = name

    def __call__(self, grid, block, *args, shared_mem: int = 0) -> None:
        """Queue a launch on the current stream.

        Python ints become int32, floats float32, None a null pointer; pass
        np.int64 etc. explicitly for other widths.
        """
        grid = grid if isinstance(grid, tuple) else (grid,)
        block = block if isinstance(block, tuple) else (block,)
        args = tuple(_arg(a) for a in args)
        if _PROFILER is None:
            self._fn(grid, block, args, shared_mem=shared_mem)
            return
        with _Timed(self.name):
            self._fn(grid, block, args, shared_mem=shared_mem)

    def set_max_dynamic_shared(self, nbytes: int) -> None:
        """Allow launches with up to `nbytes` of dynamic shared memory
        (above 48 KB needs this opt-in; sm_75 allows up to 64 KB), and ask
        for the largest shared-memory carveout so as many blocks as the
        shared memory allows can share an SM."""
        self._fn.max_dynamic_shared_size_bytes = int(nbytes)
        self._fn.preferred_shared_memory_carveout = 100

    @property
    def max_threads_per_block(self) -> int:
        return int(self._fn.max_threads_per_block)

    @property
    def num_regs(self) -> int:
        """Registers per thread (watch this when tuning occupancy)."""
        return int(self._fn.num_regs)


def ptr(buf: Buffer, offset: int = 0) -> np.uint64:
    """Device address of element `offset` of `buf`, as a kernel argument
    (for kernels that read a strided part of a buffer, e.g. the K columns of
    a fused Q|K|V projection). The buffer must outlive the launch."""
    if not 0 <= offset <= buf.size:
        raise IndexError(f"offset {offset} outside a buffer of {buf.size}")
    return np.uint64(buf.ptr + offset * buf.dtype.itemsize)


def compile_kernel(source: str, name: str, defines: dict | None = None,
                   options: tuple[str, ...] = ()) -> Kernel:
    """Compile `name` from CUDA C++ `source` for the current GPU.

    `defines` become `-DKEY=value` flags, which is how a run's sizes
    (D_MODEL=512, N_HEADS=8, ...) are baked into the kernel as constants so
    the compiler can unroll loops and size shared memory exactly.
    Results are cached per (source, name, defines, options).
    """
    _require()
    flags = ["-std=c++17"]
    flags += [f"-D{k}={_define(v)}" for k, v in sorted((defines or {}).items())]
    flags += list(options)
    key = hashlib.sha1("\0".join([source, name, *flags]).encode()).hexdigest()
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    module = _cp.RawModule(code=source, options=tuple(flags),
                           name_expressions=None)
    kernel = Kernel(module.get_function(name), name)
    _KERNEL_CACHE[key] = kernel
    return kernel


def _define(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v!r}f"
    return str(v)


# ---------------------------------------------------------------------------
# cuBLAS matrix multiply (row-major)
# ---------------------------------------------------------------------------

def _gemm_dims(a_shape, b_shape, trans_a, trans_b):
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError("gemm takes 2-D matrices; use gemm_batched")
    m, k = a_shape[::-1] if trans_a else a_shape
    k2, n = b_shape[::-1] if trans_b else b_shape
    if k != k2:
        raise ValueError(f"inner sizes differ: op(A) is {(m, k)}, "
                         f"op(B) is {(k2, n)}")
    return m, n, k


def _gemm_types(a: Buffer, b: Buffer, c: Buffer):
    if a.dtype != b.dtype:
        raise TypeError(f"A and B must share a dtype, got {a.dtype}, {b.dtype}")
    if a.dtype not in _CUDA_TYPE or c.dtype not in _CUDA_TYPE:
        raise TypeError("gemm supports float16 and float32 only")
    if a.dtype == np.float32 and c.dtype != np.float32:
        raise TypeError("float32 inputs need a float32 output")
    return _CUDA_TYPE[a.dtype], _CUDA_TYPE[c.dtype]


def _shape_label(m, n, k, trans_a, trans_b, batch=None):
    label = f"{m}x{n}x{k}" + (f" x{batch}" if batch else "")
    return label + (" A^T" if trans_a else "") + (" B^T" if trans_b else "")


def _handle():
    h = _cp.cuda.device.get_cublas_handle()
    _cublas.setStream(h, _current_stream_ptr())
    return h


# ---------------------------------------------------------------------------
# cuBLAS algorithm choice per multiply shape, decided at startup
# ---------------------------------------------------------------------------
# cuBLAS's default heuristic is not always the fastest kernel for a given
# shape. Inside `gemm_tuning()`, the first multiply of each new shape times
# every candidate algorithm (on a scratch output, so results are untouched)
# and keeps the fastest; afterwards every multiply of that shape uses it.
# On Turing (sm_75) cuBLAS honours the explicit ALGO0..15_TENSOR_OP choices
# (newer GPUs ignore them and use the default, so tuning changes nothing).

_GEMM_CANDIDATES = (_CUBLAS_GEMM_DEFAULT_TENSOR_OP,) + tuple(range(100, 116))
_GEMM_ALGO: dict[tuple, int] = {}       # shape key -> chosen algorithm
_GEMM_LABEL: dict[tuple, str] = {}      # shape key -> readable label
_GEMM_TIMES: dict[tuple, dict] = {}     # shape key -> {algo: µs}, for reports
_TUNING = 0
_TUNE_REPS = 5
_TUNE_MARGIN = 0.97     # a candidate must beat the default by 3% to replace it


class gemm_tuning:
    """Choose cuBLAS algorithms for the multiplies run inside this block.

        with driver.gemm_tuning():
            step.forward_backward(...)    # one pass: every shape gets tuned

    Each multiply shape is tuned once per process (choices are shared by
    every caller); the block waits for the GPU while it times candidates.
    """

    def __enter__(self):
        global _TUNING
        _TUNING += 1
        return self

    def __exit__(self, *exc):
        global _TUNING
        _TUNING -= 1


def gemm_algorithms() -> dict[str, dict]:
    """The tuned shapes: {label: {"algo": chosen, "us": {algo: µs}}}.
    The label is the profiler's shape label plus dtypes and beta."""
    return {_GEMM_LABEL[k]: {"algo": a, "us": dict(_GEMM_TIMES.get(k, {}))}
            for k, a in _GEMM_ALGO.items()}


def reset_gemm_algorithms() -> None:
    """Forget every tuned choice (multiplies go back to cuBLAS's default)."""
    _GEMM_ALGO.clear()
    _GEMM_LABEL.clear()
    _GEMM_TIMES.clear()


def _gemm_algo(key, label, run, c):
    """The algorithm for this multiply shape; tunes it first when inside
    `gemm_tuning()` and the shape is new. `run(algo, c)` queues the multiply
    into output buffer c."""
    algo = _GEMM_ALGO.get(key)
    if algo is not None:
        return algo
    if not _TUNING:
        return _CUBLAS_GEMM_DEFAULT_TENSOR_OP
    scratch = Buffer(_cp.empty_like(c._arr))
    start, end = _cp.cuda.Event(), _cp.cuda.Event()

    def time_us(a):
        try:
            run(a, scratch)                   # warm-up; unsupported -> error
        except Exception:  # noqa: BLE001  (cuBLAS: algorithm not supported)
            return float("inf")
        start.record()
        for _ in range(_TUNE_REPS):
            run(a, scratch)
        end.record()
        end.synchronize()
        return _cp.cuda.get_elapsed_time(start, end) * 1e3 / _TUNE_REPS

    default = _CUBLAS_GEMM_DEFAULT_TENSOR_OP
    times = {a: time_us(a) for a in _GEMM_CANDIDATES}
    best = min(times, key=times.get)
    if best != default:
        # Re-time the two finalists alternately: clocks drift during tuning.
        tb, td = float("inf"), float("inf")
        for _ in range(3):
            td = min(td, time_us(default))
            tb = min(tb, time_us(best))
        times[best], times[default] = tb, td
        if not tb < _TUNE_MARGIN * td:
            best = default
    _GEMM_ALGO[key], _GEMM_LABEL[key] = best, label
    _GEMM_TIMES[key] = {a: round(t, 2) for a, t in times.items()
                        if t != float("inf")}
    return best


def gemm(a: Buffer, b: Buffer, c: Buffer, *, trans_a: bool = False,
         trans_b: bool = False, alpha: float = 1.0, beta: float = 0.0) -> None:
    """c = alpha · op(a) @ op(b) + beta · c, all row-major.

    op(x) is x or xᵀ. Inputs float16 or float32; accumulation is always
    float32 (tensor cores on T4 for float16). A float16 product may be
    written to a float32 `c` (e.g. gradients into fp32 accumulators).
    """
    _require()
    m, n, k = _gemm_dims(a.shape, b.shape, trans_a, trans_b)
    if c.shape != (m, n):
        raise ValueError(f"c must be {(m, n)}, got {c.shape}")
    ab_type, c_type = _gemm_types(a, b, c)
    al = np.array(alpha, dtype=np.float32)
    be = np.array(beta, dtype=np.float32)
    label = _shape_label(m, n, k, trans_a, trans_b)

    # Row-major C(m,n) = op(A) op(B)  ⇔  column-major Cᵀ(n,m) = op(B)ᵀ op(A)ᵀ.
    def run(algo, out):
        _gemm_ex(a, b, out, m, n, k, trans_a, trans_b, ab_type, c_type, al,
                 be, algo)

    key = (0, m, n, k, trans_a, trans_b, ab_type, c_type, beta != 0)
    algo = _gemm_algo(key, _tuned_label(label, a, c, beta), run, c)
    with _Timed("gemm", 2.0 * m * n * k, label):
        run(algo, c)


def _tuned_label(label, a, c, beta):
    return (f"{label} {np.dtype(a.dtype).name}->{np.dtype(c.dtype).name}"
            + (" +C" if beta != 0 else ""))


def _gemm_ex(a, b, c, m, n, k, trans_a, trans_b, ab_type, c_type, al, be,
             algo=_CUBLAS_GEMM_DEFAULT_TENSOR_OP):
    _cublas.gemmEx(
        _handle(), int(trans_b), int(trans_a), n, m, k,
        al.ctypes.data, b.ptr, ab_type, b.shape[1],
        a.ptr, ab_type, a.shape[1],
        be.ctypes.data, c.ptr, c_type, n,
        _CUBLAS_COMPUTE_32F, algo)


def gemm_batched(a: Buffer, b: Buffer, c: Buffer, *, trans_a: bool = False,
                 trans_b: bool = False, alpha: float = 1.0,
                 beta: float = 0.0) -> None:
    """Batched `gemm` over the leading axis: c[i] = alpha·op(a[i])@op(b[i]) + beta·c[i].

    a, b, c are 3-D (batch, rows, cols), contiguous. Attention uses this
    with batch = batch_size · n_heads.
    """
    _require()
    if not (len(a.shape) == len(b.shape) == len(c.shape) == 3):
        raise ValueError("gemm_batched takes 3-D buffers (batch, rows, cols)")
    batch = a.shape[0]
    if b.shape[0] != batch or c.shape[0] != batch:
        raise ValueError(f"batch sizes differ: {a.shape[0]}, {b.shape[0]}, "
                         f"{c.shape[0]}")
    m, n, k = _gemm_dims(a.shape[1:], b.shape[1:], trans_a, trans_b)
    if c.shape[1:] != (m, n):
        raise ValueError(f"c must be {(batch, m, n)}, got {c.shape}")
    ab_type, c_type = _gemm_types(a, b, c)
    al = np.array(alpha, dtype=np.float32)
    be = np.array(beta, dtype=np.float32)
    label = _shape_label(m, n, k, trans_a, trans_b, batch)

    def run(algo, out):
        _gemm_strided(a, b, out, m, n, k, batch, trans_a, trans_b, ab_type,
                      c_type, al, be, algo)

    key = (batch, m, n, k, trans_a, trans_b, ab_type, c_type, beta != 0)
    algo = _gemm_algo(key, _tuned_label(label, a, c, beta), run, c)
    with _Timed("gemm_batched", 2.0 * m * n * k * batch, label):
        run(algo, c)


def _gemm_strided(a, b, c, m, n, k, batch, trans_a, trans_b, ab_type, c_type,
                  al, be, algo=_CUBLAS_GEMM_DEFAULT_TENSOR_OP):
    _cublas.gemmStridedBatchedEx(
        _handle(), int(trans_b), int(trans_a), n, m, k,
        al.ctypes.data,
        b.ptr, ab_type, b.shape[2], b.shape[1] * b.shape[2],
        a.ptr, ab_type, a.shape[2], a.shape[1] * a.shape[2],
        be.ctypes.data,
        c.ptr, c_type, n, m * n,
        batch, _CUBLAS_COMPUTE_32F, algo)


# ---------------------------------------------------------------------------
# cuBLASLt multiply with a fused epilogue (ctypes: CuPy has no binding)
# ---------------------------------------------------------------------------
# cuBLASLt runs the same tensor-core kernels as cuBLAS but can finish the
# product with an elementwise step and a column reduction inside the
# multiply, so the product is never written out and read back by a separate
# kernel. The engine uses it in the feed-forward backward:
#     g_u = mask ⊙ (alpha · g_y · W_downᵀ),   db_up = Σ_rows g_u,
# where `mask` is the 1-bit "ReLU active and kept by dropout" pattern that
# the forward kernel writes, and alpha = 1/(1-p) is the dropout scale.
#
# Enum values from cublasLt.h (CUDA 12, unchanged since 11.x).

EPILOGUES = {"default": 1, "relu": 2, "relu_aux": 130, "bias": 4,
             "relu_bias": 6, "relu_aux_bias": 134, "drelu": 136,
             "drelu_bgrad": 152}
_LT_NEEDS_AUX = {"relu_aux", "relu_aux_bias", "drelu", "drelu_bgrad"}
_LT_NEEDS_BIAS = {"bias", "relu_bias", "relu_aux_bias", "drelu_bgrad"}
_LT_DESC_TRANSA = 3
_LT_DESC_TRANSB = 4
_LT_DESC_EPILOGUE = 7
_LT_DESC_BIAS_POINTER = 8
_LT_DESC_AUX_POINTER = 11
_LT_DESC_AUX_LD = 12
_LT_DESC_BIAS_DATA_TYPE = 26
_LT_PREF_MAX_WORKSPACE = 1
_LT_WORKSPACE = 4 << 20

_lt_lib = None
_lt_error: str | None = None
_lt_handles: dict[int, tuple] = {}      # device -> (handle, workspace)
_lt_plans: dict[tuple, tuple] = {}      # multiply key -> descriptors + algo


def _lt_load():
    """The cuBLASLt library matching CuPy's CUDA major version."""
    import ctypes
    import glob
    import os
    import sys
    major = _rt.runtimeGetVersion() // 1000
    if sys.platform == "win32":
        names, sub = [f"cublasLt64_{major}.dll"], "bin"
    else:
        names, sub = [f"libcublasLt.so.{major}", "libcublasLt.so"], "lib"
    tried = []
    for name in names:          # already loaded alongside cuBLAS, or on the
        try:                    # loader path
            return ctypes.CDLL(name)
        except OSError as exc:
            tried.append(str(exc))
    dirs = []
    try:                        # pip wheels (nvidia-cublas-cu12)
        import nvidia.cublas as nv_cublas
        dirs += [os.path.join(p, sub) for p in nv_cublas.__path__]
    except Exception:  # noqa: BLE001
        pass
    for env in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(env):
            dirs += [os.path.join(os.environ[env], d) for d in ("bin", "lib64")]
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, names[0] + "*"))):
            try:
                return ctypes.CDLL(path)
            except OSError as exc:
                tried.append(str(exc))
    raise OSError("cuBLASLt not found: " + "; ".join(tried))


def _lt():
    global _lt_lib, _lt_error
    if _lt_lib is None and _lt_error is None:
        try:
            _lt_lib = _lt_load()
        except Exception as exc:  # noqa: BLE001
            _lt_error = str(exc)
    if _lt_lib is None:
        raise RuntimeError(f"cuBLASLt unavailable: {_lt_error}")
    return _lt_lib


def lt_available() -> bool:
    """True if cuBLASLt loads (`gemm_epilogue` can run)."""
    if not is_available():
        return False
    try:
        _lt()
        return True
    except RuntimeError:
        return False


def _lt_check(status, what):
    if status != 0:
        raise RuntimeError(f"cuBLASLt {what} failed (status {status})")


def _lt_handle():
    import ctypes
    dev = _cp.cuda.Device().id
    h = _lt_handles.get(dev)
    if h is None:
        handle = ctypes.c_void_p()
        _lt_check(_lt().cublasLtCreate(ctypes.byref(handle)), "create")
        h = _lt_handles[dev] = (handle, empty(_LT_WORKSPACE // 4, np.int32))
    return h


def _lt_set(desc, attr, value, ctype):
    import ctypes
    v = ctype(value)
    _lt_check(_lt().cublasLtMatmulDescSetAttribute(
        desc, ctypes.c_int(attr), ctypes.byref(v),
        ctypes.c_size_t(ctypes.sizeof(v))), f"attribute {attr}")


def _lt_result_type():
    """ctypes mirror of cublasLtMatmulHeuristicResult_t."""
    import ctypes

    class HeuristicResult(ctypes.Structure):
        _fields_ = [("algo", ctypes.c_uint64 * 8),
                    ("workspaceSize", ctypes.c_size_t),
                    ("state", ctypes.c_int),
                    ("wavesCount", ctypes.c_float),
                    ("reserved", ctypes.c_int * 4)]
    return HeuristicResult


def _lt_plan(key, m, n, k, trans_a, trans_b, ab_type, c_type, a, b, epilogue,
             bias, aux):
    """Descriptors and the heuristic's first algorithm for one multiply."""
    import ctypes
    plan = _lt_plans.get(key)
    if plan is not None:
        return plan
    lib = _lt()
    handle, _ = _lt_handle()
    desc = ctypes.c_void_p()
    _lt_check(lib.cublasLtMatmulDescCreate(
        ctypes.byref(desc), ctypes.c_int(_CUBLAS_COMPUTE_32F),
        ctypes.c_int(_CUDA_R_32F)), "descriptor")
    # Row-major C(m,n) = op(A)·op(B) is column-major Cᵀ(n,m) = op(B)ᵀ·op(A)ᵀ,
    # so cuBLASLt's first operand is our b and its second our a (as in gemm).
    _lt_set(desc, _LT_DESC_TRANSA, int(trans_b), ctypes.c_int)
    _lt_set(desc, _LT_DESC_TRANSB, int(trans_a), ctypes.c_int)
    _lt_set(desc, _LT_DESC_EPILOGUE, EPILOGUES[epilogue], ctypes.c_uint32)
    if bias is not None:      # pointers are set before the heuristic: it
        _lt_set(desc, _LT_DESC_BIAS_POINTER, bias.ptr, ctypes.c_void_p)
        _lt_set(desc, _LT_DESC_BIAS_DATA_TYPE, _CUDA_TYPE[bias.dtype],
                ctypes.c_int)     # checks their alignment
    if aux is not None:
        _lt_set(desc, _LT_DESC_AUX_POINTER, aux.ptr, ctypes.c_void_p)
        _lt_set(desc, _LT_DESC_AUX_LD, n, ctypes.c_int64)

    def layout(dtype, rows, cols):      # column-major, ld = rows
        lay = ctypes.c_void_p()
        _lt_check(lib.cublasLtMatrixLayoutCreate(
            ctypes.byref(lay), ctypes.c_int(dtype), ctypes.c_uint64(rows),
            ctypes.c_uint64(cols), ctypes.c_int64(rows)), "layout")
        return lay

    # A row-major (r, c) buffer is the column-major (c, r) matrix.
    lay_b = layout(ab_type, b.shape[1], b.shape[0])
    lay_a = layout(ab_type, a.shape[1], a.shape[0])
    lay_c = layout(c_type, n, m)
    pref = ctypes.c_void_p()
    _lt_check(lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)),
              "preference")
    wsz = ctypes.c_uint64(_LT_WORKSPACE)
    _lt_check(lib.cublasLtMatmulPreferenceSetAttribute(
        pref, ctypes.c_int(_LT_PREF_MAX_WORKSPACE), ctypes.byref(wsz),
        ctypes.c_size_t(8)), "preference attribute")
    res = _lt_result_type()()
    count = ctypes.c_int(0)
    status = lib.cublasLtMatmulAlgoGetHeuristic(
        handle, desc, lay_b, lay_a, lay_c, lay_c, pref, ctypes.c_int(1),
        ctypes.byref(res), ctypes.byref(count))
    lib.cublasLtMatmulPreferenceDestroy(pref)
    _lt_check(status, "heuristic")
    if count.value < 1:
        raise RuntimeError(f"cuBLASLt has no algorithm for {epilogue} on "
                           f"{_shape_label(m, n, k, trans_a, trans_b)}")
    plan = _lt_plans[key] = (desc, lay_b, lay_a, lay_c, res)
    return plan


def gemm_epilogue(a: Buffer, b: Buffer, c: Buffer, *, epilogue: str,
                  bias: Buffer | None = None, aux: Buffer | None = None,
                  trans_a: bool = False, trans_b: bool = False,
                  alpha: float = 1.0, beta: float = 0.0) -> None:
    """`gemm` with a cuBLASLt epilogue fused into the multiply (row-major).

    With P = alpha·op(a)@op(b) + beta·c:
      "default"        c = P
      "bias"           c = P + bias                 (bias: n entries)
      "relu_bias"      c = relu(P + bias)
      "relu_aux_bias"  c = relu(P + bias), aux = bit mask of c > 0
      "drelu"          c = P ⊙ aux
      "drelu_bgrad"    c = P ⊙ aux, bias = column sums of c (overwritten,
                       not added; bias has c's dtype)
    aux is a ReLU bit mask with one bit per element of c, in row-major
    order, least significant bit first (n must be a multiple of 128), held
    in an int32 buffer of m·n/32 entries.
    """
    import ctypes
    _require()
    if epilogue not in EPILOGUES:
        raise ValueError(f"unknown epilogue {epilogue!r}")
    m, n, k = _gemm_dims(a.shape, b.shape, trans_a, trans_b)
    if c.shape != (m, n):
        raise ValueError(f"c must be {(m, n)}, got {c.shape}")
    ab_type, c_type = _gemm_types(a, b, c)
    for name, needed, given in (("bias", _LT_NEEDS_BIAS, bias),
                                ("aux mask", _LT_NEEDS_AUX, aux)):
        if (epilogue in needed) != (given is not None):
            raise ValueError(f"epilogue {epilogue!r} "
                             f"{'needs' if epilogue in needed else 'takes no'} "
                             f"{name}")
    if bias is not None:
        if bias.size != n:
            raise ValueError(f"bias must have {n} entries, got {bias.size}")
        if epilogue == "drelu_bgrad" and bias.dtype != c.dtype:
            raise TypeError("the bias gradient must have c's dtype")
    if aux is not None:
        if n % 128:
            raise ValueError("the ReLU mask needs n divisible by 128")
        if aux.dtype != np.int32 or aux.size * 32 != m * n:
            raise ValueError(f"aux must be int32 with {m * n // 32} entries")
    lib = _lt()
    handle, ws = _lt_handle()
    key = (_cp.cuda.Device().id, m, n, k, trans_a, trans_b, ab_type, c_type,
           epilogue, None if bias is None else _CUDA_TYPE[bias.dtype])
    desc, lay_b, lay_a, lay_c, res = _lt_plan(
        key, m, n, k, trans_a, trans_b, ab_type, c_type, a, b, epilogue, bias,
        aux)
    if bias is not None:
        _lt_set(desc, _LT_DESC_BIAS_POINTER, bias.ptr, ctypes.c_void_p)
    if aux is not None:
        _lt_set(desc, _LT_DESC_AUX_POINTER, aux.ptr, ctypes.c_void_p)
    al, be = ctypes.c_float(alpha), ctypes.c_float(beta)
    label = _shape_label(m, n, k, trans_a, trans_b) + f" {epilogue}"
    with _Timed("gemm_lt", 2.0 * m * n * k, label):
        _lt_check(lib.cublasLtMatmul(
            handle, desc, ctypes.byref(al),
            ctypes.c_void_p(b.ptr), lay_b, ctypes.c_void_p(a.ptr), lay_a,
            ctypes.byref(be), ctypes.c_void_p(c.ptr), lay_c,
            ctypes.c_void_p(c.ptr), lay_c, ctypes.byref(res.algo),
            ctypes.c_void_p(ws.ptr),
            ctypes.c_size_t(min(res.workspaceSize, _LT_WORKSPACE)),
            ctypes.c_void_p(_current_stream_ptr())), "matmul")
