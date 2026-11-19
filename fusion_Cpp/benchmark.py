import ctypes
import gc
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import tracemalloc

THREADS = int(os.environ.get("BENCH_THREADS", "1"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "2"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "7"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = str(THREADS)

import numpy as np  # noqa: E402
import numba  # noqa: E402
import torch  # noqa: E402
import houssem_torch as ht  # noqa: E402

torch.set_num_threads(THREADS)
ROOT = Path(__file__).resolve().parents[1]
FOLDER = Path(__file__).resolve().parent
SOURCE = FOLDER / "fused_kernel.cpp"
LIBRARY = FOLDER / "build" / "fusion_kernel.dll"
SIZES = (64, 256, 1024, 4096, 16384, 65536)
OMP_THREAD_COUNTS = (1, 2, 4, 6)
FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)
SIZE_ARG = ctypes.c_size_t


def build_library():
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError(
            "g++ was not found on PATH. Install a MinGW-w64 toolchain first."
        )
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
        command = [
            compiler,
            "-std=c++17",
            "-O3",
            "-fopenmp",
            "-shared",
            str(SOURCE),
            "-o",
            str(LIBRARY),
        ]
        subprocess.run(command, check=True, cwd=ROOT)

    if os.name == "nt":
        os.add_dll_directory(str(Path(compiler).resolve().parent))
    library = ctypes.CDLL(str(LIBRARY))
    _configure_signatures(library)
    return library


def _configure_signatures(library):
    forward_args = [FLOAT_POINTER, FLOAT_POINTER, FLOAT_POINTER, SIZE_ARG]
    backward_args = [
        FLOAT_POINTER,
        FLOAT_POINTER,
        FLOAT_POINTER,
        FLOAT_POINTER,
        FLOAT_POINTER,
        SIZE_ARG,
    ]
    combined_args = [
        FLOAT_POINTER,
        FLOAT_POINTER,
        FLOAT_POINTER,
        FLOAT_POINTER,
        FLOAT_POINTER,
        SIZE_ARG,
    ]
    for suffix in ("", "_omp", "_avx2"):
        forward = getattr(library, f"fusion_cpp_forward{suffix}")
        forward.argtypes = forward_args
        forward.restype = None
        backward = getattr(library, f"fusion_cpp_backward{suffix}")
        backward.argtypes = backward_args
        backward.restype = None
        combined = getattr(library, f"fusion_cpp_forward_backward{suffix}")
        combined.argtypes = combined_args
        combined.restype = None
    library.fusion_cpp_avx2_supported.argtypes = []
    library.fusion_cpp_avx2_supported.restype = ctypes.c_int
    library.fusion_cpp_openmp_enabled.argtypes = []
    library.fusion_cpp_openmp_enabled.restype = ctypes.c_int
    library.fusion_cpp_set_threads.argtypes = [ctypes.c_int]
    library.fusion_cpp_set_threads.restype = None


def _pointer(array):
    return array.ctypes.data_as(FLOAT_POINTER)


def _ht_work(a, b):
    return ht.log(ht.exp(ht.relu((a * b + 0.25) / 2.0)) + 1.0)


def _torch_work(a, b):
    return torch.log(torch.exp(torch.relu((a * b + 0.25) / 2.0)) + 1.0)


def _median_ms(step, reset=lambda: None):
    for _ in range(WARMUP):
        reset()
        step()
    samples = []
    for _ in range(REPEATS):
        reset()
        start = time.perf_counter()
        step()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def _time_backward(prepare, reset):
    for _ in range(WARMUP):
        reset()
        prepare().backward()
    samples = []
    for _ in range(REPEATS):
        reset()
        loss = prepare()
        start = time.perf_counter()
        loss.backward()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def _framework_case(a_data, b_data, library_name, fused=False):
    if library_name == "Houssem":
        a = ht.tensor(a_data, requires_grad=True)
        b = ht.tensor(b_data, requires_grad=True)
        forward_a = ht.tensor(a_data, requires_grad=True)
        forward_b = ht.tensor(b_data, requires_grad=True)

        def prepare():
            with ht.fusion_mode(fused):
                return _ht_work(a, b).sum()

        def forward():
            with ht.fusion_mode(fused):
                _ht_work(forward_a, forward_b).numpy()
    else:
        a = torch.tensor(a_data, requires_grad=True)
        b = torch.tensor(b_data, requires_grad=True)
        forward_a = torch.tensor(a_data, requires_grad=True)
        forward_b = torch.tensor(b_data, requires_grad=True)

        def prepare():
            return _torch_work(a, b).sum()

        def forward():
            _torch_work(forward_a, forward_b)

    def reset():
        a.grad = None
        b.grad = None

    def combined():
        prepare().backward()

    return (
        _median_ms(forward),
        _time_backward(prepare, reset),
        _median_ms(combined, reset),
    )


def _cpp_case(library, a_data, b_data, suffix=""):
    size = a_data.size
    output = np.empty_like(a_data)
    grad_output = np.ones_like(a_data)
    grad_a = np.empty_like(a_data)
    grad_b = np.empty_like(b_data)
    a_ptr, b_ptr = _pointer(a_data), _pointer(b_data)
    output_ptr = _pointer(output)
    upstream_ptr = _pointer(grad_output)
    grad_a_ptr, grad_b_ptr = _pointer(grad_a), _pointer(grad_b)
    forward = getattr(library, f"fusion_cpp_forward{suffix}")
    backward = getattr(library, f"fusion_cpp_backward{suffix}")
    combined_kernel = getattr(library, f"fusion_cpp_forward_backward{suffix}")

    def run_forward():
        forward(a_ptr, b_ptr, output_ptr, size)

    def run_backward():
        backward(a_ptr, b_ptr, upstream_ptr, grad_a_ptr, grad_b_ptr, size)

    def run_combined():
        combined_kernel(a_ptr, b_ptr, output_ptr, grad_a_ptr, grad_b_ptr, size)

    return (
        _median_ms(run_forward),
        _median_ms(run_backward),
        _median_ms(run_combined),
    )


def _validate(library):
    rng = np.random.default_rng(42)
    for size in (1, 7, 8, 9, 17, 257):
        a_data = rng.standard_normal(size, dtype=np.float32)
        b_data = rng.standard_normal(size, dtype=np.float32)
        grad_output = rng.standard_normal(size, dtype=np.float32)
        edge_a = np.array([-20.0, -2.0, -0.25, -0.249998, 20.0],
                          dtype=np.float32)
        edge_count = min(size, edge_a.size)
        a_data[:edge_count] = edge_a[:edge_count]
        b_data[:edge_count] = 1.0

        h_a = ht.tensor(a_data, requires_grad=True)
        h_b = ht.tensor(b_data, requires_grad=True)
        with ht.fusion_mode(False):
            h_output = _ht_work(h_a, h_b)
        h_output.sum().backward()

        t_a = torch.tensor(a_data, requires_grad=True)
        t_b = torch.tensor(b_data, requires_grad=True)
        t_output = _torch_work(t_a, t_b)
        t_output.sum().backward()
        np.testing.assert_allclose(h_output.numpy(), t_output.detach().numpy(),
                                   rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(h_a.grad.numpy(), t_a.grad.numpy(),
                                   rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(h_b.grad.numpy(), t_b.grad.numpy(),
                                   rtol=2e-6, atol=1e-7)

        expected_output = h_output.numpy()
        expected_grad_a = h_a.grad.numpy()
        expected_grad_b = h_b.grad.numpy()
        for suffix in ("", "_omp") + (
            ("_avx2",) if library.fusion_cpp_avx2_supported() else ()
        ):
            if suffix == "_omp":
                library.fusion_cpp_set_threads(4)
            output = np.empty_like(a_data)
            grad_a = np.empty_like(a_data)
            grad_b = np.empty_like(b_data)
            forward = getattr(library, f"fusion_cpp_forward{suffix}")
            forward(_pointer(a_data), _pointer(b_data), _pointer(output), size)
            np.testing.assert_allclose(output, expected_output,
                                       rtol=3e-6, atol=3e-7)
            combined = getattr(library, f"fusion_cpp_forward_backward{suffix}")
            combined(_pointer(a_data), _pointer(b_data), _pointer(output),
                     _pointer(grad_a), _pointer(grad_b), size)
            np.testing.assert_allclose(output, expected_output,
                                       rtol=3e-6, atol=3e-7)
            np.testing.assert_allclose(grad_a, expected_grad_a,
                                       rtol=3e-6, atol=3e-7)
            np.testing.assert_allclose(grad_b, expected_grad_b,
                                       rtol=3e-6, atol=3e-7)

            cpp_grad_a = np.empty_like(a_data)
            cpp_grad_b = np.empty_like(b_data)
            backward = getattr(library, f"fusion_cpp_backward{suffix}")
            backward(_pointer(a_data), _pointer(b_data), _pointer(grad_output),
                     _pointer(cpp_grad_a), _pointer(cpp_grad_b), size)
            h_upstream = ht.tensor(grad_output)
            h_a.grad = None
            h_b.grad = None
            with ht.fusion_mode(False):
                (h_output * h_upstream).sum().backward()
            np.testing.assert_allclose(cpp_grad_a, h_a.grad.numpy(),
                                       rtol=3e-6, atol=3e-7)
            np.testing.assert_allclose(cpp_grad_b, h_b.grad.numpy(),
                                       rtol=3e-6, atol=3e-7)
            h_a.grad = None
            h_b.grad = None
            h_output.sum().backward()


def _houssem_peak_bytes(a_data, b_data):
    gc.collect()
    tracemalloc.start()
    a = ht.tensor(a_data, requires_grad=True)
    b = ht.tensor(b_data, requires_grad=True)
    with ht.fusion_mode(False):
        result = _ht_work(a, b)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result, a, b
    return peak


def main():
    library = build_library()
    if not library.fusion_cpp_openmp_enabled():
        raise RuntimeError("The C++ benchmark was built without OpenMP support.")
    _validate(library)
    compiler = shutil.which("g++")
    compiler_version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    has_avx2 = bool(library.fusion_cpp_avx2_supported())
    print(
        f"C++ fused pointwise benchmark: warmup={WARMUP}, repeats={REPEATS}\n"
        f"{compiler_version}\n"
        f"NumPy {np.__version__}, Numba {numba.__version__}, "
        f"PyTorch {torch.__version__}\n"
        f"AVX2 available: {has_avx2}; OpenMP enabled: True"
    )
    print(
        "C++ timings use preallocated buffers and exclude framework/autograd "
        "overhead; Houssem and PyTorch include graph creation."
    )
    print(
        f"{'size':>8} {'implementation':<22}{'threads':>8}"
        f"{'forward ms':>12}{'backward ms':>14}{'combined ms':>14}"
        f"{'vs HT eager':>13}"
    )
    for size in SIZES:
        rng = np.random.default_rng(size)
        a_data = rng.standard_normal(size, dtype=np.float32)
        b_data = rng.standard_normal(size, dtype=np.float32)
        eager = _framework_case(a_data, b_data, "Houssem")
        fused = _framework_case(a_data, b_data, "Houssem", fused=True)
        pytorch = _framework_case(a_data, b_data, "PyTorch")
        results = [
            ("Houssem eager", 1, eager),
            ("Houssem Numba", 1, fused),
            ("PyTorch eager", THREADS, pytorch),
            ("C++ scalar", 1, _cpp_case(library, a_data, b_data)),
        ]
        if has_avx2:
            results.append(
                ("C++ AVX2", 1, _cpp_case(library, a_data, b_data, "_avx2"))
            )
        for thread_count in OMP_THREAD_COUNTS:
            library.fusion_cpp_set_threads(thread_count)
            results.append(
                (
                    "C++ OpenMP",
                    thread_count,
                    _cpp_case(library, a_data, b_data, "_omp"),
                )
            )
        for name, thread_count, timing in results:
            ratio = eager[2] / timing[2] if timing[2] else float("inf")
            print(
                f"{size:>8} {name:<22}{thread_count:>8}"
                f"{timing[0]:>12.4f}{timing[1]:>14.4f}"
                f"{timing[2]:>14.4f}{ratio:>12.2f}x"
            )
        cpp_buffer_bytes = size * np.dtype(np.float32).itemsize * 5
        print(
            f"  memory estimate: C++ combined buffers={cpp_buffer_bytes} bytes "
            f"(no kernel scratch); Houssem Python-traced peak="
            f"{_houssem_peak_bytes(a_data, b_data)} bytes"
        )


if __name__ == "__main__":
    main()
