import ctypes
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

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
FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)


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
            "-march=native",
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
    size_arg = ctypes.c_size_t
    library.fusion_cpp_forward.argtypes = [
        FLOAT_POINTER, FLOAT_POINTER, FLOAT_POINTER, size_arg
    ]
    library.fusion_cpp_forward.restype = None
    library.fusion_cpp_backward.argtypes = [
        FLOAT_POINTER, FLOAT_POINTER, FLOAT_POINTER, FLOAT_POINTER,
        FLOAT_POINTER, size_arg
    ]
    library.fusion_cpp_backward.restype = None
    library.fusion_cpp_forward_backward.argtypes = [
        FLOAT_POINTER, FLOAT_POINTER, FLOAT_POINTER, FLOAT_POINTER,
        FLOAT_POINTER, size_arg
    ]
    library.fusion_cpp_forward_backward.restype = None


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


def _cpp_case(library, a_data, b_data):
    size = a_data.size
    output = np.empty_like(a_data)
    grad_output = np.ones_like(a_data)
    grad_a = np.empty_like(a_data)
    grad_b = np.empty_like(b_data)
    a_ptr, b_ptr = _pointer(a_data), _pointer(b_data)
    output_ptr = _pointer(output)
    upstream_ptr = _pointer(grad_output)
    grad_a_ptr, grad_b_ptr = _pointer(grad_a), _pointer(grad_b)

    def forward():
        library.fusion_cpp_forward(a_ptr, b_ptr, output_ptr, size)

    def backward():
        library.fusion_cpp_backward(
            a_ptr, b_ptr, upstream_ptr, grad_a_ptr, grad_b_ptr, size
        )

    def combined():
        library.fusion_cpp_forward_backward(
            a_ptr, b_ptr, output_ptr, grad_a_ptr, grad_b_ptr, size
        )

    return _median_ms(forward), _median_ms(backward), _median_ms(combined)


def _validate(library):
    rng = np.random.default_rng(42)
    a_data = rng.standard_normal(257, dtype=np.float32)
    b_data = rng.standard_normal(257, dtype=np.float32)
    cpp_output = np.empty_like(a_data)
    cpp_grad_a = np.empty_like(a_data)
    cpp_grad_b = np.empty_like(b_data)

    h_a = ht.tensor(a_data, requires_grad=True)
    h_b = ht.tensor(b_data, requires_grad=True)
    with ht.fusion_mode(False):
        h_output = _ht_work(h_a, h_b)
    h_output.sum().backward()

    library.fusion_cpp_forward_backward(
        _pointer(a_data), _pointer(b_data), _pointer(cpp_output),
        _pointer(cpp_grad_a), _pointer(cpp_grad_b), a_data.size
    )
    np.testing.assert_allclose(cpp_output, h_output.numpy(), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(cpp_grad_a, h_a.grad.numpy(), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(cpp_grad_b, h_b.grad.numpy(), rtol=2e-6, atol=1e-7)

    t_a = torch.tensor(a_data, requires_grad=True)
    t_b = torch.tensor(b_data, requires_grad=True)
    t_output = _torch_work(t_a, t_b)
    t_output.sum().backward()
    np.testing.assert_allclose(cpp_output, t_output.detach().numpy(),
                               rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(cpp_grad_a, t_a.grad.numpy(),
                               rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(cpp_grad_b, t_b.grad.numpy(),
                               rtol=2e-6, atol=1e-7)


def main():
    library = build_library()
    _validate(library)
    compiler = shutil.which("g++")
    compiler_version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    print(
        f"C++ fused pointwise benchmark: {THREADS} thread(s), "
        f"warmup={WARMUP}, repeats={REPEATS}\n"
        f"{compiler_version}\n"
        f"NumPy {np.__version__}, Numba {numba.__version__}, "
        f"PyTorch {torch.__version__}"
    )
    print(
        "C++ timings use preallocated buffers and exclude framework/autograd "
        "overhead; Houssem and PyTorch include graph creation."
    )
    print(
        f"{'size':>8} {'framework':<15}{'forward ms':>12}"
        f"{'backward ms':>14}{'combined ms':>14}{'HT eager/row':>14}"
    )
    for size in SIZES:
        rng = np.random.default_rng(size)
        a_data = rng.standard_normal(size, dtype=np.float32)
        b_data = rng.standard_normal(size, dtype=np.float32)
        eager = _framework_case(a_data, b_data, "Houssem")
        fused = _framework_case(a_data, b_data, "Houssem", fused=True)
        pytorch = _framework_case(a_data, b_data, "PyTorch")
        cpp = _cpp_case(library, a_data, b_data)
        for name, result in (
            ("Houssem eager", eager),
            ("Houssem Numba", fused),
            ("PyTorch eager", pytorch),
            ("C++ kernel", cpp),
        ):
            ratio = eager[2] / result[2] if result[2] else float("inf")
            print(
                f"{size:>8} {name:<15}{result[0]:>12.4f}"
                f"{result[1]:>14.4f}{result[2]:>14.4f}{ratio:>11.2f}x"
            )


if __name__ == "__main__":
    main()
