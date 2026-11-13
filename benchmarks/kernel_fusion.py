import gc
import os
import statistics
import time
import tracemalloc

N_THREADS = int(os.environ.get("BENCH_THREADS", "1"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "2"))
REPEAT = int(os.environ.get("BENCH_REPEATS", "7"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = str(N_THREADS)

import numpy as np  # noqa: E402
import numba  # noqa: E402
import torch  # noqa: E402
import houssem_torch as ht  # noqa: E402

torch.set_num_threads(N_THREADS)
SIZES = (64, 256, 1024, 4096, 16384, 65536)


def _ht_work(a, b):
    return ht.log(ht.exp(ht.relu((a * b + 0.25) / 2.0)) + 1.0)


def _torch_work(a, b):
    return torch.log(torch.exp(torch.relu((a * b + 0.25) / 2.0)) + 1.0)


def _median_ms(step, reset=lambda: None):
    for _ in range(WARMUP):
        reset()
        step()
    samples = []
    for _ in range(REPEAT):
        reset()
        start = time.perf_counter()
        step()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def _peak_traced_mib(step):
    gc.collect()
    tracemalloc.start()
    step()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def _time_backward(prepare, reset):
    for _ in range(WARMUP):
        reset()
        prepare().backward()

    samples = []
    for _ in range(REPEAT):
        reset()
        loss = prepare()
        start = time.perf_counter()
        loss.backward()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def _ht_case(a_data, b_data, fused):
    a = ht.tensor(a_data, requires_grad=True)
    b = ht.tensor(b_data, requires_grad=True)
    forward_a = ht.tensor(a_data, requires_grad=True)
    forward_b = ht.tensor(b_data, requires_grad=True)

    def reset():
        a.grad = None
        b.grad = None

    def prepare():
        with ht.fusion_mode(fused):
            return _ht_work(a, b).sum()

    def forward():
        with ht.fusion_mode(fused):
            _ht_work(forward_a, forward_b).numpy()

    def combined():
        prepare().backward()

    forward_ms = _median_ms(forward)
    backward_ms = _time_backward(prepare, reset)
    combined_ms = _median_ms(combined, reset)
    peak_mib = _peak_traced_mib(forward)
    return forward_ms, backward_ms, combined_ms, peak_mib


def _torch_case(a_data, b_data, compiled=None):
    a = torch.tensor(a_data, requires_grad=True)
    b = torch.tensor(b_data, requires_grad=True)
    forward_a = torch.tensor(a_data, requires_grad=True)
    forward_b = torch.tensor(b_data, requires_grad=True)
    work = compiled or _torch_work

    def reset():
        a.grad = None
        b.grad = None

    def prepare():
        return work(a, b).sum()

    def forward():
        work(forward_a, forward_b)

    def combined():
        prepare().backward()

    return (
        _median_ms(forward),
        _time_backward(prepare, reset),
        _median_ms(combined, reset),
        _peak_traced_mib(forward),
    )


def _compiled_torch_case(a_data, b_data):
    if not hasattr(torch, "compile"):
        return None

    compiled = torch.compile(_torch_work, fullgraph=True)
    a = torch.tensor(a_data, requires_grad=True)
    b = torch.tensor(b_data, requires_grad=True)
    forward_a = torch.tensor(a_data, requires_grad=True)
    forward_b = torch.tensor(b_data, requires_grad=True)
    for _ in range(WARMUP + 1):
        compiled(forward_a, forward_b)
    for _ in range(WARMUP + 1):
        a.grad = None
        b.grad = None
        compiled(a, b).sum().backward()
    return _torch_case(a_data, b_data, compiled)


def _print_rows(size, results):
    eager = results["Houssem eager"]
    for phase_index, phase in enumerate(("forward", "backward", "combined")):
        eager_ms = eager[phase_index]
        fields = [
            f"{size:>8} ",
            f"{phase:<10}",
            f"{eager_ms:>12.4f}",
        ]
        for name in ("Houssem fused", "PyTorch eager", "torch.compile"):
            result = results.get(name)
            value = "unavailable" if result is None else f"{result[phase_index]:.4f}"
            fields.append(f"{value:>14}")
        fused = results["Houssem fused"]
        speedup = eager_ms / fused[phase_index] if fused[phase_index] else float("inf")
        fields.append(f"{speedup:>10.2f}x")
        print("".join(fields))

    print(
        f"{'':>9}{'peak traced MiB':<19}"
        f"{results['Houssem eager'][3]:>12.3f}"
        f"{results['Houssem fused'][3]:>14.3f}"
        f"{'n/a':>14}{'n/a':>14}"
    )


def main():
    print(
        f"Pointwise forward/backward benchmark, {N_THREADS} thread(s), "
        f"warmup={WARMUP}, repeats={REPEAT}, "
        f"NumPy {np.__version__}, Numba {numba.__version__}, "
        f"PyTorch {torch.__version__}"
    )
    print(
        f"{'size':>8} {'phase':<10}{'HT eager ms':>12}"
        f"{'HT fused ms':>14}{'Torch ms':>14}{'compile ms':>14}"
        f"{'speedup':>12}"
    )
    print(
        "Memory is Python/NumPy tracemalloc peak; "
        "PyTorch native allocations are untracked."
    )
    compile_available = True
    for size in SIZES:
        rng = np.random.default_rng(size)
        a_data = rng.standard_normal(size, dtype=np.float32)
        b_data = rng.standard_normal(size, dtype=np.float32)
        results = {
            "Houssem eager": _ht_case(a_data, b_data, fused=False),
            "Houssem fused": _ht_case(a_data, b_data, fused=True),
            "PyTorch eager": _torch_case(a_data, b_data),
            "torch.compile": None,
        }
        if compile_available:
            try:
                results["torch.compile"] = _compiled_torch_case(a_data, b_data)
            except Exception as exc:
                compile_available = False
                print(f"torch.compile unavailable: {type(exc).__name__}: {exc}")
        _print_rows(size, results)


if __name__ == "__main__":
    main()
