import os
import statistics
import time

N_THREADS = int(os.environ.get("BENCH_THREADS", "1"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = str(N_THREADS)

import numpy as np
import torch
import houssem_torch as ht

torch.set_num_threads(N_THREADS)


def benchmark(step, reset, warmup=5, repeat=30):
    for _ in range(warmup):
        reset()
        step()

    times = []
    for _ in range(repeat):
        reset()
        start = time.perf_counter()
        step()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


CASES = (
    ("pointwise", lambda a, b: (a * b + a).sum(),
     lambda a, b: (a * b + a).sum()),
    ("matmul", lambda a, b: (a @ b).sum(),
     lambda a, b: (a @ b).sum()),
    ("chain", lambda a, b: ht.relu(a @ b).sum() * ht.exp(a.mean() * 0.1),
     lambda a, b: torch.relu(a @ b).sum() * torch.exp(a.mean() * 0.1)),
)


def main():
    rng = np.random.default_rng(0)
    print(f"CPU autograd benchmark: {N_THREADS} thread(s), "
          f"{torch.__version__=}, NumPy {np.__version__}")
    print(f"{'case':<12}{'size':>8}{'Houssem (ms)':>16}"
          f"{'PyTorch (ms)':>16}{'Houssem/PyTorch':>19}")

    for size in (64, 256):
        a = rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)
        ht_a = ht.tensor(a, requires_grad=True)
        ht_b = ht.tensor(b, requires_grad=True)
        torch_a = torch.tensor(a, requires_grad=True)
        torch_b = torch.tensor(b, requires_grad=True)

        for name, ht_fn, torch_fn in CASES:
            ht_ms = benchmark(
                lambda: ht_fn(ht_a, ht_b).backward(),
                lambda: (setattr(ht_a, "grad", None),
                         setattr(ht_b, "grad", None)),
            )
            torch_ms = benchmark(
                lambda: torch_fn(torch_a, torch_b).backward(),
                lambda: (setattr(torch_a, "grad", None),
                         setattr(torch_b, "grad", None)),
            )
            print(f"{name:<12}{size:>8}{ht_ms:>16.4f}"
                  f"{torch_ms:>16.4f}{ht_ms / torch_ms:>18.2f}x")


if __name__ == "__main__":
    main()
