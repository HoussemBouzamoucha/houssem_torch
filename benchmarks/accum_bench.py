"""Gradient accumulation: k micro-batches of B rows vs one batch of k·B rows.

Same rows, same update. Reports ms per optimizer step (median of rounds)
and the device memory each step holds: accumulation should cost about the
same time per row with the activation memory of one micro-batch.

Run from the repository root (needs CuPy and a CUDA GPU):
    python benchmarks/accum_bench.py --device 1 --micro 64 --k 4
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import houssem_torch as ht  # noqa: E402
from houssem_torch.backends.cuda import driver as drv  # noqa: E402
from houssem_torch.backends.cuda.train_step import TrainStep  # noqa: E402
from houssem_torch.models import Transformer, TransformerConfig  # noqa: E402
from trainstep_profile import make_batch, table  # noqa: E402


def timed(fn, n):
    t = drv.Timer()
    t.start()
    for _ in range(n):
        fn()
    return t.stop() / n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/transformer_base.json")
    ap.add_argument("--micro", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--src-len", type=int, default=64)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    if not drv.is_available():
        sys.exit("needs CuPy and a CUDA GPU")
    import cupy
    cfg = TransformerConfig.from_json(args.config)
    B, k, S, T = args.micro, args.k, args.src_len, args.tgt_len
    micro = [make_batch(cfg, B, S, T, 0.5, seed=i) for i in range(k)]
    big = tuple(np.concatenate(p) for p in zip(*micro))
    rows = []
    for name, rows_per_step, run in (
            (f"{k} x {B} accumulated", B, lambda s: s.accumulate_step(micro, 1e-4)),
            (f"1 x {k * B}", k * B, lambda s: s(*big, lr=1e-4))):
        cupy.get_default_memory_pool().free_all_blocks()
        ht.manual_seed(0)
        step = TrainStep(Transformer(cfg), rows_per_step, S, T,
                         max_grad_norm=1.0, device=args.device)
        timed(lambda: run(step), 2)
        ms = statistics.median(timed(lambda: run(step), args.steps)
                               for _ in range(args.rounds))
        drv.synchronize()
        gb = step.memory_bytes() / 1e9
        rows.append([name, f"{ms:.1f}", f"{ms / (k * B):.3f}", f"{gb:.2f}",
                     f"{step.last_loss():.4f}"])
        print(f"{name}: {ms:.1f} ms/step, {gb:.2f} GB", flush=True)
        step = None
    info = drv.device_info(args.device)
    lines = [f"# Gradient accumulation on {info.name}", "",
             f"`{args.config}`, {S}/{T} tokens, {k * B} rows per optimizer "
             f"step, median of {args.rounds} rounds x {args.steps} steps.", ""]
    lines += table(["run", "ms / step", "ms / row", "memory GB", "loss"], rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
