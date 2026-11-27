"""Training step with and without the cuBLASLt FFN-backward epilogue.

One `TrainStep` per batch size; the fused path is switched on and off on
the same step (`step.lt_ffn`), alternating every round so the T4's drifting
clock hits both alike. Reports the median ms per step and the GPU time of
the FFN-backward pieces from one profiled step in each mode.

Run from the repository root (needs CuPy, a CUDA GPU and cuBLASLt):
    python benchmarks/epilogue_bench.py --device 1 --batch-sizes 64 256
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import houssem_torch as ht  # noqa: E402
from houssem_torch.backends.cuda import driver as drv  # noqa: E402
from houssem_torch.backends.cuda.train_step import TrainStep  # noqa: E402
from houssem_torch.models import Transformer, TransformerConfig  # noqa: E402
from trainstep_profile import make_batch, table  # noqa: E402


def time_steps(step, batch, n):
    t = drv.Timer()
    t.start()
    for _ in range(n):
        step(*batch, lr=1e-4)
    return t.stop() / n


def ffn_bwd_us(step, batch):
    """GPU µs of the launches that differ between the two modes."""
    with drv.Profiler() as prof:
        step(*batch, lr=1e-4)
    out = defaultdict(float)
    for r in prof.records():
        if r.name in ("act_bwd", "add_into_f32", "gemm_lt", "bias_act_fwd"):
            out[r.name] += r.ms * 1e3
    return dict(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/transformer_base.json")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--src-len", type=int, default=64)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    if not drv.is_available():
        sys.exit("needs CuPy and a CUDA GPU")
    cfg = TransformerConfig.from_json(args.config)
    rows = []
    for B in args.batch_sizes:
        batch = make_batch(cfg, B, args.src_len, args.tgt_len, 0.5)
        ht.manual_seed(0)
        step = TrainStep(Transformer(cfg), B, args.src_len, args.tgt_len,
                         max_grad_norm=1.0, device=args.device)
        if not step.lt_ffn:
            sys.exit("the fused FFN epilogue is unavailable on this setup")
        ms = {True: [], False: []}
        for mode in (True, False):
            step.lt_ffn = mode
            time_steps(step, batch, 3)
        for _ in range(args.rounds):
            for mode in (True, False):
                step.lt_ffn = mode
                ms[mode].append(time_steps(step, batch, args.steps))
        prof = {}
        for mode in (True, False):
            step.lt_ffn = mode
            prof[mode] = ffn_bwd_us(step, batch)
        step.lt_ffn = True
        fused, plain = (statistics.median(ms[m]) for m in (True, False))
        rows.append([B, f"{plain:.1f}", f"{fused:.1f}", f"{plain / fused:.3f}",
                     f"{plain - fused:.2f}",
                     ", ".join(f"{k} {v:.0f}" for k, v in sorted(prof[False].items())),
                     ", ".join(f"{k} {v:.0f}" for k, v in sorted(prof[True].items()))])
        print(f"batch {B}: plain {plain:.1f} ms, fused {fused:.1f} ms", flush=True)
        del step
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    info = drv.device_info(args.device)
    lines = [f"# FFN-backward epilogue on {info.name}", "",
             f"`{args.config}`, {args.src_len}/{args.tgt_len} tokens, median of "
             f"{args.rounds} rounds x {args.steps} steps, modes alternating.", ""]
    lines += table(["batch", "plain ms", "fused ms", "speedup", "saved ms",
                    "plain µs (profiled)", "fused µs (profiled)"], rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
