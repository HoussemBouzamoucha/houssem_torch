"""The training step's matrix multiplies, one by one: ours, other layouts, PyTorch.

Records every multiply one `TrainStep` makes (shape, transposes, dtypes,
beta), then times each one in isolation, all variants back to back in the
same round so a drifting clock (the T4 is power-capped) hits every column
alike; the median over rounds is reported. Variants per multiply:

  ours        exactly what the step runs (cuBLAS default algorithm)
  ours alt    the weight stored the other way round: weights (out, in) like
              PyTorch's nn.Linear, so the forward is x·Wᵀ and the input
              gradient g·W; a weight gradient becomes gᵀ·x into (out, in)
  ours fp16   a float32 output (weight gradients) written as float16 instead
  torch       torch.matmul on our layout, fp16 in and out (as under autocast)
  torch alt   torch.matmul on the alt layout (what nn.Linear runs)

Run from the repository root (needs CuPy and a CUDA GPU; PyTorch optional):
    python benchmarks/gemm_bench.py
    python benchmarks/gemm_bench.py --batch 64 --device 1
Writes gemm_bench_report.md and gemm_bench_results.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
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

VARIANTS = ("ours", "ours alt", "ours fp16", "torch", "torch alt")


def record_calls(step, batch):
    """Counter of {(batch, m, n, k, trans_a, trans_b, ab, c, beta): calls}
    for one training step."""
    calls = Counter()
    gemm, gemm_batched = drv.gemm, drv.gemm_batched

    def rec(a, b, c, nb, trans_a, trans_b, beta):
        shp_a = a.shape[-2:]
        shp_b = b.shape[-2:]
        m, k = shp_a[::-1] if trans_a else shp_a
        n = shp_b[0] if trans_b else shp_b[1]
        calls[(nb, m, n, k, trans_a, trans_b, np.dtype(a.dtype).name,
               np.dtype(c.dtype).name, beta != 0)] += 1

    def g2(a, b, c, *, trans_a=False, trans_b=False, alpha=1.0, beta=0.0):
        rec(a, b, c, 0, trans_a, trans_b, beta)
        gemm(a, b, c, trans_a=trans_a, trans_b=trans_b, alpha=alpha, beta=beta)

    def g3(a, b, c, *, trans_a=False, trans_b=False, alpha=1.0, beta=0.0):
        rec(a, b, c, a.shape[0], trans_a, trans_b, beta)
        gemm_batched(a, b, c, trans_a=trans_a, trans_b=trans_b, alpha=alpha,
                     beta=beta)

    drv.gemm, drv.gemm_batched = g2, g3
    try:
        step(*batch, lr=1e-4)
        drv.synchronize()
    finally:
        drv.gemm, drv.gemm_batched = gemm, gemm_batched
    return calls


def label(key):
    nb, m, n, k, ta, tb, ab, c, beta = key
    s = f"{m}x{n}x{k}" + (f" x{nb}" if nb else "")
    s += (" A^T" if ta else "") + (" B^T" if tb else "")
    return s + (f" {ab}->{c}" if c != ab else "") + (" +C" if beta else "")


def role(key, vocab):
    nb, m, n, k, ta, tb, *_ = key
    if nb:
        return "attention"
    if m == 1:
        return "bias grad"
    if ta:
        return "weight grad"
    if tb:
        return "logits forward" if n == vocab else "input grad"
    return "logits input grad" if k == vocab else "forward"


def alt_layout(key):
    """(m, n, k, trans_a, trans_b) of the same product with the weight stored
    transposed; None when there is no weight (attention, bias gradients)."""
    nb, m, n, k, ta, tb, *_ = key
    if nb or m == 1:
        return None
    if ta:      # C = Aᵀ·B, C is the weight gradient: compute Cᵀ = Bᵀ·A
        return n, m, k, True, False
    return m, n, k, False, not tb   # B is the weight: flip how it is stored


def operands(m, n, k, ta, tb, nb):
    """Shapes of A and B as stored, for C(m, n) = op(A)·op(B)."""
    lead = (nb,) if nb else ()
    return lead + ((k, m) if ta else (m, k)), lead + ((n, k) if tb else (k, n))


class OursRunner:
    def __init__(self, m, n, k, ta, tb, nb, ab, c, beta):
        import cupy
        sa, sb = operands(m, n, k, ta, tb, nb)
        rnd = cupy.random.default_rng(0)
        self.a = drv.Buffer(rnd.standard_normal(sa, dtype=np.float32).astype(ab))
        self.b = drv.Buffer(rnd.standard_normal(sb, dtype=np.float32).astype(ab))
        self.c = drv.Buffer(cupy.zeros(((nb,) if nb else ()) + (m, n), dtype=c))
        self.f = drv.gemm_batched if nb else drv.gemm
        self.kw = dict(trans_a=ta, trans_b=tb, beta=1.0 if beta else 0.0)
        self.cupy = cupy

    def __call__(self, reps):
        start, end = self.cupy.cuda.Event(), self.cupy.cuda.Event()
        self.f(self.a, self.b, self.c, **self.kw)
        start.record()
        for _ in range(reps):
            self.f(self.a, self.b, self.c, **self.kw)
        end.record()
        end.synchronize()
        return self.cupy.cuda.get_elapsed_time(start, end) * 1e3 / reps


class TorchRunner:
    def __init__(self, m, n, k, ta, tb, nb, device):
        import torch
        sa, sb = operands(m, n, k, ta, tb, nb)
        dev = torch.device(f"cuda:{device}")
        a = torch.randn(*sa, device=dev, dtype=torch.float16)
        b = torch.randn(*sb, device=dev, dtype=torch.float16)
        self.a, self.b = (a.mT if ta else a), (b.mT if tb else b)
        self.torch = torch

    def __call__(self, reps):
        torch = self.torch
        torch.matmul(self.a, self.b)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(reps):
            torch.matmul(self.a, self.b)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e3 / reps


def runners(key, device, has_torch):
    nb, m, n, k, ta, tb, ab, c, beta = key
    out = {"ours": OursRunner(m, n, k, ta, tb, nb, ab, c, beta)}
    alt = alt_layout(key)
    if alt:
        out["ours alt"] = OursRunner(*alt, nb, ab, c, beta)
    if c != ab:
        out["ours fp16"] = OursRunner(m, n, k, ta, tb, nb, ab, ab, False)
    if has_torch and ab == "float16":
        out["torch"] = TorchRunner(m, n, k, ta, tb, nb, device)
        if alt:
            out["torch alt"] = TorchRunner(*alt, nb, device)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/transformer_base.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--src-len", type=int, default=64)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="gemm_bench_report.md")
    args = ap.parse_args()
    if not drv.is_available():
        sys.exit("needs CuPy and a CUDA GPU")
    try:
        import torch
        torch.cuda.set_device(args.device)
        has_torch = True
    except ImportError:
        has_torch = False

    cfg = TransformerConfig.from_json(args.config)
    batch = make_batch(cfg, args.batch, args.src_len, args.tgt_len, 0.5)
    ht.manual_seed(0)
    drv.reset_gemm_algorithms()
    step = TrainStep(Transformer(cfg), args.batch, args.src_len, args.tgt_len,
                     max_grad_norm=1.0, device=args.device, tune_gemms=False)
    for _ in range(3):
        step(*batch, lr=1e-4)
    calls = record_calls(step, batch)
    del step
    import cupy
    cupy.get_default_memory_pool().free_all_blocks()

    keys = sorted(calls)
    run = {key: runners(key, args.device, has_torch) for key in keys}
    times = defaultdict(lambda: defaultdict(list))
    for _ in range(args.rounds):
        for key in keys:
            for name, r in run[key].items():
                times[key][name].append(r(args.reps))
    med = {key: {v: statistics.median(ts) * calls[key]
                 for v, ts in times[key].items()} for key in keys}

    rows, data = [], []
    tot = defaultdict(float)
    best_tot = 0.0
    for key in sorted(keys, key=lambda k: -med[k]["ours"]):
        m = med[key]
        ours_choices = [m[v] for v in ("ours", "ours alt", "ours fp16") if v in m]
        best = min(ours_choices)
        best_tot += best
        for v in VARIANTS:
            tot[v] += m.get(v, m["ours"] if v != "torch alt" else
                            m.get("torch", m["ours"]))
        cells = [f"{m[v]:.0f}" if v in m else "-" for v in VARIANTS]
        r = role(key, cfg.vocab_size)
        rows.append([label(key), r, str(calls[key])] + cells)
        data.append(dict(label=label(key), role=r, calls=calls[key],
                         key=list(key), us=m))
    rows.append(["**total**", "", ""] + [f"{tot[v]:.0f}" for v in VARIANTS])

    info = drv.device_info(args.device)
    lines = [f"# Training-step matrix multiplies on {info.name}", "",
             f"Batch {args.batch} x {args.src_len}/{args.tgt_len}, "
             f"`{args.config}`. µs per training step (all calls of the "
             f"multiply), median of {args.rounds} rounds of {args.reps}; every "
             "variant of a multiply is timed in the same round. Totals fill a "
             "missing variant with the plain one (ours or torch)."
             + ("" if has_torch else " PyTorch not installed."), "",
             f"Best of ours / ours alt / ours fp16 per multiply: "
             f"**{best_tot:.0f} µs** vs {tot['ours']:.0f} µs now.", ""]
    lines += table(["multiply (m x n x k)", "role", "calls"] + list(VARIANTS),
                   rows)
    report = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.splitext(args.out)[0].replace("_report", "")
              + "_results.json", "w", encoding="utf-8") as f:
        json.dump(dict(gpu=info.name, batch=args.batch, src_len=args.src_len,
                       tgt_len=args.tgt_len, rounds=args.rounds, reps=args.reps,
                       best_total_us=best_tot, totals_us=tot, rows=data),
                  f, indent=2, default=float)
    print(report)


if __name__ == "__main__":
    main()
