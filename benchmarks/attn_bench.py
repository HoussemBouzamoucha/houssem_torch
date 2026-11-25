"""Time the fused attention kernels alone (attn_fwd, attn_bwd).

Default shape is transformer_base at the Kaggle benchmark's batch:
B=64, H=8, L=64, DK=64, dropout 0.1, with the Q/K/V bias gradients on.
Reports the median of several timed runs per kernel, for the non-causal
(encoder self / cross) and causal (decoder self) variants.

    python benchmarks/attn_bench.py              # default shape
    python benchmarks/attn_bench.py --B 32 --L 48
    python benchmarks/attn_bench.py --ref afbb189 --nwb 4,8,16
        (also time kernels.cu as of a git revision, and sweep the backward
        kernel's warps per block)
"""
import argparse
import math
import subprocess
from pathlib import Path

import numpy as np

from houssem_torch.backends.cuda import driver
from houssem_torch.backends.cuda.train_step import (
    _ATT_BWD_WARPS, _ATT_WARPS, _SOURCE, _pad16, attn_smem)


def bench(B, H, L, DK, causal, p, reps, iters, source=_SOURCE,
          nwb=_ATT_BWD_WARPS):
    d = H * DK
    rng = np.random.default_rng(0)
    f16 = lambda a: a.astype(np.float16)  # noqa: E731
    qkv = driver.from_numpy(f16(rng.standard_normal((B * L, 3 * d))))
    bias = driver.from_numpy(f16(rng.standard_normal(3 * d) * 0.1))
    kid = rng.integers(3, 50, (B, L)).astype(np.int32)
    kid[:, L - L // 4:] = 0                     # some trailing padding
    kid = driver.from_numpy(kid)
    dout = driver.from_numpy(f16(rng.standard_normal((B * L, d))))
    out = driver.zeros((B * L, d), np.float16)
    lse = driver.zeros(2 * B * H * L, np.float32)
    gqkv = driver.zeros((B * L, 3 * d), np.float16)
    gb = driver.zeros(3 * d)

    defs = dict(USE_HALF=1, ATT=1, AQ=L, AK=L, AQP=_pad16(L), AKP=_pad16(L),
                DK=DK, CAUSAL=causal, NW=_ATT_WARPS, NWB=nwb)
    kf = driver.compile_kernel(source, "attn_fwd", defs)
    kb = driver.compile_kernel(source, "attn_bwd", defs)
    fwd_smem, bwd_smem, _ = attn_smem(L, L, DK)
    kf.set_max_dynamic_shared(fwd_smem)
    kb.set_max_dynamic_shared(bwd_smem)
    ptr = driver.ptr
    qkv_args = [ptr(qkv, 0), 3 * d, ptr(bias, 0),
                ptr(qkv, d), 3 * d, ptr(bias, d),
                ptr(qkv, 2 * d), 3 * d, ptr(bias, 2 * d)]
    alpha = 1.0 / math.sqrt(DK)
    threads = 32 * _ATT_WARPS

    def fwd():
        kf(B * H, threads, *qkv_args, kid, out, d, lse, H, 0, np.uint32(7), p,
           alpha, shared_mem=fwd_smem)

    def bwd():
        kb(B * H, 32 * nwb, *qkv_args, kid, out, d, dout, d, lse,
           ptr(gqkv, 0), 3 * d, ptr(gqkv, d), 3 * d, ptr(gqkv, 2 * d), 3 * d,
           ptr(gb, 0), ptr(gb, d), ptr(gb, 2 * d),
           H, 0, np.uint32(7), p, alpha, shared_mem=bwd_smem)

    res = {}
    for name, fn in (("fwd", fwd), ("bwd", bwd)):
        for _ in range(3):
            fn()
        driver.synchronize()
        times = []
        for _ in range(reps):
            t = driver.Timer()
            t.start()
            for _ in range(iters):
                fn()
            times.append(t.stop() / iters)
        res[name] = float(np.median(times))
    return res, fwd_smem, bwd_smem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--DK", type=int, default=64)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ref", help="also time kernels.cu at this git revision "
                    "(its backward with the forward's warp count)")
    ap.add_argument("--nwb", default=str(_ATT_BWD_WARPS),
                    help="comma-separated backward warps per block to try")
    a = ap.parse_args()
    info = driver.device_info()
    print(f"{info.name}: B={a.B} H={a.H} L={a.L} DK={a.DK} p={a.p}")
    variants = []
    if a.ref:
        root = Path(__file__).resolve().parents[1]
        old = subprocess.run(
            ["git", "show", f"{a.ref}:houssem_torch/backends/cuda/kernels.cu"],
            cwd=root, check=True, capture_output=True, text=True).stdout
        variants.append((f"{a.ref} (bwd {_ATT_WARPS} warps)", old, _ATT_WARPS))
    for n in (int(x) for x in a.nwb.split(",")):
        variants.append((f"current (bwd {n} warps)", _SOURCE, n))
    for label, src, nwb in variants:
        print(f"{label}:")
        for causal in (0, 1):
            r, fs, bs = bench(a.B, a.H, a.L, a.DK, causal, a.p, a.reps,
                              a.iters, src, nwb)
            print(f"  causal={causal}: attn_fwd {r['fwd'] * 1e3:8.1f} us "
                  f"({fs / 1024:.1f} KB) | attn_bwd {r['bwd'] * 1e3:8.1f} us "
                  f"({bs / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
