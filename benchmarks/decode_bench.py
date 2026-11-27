"""GPU greedy decoding speed (Translator, key/value cache).

transformer_base with random weights (so sentences rarely stop early: every
step decodes the full batch). Reports ms per decoding step and sentences
per second for full-length outputs.

Run from the repository root (needs CuPy and a CUDA GPU):
    python benchmarks/decode_bench.py --device 1 --batch-sizes 32 64 128
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import houssem_torch as ht  # noqa: E402
from houssem_torch.backends.cuda import driver as drv  # noqa: E402
from houssem_torch.backends.cuda.decoder import Translator  # noqa: E402
from houssem_torch.backends.cuda.train_step import TrainStep  # noqa: E402
from houssem_torch.models import Transformer, TransformerConfig  # noqa: E402
from trainstep_profile import make_batch, table  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/transformer_base.json")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--src-len", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=63)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    if not drv.is_available():
        sys.exit("needs CuPy and a CUDA GPU")
    cfg = TransformerConfig.from_json(args.config)
    ht.manual_seed(0)
    step = TrainStep(Transformer(cfg), 8, args.src_len, args.src_len,
                     device=args.device)
    rows = []
    for B in args.batch_sizes:
        src = make_batch(cfg, B, args.src_len, args.src_len, 0.5)[0]
        tr = Translator(step, B, args.src_len)
        tr.translate_greedy(src, max_new_tokens=4)              # warm-up
        drv.synchronize()
        t0 = time.perf_counter()
        out = tr.translate_greedy(src, max_new_tokens=args.new_tokens)
        drv.synchronize()
        s = time.perf_counter() - t0
        steps = out.shape[1]
        rows.append(["greedy", B, B, steps, f"{s * 1e3 / steps:.2f}",
                     f"{B / s:.0f}"])
        print(f"greedy, batch {B}: {steps} steps in {s:.2f} s", flush=True)
        n = B // args.beam
        t0 = time.perf_counter()
        out = tr.translate_beam(src[:n], beam=args.beam,
                                max_new_tokens=args.new_tokens)
        drv.synchronize()
        s = time.perf_counter() - t0
        rows.append([f"beam {args.beam}", B, n, out.shape[1],
                     f"{s * 1e3 / out.shape[1]:.2f}", f"{n / s:.0f}"])
        print(f"beam {args.beam}, {n} sentences: {s:.2f} s", flush=True)
    info = drv.device_info(args.device)
    lines = [f"# Greedy decoding on {info.name}", "",
             f"`{args.config}`, random weights, sources of {args.src_len} "
             "tokens; time includes the encoder pass and the host loop.", ""]
    lines += table(["search", "rows", "sentences", "steps", "ms / step",
                    "sentences / s"], rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
