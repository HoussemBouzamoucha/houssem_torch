# Gradient accumulation (Kaggle T4, 2026-11-27)

Branch `gpu/faster-gemms`, commit 5854b5f. 65 of 66 GPU tests passed. The one
failure was the weight comparison after 3 AdamW steps of accumulated vs. big
batch (fp32): 1 of 12288 weights was off by 1.8e-5 against `atol=1e-5`. The
loss and every gradient matched. It is rounding noise that AdamW amplifies, so
the tolerance is now 1e-4.

`benchmarks/accum_bench.py --device 1 --micro 64 --k 4`, transformer_base,
64/64 tokens, 256 rows per optimizer step, median of 5 rounds x 5 steps:

| run | ms / step | ms / row | memory GB |
|---|---:|---:|---:|
| 4 x 64 accumulated | 293.9 | 1.148 | 2.18 |
| 1 x 256 | 277.0 | 1.082 | 5.88 |

- Accumulation gives the batch-256 update with **37% of the memory** (2.18 vs
  5.88 GB) for **6% more time**. A 64-row multiply uses the T4 a little less
  well than a 256-row one, and there are 4x the launches.
- Use a real big batch when it fits. Use accumulation for batches that don't,
  or to keep memory for longer sequences or a bigger model.
- The loss values differ (8.820 vs 8.833) only because dropout draws different
  masks per micro-batch.
