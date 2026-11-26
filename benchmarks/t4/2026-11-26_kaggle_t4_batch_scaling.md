# Batch-size scaling, Kaggle T4 (2026-11-26): houssem_torch vs torch.compile

[`benchmarks/batch_scaling_kaggle.py`](../batch_scaling_kaggle.py) at commit `df793b4`
on branch `gpu/gemm-tuning-buckets`, with both engines on cuda:1. All 55 GPU tests pass.

Setup: token-budget length buckets 16/32/48/64, 6000 steps per run. Every run
uses the same initial weights, the same data order and the same Noam schedule
(warmup 2000). Scale 1 is 1493 target tokens per step, the same as one 64-pair batch.
The order of the two engines alternated between scales. Validation time is
excluded from all timings. All numbers are also in [`history.json`](history.json).

## Bottom line

- houssem_torch is faster at every size: **1.28× / 1.22× / 1.20×** in steady
  ms/step at ×1/×2/×4, and **1.30× / 1.45× / 1.32×** in total training time
  (torch.compile's compile time is included in the totals).
- The steady advantage **levels off at about 1.2×**. Bigger batches give the
  matrix multiplies a bigger share of each step, and both engines run those
  on cuBLAS. Our fused kernels are a constant saving, so they become a smaller
  fraction of a longer step.
- **The results match at every size.** The two engines' validation curves agree
  to about 0.01 (and to 0.05 at worst, ×2 step 2000).
- **At ×2 and ×4, houssem_torch uses less memory** (2.25 vs 2.50 GB, and 3.61 vs
  4.24 GB). At ×1 it uses more (2.03 vs 1.62 GB), because the shared activation
  block is sized for the 64×64 validation bucket.

## Results

| scale | pairs per batch (16/32/48/64) | ms/step ht / torch | speedup | target tok/s ht / torch | wall s ht / torch | val ht / torch | memory GB ht / torch |
|---:|---|---:|---:|---:|---:|---:|---:|
| ×1 | 152/64/40/32 | 49.5 / 63.6 | 1.28× | 29,858 / 22,899 | 299 / 389 | 4.118 / 4.129 | 2.03 / 1.62 |
| ×2 | 304/136/80/56 | 91.6 / 111.8 | 1.22× | 32,351 / 22,357 | 550 / 796 | 3.640 / 3.649 | 2.25 / 2.50 |
| ×4 | 608/272/160/112 | 173.7 / 208.0 | 1.20× | 34,121 / 25,882 | 1044 / 1376 | 3.601 / 3.607 | 3.61 / 4.24 |

Throughput in the table is based on total time. In steady state, houssem_torch
went from 30.1k to 32.4k to 34.2k target tokens/s, and torch.compile from 23.4k
to 26.5k to 28.5k. Both get more efficient on bigger batches.

torch.compile's compile time: about 8 s at ×1 (from a warm cache), and about
125 s at ×2 and at ×4. It builds 4 graphs per scale, one per bucket shape.
houssem_torch's setup takes 1.3–1.9 s.

## Validation loss (houssem_torch; torch.compile within 0.01–0.05 at every point)

| step | ×1 | ×2 | ×4 |
|---:|---:|---:|---:|
| 1000 | 6.202 | 5.846 | 5.429 |
| 2000 | 5.462 | 4.753 | 4.193 |
| 3000 | 4.868 | 4.111 | 3.753 |
| 4000 | 4.514 | 3.852 | 3.606 |
| 5000 | 4.278 | 3.727 | **3.575** |
| 6000 | 4.118 | 3.640 | 3.601 ↑ |

- **The val loss rise at ×4 is overfitting.** The opus_books training set has about 2.7M target
  tokens. In 6000 steps, ×4 makes about 13 passes over it, against 6.6 for ×2 and
  3.3 for ×1. Its training-batch loss (~2.5) is far below its val loss (~3.6),
  and torch shows exactly the same rise. This is a property of the dataset,
  not a problem with batch size.
- **At equal training time, the three sizes give about the same loss.** At about 300 s:
  ×1 reaches 4.118 (299 s), ×2 4.111 (276 s), and ×4 about 4.19 (351 s).
  Per token processed they learn about equally well; bigger batches mainly get
  more out of the GPU (+14% tokens/s from ×1 to ×4).
- **Best model in this run:** ×4 at step 5000 (val 3.575, 870 s). ×2 at step 6000
  (3.640, 550 s) is the better trade-off between time and quality.

## Next

- To make big batches faster, the next target is the matrix multiplies: they
  dominate at large scales, and ours are not faster than PyTorch's.
- To train longer at ×4, use a larger dataset (or regularization).
  opus_books overfits after about 10 passes.
- The bigger-batch scales still use the batch-64 learning-rate schedule. A higher
  learning rate at ×4 may converge faster before the overfitting sets in.
