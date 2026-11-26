# houssem_torch vs PyTorch, Kaggle T4 (2026-11-26): cuBLAS tuning and length buckets

One run of [`benchmarks/cuda_train_kaggle.py`](../cuda_train_kaggle.py) plus
[`benchmarks/gemm_bench.py`](../gemm_bench.py), branch `gpu/gemm-tuning-buckets`
at commit `b920e4a`. Both engines ran on cuda:1. Setup: transformer_base, fp16,
batch 64, opus_books fr→en. All 53 GPU tests pass on the T4. The numbers are also in
[`history.json`](history.json).

## Bottom line

- **Length buckets work.** With the same 3000 batches, houssem_torch takes
  **154 s instead of 268 s (1.74× faster)**. Its steady speed is 51.3 ms/step,
  against 64.7 for torch.compile on the same buckets (1.26×).
- **Val loss got worse with buckets, for both engines equally** (5.02 vs 4.75 at fixed
  64/64). The cause is the batching (64 pairs per batch whatever their length), not
  the engine. Fix: batch to a token budget instead (see Next).
- **cuBLAS algorithm tuning gained nothing.** The default algorithm was the
  fastest for 16 of 18 shapes. The matmul gap to PyTorch is now about 3% (54.2 vs
  52.6 ms/step), which is within noise. Algorithm choice was not the cause.

## Fixed 64/64 batches (ms/step)

| batch | houssem_torch | torch.compile | eager | speedup vs compile |
|---:|---:|---:|---:|---:|
| 32 | 45.2 | 58.2 | 70.5 | 1.29× |
| 64 | 84.0 | 106.3 | 128.1 | **1.27×** |
| 96 | 136.4 | 151.7 | 187.8 | 1.11× |

Training, 3000 steps: **268 vs 325 s (1.21×)**, val loss 4.752 vs 4.792. Startup
0.6 s vs 30.6 s. Memory 2.03 vs 2.19 GB.

## Length buckets (16/32/48/64, 64 pairs per batch, same batches for both)

Share of batches by bucket: 16: 34%, 32: 38%, 48: 20%, 64: 8%. In the fixed 64/64
batches, only 38% of the target slots are real tokens.

| engine | wall s | steady ms/step | val loss | fixed: wall s | fixed: ms/step | fixed: val |
|---|---:|---:|---:|---:|---:|---:|
| houssem_torch | **154** | **51.3** | 5.018 | 268 | 89.0 | 4.752 |
| torch.compile | 286 | 64.7 | 5.015 | 325 | 107.5 | 4.792 |

- torch.compile's wall time includes compiling one graph per bucket, about 95 s
  before step 250. houssem_torch's startup with four buckets is 8.5 s, which
  includes compiling the kernels and tuning.
- Memory: houssem_torch uses 3.76 GB with buckets, because each bucket has its own
  activation buffers. torch.compile peaks at 2.19 GB.
- Why val loss rose: with a fixed 64 pairs per step, a 16-token batch averages its
  loss over about 4× fewer tokens than a 64-token batch. Long sentences get less
  weight per token, and the gradient is noisier (the training loss jumps around
  more). Both engines track each other step for step, so the engine is not the cause.

## Matrix multiplies (`gemm_bench.py`, µs per training step)

| | default | tuned | torch.matmul |
|---|---:|---:|---:|
| total | 58,920 | 60,112 | 57,962 |

- Tuning picked a non-default algorithm twice: 106 for `512x2048x4096 A^T`
  (−8%) and 103 for `4096x1536x512`, which measured +6% slower inside the step.
  The tuned total is 2% *slower* because it ran second, on a hotter GPU. cuBLAS
  does honour the explicit algorithms on sm_75, but its default is already the
  best choice.
- The per-shape torch.matmul numbers are not reliable. The shapes were timed one
  after another while the power-capped clock drifted: early rows look 1.7×
  faster than ours and late rows 1.4× slower. Only the total compares, and it is
  within 2%.

## Next

1. **Token-budget buckets:** more pairs for short buckets, e.g. (256, 16), (128, 32),
   (80, 48), (64, 64). `BucketedTrainStep` already accepts `(batch, src, tgt)`.
   Compare at an equal number of tokens or epochs, not equal steps.
2. **Turn tuning off by default** (`tune_gemms=False`). It costs startup time for
   no gain. Keep it as an option.
3. **Bucket memory:** let the buckets share one activation arena, since only one
   runs at a time. That brings memory back to about 2 GB.
4. **gemm_bench:** time ours and torch's back to back for each shape, so clock drift
   cancels out.
