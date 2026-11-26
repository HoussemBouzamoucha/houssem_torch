# houssem_torch GPU engine: summary

*As of 2026-11-26, after that day's T4 run. Hardware: Kaggle Tesla T4. Model: transformer_base (52.5M parameters), fp16, batch 64 × 64/64 tokens.*

## What was built

houssem_torch is a standalone library for training Transformers. Its GPU engine
does not use PyTorch: it calls NVIDIA's libraries (cuBLAS) for plain matrix
multiplies and uses our own CUDA kernels for everything else.

## Key decisions and their outcome

| Decision | Why | Outcome |
|---|---|---|
| **Compiled training step** (`TrainStep`): the whole step is built once for a fixed batch shape. Memory, kernels and launch order are all set at startup. | No per-operation dispatch, and no wait to compile at startup. | Startup **0.5 s** vs **~30 s** for torch.compile. Same loss curves as PyTorch. |
| **Thin CuPy driver** (`driver.py`), the only module that touches the GPU. | Keeps CuPy swappable and out of the rest of the code. | Worked as intended. No CuPy types leak out. |
| **Fuse the memory-bound work** (LayerNorm + residual + dropout, bias + activation, loss, AdamW), with dropout masks regenerated from a hash instead of stored. | That work is limited by memory speed, so combining steps saves reading and writing memory. | We beat torch.compile in every one of these categories. For example, LayerNorm + residual takes 5.4 vs 12.2 ms/step. |
| **Our own fused attention kernels** (tensor cores via inline PTX, with the attention matrix never stored in memory). | Attention was the only category where we lost (16.8 vs 10.2 ms). | Backward kernel 873 → 279 µs. Attention now **beats PyTorch's**: 8.9 vs 9.6 ms/step. |
| **Async batch uploads** from pinned memory. | Before, each step waited for the previous step to finish. | Training time now matches the step benchmark. Host-side cost is 0.4 ms/step. |
| **Always time both engines on the same card** (`HT_GPU=1 TORCH_GPU=1`). | Kaggle's cuda:0 runs hotter and slower, which skewed earlier comparisons. | Fair result: **1.20–1.22× faster than torch.compile** per step, and **1.21×** on total training time (263 vs 317 s for 3000 steps). |
| **Run all GPU work on Kaggle**, nothing on the local GPU. | The local GTX 1650 has no tensor cores, so its timings don't predict the T4. | Every number comes from the T4. All runs are logged in `benchmarks/t4/history.json`. |

## Latest (branch `gpu/gemm-tuning-buckets`, Kaggle T4 run 2026-11-26)

| Decision | Why | Outcome |
|---|---|---|
| **Length buckets** (`BucketedTrainStep`): each batch runs at the smallest of the 16/32/48/64 widths that fits it, with every bucket sharing one set of weights. | About 62% of each 64/64 batch was padding. | First run, 64 pairs per batch: 1.74× faster, but val loss worse (5.02 vs 4.75) and memory 3.76 GB. |
| **Token-budget buckets** (more pairs in short buckets) **and one shared activation block** | Fix the val loss and the memory. | 3000 steps in **149 s instead of 271 s (1.82×)**. **1.29×** faster than torch.compile on the same buckets. Memory **2.03 GB** (= fixed step). Val loss gap down to +0.09 (4.87 vs 4.78), the same for both engines. |
| **cuBLAS algorithm tuning at startup** | Matmuls were about 10% slower than PyTorch's. | **No gain.** The default algorithm was the fastest for 16 of 18 shapes. The matmul gap is now about 3% (54.2 vs 52.6 ms), so algorithm choice was never the issue. |

Fixed 64/64 in the same run: **84.0 vs 106.3 ms/step (1.27×)**, training 268 vs 325 s.
Full write-up: `benchmarks/t4/2026-11-26_kaggle_t4_gemm_tuning_buckets.md`.

Tuning is now off by default.

**Batch-size scaling** (token-budget buckets, 6000 steps, ×1/×2/×4 the tokens of a 64-pair batch;
`benchmarks/t4/2026-11-26_kaggle_t4_batch_scaling.md`):

| scale | ms/step ht / torch | steady speedup | total time ht / torch | val ht / torch | memory GB ht / torch |
|---:|---:|---:|---:|---:|---:|
| ×1 | 49.5 / 63.6 | 1.28× | 299 / 389 s | 4.118 / 4.129 | 2.03 / 1.62 |
| ×2 | 91.6 / 111.8 | 1.22× | 550 / 796 s | 3.640 / 3.649 | 2.25 / 2.50 |
| ×4 | 173.7 / 208.0 | 1.20× | 1044 / 1376 s | 3.601 / 3.607 | 3.61 / 4.24 |

houssem_torch stays faster at every size, but the steady gain levels off at about 1.2×,
because cuBLAS matmuls (the same in both engines) dominate big batches. The two
engines' val losses match at every size. ×4 overfits opus_books after step 5000 (~13 passes over the data).

**Next:** faster matmuls (the only lever left at big batches), a larger
dataset for long ×4 runs, gradient accumulation.

## Known limits

- The fused attention kernels only handle sequences up to about 64 tokens. Longer sequences and fp32 fall back to the unfused path.
- Only pre-norm is supported; post-norm isn't written.
- No gradient accumulation yet: one update per batch, so the batch size is limited by GPU memory.
