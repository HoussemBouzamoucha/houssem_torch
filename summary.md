# houssem_torch GPU engine: summary

*As of 2026-11-27, after that day's T4 runs. Hardware: Kaggle Tesla T4. Model: transformer_base (52.5M parameters), fp16, batch 64 × 64/64 tokens.*

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

## Latest (branch `gpu/faster-gemms`, Kaggle T4 runs 2026-11-27)

| Decision | Why | Outcome |
|---|---|---|
| **Time each matmul in isolation**, ours and torch.matmul back to back in the same round (`gemm_bench.py`). | The earlier "matmuls ~10% slower" came from timing the two engines at different moments. | **Our matmuls match PyTorch's** (58.3 vs 56.9 ms/step at batch 64). The only difference is that we write weight gradients in fp32 and accumulate them in place. Storing weights as (out, in) like `nn.Linear` gains nothing. The speedup levels off at big batches because both engines run the same cuBLAS kernels, which take a growing share of the step. Our matmuls are not slower. |
| **cuBLASLt epilogues** (ctypes binding, since CuPy has none). The FFN backward's ReLU + dropout mask + bias gradient are fused into the multiply, and a 1-bit mask is written by the forward kernel. | Remove the separate `act_bwd` pass over the M×d_ff gradient. | **Correct, but slower on the T4.** On Turing the epilogue kernels run 1.35–2.5× slower than plain cuBLAS, even after timing 8 algorithms per shape, so the step is 2–4% slower. Off by default (`fused_epilogues=False`). Kept for newer GPUs. |
| **Gradient accumulation** (`accumulate_step(micro_batches, lr)`). Every micro-batch's loss is divided by the target tokens of all of them, followed by one update. Works across buckets. | Batch size was limited by GPU memory. | 4×64 gives the same update as 1×256 (66/66 GPU tests, checked against one big batch) with **2.18 vs 5.88 GB** for **+6% time** (293 vs 275 ms/step). |

Write-ups: `benchmarks/t4/2026-11-27_kaggle_t4_gemm_layouts.md`, `…_lt_epilogues.md`,
`…_grad_accumulation.md`. On the T4, the engine's speed work is essentially done.

## Training quality (branch `train/quality`, Kaggle T4 run 2026-11-28)

opus-100 fr->en (933k pairs), 10000 optimizer steps, BLEU on the 2000 test
sentences with our own GPU decoder (KV-cache greedy and beam search):

| run | tokens / step | val loss | BLEU greedy | BLEU beam 4 | wall s | memory GB |
|---|---:|---:|---:|---:|---:|---:|
| x4, Noam factor 1 | 5,872 | 3.318 | 26.62 | 27.18 | 2112 | 5.61 |
| x4, Noam factor 2 | 5,872 | 3.314 | 26.18 | 26.97 | 2108 | 5.61 |
| x8 (2 x x4 accumulated), factor 2.83 | 11,745 | 3.170 | 28.07 | 28.87 | 4139 | 5.61 |

- Doubling the LR at x4 changes nothing, so the default Noam factor stays 1.0.
- x8 is the best model, but only because it sees 2x the tokens. At equal
  compute (x8 at step 5000) it is slightly behind x4 (BLEU 25.72 vs 26.62).
- No run has converged yet. Beam 4 adds +0.6 to +0.8 BLEU.

Write-up: `benchmarks/t4/2026-11-28_kaggle_t4_opus100_quality.md`.

**Next:** CUDA Graphs for the TrainStep.

## Known limits

- The fused attention kernels only handle sequences up to about 64 tokens. Longer sequences and fp32 fall back to the unfused path.
- Only pre-norm is supported; post-norm isn't written.
- cuBLASLt epilogues are slower than plain cuBLAS on Turing, so they're off. Newer GPUs may change that.
