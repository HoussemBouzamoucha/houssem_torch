# Research notes: beating general-purpose compilers on fixed-config transformer training

*Last updated: 2026-11-23. Literature status is as of 2026-11-21. First T4
measurements: §7 (they revise the order of work in §5).*

houssem_torch only trains transformers, and every run is fixed by a config
file: the architecture and all tensor shapes are known when the run starts.
These notes collect where that specialization can beat PyTorch (including
`torch.compile`), which research directions apply, and what is still open.

Target hardware: NVIDIA T4 (Turing, sm_75): fp16 and INT8 tensor cores, no
bf16, 16 GB.

---

## 1. What PyTorch already does

From PyTorch's blog post
[Why is PyTorch compile so fast? Kernel fusion](https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/),
`torch.compile` (TorchInductor, generating Triton kernels) fuses:

| Kind | What it does |
|---|---|
| Vertical (pointwise) | chains such as mul → add → sigmoid become one kernel (the post's example: 3 launches → 1, memory operations 8 → 4) |
| Reduction | ops before and after a sum / max / mean join the reduction kernel |
| GEMM epilogue | bias and activation run inside the matrix multiply, right after it |
| GEMM prologue | preprocessing (e.g. normalization) runs as data is loaded into the matrix multiply |
| Horizontal | independent ops that read the same input run together and load it once |

The post doesn't discuss attention. Its "up to 10x faster" has no
measurement behind it. `TORCH_LOGS="output_code"` prints the generated
kernels.

**Where it stops:**
- Matrix multiplies run on cuBLAS / CUTLASS. Inductor fuses small ops around
  one matrix multiply, not chains of two or more.
- It must handle any shape: variable sequence lengths cause recompiles or
  break CUDA graphs (`mode="reduce-overhead"`).
- It plans memory at runtime, with heuristics.

**The honest framing:** houssem_torch won't beat a general compiler in
general, nor cuBLAS on plain matrix multiplies. It can win on one fixed
workload, by using facts a general compiler can't assume.

---

## 2. Opportunities from a fixed config

Ranked by expected payoff for an example target config: d_model 256, 4 + 4
layers, d_ff 512, 32k joint vocabulary, 8192 tokens per batch, on a T4.

### 2.1 Fused output projection + cross-entropy, never storing the logits
- The (8192 × 256) @ (256 × 32k) output projection plus its backward is about
  **400 GFLOP per step**. The rest of the model is about **250 GFLOP**, so the
  output layer and its loss are the largest single cost.
- The logits are 8192 × 32k ≈ 262M values: about 0.5 GB in fp16, before the
  softmax and gradient copies.
- The kernel: work through the vocabulary in tiles, keep running log-sum-exp
  values, apply the exact label-smoothing mode (including `non_target`), and
  write the gradient back directly. Payoff: much less memory traffic, and the
  freed memory goes to bigger batches.
- *Measured (§7), at d_model 512 / 16k vocab:* the head is 12% of a compiled
  step, and most of that is its unavoidable matrix multiplies. The speed gain
  is small there (~2–3%); the memory gain is large. The head's share grows
  as d_model shrinks or the vocabulary grows, as in the example config above.

### 2.2 Attention written for Turing
- The official FlashAttention-2 targets Ampere and newer.
- The config fixes the head size (64), the maximum length (128) and the
  masking pattern (padding only, or causal + padding).
- That allows a FlashAttention-style kernel tuned for sm_75 fp16 tensor
  cores. It would build masks from sequence lengths (no stored mask arrays)
  and fuse in the attention dropout.
- *Update from §4:* Turing-tuned FlashAttention already exists outside
  PyTorch's defaults. Measure what PyTorch actually picks on a T4 first.

### 2.3 Fixed shapes, one CUDA graph per length bucket
- Small models (7–20M parameters) are often limited by kernel-launch
  overhead, not by compute.
- Choose a few length buckets from the config, compile kernels for each, and
  capture the whole training step as one CUDA graph per bucket.
- *Measured (§7):* at d_model 512 and batch 64 × 64 the GPU is idle only 1%
  of the step, and `reduce-overhead` is no faster than `compile`. This only
  pays for much smaller models or batches.

### 2.4 Fusing across two matrix multiplies
- **Feed-forward block:** compute `relu(x W1)` one tile of d_ff at a time and
  add each tile's product with `W2` straight into the output. The (N, d_ff)
  intermediate never reaches GPU memory; backward recomputes it, which is
  cheap at d_ff = 512.
- **Residual + dropout + LayerNorm** (the post-norm block) folded into the
  matrix multiply before it, as a single pass.

### 2.5 Memory planned ahead of time
- Every activation size is known at startup: lay out all GPU memory once, and
  choose per layer what to store and what to recompute for the batch size.

### What won't give an edge
- **Plain matrix multiplies:** use cuBLAS / CUTLASS.
- **Fused Adam:** PyTorch already has `Adam(fused=True)` on CUDA.
- **Multi-GPU gradient all-reduce:** DDP already overlaps it with backward.
- **fp16 tensor-core use:** a T4 has nothing better, and PyTorch uses it the
  same way.

### Expected gains (estimates, not yet measured)
- About **1.3–2×** over PyTorch without `torch.compile` on this config.
- Smaller over `torch.compile`, mostly from §2.1–2.3.

---

## 3. Research directions

### 3.1 Kernel superoptimization
- **What it is:** instead of hand-written fusion rules, search over ways to
  split and fuse the computation across all three GPU levels (whole kernel,
  thread block, thread). Keep only candidates proven equivalent, and keep the
  fastest.
- **Papers:**
  - **Mirage** (Wu et al., OSDI 2025) finds fused kernels nobody had written
    by hand, and reports up to 3.3–3.5× over existing optimizers.
  - Earlier graph-rewriting work: TASO (SOSP 2019), PET (OSDI 2021), Tensat
    (MLSys 2021).
  - Newer work on the search itself: Prism / equality saturation (2026).
- **Why it fits:** search is too expensive for a framework that sees new
  shapes all the time. With one config and a long run, a few minutes of
  search at startup can pay off many times over.

### 3.2 Fusing chains of matrix multiplies
- GEMM → activation → GEMM chains (the feed-forward block, attention), fused
  using the GPU memory hierarchy so intermediates stay on chip.
- **Papers:** Chimera (HPCA 2023), BOLT (MLSys 2022), Welder (OSDI 2023),
  Souffle (ASPLOS 2024), Korch (ASPLOS 2024: picks which kernels to run by
  solving an optimization problem).

### 3.3 Optimal recompute-vs-store planning
- **Checkmate** (MLSys 2020) solves an integer linear program for which
  activations to keep and which to recompute, under a memory budget.
  Limitation: solve time grows quickly with graph size.
- **Successors and alternatives:** Rockmate (ILP per block, combined with
  Rotor), Moccasin (ICML 2023), the Beaumont et al. dynamic program
  (ACM TOMS 2024), DTR (dynamic, no ahead-of-time knowledge), TapeSAC (2026).
- **Why it fits:** fixed shapes let you solve once at startup and lay memory
  out statically.

### 3.4 Cross-entropy without materializing the logits
- **Cut Cross-Entropy** (Wijmans et al., Apple, ICLR 2025): memory
  O(N + |V|) instead of O(N × |V|), and it skips vocabulary tiles whose
  gradient is negligible. This is the research form of §2.1.

### 3.5 Megakernels
- **What it is:** one persistent kernel stays resident and works through the
  whole layer or step itself, instead of hundreds of separate launches.
- **Prior work:** Hazy Research's "No Bubbles" Llama megakernel (2025) and
  Mirage Persistent Kernel (MPK, 2025). Both target inference.
- **Why it fits:** small model + fixed shapes is exactly where launch
  overhead dominates.

### 3.6 INT8 tensor-core training
- **The opening:** a T4 has INT8 tensor cores at about 2× its fp16 peak
  (about 130 vs 65 TOPS). PyTorch doesn't use them for training.
- **Papers:**
  - **SwitchBack** (Wortsman et al., NeurIPS 2023): the forward pass and the
    input gradient in INT8, the weight gradient in 16-bit. Applied to vision
    transformers; more than 0.5% accuracy loss on DeiT-tiny/small in one
    comparison.
  - **Jetfire** (ICML 2024): INT8 data flow with per-block quantization,
    1.42× end-to-end speedup and 1.49× less memory than fp16 on a transformer
    block.
  - **Accurate INT8 Training Through Dynamic Block-Level Fallback** (2025):
    reports lossless INT8 training on Llama-3.1 and Qwen-2.5.
  - **Delta-Matching** (2026): native 8-bit training.
- **The risk:** possible loss of model quality; any claim needs task-metric
  parity (e.g. BLEU).

---

## 4. Literature status (search of 2026-11-21)

### Already taken: not a differentiator
| Direction | Status |
|---|---|
| Fused output layer + cross-entropy (§2.1, §3.4) | PyTorch has an experimental `F.linear_cross_entropy`; `torch.compile` can chunk linear + CE. PyTorch's path is only chunked (each chunk's logits still go to GPU memory), and an open issue reports bf16 loss accumulation in it. torchtune, TorchTitan (Liger, opt-in) and NeMo Automodel also ship variants. **Still worth building for the config; engineering, not research.** |
| Optimal remat planning (§3.3) | PyTorch appears to ship an ILP-based selective activation checkpointing tool (`torch/distributed/_tools/sac_ilp.py`, cited by TapeSAC at v2.13.0; source not checked). Research continues with Rockmate, Moccasin, TapeSAC. **Useful, not novel.** |
| FlashAttention on Turing (§2.2) | Exists outside the official repo: `flash-attention-turing`, and the `flash-attention-triton` package with tuned sm_75 configs (including a backward config). FlashInfer's templates cover sm75–sm90a for inference. Turing-tuned kernels exist; they just aren't PyTorch's default. |

### Still open: fits houssem_torch
| Direction | What exists | Gap |
|---|---|---|
| **Training megakernel for a dense transformer on a commodity GPU** (§3.5) | gpt2.cu (GPT-2 forward, backward and AdamW in one persistent kernel; checked against PyTorch; proof of concept). Training megakernels otherwise target mixture-of-experts on large clusters: UniEP, Cursor's Mixture-of-Kittens, KOSMOS. MPK focuses on inference. ForgeTrain (2026) fuses backward steps with an AI agent. | A dense encoder-decoder training megakernel, specialized from a config, on a T4: appears unexplored. |
| **INT8 training on Turing** (§3.6) | Jetfire, block-level fallback, SwitchBack: all evaluated on newer GPUs. | No evaluation on a T4 or on Turing found. |
| **Superoptimization for training** (§3.1) | Mirage shows training only for one LoRA fine-tuning kernel; Prism is about search methods. | Startup-time search over fused forward + backward training kernels for one fixed config. |

---

## 5. Recommendation

**The strongest original angle combines §3.5 and §3.1:** at startup, compile
the config into megakernels, possibly found by search. The claim to test:
faster training than `torch.compile` on a T4 for a fixed transformer config.
PyTorch can't easily adopt this without giving up on supporting any shape.

INT8 training (§3.6) is a separate, high-risk experiment. It's cheap to try
once a GPU backend exists, and it needs a quality check alongside it.

*Revised 2026-11-23 after the T4 measurements (§7).* The megakernel idea
stands, but its case is now memory traffic, not launch overhead: the GPU is
already 99% busy. The concrete gap is that on a T4 Inductor fuses nothing
into matrix multiplies (§7.3).

**Order of work:**
1. ~~**Measure.**~~ Done 2026-11-23 for `transformer_base`, see §7. Still to
   do: the same run on the actual training config and batch shape.
2. **GPU basics:** device abstraction, GPU storage, cuBLAS matrix multiplies.
3. **First kernel: fused residual + dropout + LayerNorm**, forward and
   backward (~13% of a compiled step). Simple, memory-bound, and the
   building block for step 4.
4. **GEMM epilogue fusion:** bias, ReLU, dropout and the residual add inside
   the matrix multiply (CUTLASS epilogues or own kernels). Main thesis
   candidate: Inductor can't do it on a T4.
5. **Fused attention for short sequences on sm_75** (~10% of a step, backward
   is 2/3 of it). Head size 64 and length ≤ 128 fit one head in shared memory.
6. **Logits-free cross-entropy** (memory) and **AdamW with the GradScaler
   unscale folded in**.
7. **Static memory planner**, then the **megakernel prototype** for one
   encoder/decoder layer, measured against `torch.compile`. Read gpt2.cu's
   code and the MPK paper in full first.

---

## 6. Caveats
- Most of the 2026 sources are preprints or GitHub repos. Their speedups are
  self-reported on hardware other than a T4.
- §4 is based on search summaries, not full readings of every paper.
- Not confirmed: whether Mirage / MPK support training, and the current state
  of `sac_ilp.py` in PyTorch.
- The FLOP figures in §2.1 and the gains in §2 are back-of-envelope estimates.

---

## 7. T4 measurements (2026-11-23)

Script: [`benchmarks/t4_profile.py`](benchmarks/t4_profile.py). Raw output:
[`benchmarks/t4/2026-11-23_colab_t4.md`](benchmarks/t4/2026-11-23_colab_t4.md).
One Colab T4, PyTorch 2.14.1 / CUDA 13.0 / Triton 3.8.0, `transformer_base`
(d_model 512, 6 + 6 layers, d_ff 2048, 16k vocab), batch 64 × 64 source and
target tokens, fp16 + GradScaler, fused AdamW. This measures **PyTorch
only**: houssem_torch has no GPU backend yet.

### 7.1 Step time

| mode | attention `sdpa` | attention `math` |
|---|---:|---:|
| eager | 121.5 ms | 152.8 ms |
| `torch.compile` | **101.3 ms (1.20×)** | 114.7 ms (1.33×) |
| `reduce-overhead` | 102.3 ms (1.19×) | 117.1 ms (1.30×) |

`sdpa` = PyTorch's fused attention (`fmha_cutlass` memory-efficient kernels
on sm_75); `math` = explicit `softmax(QKᵀ)V`, the way houssem_torch computes
it. Compile time on the first step: 41–86 s.

### 7.2 Where the compiled step's time goes (`sdpa`)

| | share of GPU busy time |
|---|---:|
| matrix multiplies (cuBLAS) | 49% |
| residual + dropout + LayerNorm (Inductor-fused) | 13% |
| AdamW + GradScaler | 11% |
| attention (backward ~2/3) | 10% |
| other elementwise, dropout / RNG | 13% |
| loss (log-softmax / NLL) | 4% |
| **GPU idle between kernels** | **1% of the span** |

The profiler inflates kernel times (compiled GPU span 125 ms vs 101 ms
un-profiled), so the shares are reliable and the per-category ms are not.

### 7.3 Findings

1. **Launch overhead is not the problem at this size.** Idle is 1% in every
   mode, and CUDA graphs (`reduce-overhead`) add nothing over `compile`. The
   megakernel case has to rest on memory traffic.
2. **Matrix multiplies are already efficient.** Estimated ~1.3 TFLOP per step
   (≈ 440 GFLOP forward × 3); at ~48 ms of real matmul time that is
   ~27 TFLOPS, a good fraction of what a T4 sustains. Beating cuBLAS on
   plain matrix multiplies is not the goal (§1 still holds).
3. **About half the compiled step is memory-bound work around the matrix
   multiplies.** That is the headroom for houssem_torch.
4. **On a T4, Inductor fuses nothing into matrix multiplies.** Its log says
   `Not enough SMs to use max_autotune_gemm mode`: Triton GEMM templates (and
   with them GEMM epilogue / prologue fusion, §1) need a GPU with more SMs
   than the T4's 40, so every matmul goes to cuBLAS and its output makes a
   round trip through memory to separate bias / dropout / residual / LayerNorm
   kernels. A fixed-config engine can fuse those. *To confirm in the Inductor
   source: the exact SM threshold, and whether default (non-max-autotune)
   mode ever fuses into GEMMs on any GPU.*
5. **The logits-free loss is a memory win first.** Head alone (output
   projection + loss, fwd + bwd): 12.4 ms compiled, 12% of the step. Its
   three matrix multiplies (~206 GFLOP) take ~7.6 ms of that, leaving ~5 ms of
   logits traffic, and Cut Cross-Entropy-style kernels recompute the logits in
   backward (~+2.5 ms). Net speed gain ~2–3%; memory 420 MB → small.
6. **Fused attention is required from day one.** Explicit attention costs 13%
   on the compiled step (114.7 vs 101.3 ms) and 26% in eager.
7. **The optimizer is ~11% per step** and bandwidth-bound (fp32 params, grads,
   moments for 52.5M parameters ≈ 1.5 GB of traffic). Gradient accumulation
   divides that share; folding the GradScaler unscale and inf check into the
   AdamW kernel saves one more pass over the gradients.
8. **Eager PyTorch's extra cost is mostly fp16 casts and unfused elementwise
   ops.** Autocast copies each fp32 weight to fp16 on every use (~200 copy
   kernels per step); compile removes most of them. An engine that keeps an
   fp16 copy of the weights, refreshed once per optimizer step, avoids them
   too.

### 7.4 Still to measure
- The actual training config and batch shape (vocabulary, no attention bias,
  tokens per batch, gradient accumulation).
- Several batch sizes: idle time and the head's share depend on them.
- Repeat runs on Kaggle's T4s to bound the noise (one Colab run so far).

### 7.5 First houssem_torch GPU step on a T4 (2026-11-24)

Raw output: [`benchmarks/t4/2026-11-24_kaggle_t4_trainstep.md`](benchmarks/t4/2026-11-24_kaggle_t4_trainstep.md).
`TrainStep` v1 (compiled whole step, cuBLAS + own fused kernels via CuPy),
same model, batch shape and settings as §7.1, real opus_books batches,
identical initial weights for both engines.

| | houssem_torch | `torch.compile` | eager |
|---|---:|---:|---:|
| ms/step | 101.8 | 100.7 | 120.9 |
| setup / compile | 0.5 s | 42.9 s | — |
| val loss after 3000 steps | 4.764 | 4.790 | — |

1. **Correct and trains identically.** 25 GPU tests pass on the T4; same
   loss from the same weights (10.32002 vs 10.31998); loss curves agree
   within noise over 3000 steps.
2. **Parity with `torch.compile` using explicit attention.** v1 computes
   attention as batched QKᵀ, a softmax kernel and batched PV, which cost
   PyTorch 13% on the compiled step (§7.3.6). So the rest of the step is
   already ahead of Inductor's, and fused attention is the first lever.
3. **Known headroom in v1:** bias gradients as 66 extra tiny matmuls; head
   split/merge copies (84 launches); full fp16 logits written and read back;
   no fusion into the matmuls; optimizer reads/writes ~1.4 GB per step.
4. **Padding is the largest real-world cost.** Real batches fill only
   1471 of 4096 target slots (36%). A plan per length bucket (one compiled
   step per shape, built in ~0.5 s each) fits the startup-specialization
   design; `torch.compile` pays a recompile per new shape.

### 7.6 Where the GPU step's time goes (2026-11-24)

Raw output: [`benchmarks/t4/2026-11-24_kaggle_t4_trainstep_profile.md`](benchmarks/t4/2026-11-24_kaggle_t4_trainstep_profile.md)
(`benchmarks/trainstep_profile.py`, CUDA events around each of the 592
launches; profiled sum 112.4 ms vs 111.4 ms unprofiled, so the GPU is never
idle). That session's T4 was ~10% slower than the benchmark session (111.4
vs 101.8 ms, same shapes and code): compare shares, and always time PyTorch
in the same session.

| work | ms | share | note |
|---|---:|---:|---|
| linear matmuls (cuBLAS) | 45.2 | 40% | 20–26 TFLOP/s, fine |
| **head split / merge copies** | **13.8** | **12%** | pure layout copies |
| **attention batched matmuls** | **10.7** | **10%** | 64×64×64 tiles: 2.7 TFLOP/s |
| optimizer | 9.0 | 8% | AdamW ~170 GB/s |
| residual + dropout + LayerNorm | 7.8 | 7% | ~155 GB/s |
| FFN bias + activation + dropout | 7.4 | 7% | ~150 GB/s |
| output projection matmuls | 7.0 | 6% | 27–32 TFLOP/s |
| cross-entropy | 4.0 | 4% | ~135 GB/s (3 reads of 134 MB logits) |
| bias-gradient matmuls | 3.5 | 3% | 0.1–0.2 TFLOP/s |
| attention softmax | 2.8 | 2% | |

Findings:
1. **Attention as written is the biggest waste: 27 ms (25%)**, as head
   copies 13.8, tiny batched matmuls 10.7 and softmax 2.8. The useful work is
   ~0.2 TFLOP. A fused kernel that reads Q/K/V straight from the projection
   output, keeps one (batch, head) tile in shared memory (64 × 64 fits
   easily) and writes the context back in merged layout removes all three.
   Estimate: 27 → ~8–10 ms.
2. **Our memory-bound kernels run at ~135–170 GB/s, about half the T4's
   320 GB/s.** They use scalar fp16 loads and per-element index math.
   128-bit vectorized loads (8 × fp16) and loading each cross-entropy row
   once into shared memory should give ~1.6–2× on ~27 ms of kernels
   (AdamW, LayerNorm fwd/bwd, bias + activation fwd/bwd, cross-entropy).
   Estimate: save ~10–12 ms.
3. **Bias gradients don't belong in cuBLAS** (0.1 TFLOP/s): the column sums
   can be done inside kernels that already read those gradients (`ln_bwd`
   writes the residual branch's gradient, `act_bwd` the FFN one, the
   attention backward the Q/K/V one). Save ~3 ms.
4. **cuBLAS linear matmuls are fine** at 20–26 TFLOP/s; not a target.
5. **Batch size barely matters:** 25.3k / 26.8k / 28.5k / 26.7k tokens/s at
   batch 32 / 64 / 96 / 128. The step scales linearly, so there is no launch
   overhead to amortize; 96 is marginally best.

Together 1–3 put the step at ~75–85 ms against `torch.compile`'s 100.7 ms,
which matches the §7.5 estimate. Length buckets (§7.5.4) come on top.

---

## Sources

**PyTorch and compilers**
- [Why is PyTorch compile so fast? Kernel fusion (PyTorch blog)](https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/)
- [PyTorch issue #124480: fused `F.linear_cross_entropy`](https://github.com/pytorch/pytorch/issues/124480)
- [PyTorch issue #201659: bf16 loss accumulation in the chunked path](https://github.com/pytorch/pytorch/issues/201659)
- [torchtune: linear cross-entropy PR](https://github.com/meta-pytorch/torchtune/pull/2507)
- [TorchTitan: opt-in Liger fused linear CE](https://github.com/pytorch/torchtitan/pull/5099)
- [NVIDIA NeMo Automodel: fused-loss issue](https://github.com/NVIDIA-NeMo/Automodel/issues/4099)

**Superoptimization**
- [Mirage (OSDI 2025)](https://www.usenix.org/system/files/osdi25-wu-mengdi.pdf)
- [Mirage arXiv abstract](https://arxiv.org/abs/2405.05751)
- [Mirage / MPK repository](https://github.com/guanzheh/mirage)
- [Prism / equality saturation for tensor program superoptimization](https://arxiv.org/pdf/2609.12330)

**Megakernels**
- [gpt2.cu: full GPT-2 training in one persistent kernel](https://github.com/vnnm404/gpt2.cu)
- [UniEP: MoE training megakernel](https://arxiv.org/html/2604.19241)
- [Cursor's Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens)
- [KOSMOS](https://arxiv.org/html/2611.07477v1)
- [ForgeTrain](https://arxiv.org/pdf/2609.13645)
- [MLP megakernel repo](https://github.com/Marcelo5444/mlp-megakernel)
- [Megakernels overview](https://theorempath.com/topics/megakernels)

**Cross-entropy**
- [Cut Cross-Entropy: Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/pdf/2411.09009)

**Attention on Turing**
- [Official FlashAttention repo](https://github.com/dao-ailab/flash-attention)
- [flash-attention-triton (Turing support)](https://pypi.org/project/flash-attention-triton/)
- [FlashInfer](https://arxiv.org/pdf/2501.01005)
- [FlashAttention (original paper)](https://arxiv.org/pdf/2205.14135)

**INT8 training**
- [Jetfire](https://arxiv.org/pdf/2403.12422)
- [Accurate INT8 Training Through Dynamic Block-Level Fallback](https://arxiv.org/pdf/2503.08040)
- [Delta-Matching: native 8-bit training](https://arxiv.org/pdf/2609.37852)

**Memory planning / rematerialization**
- [Checkmate](https://ar5iv.labs.arxiv.org/html/1910.02653)
- [Rockmate](https://arxiv.org/pdf/2307.01236)
- [TapeSAC](https://arxiv.org/pdf/2611.04897)
- [Dynamic Tensor Rematerialization](https://arxiv.org/pdf/2006.09616)
- [PyTorch min-cut recomputation with AOTAutograd](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)
