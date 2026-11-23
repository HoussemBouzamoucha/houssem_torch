# Research notes: beating general-purpose compilers on fixed-config transformer training

*Last updated: 2026-11-21. Literature status is as of that date.*

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

**Order of work:**
1. **Measure.** Profile the target config on a T4 in three modes: PyTorch as
   is, `torch.compile`, and `torch.compile(mode="reduce-overhead")`. Use
   `torch.profiler` or `TORCH_LOGS="output_code"`. The time split (matrix
   multiplies / attention / loss / other / idle between kernels) decides what
   to build first.
2. **GPU basics:** device abstraction, GPU storage, cuBLAS matrix multiplies.
   First hand-written kernel: Cut Cross-Entropy.
3. **Static memory planner**, with remat optimization in the style of
   Checkmate.
4. **Megakernel prototype** for one encoder/decoder layer, then the whole
   step, measured against `torch.compile`. Read gpt2.cu's code and the MPK
   paper in full first.

---

## 6. Caveats
- Most of the 2026 sources are preprints or GitHub repos. Their speedups are
  self-reported on hardware other than a T4.
- §4 is based on search summaries, not full readings of every paper.
- Not confirmed: whether Mirage / MPK support training, and the current state
  of `sac_ilp.py` in PyTorch.
- The FLOP figures in §2.1 and the gains in §2 are back-of-envelope estimates.

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
