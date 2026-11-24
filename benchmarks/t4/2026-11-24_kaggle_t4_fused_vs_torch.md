# houssem_torch TrainStep (fused attention) vs PyTorch, Kaggle T4 (2026-11-24)

Script: [`benchmarks/cuda_train_kaggle.py`](../cuda_train_kaggle.py) at commit
`839ffd6`. Kaggle, 2 × Tesla T4 (driver 580.178), PyTorch 2.11.0+cu130;
houssem_torch on cuda:0, PyTorch on cuda:1, same session. 32/32 GPU tests
pass on the T4. Same initial weights for both engines; eval loss from them:
houssem fp16 10.32006, torch fp32 10.31998, torch fp16 10.32001.

All three attention kinds ran fused (encoder self and cross attention share
the (64, 64, non-causal) kernel; the run printed "2/3 shapes", a message
bug fixed afterwards).

## Scoreboard

```
houssem_torch vs PyTorch: Tesla T4, transformer_base, batch 64, fp16
Speed      houssem_torch   84.1 ms/step | torch.compile  101.7 | eager  123.1
           -> houssem_torch is 1.21x torch.compile's speed and 1.46x eager's
Startup    houssem_torch   0.5 s | torch.compile  34.5 s (compiling)
Memory     houssem_torch 2.03 GB | torch.compile 2.19 GB
Training   3000 steps, same data: houssem_torch 276 s, val loss 4.771 |
           PyTorch compile 335 s, val loss 4.781
```

## Speed (median of 3 × 30 steps, real batches, 1471 target tokens/batch)

| engine | ms/step | tokens/s | time vs houssem_torch | setup s | memory GB |
|---|---:|---:|---:|---:|---:|
| houssem_torch | 84.1 | 17493 | 1.00x | 0.5 | 2.03 |
| torch.compile | 101.7 | 14465 | 1.21x | 34.5 | 2.19 |
| PyTorch eager | 123.1 | 11957 | 1.46x | 0.2 | 2.68 |

| batch | houssem_torch | torch.compile | PyTorch eager | speedup vs torch.compile |
|---:|---:|---:|---:|---:|
| 32 | 45.8 | 56.5 | 68.5 | 1.24x |
| 64 | 84.1 | 101.7 | 123.1 | 1.21x |
| 96 | 121.3 | 160.5 | 183.8 | 1.32x |

## Where the time goes (ms/step, scaled to each engine's step time)

PyTorch kernels are bucketed by name (Triton fused kernels by the ops in
their names), so its split is approximate; totals are measured.

| work | houssem_torch | torch.compile | PyTorch eager |
|---|---:|---:|---:|
| matrix multiplies | 46.7 | 50.9 | 48.1 |
| attention | 16.8 | 10.2 | 17.4 |
| LayerNorm + residual | 5.3 | 11.5 | 5.4 |
| activation, dropout, bias | 5.4 | 12.7 | 42.4 |
| loss | 1.1 | 4.6 | 0.0 |
| optimizer | 7.6 | 11.5 | 9.4 |
| embedding, other | 1.1 | 0.4 | 0.4 |

## Training, 3000 steps each, same init and batches

| engine | wall s | final train loss | val loss |
|---|---:|---:|---:|
| houssem_torch TrainStep | 276 | 5.0159 | 4.7712 |
| PyTorch compile | 335 | 5.0433 | 4.7815 |

| step | houssem loss | torch loss |
|---:|---:|---:|
| 250 | 7.3970 | 7.3922 |
| 500 | 6.8128 | 6.8141 |
| 750 | 6.5399 | 6.5327 |
| 1000 | 6.2478 | 6.2477 |
| 1250 | 5.9917 | 5.9844 |
| 1500 | 5.6067 | 5.5951 |
| 1750 | 5.6437 | 5.6409 |
| 2000 | 5.3405 | 5.3193 |
| 2250 | 5.1142 | 5.1248 |
| 2500 | 5.1869 | 5.2107 |
| 2750 | 4.9731 | 4.9936 |
| 3000 | 5.0159 | 5.0433 |

Translations after 3000 steps (~1.7 epochs) remain undertrained for both
engines (repetition loops); not an engine difference.
