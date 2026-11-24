# houssem_torch TrainStep vs PyTorch: first Kaggle T4 run (2026-11-24)

Script: [`benchmarks/cuda_train_kaggle.py`](../cuda_train_kaggle.py) at commit
`0af6946` (branch `gpu/driver-layer`). Kaggle, 2 × Tesla T4 (driver 580.178),
PyTorch 2.11.0+cu130. houssem_torch on cuda:0, PyTorch on cuda:1.

Before the benchmark: all 25 GPU tests pass on the T4 (first fp16 run on
tensor cores), and from the same weights the two engines agree on the loss
of one real batch (dropout off): houssem fp16 10.32002, torch fp32 10.31998,
torch fp16 10.32001.

Model: `transformer_base`, vocab 16384 (joint BPE), 52,529,152 parameters.
Batch 64 × 64 source / 64 target tokens, fp16 mixed precision, AdamW
(0.9, 0.98, 1e-9, wd 0.01), clip 1.0, dynamic loss scaling.
Data: opus_books fr→en, 115,790 training pairs that fit 64/64 tokens.

## Speed (median of 3 × 30 steps, real batches, 1471 target tokens/batch)

| engine | ms/step | tokens/s | vs best PyTorch | setup s | memory GB |
|---|---:|---:|---:|---:|---:|
| houssem_torch TrainStep (fp16) | 101.8 | 14451 | 0.99x | 0.5 | 2.26 |
| PyTorch eager (AMP fp16) | 120.9 | 12171 | 0.83x | 0.5 | 2.68 |
| PyTorch compile (AMP fp16) | 100.7 | 14617 | 1.00x | 42.9 | 2.19 |

Inductor again logged `Not enough SMs to use max_autotune_gemm mode`.

## Training, 3000 steps each, same init and batches

| engine | wall s | final train loss | val loss |
|---|---:|---:|---:|
| houssem_torch TrainStep | 335 | 5.0101 | 4.7641 |
| PyTorch compile | 331 | 5.0371 | 4.7896 |

| step | houssem loss | torch loss |
|---:|---:|---:|
| 250 | 7.3968 | 7.3917 |
| 500 | 6.8123 | 6.8126 |
| 750 | 6.5388 | 6.5352 |
| 1000 | 6.2510 | 6.2392 |
| 1250 | 5.9933 | 5.9845 |
| 1500 | 5.5951 | 5.6273 |
| 1750 | 5.6692 | 5.6568 |
| 2000 | 5.3379 | 5.3067 |
| 2250 | 5.1280 | 5.1292 |
| 2500 | 5.1865 | 5.1792 |
| 2750 | 4.9725 | 4.9739 |
| 3000 | 5.0101 | 5.0371 |

The houssem_torch loss scale grew 65536 → 131072 at step 2000 with no
skipped steps. The training wall time includes building each batch in
Python (~10 ms/step for both engines over the benchmark's ms/step).

## Sample translations (houssem_torch weights, greedy, CPU)

- src: Nous fumions, nous blaguions tout le jour...
  - ref: We smoked and joked all the livelong day."
  - hyp: We were hungry, and we were all the day--"
- src: -- Vous n'abusez pas, et votre présence ne gêne en rien mon programme...
  - ref: "You do not intrude, nor do you in the least embarrass my project.
  - hyp: "You don't go out, and your presence is not to be in my programs."
- src: Laurent se donna la tâche de passer chaque matin par la Morgue, en se rendant à son bureau.
  - ref: In the meantime Laurent imposed on himself the task of passing each morning by the Morgue, on the way to his office.
  - hyp: Laurent gave the task of each morning to pass the morning by Many Many.
