# Modules, optimizers and the Transformer model

This page covers the layer above the ops: building blocks (`houssem_torch.nn`),
optimizers (`houssem_torch.optim`) and a ready-made encoder-decoder
Transformer (`houssem_torch.models`). Every class and function also has a
docstring with the details.

None of these needed new backward passes. They're built entirely from ops
whose gradients already exist ([ops/grads.py](../houssem_torch/ops/grads.py)),
so autograd handles them automatically.

---

## `nn.Module`: the base of every layer

A module is an object that owns **parameters** (trainable tensors) and/or
other **modules** (sub-layers). Assigning one as an attribute registers it,
with no extra bookkeeping:

```python
from houssem_torch import nn

class Block(nn.Module):
    def __init__(self, d):
        self.norm = nn.LayerNorm(d)      # sub-module, registered
        self.proj = nn.Linear(d, d)      # sub-module, registered

    def forward(self, x):
        return x + self.proj(self.norm(x))

block = Block(64)
y = block(x)                             # same as block.forward(x)
```

| Method | What it does |
|---|---|
| `forward(...)` | the computation; every module implements it |
| `parameters()` | every parameter, each **once**: pass it to the optimizer |
| `named_parameters()` | `(name, parameter)` pairs, e.g. `"encoder.0.ffn.up.weight"` |
| `num_parameters()` | total trainable values |
| `train()` / `eval()` | switch dropout on/off for the module and all its sub-modules |
| `zero_grad()` | clear every gradient |
| `state_dict()` | `{name: array}` copy of all weights: save with `np.savez(path, **sd)` |
| `load_state_dict(sd)` | copy weights back in place; shapes and names are checked |

**Shared parameters** (the same parameter reachable under two names, like the
transformer's shared embedding) are listed once, under the name where they
were first defined, so the optimizer never updates them twice.

`nn.ModuleList([...])` holds a list of modules, such as the N stacked layers.
Its items get names `0`, `1`, ….

## Building blocks

| Module | Input → output | What it is |
|---|---|---|
| `Linear(in, out)` | `(…, in)` → `(…, out)` | `x @ W + b`; `W` is `(in, out)`, Xavier-initialized; `b` starts at 0 |
| `LayerNorm(d)` | `(…, d)` → `(…, d)` | normalize each vector to mean 0 / variance 1, then learned scale and shift |
| `Embedding(V, d)` | ids `(…)` → `(…, d)` | lookup table, one learned vector per token, initialized `N(0, 1/d)` |
| `Dropout(p)` | same shape | zero a fraction `p` while training (rest scaled by `1/(1-p)`); identity in eval |
| `FeedForward(d, d_ff, dropout, activation)` | `(…, d)` → `(…, d)` | `Linear → relu/gelu → dropout → Linear` |
| `MultiHeadAttention(d, h, dropout, cross)` | `(B, Tq, d)` → `(B, Tq, d)` | see below |

`Linear` flattens all leading dimensions before multiplying, so `(B, T, d)`
runs as one `(B·T, d) @ (d, out)` product. That's faster, and the weight
gradient needs no extra reduction.

### `MultiHeadAttention`

One class, two layouts:

- **Self-attention** (`cross=False`): queries, keys and values all come from
  `x`, through one fused projection `w_qkv (d, 3d)`. Used by the encoder
  (`causal=False`) and the decoder (`causal=True`).
- **Cross-attention** (`cross=True`): queries come from `x` (the target side),
  keys and values from `memory` (the encoder's output for the source), through
  `w_q (d, d)` and
  `w_kv (d, 2d)`.

```python
attn(x, memory=None, key_padding_mask=None, causal=False)
```

- `key_padding_mask`: `(B, Tk)` bool, True where a key is padding. No query
  attends to it.
- `causal`: query `i` sees only keys `0..i`.

Masked scores are set to `-1e9`, not `-inf`, so an all-padding row gives a
uniform distribution rather than NaN. `build_attention_mask` combines both
masks into one array that broadcasts against the `(B, h, Tq, Tk)` scores.

### `clip_grad_norm(parameters, max_norm)`

Measures the combined size (global L2 norm) of all gradients. If it's larger
than `max_norm`, it scales them all down together: the update keeps its
direction but its length is capped. It returns the norm before clipping,
which is worth logging: frequent clipping is an early warning sign.

---

## Optimizers

Every optimizer follows the same three-call rhythm per step:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()        # updates parameters in place
```

| Optimizer | Update rule | Use |
|---|---|---|
| `SGD(params, lr, momentum)` | `v = μ·v + g;  p -= lr·v` | baseline |
| `Adam(params, lr, betas=(0.9, 0.98), eps=1e-9)` | per-parameter adaptive step from running averages of `g` and `g²`, with bias correction | **the paper's optimizer** |
| `AdamW(..., weight_decay=0.01)` | Adam plus decoupled decay `p -= lr·wd·p` | the modern default |

Adam's defaults are the transformer paper's. Each optimizer's
`state_dict()` / `load_state_dict()` saves and restores its learning rate,
step counts and moving averages, so a resumed run continues exactly. All
four optimizers are tested to match PyTorch's step for step.

### Learning-rate schedule

`noam_lr(step, d_model, warmup_steps, factor)` is the paper's schedule:

```
lr = factor · d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)
```

It rises linearly for `warmup_steps` steps, then decays like `1/√step`.
`NoamSchedule(optimizer, d_model, warmup_steps)` applies it: call
`schedule.step()` once per training step, before `optimizer.step()`.

---

## The ready-made Transformer (`houssem_torch.models`)

A complete encoder-decoder (sequence-to-sequence) model, e.g. for
translation. Use it directly, or treat it as a reference and build your own
architecture from the blocks above in your own project: the blocks are where
the optimized kernels live, so a model built from them gets the same speed.

### The config file

All sizes and settings live in one JSON file, loaded into a
`TransformerConfig`. An example with Transformer-base sizes and a 16k
vocabulary is [configs/transformer_base.json](../configs/transformer_base.json)
(52.5M parameters):

```python
from houssem_torch.models import Transformer, TransformerConfig

cfg = TransformerConfig.from_json("configs/transformer_base.json")
model = Transformer(cfg)
```

| Field | Meaning |
|---|---|
| `vocab_size` | shared source+target vocabulary (e.g. 8192 or 16384) |
| `d_model`, `n_heads`, `d_ff` | widths; `d_model` must divide by `n_heads` |
| `n_encoder_layers`, `n_decoder_layers` | stack depths |
| `dropout` | used everywhere; 0 disables it |
| `max_len` | longest sequence supported |
| `activation` | `"relu"` (paper) or `"gelu"` |
| `label_smoothing` | ε for the loss (paper: 0.1) |
| `pad_id`, `bos_id`, `eos_id` | special token ids |

The config is checked when it loads: unknown keys (typos), indivisible
widths and clashing special ids are errors, not silent surprises.

### The layers

**`EncoderLayer`** (pre-norm, with residual connections):
```
x = x + dropout(SelfAttention(LayerNorm(x)))        every word sees every word
x = x + dropout(FeedForward(LayerNorm(x)))
```

**`DecoderLayer`:**
```
y = y + dropout(MaskedSelfAttention(LayerNorm(y)))  sees only earlier target tokens
y = y + dropout(CrossAttention(LayerNorm(y), memory))  looks at the source
y = y + dropout(FeedForward(LayerNorm(y)))
```

The `x + …` residual paths let information and gradients pass straight
through the stack, so each layer learns a refinement. Normalizing *before*
each block ("pre-norm") trains more stably from scratch than the paper's
original "post-norm".

**`sinusoidal_position_encoding(max_len, d)`** is the paper's fixed sine and
cosine position signal, added to the embeddings so attention can tell word
order.

### `Transformer`

| Method | Does |
|---|---|
| `loss(src, tgt_in, tgt_out)` | the training loss: mean label-smoothed cross-entropy, padding ignored |
| `forward(src, tgt_in)` / `model(src, tgt_in)` | logits `(B, T, V)` for the whole target sequence at once (teacher forcing) |
| `encode(src)` | source ids → `memory (B, S, d)` and the padding mask |
| `decode(tgt_in, memory, src_pad)` | target prefix + memory → logits |
| `translate_greedy(src, max_new_tokens)` | inference: repeatedly pick the most likely next token until `<eos>` |
| `embed(ids)` | embedding × √d + position encoding, then dropout |
| `pad_mask(ids)` | `(B, L)` bool, True at padding |

Token inputs are integer NumPy arrays:
- `src (B, S)`: source ids, padded with `pad_id`.
- `tgt_in (B, T)`: target ids **shifted right**: `<bos> w1 w2 …`.
- `tgt_out (B, T)`: target labels: `w1 w2 … <eos>`.

One embedding table serves the encoder input, the decoder input **and** the
output projection (`logits = h @ Eᵀ`). It's a single parameter; its gradient
is the sum from all three uses.

`translate_greedy` is deliberately simple: it re-runs the decoder over the
whole prefix at every step. Key/value caching and beam search belong to the
fast engine.

### A training step

```python
import houssem_torch as ht
from houssem_torch import optim

ht.manual_seed(0)
model = Transformer(cfg)
opt = optim.Adam(model.parameters())
schedule = optim.NoamSchedule(opt, d_model=cfg.d_model, warmup_steps=4000)

for src, tgt_in, tgt_out in batches:
    model.train()
    opt.zero_grad()
    loss = model.loss(src, tgt_in, tgt_out)
    loss.backward()
    ht.nn.clip_grad_norm(model.parameters(), max_norm=1.0)
    schedule.step()
    opt.step()
```

[examples/train_toy_reverse.py](../examples/train_toy_reverse.py) runs this
loop on a toy task: reverse a sequence of tokens. It uses variable lengths
with padding, both masks, cross-attention, label smoothing and warmup, and
reaches about 99% exact match in roughly two minutes on CPU.

---

## How it's verified

| Test file | Checks |
|---|---|
| `tests/test_nn.py` | parameter naming and de-duplication, train/eval, state dicts; `Linear` and every attention mode against PyTorch; SGD, Adam, AdamW and clipping against PyTorch; optimizer resume |
| `tests/test_transformer_model.py` | config validation; **the full model's loss and all 69 parameter gradients match an independent PyTorch implementation**; the tiny model learns to reverse sequences (>90% exact match) |
