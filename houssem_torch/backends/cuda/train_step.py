"""The compiled training step: one whole Transformer training step on a GPU.

Built once from a model and a fixed batch shape, then called every step:

    model = Transformer(cfg)                        # CPU model: config + weights
    step = TrainStep(model, batch_size=64, src_len=64, tgt_len=64)
    for src, tgt_in, tgt_out in batches:            # int arrays (B,S), (B,T), (B,T)
        step(src, tgt_in, tgt_out, lr=schedule(i))  # forward, backward, AdamW
    print(step.last_loss())
    model.load_state_dict(step.state_dict())        # weights back on the CPU

Everything is decided at startup ("specialize at startup"):
  * Memory: every buffer is allocated once. All parameters live in one
    flat fp32 buffer (master weights) plus a flat working copy in the
    compute precision; gradients and AdamW moments are flat fp32 buffers
    too, so zeroing, the gradient norm and the optimizer are one launch each.
  * Kernels: compiled with the run's sizes baked in (d_model, sequence
    lengths, activation, causal or not).
  * Order: a fixed sequence of cuBLAS multiplies and our fused kernels.
    Backward is written per layer from the formulas in ops/grads.py.

Fused kernels (see kernels.cu): embedding + LayerNorm; residual + dropout +
LayerNorm (forward) and LayerNorm backward + residual + the previous
dropout (backward); bias + activation + dropout; masked softmax + dropout;
cross-entropy forward + backward in one pass; gradient norm, loss scaling,
clipping and AdamW. Dropout masks are regenerated in backward from a hash,
never stored.

Mixed precision (precision="fp16"): activations and working weights in
fp16, all statistics and gradient accumulation in fp32, fp32 master
weights, dynamic loss scaling decided on the GPU (no host sync per step).

Matches houssem_torch's CPU Transformer (pre-norm) exactly up to float
rounding: same loss, same gradients, same AdamW update. Dropout uses the
GPU's own random masks, so with dropout > 0 individual steps differ from
the CPU while following the same distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np

from houssem_torch.backends.cuda import driver as drv
from houssem_torch.models.transformer import (Transformer,
                                              sinusoidal_position_encoding)

_SOURCE = Path(__file__).with_name("kernels.cu").read_text(encoding="utf-8")
_THREADS = 256
_ALIGN = 64          # parameter offsets, in elements (128 B in fp16)
_LN_EPS = 1e-5       # houssem_torch.nn.LayerNorm default
_ATT_WARPS = 4       # warps per fused-attention forward block
_ATT_BWD_WARPS = 16  # warps per fused-attention backward block (T4: 16 > 8 > 4)
_ATT_QREG = 0        # 1: forward reads Q into registers (less smem; T4: ~5 us slower)
_SMEM_LIMIT = 65536  # dynamic + static shared memory per block on sm_75
_CE_THREADS = 512    # register-resident cross-entropy block size


def _vec_width(n, half, limit=None):
    """Widest vector (16 bytes max) whose element count divides n."""
    for w in ((8, 4, 2, 1) if half else (4, 2, 1)):
        if n % w == 0 and (limit is None or w <= limit):
            return w
    return 1


def _pad16(n):
    return -(-n // 16) * 16


def attn_smem(lq, lk, dk, qreg=_ATT_QREG):
    """(forward, backward) dynamic shared bytes of the fused attention
    kernels, and the backward's static shared bytes."""
    qp, kp = _pad16(lq), _pad16(lk)
    lds, ldp = dk + 8, kp + 8
    fwd = 2 * ((0 if qreg else qp) + 2 * kp) * lds
    bwd = 2 * ((2 * qp + 2 * kp) * lds + 2 * qp * ldp)
    static = 4 * kp + 4 * qp + 12 * dk
    return fwd, bwd, static


@dataclass
class _Sub:
    """One residual sublayer: x = x + dropout(f(LayerNorm(x)) + bias)."""
    kind: str               # "self", "cross" or "ffn"
    prefix: str             # parameter prefix, e.g. "decoder.1.cross_attn"
    norm: str               # its LayerNorm, e.g. "decoder.1.norm_cross"
    causal: bool = False
    out_bias: str | None = None
    bufs: dict = field(default_factory=dict)   # saved activations


@dataclass
class _Stack:
    name: str               # "enc" or "dec"
    L: int                  # sequence length
    M: int                  # rows = batch · L
    ids: drv.Buffer         # (B, L) int32 token ids
    subs: list
    final: str              # final LayerNorm
    x: list = field(default_factory=list)      # residual stream per site
    h: list = field(default_factory=list)      # LayerNorm outputs per site
    mean: list = field(default_factory=list)
    rstd: list = field(default_factory=list)


class TrainStep:
    """One compiled forward + backward + AdamW step for a fixed batch shape.

    model:        houssem_torch Transformer (pre-norm) giving config and the
                  starting weights.
    batch_size, src_len, tgt_len: the fixed batch shape (pad batches to it).
    precision:    "fp16" (mixed precision, fp32 master weights) or "fp32".
    betas, eps, weight_decay: AdamW, same meaning and defaults as
                  houssem_torch.optim.AdamW.
    max_grad_norm: clip the global gradient norm (None: no clipping), like
                  houssem_torch.nn.clip_grad_norm.
    seed:         dropout randomness.
    init_scale, growth_interval: dynamic loss scaling (fp16 only).
    fused_attention: use the fused tensor-core attention kernels where the
                  shape allows (fp16, d_k divisible by 8, sequences up to
                  ~64 tokens); False forces the unfused path everywhere.
    """

    def __init__(self, model: Transformer, batch_size: int, src_len: int,
                 tgt_len: int, *, precision: str = "fp16",
                 betas=(0.9, 0.98), eps: float = 1e-9,
                 weight_decay: float = 0.01, max_grad_norm: float | None = None,
                 seed: int = 0, init_scale: float = 2.0 ** 16,
                 growth_interval: int = 2000, device: int = 0,
                 fused_attention: bool = True):
        if not drv.is_available():
            raise RuntimeError("TrainStep needs a CUDA GPU and CuPy.")
        if precision not in ("fp16", "fp32"):
            raise ValueError(f"precision must be 'fp16' or 'fp32', got {precision!r}")
        cfg = model.cfg
        if cfg.norm_position != "pre":
            raise NotImplementedError("the GPU step supports norm_position='pre' "
                                      "only (post-norm is not written yet)")
        if cfg.d_model % 32:
            raise ValueError("the GPU step needs d_model divisible by 32")
        if max(src_len, tgt_len) > cfg.max_len:
            raise ValueError(f"sequence length exceeds max_len {cfg.max_len}")
        drv.set_device(device)
        self.cfg = cfg
        self.B, self.S, self.T = batch_size, src_len, tgt_len
        self.precision = precision
        self.half = precision == "fp16"
        self.dtype = np.float16 if self.half else np.float32
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.adam_eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = max_grad_norm
        self.growth_interval = growth_interval
        self.fused_attention = bool(fused_attention)
        self.d = cfg.d_model
        self.H = cfg.n_heads
        self.dk = cfg.d_model // cfg.n_heads
        self.emb_scale = float(np.sqrt(cfg.d_model))
        self._rng = np.random.default_rng(seed)
        self._bufs: dict = {}
        info = drv.device_info(device)
        self._sms = info.multiprocessors

        self._build_parameters(model)
        self._compile()
        self._build_plan()

        self.scalars = drv.zeros(10, np.float32)
        s = np.zeros(10, np.float32)
        s[0] = init_scale if self.half else 1.0
        self.scalars.copy_from(s)
        self._count = 1

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_parameters(self, model):
        self.names, self.shapes, self.offsets = [], {}, {}
        n = 0
        for name, p in model.named_parameters():
            self.names.append(name)
            self.shapes[name] = p.shape
            self.offsets[name] = n
            n += -(-int(np.prod(p.shape)) // _ALIGN) * _ALIGN
        self.n_params = n
        host = np.zeros(n, np.float32)
        for name, p in model.named_parameters():
            o = self.offsets[name]
            host[o:o + p.numpy().size] = p.numpy().reshape(-1)
        self.master = drv.from_numpy(host)
        self.work = drv.from_numpy(host, self.dtype) if self.half else self.master
        self.grad = drv.zeros(n, np.float32)
        self.m = drv.zeros(n, np.float32)
        self.v = drv.zeros(n, np.float32)
        self.W = {k: self._param_view(self.work, k) for k in self.names}
        self.G = {k: self._param_view(self.grad, k) for k in self.names}

    def _param_view(self, flat, name):
        o, shape = self.offsets[name], self.shapes[name]
        return flat[o:o + int(np.prod(shape))].view(*shape)

    def _kernel(self, name, **defines):
        return drv.compile_kernel(_SOURCE, name,
                                  {"USE_HALF": int(self.half), **defines})

    def _compile(self):
        cfg, d, half = self.cfg, self.d, self.half
        self.ln_warps = max(1, min(8, 6144 // d))
        vw = _vec_width(d // 32, half)
        evw = _vec_width(cfg.d_ff, half)
        if cfg.d_ff // evw > 1024:
            raise ValueError(f"d_ff {cfg.d_ff} too wide for the FFN kernels")
        act = {"relu": 0, "gelu": 1}[cfg.activation]
        self.k_embed_ln = self._kernel("add_ln_fwd", D=d, VW=vw, MODE=0)
        self.k_res_ln = self._kernel("add_ln_fwd", D=d, VW=vw, MODE=1)
        self.k_ln_bwd = self._kernel("ln_bwd", D=d, VW=vw, LN_WARPS=self.ln_warps)
        self.k_embed_bwd = self._kernel("embed_bwd")
        self.evw = evw
        self.k_bias_act = self._kernel("bias_act_fwd", ACT=act, EVW=evw)
        self.k_act_bwd = self._kernel("act_bwd", ACT=act, EVW=evw)
        self.k_split = self._kernel("split_heads")
        self.k_merge = self._kernel("merge_heads")
        # Attention: fused kernels per (queries, keys, causal) where they fit,
        # the unfused path (cuBLAS + softmax kernels) elsewhere.
        self.k_softmax, self.k_softmax_bwd, self.k_attn = {}, {}, {}
        for lq, lk, causal in {(self.S, self.S, 0), (self.T, self.T, 1),
                               (self.T, self.S, 0)}:
            fused = self._fused_attention_plan(lq, lk)
            if fused is not None:
                fwd_smem, bwd_smem = fused
                defs = dict(ATT=1, AQ=lq, AK=lk, AQP=_pad16(lq), AKP=_pad16(lk),
                            DK=self.dk, CAUSAL=causal, NW=_ATT_WARPS,
                            NWB=_ATT_BWD_WARPS, QREG=_ATT_QREG)
                kf = self._kernel("attn_fwd", **defs)
                kb = self._kernel("attn_bwd", **defs)
                kf.set_max_dynamic_shared(fwd_smem)
                kb.set_max_dynamic_shared(bwd_smem)
                self.k_attn[lq, lk, causal] = (kf, kb, fwd_smem, bwd_smem)
            else:
                self.k_softmax[lk, causal] = self._kernel(
                    "softmax_fwd", LK=lk, CAUSAL=causal)
                self.k_softmax_bwd[lk, causal] = self._kernel(
                    "softmax_bwd", LK=lk, CAUSAL=causal)
        V = cfg.vocab_size
        ce_vw = _vec_width(V, half)
        if -(-V // (_CE_THREADS * ce_vw)) * ce_vw <= 64:
            self.k_ce_reg = self._kernel("cross_entropy_reg", CE_V=V,
                                         CE_T=_CE_THREADS, CE_VW=ce_vw)
        else:
            self.k_ce_reg = None
        self.k_ce = self._kernel("cross_entropy")
        self.k_reset = self._kernel("reset_scalars")
        self.k_sumsq = self._kernel("sum_squares")
        self.k_prepare = self._kernel("step_prepare")
        self.k_adamw = self._kernel("adamw")

    def _fused_attention_plan(self, lq, lk):
        """(forward, backward) dynamic shared bytes of the fused attention
        kernels for this shape, or None if they don't apply."""
        if not (self.fused_attention and self.half and self.dk % 8 == 0):
            return None
        fwd, bwd, static = attn_smem(lq, lk, self.dk)
        if bwd + static > _SMEM_LIMIT:
            return None
        return fwd, bwd

    def _attn(self, sub, st):
        """Fused kernels for this attention sublayer, or None."""
        if sub.kind not in ("self", "cross"):
            return None
        lk = self.S if sub.kind == "cross" else st.L
        return self.k_attn.get((st.L, lk, int(sub.causal)))

    def _buf(self, key, shape, dtype=None):
        """A buffer allocated on first use and reused every step."""
        dtype = self.dtype if dtype is None else dtype
        full = (key, tuple(shape), np.dtype(dtype).name)
        b = self._bufs.get(full)
        if b is None:
            b = self._bufs[full] = drv.zeros(shape, dtype)
        return b

    def _build_plan(self):
        cfg, B, d = self.cfg, self.B, self.d
        bias = cfg.attention_bias
        self.src_ids = drv.zeros((B, self.S), np.int32)
        self.tgt_in_ids = drv.zeros((B, self.T), np.int32)
        self.tgt_out_ids = drv.zeros((B, self.T), np.int32)
        self.pe = drv.from_numpy(sinusoidal_position_encoding(cfg.max_len, d))
        ones = np.ones(B * max(self.S, self.T), self.dtype)
        self.ones = drv.from_numpy(ones)

        enc_subs, dec_subs = [], []
        for i in range(cfg.n_encoder_layers):
            p = f"encoder.{i}"
            enc_subs.append(_Sub("self", f"{p}.self_attn", f"{p}.norm_attn",
                                 out_bias=f"{p}.self_attn.w_o.bias" if bias else None))
            enc_subs.append(_Sub("ffn", f"{p}.ffn", f"{p}.norm_ffn",
                                 out_bias=f"{p}.ffn.down.bias"))
        for i in range(cfg.n_decoder_layers):
            p = f"decoder.{i}"
            dec_subs.append(_Sub("self", f"{p}.self_attn", f"{p}.norm_self",
                                 causal=True,
                                 out_bias=f"{p}.self_attn.w_o.bias" if bias else None))
            dec_subs.append(_Sub("cross", f"{p}.cross_attn", f"{p}.norm_cross",
                                 out_bias=f"{p}.cross_attn.w_o.bias" if bias else None))
            dec_subs.append(_Sub("ffn", f"{p}.ffn", f"{p}.norm_ffn",
                                 out_bias=f"{p}.ffn.down.bias"))
        self.enc = _Stack("enc", self.S, B * self.S, self.src_ids, enc_subs,
                          "encoder_norm")
        self.dec = _Stack("dec", self.T, B * self.T, self.tgt_in_ids, dec_subs,
                          "decoder_norm")

        # Dropout sites, each with a fresh seed every step.
        self._n_seeds = 0

        def slot():
            self._n_seeds += 1
            return self._n_seeds - 1

        for st in (self.enc, self.dec):
            st.emb_seed = slot()
            for sub in st.subs:
                sub.res_seed, sub.inner_seed = slot(), slot()
            # Saved activations: one residual-stream site per LayerNorm.
            for _ in range(len(st.subs) + 1):
                st.x.append(drv.zeros((st.M, d), self.dtype))
                st.h.append(drv.zeros((st.M, d), self.dtype))
                st.mean.append(drv.zeros(st.M, np.float32))
                st.rstd.append(drv.zeros(st.M, np.float32))
            for sub in st.subs:
                self._alloc_sub(st, sub)
        self.logits = drv.zeros((self.dec.M, cfg.vocab_size), self.dtype)
        self._seeds = np.zeros(self._n_seeds, np.uint32)

    def _alloc_sub(self, st, sub):
        B, H, dk, d = self.B, self.H, self.dk, self.d
        BH, L = B * H, st.L
        z = lambda *shape: drv.zeros(shape, self.dtype)  # noqa: E731
        if self._attn(sub, st) is not None:
            # fused: keep the raw projections, the context and one
            # log-sum-exp per query row; nothing of size L x L.
            if sub.kind == "self":
                sub.bufs.update(qkv=z(st.M, 3 * d))
            else:
                sub.bufs.update(q=z(st.M, d), kv=z(self.B * self.S, 2 * d))
            sub.bufs.update(ctx=z(st.M, d),
                            lse=drv.zeros(2 * B * H * L, np.float32))
        elif sub.kind == "self":
            sub.bufs.update(heads=z(3, BH, L, dk), P=z(BH, L, L), Pd=z(BH, L, L),
                            ctx=z(st.M, d))
        elif sub.kind == "cross":
            S = self.S
            sub.bufs.update(qh=z(1, BH, L, dk), kvh=z(2, BH, S, dk),
                            P=z(BH, L, S), Pd=z(BH, L, S), ctx=z(st.M, d))
        else:
            dff = self.cfg.d_ff
            sub.bufs.update(u=z(st.M, dff), a=z(st.M, dff))

    # ------------------------------------------------------------------
    # Launch helpers
    # ------------------------------------------------------------------

    def _grid(self, n):
        return max(1, min(-(-n // _THREADS), self._sms * 32))

    def _warp_rows(self, rows):
        return -(-rows * 32 // _THREADS)

    def _ffn_grid(self, rows, cols):
        """(blocks, threads, rows per block) for bias_act_fwd / act_bwd."""
        rpb = max(1, -(-rows // (self._sms * 2)))
        return -(-rows // rpb), cols // self.evw, rpb

    def _colsum(self, g, out, rows):
        """out (fp32, (n,)) += sum of g's rows, as a cuBLAS ones-vector multiply."""
        drv.gemm(self.ones[:rows].view(1, rows), g, out.view(1, out.shape[0]),
                 beta=1.0)

    def _split(self, src, bias, dst, L, parts):
        n = self.B * L * parts * self.d
        self.k_split(self._grid(n), _THREADS, src, bias, dst, self.B, L,
                     self.H, self.dk, parts)

    def _merge(self, src, dst, L, parts):
        n = self.B * L * parts * self.d
        self.k_merge(self._grid(n), _THREADS, src, dst, self.B, L, self.H,
                     self.dk, parts)

    def _seed(self, i):
        return np.uint32(self._seeds[i])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, train):
        cfg = self.cfg
        self._p_res = self._p_emb = self._p_attn = float(cfg.dropout) if train else 0.0
        self._p_ffn = float(cfg.ffn_dropout_rate) if train else 0.0
        with drv.region("fwd"):
            self._fwd_stack(self.enc)
            self._fwd_stack(self.dec)
        with drv.region("fwd"), drv.region("head"):
            self._head_forward()

    def _head_forward(self):
        cfg, dec = self.cfg, self.dec
        drv.gemm(dec.h[-1], self.W["embedding.weight"], self.logits, trans_b=True)
        pad = cfg.pad_id
        args = (float(cfg.label_smoothing),
                int(cfg.label_smoothing_mode == "non_target"), pad, pad,
                self.scalars, 1.0 / self._count, self.scalars[2:3])
        if self.k_ce_reg is not None:
            self.k_ce_reg(dec.M, _CE_THREADS, self.logits, self.tgt_out_ids, *args)
        else:
            self.k_ce(dec.M, _THREADS, self.logits, self.tgt_out_ids,
                      cfg.vocab_size, *args)

    def _ln_names(self, st, k):
        name = st.subs[k].norm if k < len(st.subs) else st.final
        return self.W[name + ".weight"], self.W[name + ".bias"]

    def _fwd_stack(self, st):
        with drv.region(st.name):
            self._fwd_stack_inner(st)

    def _fwd_stack_inner(self, st):
        w, b = self._ln_names(st, 0)
        with drv.region("embed"):
            self._embed_ln(st, w, b)
        for k, sub in enumerate(st.subs):
            with drv.region(sub.kind):
                y = self._sub_fwd(st, k, sub)
            with drv.region("residual_ln"):
                self._residual_ln(st, k, sub, y)

    def _embed_ln(self, st, w, b):
        self.k_embed_ln(self._warp_rows(st.M), _THREADS,
                        None, None, None, st.ids, self.W["embedding.weight"],
                        self.pe, st.L, self.emb_scale,
                        st.x[0], st.h[0], st.mean[0], st.rstd[0], w, b,
                        st.M, self._seed(st.emb_seed), self._p_emb, _LN_EPS)

    def _residual_ln(self, st, k, sub, y):
        w, b = self._ln_names(st, k + 1)
        ybias = self.W[sub.out_bias] if sub.out_bias else None
        self.k_res_ln(self._warp_rows(st.M), _THREADS,
                      st.x[k], y, ybias, None, None, None, 0, 0.0,
                      st.x[k + 1], st.h[k + 1], st.mean[k + 1],
                      st.rstd[k + 1], w, b, st.M,
                      self._seed(sub.res_seed), self._p_res, _LN_EPS)

    def _sub_fwd(self, st, k, sub):
        W, d, H, dk, BH = self.W, self.d, self.H, self.dk, self.B * self.H
        h, L, M, pre = st.h[k], st.L, st.M, sub.prefix
        y = self._buf(f"{st.name}.y", (M, d))
        alpha = 1.0 / math.sqrt(dk)
        if sub.kind == "ffn":
            u, a = sub.bufs["u"], sub.bufs["a"]
            drv.gemm(h, W[f"{pre}.up.weight"], u)
            blocks, threads, rpb = self._ffn_grid(M, self.cfg.d_ff)
            self.k_bias_act(blocks, threads, u, W[f"{pre}.up.bias"], a, M,
                            self.cfg.d_ff, rpb, self._seed(sub.inner_seed),
                            self._p_ffn)
            drv.gemm(a, W[f"{pre}.down.weight"], y)
            return y
        fused = self._attn(sub, st)
        if fused is not None:
            return self._attn_fwd_fused(st, k, sub, fused, y)
        if sub.kind == "self":
            qkv = self._buf(f"{st.name}.qkv", (M, 3 * d))
            heads = sub.bufs["heads"]
            drv.gemm(h, W[f"{pre}.w_qkv.weight"], qkv)
            self._split(qkv, W.get(f"{pre}.w_qkv.bias"), heads, L, 3)
            q, kk, v = heads[0], heads[1], heads[2]
            kid, LK = st.ids, L
        else:  # cross: queries from the decoder, keys/values from memory
            mem, Ms = self.enc.h[-1], self.enc.M
            qbuf = self._buf("dec.q", (M, d))
            kv = self._buf("dec.kv", (Ms, 2 * d))
            drv.gemm(h, W[f"{pre}.w_q.weight"], qbuf)
            self._split(qbuf, W.get(f"{pre}.w_q.bias"), sub.bufs["qh"], L, 1)
            drv.gemm(mem, W[f"{pre}.w_kv.weight"], kv)
            self._split(kv, W.get(f"{pre}.w_kv.bias"), sub.bufs["kvh"], self.S, 2)
            q, kk, v = sub.bufs["qh"][0], sub.bufs["kvh"][0], sub.bufs["kvh"][1]
            kid, LK = self.src_ids, self.S
        P, Pd = sub.bufs["P"], sub.bufs["Pd"]
        scores = self._buf(f"scores.{L}x{LK}", (BH, L, LK))
        drv.gemm_batched(q, kk, scores, trans_b=True, alpha=alpha)
        rows = BH * L
        self.k_softmax[LK, int(sub.causal)](
            self._warp_rows(rows), _THREADS, scores, P, Pd, kid, rows, L, H,
            self.cfg.pad_id, self._seed(sub.inner_seed), self._p_attn)
        o = self._buf(f"{st.name}.o", (1, BH, L, dk))
        drv.gemm_batched(Pd, v, o[0])
        self._merge(o, sub.bufs["ctx"], L, 1)
        drv.gemm(sub.bufs["ctx"], W[f"{pre}.w_o.weight"], y)
        return y

    def _qkv_parts(self, sub):
        """Kernel arguments locating Q, K and V inside the raw projection
        outputs (fused path): pointer, row stride, bias pointer, three times."""
        W, d, pre = self.W, self.d, sub.prefix
        if sub.kind == "self":
            qkv, bias = sub.bufs["qkv"], W.get(f"{pre}.w_qkv.bias")
            srcs = [(qkv, 3 * d, i * d) for i in range(3)]
            biases = [(bias, i * d) for i in range(3)]
        else:
            q, kv = sub.bufs["q"], sub.bufs["kv"]
            bq, bkv = W.get(f"{pre}.w_q.bias"), W.get(f"{pre}.w_kv.bias")
            srcs = [(q, d, 0), (kv, 2 * d, 0), (kv, 2 * d, d)]
            biases = [(bq, 0), (bkv, 0), (bkv, d)]
        args = []
        for (buf, stride, off), (b, boff) in zip(srcs, biases):
            args += [drv.ptr(buf, off), stride,
                     drv.ptr(b, boff) if b is not None else None]
        return args

    def _attn_fwd_fused(self, st, k, sub, fused, y):
        W, d, pre = self.W, self.d, sub.prefix
        h = st.h[k]
        if sub.kind == "self":
            drv.gemm(h, W[f"{pre}.w_qkv.weight"], sub.bufs["qkv"])
            kid = st.ids
        else:
            drv.gemm(h, W[f"{pre}.w_q.weight"], sub.bufs["q"])
            drv.gemm(self.enc.h[-1], W[f"{pre}.w_kv.weight"], sub.bufs["kv"])
            kid = self.src_ids
        kf, _, fwd_smem, _ = fused
        kf(self.B * self.H, _ATT_WARPS * 32, *self._qkv_parts(sub), kid,
           sub.bufs["ctx"], d, sub.bufs["lse"], self.H, self.cfg.pad_id,
           self._seed(sub.inner_seed), self._p_attn, 1.0 / math.sqrt(self.dk),
           shared_mem=fwd_smem)
        drv.gemm(sub.bufs["ctx"], W[f"{pre}.w_o.weight"], y)
        return y

    def _attn_bwd_fused(self, st, k, sub, fused, g_y, g_h):
        W, G, d, pre = self.W, self.G, self.d, sub.prefix
        h, M = st.h[k], st.M
        ctx = sub.bufs["ctx"]
        drv.gemm(ctx, g_y, G[f"{pre}.w_o.weight"], trans_a=True, beta=1.0)
        g_ctx = self._buf(f"{st.name}.gctx", (M, d))
        drv.gemm(g_y, W[f"{pre}.w_o.weight"], g_ctx, trans_b=True)
        if sub.kind == "self":
            g_qkv = self._buf(f"{st.name}.gqkv", (M, 3 * d))
            outs = [(g_qkv, 3 * d, i * d) for i in range(3)]
            gb = G.get(f"{pre}.w_qkv.bias")
            gbias = [(gb, i * d) for i in range(3)]
            kid = st.ids
        else:
            g_q = self._buf("dec.gq", (M, d))
            g_kv = self._buf("dec.gkv", (self.enc.M, 2 * d))
            outs = [(g_q, d, 0), (g_kv, 2 * d, 0), (g_kv, 2 * d, d)]
            gq, gkv = G.get(f"{pre}.w_q.bias"), G.get(f"{pre}.w_kv.bias")
            gbias = [(gq, 0), (gkv, 0), (gkv, d)]
            kid = self.src_ids
        out_args = []
        for buf, stride, off in outs:
            out_args += [drv.ptr(buf, off), stride]
        bias_args = [drv.ptr(b, off) if b is not None else None for b, off in gbias]
        _, kb, _, bwd_smem = fused
        kb(self.B * self.H, _ATT_BWD_WARPS * 32, *self._qkv_parts(sub), kid,
           ctx, d, g_ctx, d, sub.bufs["lse"], *out_args, *bias_args,
           self.H, self.cfg.pad_id, self._seed(sub.inner_seed), self._p_attn,
           1.0 / math.sqrt(self.dk), shared_mem=bwd_smem)
        if sub.kind == "self":
            drv.gemm(h, g_qkv, G[f"{pre}.w_qkv.weight"], trans_a=True, beta=1.0)
            drv.gemm(g_qkv, W[f"{pre}.w_qkv.weight"], g_h, trans_b=True)
        else:
            drv.gemm(h, g_q, G[f"{pre}.w_q.weight"], trans_a=True, beta=1.0)
            drv.gemm(g_q, W[f"{pre}.w_q.weight"], g_h, trans_b=True)
            drv.gemm(self.enc.h[-1], g_kv, G[f"{pre}.w_kv.weight"], trans_a=True,
                     beta=1.0)
            drv.gemm(g_kv, W[f"{pre}.w_kv.weight"], self._g_mem, trans_b=True,
                     beta=1.0)                                 # memory grad
        return g_h

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def _backward(self):
        dec, enc = self.dec, self.enc
        E, gE = self.W["embedding.weight"], self.G["embedding.weight"]
        g_h = self._buf("dec.gh_final", (dec.M, self.d))
        with drv.region("bwd"), drv.region("head"):
            drv.gemm(self.logits, E, g_h)                                # dh = dl·E
            drv.gemm(self.logits, dec.h[-1], gE, trans_a=True, beta=1.0)  # dE += dlᵀ·h
        g_mem = self._buf("enc.gh_final", (enc.M, self.d))
        with drv.region("bwd"):
            g_mem.fill_(0)
            self._g_mem = g_mem
            with drv.region("dec"):
                self._bwd_stack(dec, g_h)
            with drv.region("enc"):
                self._bwd_stack(enc, g_mem)

    def _ln_bwd(self, st, k, g_h, accum, prev_seed):
        """LayerNorm site k backward; also writes the gradient of sublayer
        k-1's output (and sums its output bias's gradient) when prev_seed
        is given."""
        name = st.subs[k].norm if k < len(st.subs) else st.final
        prev_bias = st.subs[k - 1].out_bias if k > 0 else None
        g_x = self._buf(f"{st.name}.gx", (st.M, self.d))
        g_y = self._buf(f"{st.name}.gy", (st.M, self.d))
        blocks = max(1, min(-(-st.M // self.ln_warps), self._sms * 2))
        self.k_ln_bwd(blocks, self.ln_warps * 32, g_h, st.x[k], st.mean[k],
                      st.rstd[k], self.W[name + ".weight"], g_x, int(accum),
                      self.G[name + ".weight"], self.G[name + ".bias"],
                      g_y if prev_seed is not None else None,
                      self.G[prev_bias] if (prev_seed is not None and prev_bias)
                      else None,
                      self._seed(prev_seed) if prev_seed is not None else np.uint32(0),
                      self._p_res, st.M)
        return g_x, g_y

    def _bwd_stack(self, st, g_h_final):
        K = len(st.subs)
        prev = st.subs[K - 1].res_seed if K else None
        with drv.region("residual_ln"):
            g_x, g_y = self._ln_bwd(st, K, g_h_final, False, prev)
        for k in range(K - 1, -1, -1):
            with drv.region(st.subs[k].kind):
                g_h = self._sub_bwd(st, k, st.subs[k], g_y)
            prev = st.subs[k - 1].res_seed if k > 0 else None
            with drv.region("residual_ln"):
                self._ln_bwd(st, k, g_h, True, prev)
        pad_idx = self.cfg.pad_id if self.cfg.embedding_padding_idx else -1
        n = st.M * self.d
        with drv.region("embed"):
            self._embed_bwd(st, g_x, n, pad_idx)

    def _embed_bwd(self, st, g_x, n, pad_idx):
        self.k_embed_bwd(self._grid(n), _THREADS, g_x, st.ids,
                         self.G["embedding.weight"], st.M, self.d,
                         self._seed(st.emb_seed), self._p_emb, self.emb_scale,
                         pad_idx)

    def _sub_bwd(self, st, k, sub, g_y):
        """g_y: gradient of the sublayer output (before its bias, after the
        residual dropout's mask). Returns the gradient of its input h."""
        W, G, d, H, dk, BH = self.W, self.G, self.d, self.H, self.dk, self.B * self.H
        h, L, M, pre = st.h[k], st.L, st.M, sub.prefix
        g_h = self._buf(f"{st.name}.gh", (M, d))
        alpha = 1.0 / math.sqrt(dk)
        # The output bias's gradient was already summed by ln_bwd.
        if sub.kind == "ffn":
            dff = self.cfg.d_ff
            drv.gemm(sub.bufs["a"], g_y, G[f"{pre}.down.weight"], trans_a=True,
                     beta=1.0)
            g_a = self._buf(f"{st.name}.ga", (M, dff))
            drv.gemm(g_y, W[f"{pre}.down.weight"], g_a, trans_b=True)
            blocks, threads, rpb = self._ffn_grid(M, dff)
            self.k_act_bwd(blocks, threads, g_a, sub.bufs["u"],
                           G[f"{pre}.up.bias"], M, dff, rpb,
                           self._seed(sub.inner_seed), self._p_ffn)
            drv.gemm(h, g_a, G[f"{pre}.up.weight"], trans_a=True, beta=1.0)
            drv.gemm(g_a, W[f"{pre}.up.weight"], g_h, trans_b=True)
            return g_h

        fused = self._attn(sub, st)
        if fused is not None:
            return self._attn_bwd_fused(st, k, sub, fused, g_y, g_h)
        # attention output projection
        ctx = sub.bufs["ctx"]
        drv.gemm(ctx, g_y, G[f"{pre}.w_o.weight"], trans_a=True, beta=1.0)
        g_ctx = self._buf(f"{st.name}.gctx", (M, d))
        drv.gemm(g_y, W[f"{pre}.w_o.weight"], g_ctx, trans_b=True)
        g_o = self._buf(f"{st.name}.go", (1, BH, L, dk))
        self._split(g_ctx, None, g_o, L, 1)
        if sub.kind == "self":
            heads = sub.bufs["heads"]
            q, kk, v = heads[0], heads[1], heads[2]
            kid, LK = st.ids, L
            g_heads = self._buf(f"{st.name}.gheads", (3, BH, L, dk))
            g_q, g_k, g_v = g_heads[0], g_heads[1], g_heads[2]
        else:
            q, kk, v = sub.bufs["qh"][0], sub.bufs["kvh"][0], sub.bufs["kvh"][1]
            kid, LK = self.src_ids, self.S
            g_qh = self._buf("dec.gqh", (1, BH, L, dk))
            g_kvh = self._buf("dec.gkvh", (2, BH, self.S, dk))
            g_q, g_k, g_v = g_qh[0], g_kvh[0], g_kvh[1]
        P, Pd = sub.bufs["P"], sub.bufs["Pd"]
        dP = self._buf(f"dscores.{L}x{LK}", (BH, L, LK))
        drv.gemm_batched(g_o[0], v, dP, trans_b=True)          # dPd = dO·Vᵀ
        drv.gemm_batched(Pd, g_o[0], g_v, trans_a=True)        # dV = Pdᵀ·dO
        rows = BH * L
        self.k_softmax_bwd[LK, int(sub.causal)](
            self._warp_rows(rows), _THREADS, dP, P, kid, rows, L, H,
            self.cfg.pad_id, self._seed(sub.inner_seed), self._p_attn)
        drv.gemm_batched(dP, kk, g_q, alpha=alpha)             # dQ = a·dS·K
        drv.gemm_batched(dP, q, g_k, trans_a=True, alpha=alpha)  # dK = a·dSᵀ·Q

        if sub.kind == "self":
            g_qkv = self._buf(f"{st.name}.gqkv", (M, 3 * d))
            self._merge(g_heads, g_qkv, L, 3)
            if f"{pre}.w_qkv.bias" in G:
                self._colsum(g_qkv, G[f"{pre}.w_qkv.bias"], M)
            drv.gemm(h, g_qkv, G[f"{pre}.w_qkv.weight"], trans_a=True, beta=1.0)
            drv.gemm(g_qkv, W[f"{pre}.w_qkv.weight"], g_h, trans_b=True)
            return g_h

        mem, Ms = self.enc.h[-1], self.enc.M
        g_qm = self._buf("dec.gq", (M, d))
        self._merge(g_qh, g_qm, L, 1)
        if f"{pre}.w_q.bias" in G:
            self._colsum(g_qm, G[f"{pre}.w_q.bias"], M)
        drv.gemm(h, g_qm, G[f"{pre}.w_q.weight"], trans_a=True, beta=1.0)
        drv.gemm(g_qm, W[f"{pre}.w_q.weight"], g_h, trans_b=True)
        g_kv = self._buf("dec.gkv", (Ms, 2 * d))
        self._merge(g_kvh, g_kv, self.S, 2)
        if f"{pre}.w_kv.bias" in G:
            self._colsum(g_kv, G[f"{pre}.w_kv.bias"], Ms)
        drv.gemm(mem, g_kv, G[f"{pre}.w_kv.weight"], trans_a=True, beta=1.0)
        drv.gemm(g_kv, W[f"{pre}.w_kv.weight"], self._g_mem, trans_b=True,
                 beta=1.0)                                     # memory grad
        return g_h

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _load_batch(self, src, tgt_in, tgt_out):
        src, tgt_in, tgt_out = (np.asarray(a) for a in (src, tgt_in, tgt_out))
        want = {"src": (self.B, self.S), "tgt_in": (self.B, self.T),
                "tgt_out": (self.B, self.T)}
        for name, a in zip(want, (src, tgt_in, tgt_out)):
            if a.shape != want[name]:
                raise ValueError(f"{name} must have shape {want[name]} (the "
                                 f"shape this step was built for), got {a.shape}")
        self.src_ids.copy_from(src)
        self.tgt_in_ids.copy_from(tgt_in)
        self.tgt_out_ids.copy_from(tgt_out)
        self._count = max(int((tgt_out != self.cfg.pad_id).sum()), 1)

    def forward_backward(self, src, tgt_in, tgt_out, train: bool = True) -> None:
        """Loss and gradients for one batch (no parameter update)."""
        self._load_batch(src, tgt_in, tgt_out)
        self._seeds = self._rng.integers(0, 2 ** 32, size=self._n_seeds,
                                         dtype=np.uint32)
        with drv.region("optimizer"):
            self.grad.fill_(0)
            self.k_reset(1, 1, self.scalars)
        self._forward(train)
        self._backward()

    def update(self, lr: float) -> None:
        """Gradient norm, loss-scale bookkeeping, clipping and AdamW."""
        with drv.region("optimizer"):
            self._update(lr)

    def _update(self, lr):
        n = self.n_params
        n4 = n // 4
        self.k_sumsq(self._grid(n4), _THREADS, self.grad, n4, self.scalars[1:2])
        self.k_prepare(1, 1, self.scalars, self.beta1, self.beta2,
                       float(self.max_grad_norm or 0.0), int(self.half),
                       int(self.growth_interval))
        self.k_adamw(self._grid(n4), _THREADS, self.master, self.grad, self.m,
                     self.v, self.work, n4, self.scalars, float(lr), self.beta1,
                     self.beta2, self.adam_eps, self.weight_decay, 1)

    def __call__(self, src, tgt_in, tgt_out, lr: float) -> None:
        """One training step. Queued on the GPU: returns without waiting."""
        self.forward_backward(src, tgt_in, tgt_out, train=True)
        self.update(lr)

    def evaluate(self, src, tgt_in, tgt_out) -> float:
        """Loss without dropout and without updating anything (waits)."""
        self._load_batch(src, tgt_in, tgt_out)
        self.k_reset(1, 1, self.scalars)
        self._forward(train=False)
        return float(self.scalars[2:3].numpy()[0])

    def last_loss(self) -> float:
        """Loss of the most recent step (waits for the GPU)."""
        return float(self.scalars[2:3].numpy()[0])

    def stats(self) -> dict:
        """Loss, gradient norm, loss scale, skipped flag and step count."""
        s = self.scalars.numpy()
        return {"loss": float(s[2]), "grad_norm": float(s[9]),
                "loss_scale": float(s[0]), "skipped": bool(s[4]),
                "steps": int(s[5])}

    def gradients(self) -> dict:
        """{name: fp32 gradient} of the last forward_backward, unscaled."""
        scale = float(self.scalars[0:1].numpy()[0])
        flat = self.grad.numpy() / scale
        return {k: flat[self.offsets[k]:self.offsets[k] + int(np.prod(s))]
                .reshape(s) for k, s in self.shapes.items()}

    def state_dict(self) -> dict:
        """{name: fp32 master weights}, loadable into the CPU Transformer."""
        flat = self.master.numpy()
        return {k: flat[self.offsets[k]:self.offsets[k] + int(np.prod(s))]
                .reshape(s).copy() for k, s in self.shapes.items()}

    def memory_bytes(self) -> int:
        """Device memory held by this step's buffers."""
        return drv.memory_in_use()
