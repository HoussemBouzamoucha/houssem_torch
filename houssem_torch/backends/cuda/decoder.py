"""Translation on the GPU: incremental decoding with a key/value cache.

    tr = Translator(step, batch_size=64, src_len=64)   # step: a TrainStep
    ids = tr.translate_greedy(src)                      # like Transformer's

The encoder runs once per batch; each decoding step feeds one new token per
sentence through the decoder (B rows, not B·T). Self-attention appends the
token's key and value to a per-layer cache and attends to all earlier ones;
cross-attention reads the memory's K/V projections, computed once per
batch. It reuses the training step's weights (the fp16 working copy in
mixed precision) and its compiled kernels, so a model can be evaluated
between training steps without copying anything.
"""
from __future__ import annotations

import math

import numpy as np

from houssem_torch.backends.cuda import driver as drv
from houssem_torch.backends.cuda.train_step import (_LN_EPS, _THREADS,
                                                    BucketedTrainStep,
                                                    TrainStep)

_ATTN_THREADS = 128


class Translator:
    """Greedy translation of up to `batch_size` sentences of up to
    `src_len` source tokens at a time, with the weights of `step`."""

    def __init__(self, step: TrainStep | BucketedTrainStep, batch_size: int,
                 src_len: int):
        if isinstance(step, BucketedTrainStep):
            step = step.main
        cfg = step.cfg
        if src_len > cfg.max_len:
            raise ValueError(f"src_len exceeds max_len {cfg.max_len}")
        # A step of this source shape (sharing the weights) runs the encoder.
        self.enc_step = step.with_shape(batch_size, src_len, src_len)
        self.step, self.cfg = step, cfg
        self.B, self.S = batch_size, src_len
        self.Lmax = cfg.max_len
        d, B = cfg.d_model, batch_size
        self.d, self.H, self.dk = d, cfg.n_heads, d // cfg.n_heads
        dt = step.dtype
        self.k_attn = step._kernel("decode_attn", DEC_DK=self.dk,
                                   DEC_LMAX=max(self.Lmax, src_len))
        self.k_argmax = step._kernel("argmax_rows")
        self.subs = self.enc_step.dec.subs
        self.ids = drv.zeros(B, np.int32)
        self.next = drv.zeros(B, np.int32)
        self.x = [drv.zeros((B, d), dt), drv.zeros((B, d), dt)]
        self.h = drv.zeros((B, d), dt)
        self.mean, self.rstd = drv.zeros(B, np.float32), drv.zeros(B, np.float32)
        self.y = drv.zeros((B, d), dt)
        self.ctx = drv.zeros((B, d), dt)
        self.qkv = drv.zeros((B, 3 * d), dt)
        self.q = drv.zeros((B, d), dt)
        self.u = drv.zeros((B, cfg.d_ff), dt)
        self.a = drv.zeros((B, cfg.d_ff), dt)
        self.logits = drv.zeros((B, cfg.vocab_size), dt)
        self.cache = {}      # self-attention prefix -> (K, V), (B, Lmax, d)
        self.mem_kv = {}     # cross-attention prefix -> (B·S, 2d) projection
        for sub in self.subs:
            if sub.kind == "self":
                self.cache[sub.prefix] = (drv.zeros((B, self.Lmax, d), dt),
                                          drv.zeros((B, self.Lmax, d), dt))
            elif sub.kind == "cross":
                self.mem_kv[sub.prefix] = drv.zeros((B * src_len, 2 * d), dt)

    # ------------------------------------------------------------------

    def _fit(self, src):
        """Pad a (n ≤ B, w ≤ S) source batch to (B, S) with pad rows."""
        src = np.asarray(src)
        n, w = src.shape
        if n > self.B or w > self.S:
            raise ValueError(f"at most {self.B} sentences of {self.S} tokens, "
                             f"got {src.shape}")
        out = np.full((self.B, self.S), self.cfg.pad_id, np.int64)
        out[:n, :w] = src
        return out, n

    def _start(self, src):
        """Encoder pass and the cross-attention K/V of every layer."""
        self.src, n = self._fit(src)
        mem = self.enc_step.encode(self.src)
        W = self.step.W
        for prefix, kv in self.mem_kv.items():
            drv.gemm(mem, W[f"{prefix}.w_kv.weight"], kv)
        return n

    def _ln(self, k):
        name = self.subs[k].norm if k < len(self.subs) else self.enc_step.dec.final
        return self.step.W[name + ".weight"], self.step.W[name + ".bias"]

    def _step(self, t):
        """Logits (B, V) for position t, given self.ids = the tokens at t."""
        es, W, d, B = self.enc_step, self.step.W, self.d, self.B
        warps = es._warp_rows(B)
        w, b = self._ln(0)
        cur = 0
        es.k_embed_ln(warps, _THREADS, None, None, None, self.ids,
                      W["embedding.weight"], drv.ptr(es.pe, t * d), 1,
                      es.emb_scale, self.x[cur], self.h, self.mean, self.rstd,
                      w, b, B, np.uint32(0), 0.0, _LN_EPS)
        scale = 1.0 / math.sqrt(self.dk)
        for k, sub in enumerate(self.subs):
            pre = sub.prefix
            if sub.kind == "ffn":
                drv.gemm(self.h, W[f"{pre}.up.weight"], self.u)
                blocks, threads, rpb = es._ffn_grid(B, self.cfg.d_ff)
                es.k_bias_act(blocks, threads, self.u, W[f"{pre}.up.bias"],
                              self.a, None, B, self.cfg.d_ff, rpb, np.uint32(0),
                              0.0)
                drv.gemm(self.a, W[f"{pre}.down.weight"], self.y)
            elif sub.kind == "self":
                K, V = self.cache[pre]
                bias = W.get(f"{pre}.w_qkv.bias")
                drv.gemm(self.h, W[f"{pre}.w_qkv.weight"], self.qkv)
                self.k_attn(B * self.H, _ATTN_THREADS,
                            self.qkv, 3 * d, bias,
                            drv.ptr(self.qkv, d), drv.ptr(self.qkv, 2 * d), 3 * d,
                            None if bias is None else drv.ptr(bias, d),
                            None if bias is None else drv.ptr(bias, 2 * d), t,
                            K, V, d, self.Lmax * d, None, None,
                            None, 0, t + 1, self.ctx, d, self.H, scale)
                drv.gemm(self.ctx, W[f"{pre}.w_o.weight"], self.y)
            else:
                kv = self.mem_kv[pre]
                bq, bkv = W.get(f"{pre}.w_q.bias"), W.get(f"{pre}.w_kv.bias")
                drv.gemm(self.h, W[f"{pre}.w_q.weight"], self.q)
                self.k_attn(B * self.H, _ATTN_THREADS, self.q, d, bq,
                            None, None, 0, None, None, 0,
                            kv, drv.ptr(kv, d), 2 * d, self.S * 2 * d,
                            bkv, None if bkv is None else drv.ptr(bkv, d),
                            es.src_ids, self.cfg.pad_id, self.S,
                            self.ctx, d, self.H, scale)
                drv.gemm(self.ctx, W[f"{pre}.w_o.weight"], self.y)
            w, b = self._ln(k + 1)
            ybias = W[sub.out_bias] if sub.out_bias else None
            es.k_res_ln(warps, _THREADS, self.x[cur], self.y, ybias, None, None,
                        None, 0, 0.0, self.x[1 - cur], self.h, self.mean,
                        self.rstd, w, b, B, np.uint32(0), 0.0, _LN_EPS)
            cur = 1 - cur
        drv.gemm(self.h, W["embedding.weight"], self.logits, trans_b=True)
        return self.logits

    # ------------------------------------------------------------------

    def translate_greedy(self, src, max_new_tokens: int | None = None) -> np.ndarray:
        """Same contract as Transformer.translate_greedy: int ids
        (n, ≤ max_new_tokens) without the leading <bos>, pad_id after a
        sentence's <eos>. src: (n ≤ batch_size, ≤ src_len) token ids."""
        cfg = self.cfg
        max_new = min(max_new_tokens or cfg.max_len - 1, self.Lmax)
        n = self._start(src)
        tokens = np.full(self.B, cfg.bos_id, np.int32)
        finished = np.zeros(self.B, bool)
        finished[n:] = True                     # padding rows
        out = []
        for t in range(max_new):
            self.ids.copy_from(tokens)
            self.k_argmax(self.B, 256, self._step(t), cfg.vocab_size, self.next)
            nxt = self.next.numpy()
            nxt = np.where(finished, cfg.pad_id, nxt).astype(np.int32)
            out.append(nxt)
            finished |= nxt == cfg.eos_id
            tokens = nxt
            if finished.all():
                break
        return np.stack(out, axis=1)[:n].astype(np.int64)

    def decode_logits(self, src, tgt_in) -> np.ndarray:
        """Logits (n, T, V) of teacher-forced decoding, one token at a time
        through the cache (for checking it against a full forward pass)."""
        n = self._start(src)
        tgt_in = np.asarray(tgt_in)
        T = tgt_in.shape[1]
        if T > self.Lmax:
            raise ValueError(f"at most {self.Lmax} target tokens")
        full = np.full((self.B, T), self.cfg.pad_id, np.int32)
        full[:n] = tgt_in
        out = []
        for t in range(T):
            self.ids.copy_from(full[:, t])
            out.append(self._step(t).numpy().astype(np.float32)[:n])
        return np.stack(out, axis=1)
