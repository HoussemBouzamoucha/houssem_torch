"""Fused tensor-core attention kernels against a NumPy reference.

The reference reproduces the kernels' dropout hash exactly, so forward
output, log-sum-exp, dQ / dK / dV and the bias gradients are compared with
dropout on. Inputs are fp16 (what the kernels read); the reference runs in
float64 on those same fp16 values. Skipped without CuPy and a CUDA GPU.
"""
import math

import numpy as np
import pytest

from houssem_torch.backends.cuda import driver

pytestmark = pytest.mark.skipif(not driver.is_available(),
                                reason="needs CuPy and a CUDA GPU")

from houssem_torch.backends.cuda.train_step import (  # noqa: E402
    _SOURCE, _ATT_BWD_WARPS, _ATT_WARPS, _pad16, attn_smem)

MASK_VALUE = -1e9
PAD = 0


def mix32(h):
    h = h.astype(np.uint64) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def drop_mult(seed, idx, p):
    """kernels.cu drop_mult, vectorized over idx."""
    if p <= 0:
        return np.ones(idx.shape)
    i = (idx.astype(np.uint64) * 0x9E3779B9) & 0xFFFFFFFF
    h = mix32(i ^ mix32(np.array([seed], np.uint64))[0])
    u = (h >> 8).astype(np.float64) / 16777216.0
    keep = u >= np.float32(p)
    return np.where(keep, np.float64(np.float32(1.0) / np.float32(1.0 - p)), 0.0)


def reference(q, k, v, kid, causal, p, seed, dout, alpha):
    """q (B,H,LQ,DK), k/v (B,H,LK,DK) float64; returns o, lse, dq, dk, dv."""
    B, H, LQ, _ = q.shape
    LK = k.shape[2]
    s = np.einsum("bhqd,bhkd->bhqk", q, k) * alpha
    masked = (kid == PAD)[:, None, None, :]
    if causal:
        masked = masked | (np.arange(LK)[None, :] > np.arange(LQ)[:, None])
    s = np.where(masked, MASK_VALUE, s)
    mx = s.max(-1, keepdims=True)
    e = np.exp(s - mx)
    P = e / e.sum(-1, keepdims=True)
    lse = (mx + np.log(e.sum(-1, keepdims=True)))[..., 0]
    bh = np.arange(B * H).reshape(B, H)[:, :, None, None]
    idx = (bh * LQ + np.arange(LQ)[None, None, :, None]) * LK + \
        np.arange(LK)[None, None, None, :]
    m = drop_mult(seed, idx, p)
    Pd = P * m
    o = np.einsum("bhqk,bhkd->bhqd", Pd, v)
    dPd = np.einsum("bhqd,bhkd->bhqk", dout, v)
    dP = dPd * m
    dS = P * (dP - (dP * P).sum(-1, keepdims=True))
    dS = np.where(masked, 0.0, dS)
    dq = np.einsum("bhqk,bhkd->bhqd", dS, k) * alpha
    dk = np.einsum("bhqk,bhqd->bhkd", dS, q) * alpha
    dv = np.einsum("bhqk,bhqd->bhkd", Pd, dout)
    return o, lse, dq, dk, dv


def heads(x, B, L, H, dk):
    """(B*L, H*dk) -> (B, H, L, dk)"""
    return x.reshape(B, L, H, dk).transpose(0, 2, 1, 3)


def merged(x):
    B, H, L, dk = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B * L, H * dk)


CASES = {
    #           B, H, LQ, LK, DK, causal, p,   bias
    "self":     (3, 2, 16, 16, 16, 0, 0.0, True),
    "self_pad": (2, 3, 13, 13, 32, 0, 0.2, True),
    "causal":   (2, 2, 21, 21, 16, 1, 0.1, False),
    "cross":    (2, 2, 9, 24, 64, 0, 0.15, True),
    "base":     (2, 8, 64, 64, 64, 1, 0.1, True),
}


# (backward warps, forward Q in registers)
VARIANTS = sorted({(4, 1), (8, 1), (16, 1), (_ATT_BWD_WARPS, 0)})


@pytest.mark.parametrize("nwb,qreg", VARIANTS)
@pytest.mark.parametrize("case", CASES)
def test_fused_attention_matches_reference(case, nwb, qreg):
    B, H, LQ, LK, DK, causal, p, with_bias = CASES[case]
    d = H * DK
    rng = np.random.default_rng(0)
    f16 = lambda a: a.astype(np.float16)  # noqa: E731
    # Q from a (B*LQ, d) buffer; K and V side by side in one (B*LK, 2d)
    # buffer, like the cross-attention projection output.
    qbuf = f16(rng.standard_normal((B * LQ, d)))
    kvbuf = f16(rng.standard_normal((B * LK, 2 * d)))
    bias_q = f16(rng.standard_normal(d) * 0.1) if with_bias else None
    bias_kv = f16(rng.standard_normal(2 * d) * 0.1) if with_bias else None
    kid = rng.integers(3, 50, (B, LK)).astype(np.int32)
    kid[0, LK - 3:] = PAD                    # trailing padding
    if B > 1 and not causal:
        kid[1, :] = PAD                      # a sentence of only padding
    dout = f16(rng.standard_normal((B * LQ, d)))

    def with_b(x, b):
        return x.astype(np.float64) + (0 if b is None else b.astype(np.float64))
    # The kernels add the bias in fp32 and round to fp16 in shared memory.
    qf = f16(with_b(qbuf, bias_q)).astype(np.float64)
    kf = f16(with_b(kvbuf[:, :d], None if bias_kv is None else bias_kv[:d]))
    vf = f16(with_b(kvbuf[:, d:], None if bias_kv is None else bias_kv[d:]))
    alpha = 1.0 / math.sqrt(DK)
    seed = 1234
    o, lse, dq, dk, dv = reference(
        heads(qf, B, LQ, H, DK), heads(kf.astype(np.float64), B, LK, H, DK),
        heads(vf.astype(np.float64), B, LK, H, DK), kid, causal, p, seed,
        heads(dout.astype(np.float64), B, LQ, H, DK), alpha)

    defs = dict(USE_HALF=1, ATT=1, AQ=LQ, AK=LK, AQP=_pad16(LQ), AKP=_pad16(LK),
                DK=DK, CAUSAL=causal, NW=_ATT_WARPS, NWB=nwb, QREG=qreg)
    kf_, kb_ = (driver.compile_kernel(_SOURCE, n, defs)
                for n in ("attn_fwd", "attn_bwd"))
    fwd_smem, bwd_smem, _ = attn_smem(LQ, LK, DK, qreg)
    kf_.set_max_dynamic_shared(fwd_smem)
    kb_.set_max_dynamic_shared(bwd_smem)

    gq, gkv = driver.from_numpy(qbuf), driver.from_numpy(kvbuf)
    gbq = driver.from_numpy(bias_q) if with_bias else None
    gbkv = driver.from_numpy(bias_kv) if with_bias else None
    gkid = driver.from_numpy(kid)
    out = driver.zeros((B * LQ, d), np.float16)
    glse = driver.zeros(2 * B * H * LQ, np.float32)  # (row max, log sum)
    ptr = driver.ptr
    qkv_args = [ptr(gq), d, ptr(gbq) if gbq else None,
                ptr(gkv), 2 * d, ptr(gbkv) if gbkv else None,
                ptr(gkv, d), 2 * d, ptr(gbkv, d) if gbkv else None]
    kf_(B * H, 32 * _ATT_WARPS, *qkv_args, gkid, out, d, glse, H, PAD,
        np.uint32(seed), p, alpha, shared_mem=fwd_smem)
    np.testing.assert_allclose(out.numpy().astype(np.float64), merged(o),
                               atol=2e-2, rtol=2e-2)
    stats = glse.numpy().astype(np.float64).reshape(B, H, LQ, 2)
    np.testing.assert_allclose(stats.sum(-1), lse, atol=2e-3, rtol=1e-4)

    gdout = driver.from_numpy(dout)
    dq_b = driver.zeros((B * LQ, d), np.float16)
    dkv_b = driver.zeros((B * LK, 2 * d), np.float16)
    dbq = driver.zeros(d) if with_bias else None
    dbkv = driver.zeros(2 * d) if with_bias else None
    kb_(B * H, 32 * nwb, *qkv_args, gkid, out, d, gdout, d, glse,
        ptr(dq_b), d, ptr(dkv_b), 2 * d, ptr(dkv_b, d), 2 * d,
        ptr(dbq) if dbq else None, ptr(dbkv) if dbkv else None,
        ptr(dbkv, d) if dbkv else None,
        H, PAD, np.uint32(seed), p, alpha, shared_mem=bwd_smem)

    def close(got, ref, name):
        scale = np.abs(ref).max() + 1e-6
        np.testing.assert_allclose(got, ref, atol=2e-2 * scale, rtol=3e-2,
                                   err_msg=name)
    close(dq_b.numpy().astype(np.float64), merged(dq), "dq")
    dkv = dkv_b.numpy().astype(np.float64)
    close(dkv[:, :d], merged(dk), "dk")
    close(dkv[:, d:], merged(dv), "dv")
    if with_bias:
        close(dbq.numpy(), merged(dq).sum(0), "dbias q")
        # The key bias's true gradient is 0 (softmax ignores a per-row
        # constant): check the kernel's rounding noise against the size of
        # the dK values it sums.
        noise = np.abs(dbkv.numpy()[:d] - merged(dk).sum(0)).max()
        assert noise < 1e-2 * np.abs(merged(dk)).sum(0).max(), noise
        close(dbkv.numpy()[d:], merged(dv).sum(0), "dbias v")
