// houssem_torch GPU kernels for the compiled training step.
//
// Compiled at startup with NVRTC (driver.compile_kernel). Compile-time
// constants, set per run from the config:
//   USE_HALF   1: activations / working weights are fp16, 0: fp32
//   D          d_model            (add_ln_fwd, ln_bwd; multiple of 32)
//   LN_WARPS   warps per block in ln_bwd
//   LK         keys per attention row (softmax_fwd / softmax_bwd)
//   CAUSAL     1: query i only sees keys 0..i (softmax_fwd / softmax_bwd)
//   ACT        0: relu, 1: gelu (tanh approximation)
//   MODE       add_ln_fwd: 0 = embedding input, 1 = residual input
//
// All reductions (LayerNorm statistics, softmax, loss, norms) and every
// gradient that is accumulated across rows run in fp32.
//
// Dropout masks are never stored: keep/drop for element i of a dropout
// site is a hash of (seed, i), so the backward pass regenerates exactly the
// mask the forward pass used. Each site gets a fresh seed every step.

#include <cuda_fp16.h>

#if USE_HALF
typedef __half T;
__device__ __forceinline__ float tof(T x) { return __half2float(x); }
__device__ __forceinline__ T fromf(float x) { return __float2half(x); }
#else
typedef float T;
__device__ __forceinline__ float tof(T x) { return x; }
__device__ __forceinline__ T fromf(float x) { return x; }
#endif

// Same value as houssem_torch.nn.attention.MASK_VALUE. Scores are masked in
// fp32 registers, so it is safe in fp16 mode too.
#define MASK_VALUE (-1e9f)
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned mix32(unsigned h) {  // murmur3 finalizer
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

// Dropout multiplier for element i: 0 (dropped, probability p) or
// 1/(1-p) (kept). p <= 0 means no dropout.
__device__ __forceinline__ float drop_mult(unsigned seed, unsigned i, float p) {
    if (p <= 0.f) return 1.f;
    float u = (mix32(i * 0x9E3779B9u ^ mix32(seed)) >> 8) * (1.0f / 16777216.0f);
    return u >= p ? 1.f / (1.f - p) : 0.f;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum over the whole block (blockDim.x a multiple of 32, <= 1024).
__device__ float block_sum(float v, float* shared32) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) shared32[warp] = v;
    __syncthreads();
    int nwarps = blockDim.x >> 5;
    v = (threadIdx.x < nwarps) ? shared32[threadIdx.x] : 0.f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) shared32[0] = v;
    __syncthreads();
    return shared32[0];
}

__device__ float block_max(float v, float* shared32) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) shared32[warp] = v;
    __syncthreads();
    int nwarps = blockDim.x >> 5;
    v = (threadIdx.x < nwarps) ? shared32[threadIdx.x] : -INFINITY;
    if (warp == 0) v = warp_max(v);
    if (threadIdx.x == 0) shared32[0] = v;
    __syncthreads();
    return shared32[0];
}

// ---------------------------------------------------------------------------
// Residual stream: (embedding | residual + dropout) fused with LayerNorm
// ---------------------------------------------------------------------------
#ifdef D
#define VPT (D / 32)

// One warp per row of width D. Lane owns columns lane, lane+32, ...
//   MODE 0: x_out = dropout(table[ids[row]] * emb_scale + pe[row % L])
//   MODE 1: x_out = x_in + dropout(y + ybias)          (ybias may be null)
// then h = LayerNorm(x_out) * w + b, saving mean and rstd per row.
extern "C" __global__ void add_ln_fwd(
        const T* __restrict__ x_in, const T* __restrict__ y,
        const T* __restrict__ ybias,
        const int* __restrict__ ids, const T* __restrict__ table,
        const float* __restrict__ pe, int L, float emb_scale,
        T* __restrict__ x_out, T* __restrict__ h,
        float* __restrict__ mean_out, float* __restrict__ rstd_out,
        const T* __restrict__ w, const T* __restrict__ b,
        int rows, unsigned seed, float p, float eps) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    size_t base = (size_t)row * D;
    float v[VPT];
    float sum = 0.f;
#if MODE == 0
    const T* trow = table + (size_t)ids[row] * D;
    const float* prow = pe + (size_t)(row % L) * D;
#endif
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        int c = lane + 32 * j;
        unsigned idx = (unsigned)(base + c);
#if MODE == 0
        float t = (tof(trow[c]) * emb_scale + prow[c]) * drop_mult(seed, idx, p);
#else
        float yy = tof(y[base + c]) + (ybias ? tof(ybias[c]) : 0.f);
        float t = tof(x_in[base + c]) + yy * drop_mult(seed, idx, p);
#endif
        T stored = fromf(t);
        x_out[base + c] = stored;
        v[j] = tof(stored);  // statistics of the value backward will see
        sum += v[j];
    }
    float mean = warp_sum(sum) * (1.f / D);
    float sq = 0.f;
#pragma unroll
    for (int j = 0; j < VPT; ++j) { float t = v[j] - mean; sq += t * t; }
    float rstd = rsqrtf(warp_sum(sq) * (1.f / D) + eps);
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        int c = lane + 32 * j;
        h[base + c] = fromf((v[j] - mean) * rstd * tof(w[c]) + tof(b[c]));
    }
    if (lane == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
}

// Backward of one LayerNorm site of the residual stream, fused with the
// dropout of the residual branch that produced its input:
//   g_x  = (accum ? g_x : 0) + LayerNormBackward(g_h)
//   dw  += sum_rows g_h * x_hat,   db += sum_rows g_h       (fp32, atomics)
//   if g_y_prev: g_y_prev = dropout_mask(seed_prev) * g_x  (grad of the
//                previous sublayer's output, before its bias)
// Each warp walks rows with a grid stride, keeping its column partial sums
// for dw/db in registers; the block combines them in shared memory and
// issues one atomic per column.
#ifdef LN_WARPS
extern "C" __global__ void ln_bwd(
        const T* __restrict__ g_h, const T* __restrict__ x,
        const float* __restrict__ mean, const float* __restrict__ rstd,
        const T* __restrict__ w, T* __restrict__ g_x, int accum,
        float* __restrict__ dw, float* __restrict__ db,
        T* __restrict__ g_y_prev, unsigned seed_prev, float p_prev, int rows) {
    __shared__ float s_dw[LN_WARPS][D];
    __shared__ float s_db[LN_WARPS][D];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float pdw[VPT], pdb[VPT], wv[VPT];
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        pdw[j] = 0.f; pdb[j] = 0.f; wv[j] = tof(w[lane + 32 * j]);
    }
    int stride = gridDim.x * LN_WARPS;
    for (int row = blockIdx.x * LN_WARPS + warp; row < rows; row += stride) {
        size_t base = (size_t)row * D;
        float mu = mean[row], rs = rstd[row];
        float xh[VPT], gh[VPT];
        float s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            int c = lane + 32 * j;
            float g = tof(g_h[base + c]);
            xh[j] = (tof(x[base + c]) - mu) * rs;
            pdb[j] += g;
            pdw[j] += g * xh[j];
            gh[j] = g * wv[j];
            s1 += gh[j];
            s2 += gh[j] * xh[j];
        }
        s1 = warp_sum(s1) * (1.f / D);
        s2 = warp_sum(s2) * (1.f / D);
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            int c = lane + 32 * j;
            float dx = rs * (gh[j] - s1 - xh[j] * s2);
            if (accum) dx += tof(g_x[base + c]);
            T stored = fromf(dx);
            g_x[base + c] = stored;
            if (g_y_prev)
                g_y_prev[base + c] = fromf(tof(stored) *
                                           drop_mult(seed_prev, (unsigned)(base + c), p_prev));
        }
    }
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        s_dw[warp][lane + 32 * j] = pdw[j];
        s_db[warp][lane + 32 * j] = pdb[j];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float a = 0.f, bsum = 0.f;
#pragma unroll
        for (int k = 0; k < LN_WARPS; ++k) { a += s_dw[k][c]; bsum += s_db[k][c]; }
        atomicAdd(dw + c, a);
        atomicAdd(db + c, bsum);
    }
}
#endif  // LN_WARPS
#endif  // D

// Embedding backward: dE[ids[row]] += emb_scale * dropout_mask * g_x[row].
// Rows whose id is padding_idx (>= 0) send nothing.
extern "C" __global__ void embed_bwd(
        const T* __restrict__ g_x, const int* __restrict__ ids,
        float* __restrict__ dE, int rows, int d, unsigned seed, float p,
        float emb_scale, int padding_idx) {
    size_t n = (size_t)rows * d;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        int row = (int)(i / d), c = (int)(i % d);
        int id = ids[row];
        if (id == padding_idx) continue;
        float g = tof(g_x[i]) * drop_mult(seed, (unsigned)i, p) * emb_scale;
        atomicAdd(dE + (size_t)id * d + c, g);
    }
}

// ---------------------------------------------------------------------------
// Feed-forward: bias + activation + dropout
// ---------------------------------------------------------------------------
#ifdef ACT
__device__ __forceinline__ float act_f(float x) {
#if ACT == 0
    return fmaxf(x, 0.f);
#else
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
#endif
}

__device__ __forceinline__ float act_grad(float x) {
#if ACT == 0
    return x > 0.f ? 1.f : 0.f;
#else
    const float c = 0.7978845608028654f;
    float x2 = x * x;
    float t = tanhf(c * (x + 0.044715f * x2 * x));
    float du = c * (1.f + 3.f * 0.044715f * x2);
    return 0.5f * ((1.f + t) + x * (1.f - t * t) * du);
#endif
}

// u += bias (kept for backward);  a = dropout(act(u)).
extern "C" __global__ void bias_act_fwd(
        T* __restrict__ u, const T* __restrict__ bias, T* __restrict__ a,
        int rows, int cols, unsigned seed, float p) {
    size_t n = (size_t)rows * cols;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        T ub = fromf(tof(u[i]) + tof(bias[i % cols]));
        u[i] = ub;
        a[i] = fromf(act_f(tof(ub)) * drop_mult(seed, (unsigned)i, p));
    }
}

// g (grad of a) -> grad of u, in place: g * dropout_mask * act'(u).
extern "C" __global__ void act_bwd(
        T* __restrict__ g, const T* __restrict__ u, int rows, int cols,
        unsigned seed, float p) {
    size_t n = (size_t)rows * cols;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        g[i] = fromf(tof(g[i]) * drop_mult(seed, (unsigned)i, p) * act_grad(tof(u[i])));
    }
}
#endif  // ACT

// ---------------------------------------------------------------------------
// Attention: head layout changes, masked softmax + dropout
// ---------------------------------------------------------------------------

// src (B*L, NP*H*DK) [+ bias] -> dst (NP, B, H, L, DK).  NP = 3 for Q|K|V.
extern "C" __global__ void split_heads(
        const T* __restrict__ src, const T* __restrict__ bias,
        T* __restrict__ dst, int B, int L, int H, int DK, int NP) {
    int dm = H * DK, width = NP * dm;
    size_t n = (size_t)B * L * width;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        int row = (int)(i / width), c = (int)(i % width);
        int part = c / dm, cc = c % dm, head = cc / DK, k = cc % DK;
        int b = row / L, l = row % L;
        float v = tof(src[i]) + (bias ? tof(bias[c]) : 0.f);
        dst[((((size_t)part * B + b) * H + head) * L + l) * DK + k] = fromf(v);
    }
}

// Inverse of split_heads (no bias): src (NP, B, H, L, DK) -> dst (B*L, NP*H*DK).
extern "C" __global__ void merge_heads(
        const T* __restrict__ src, T* __restrict__ dst,
        int B, int L, int H, int DK, int NP) {
    int dm = H * DK, width = NP * dm;
    size_t n = (size_t)B * L * width;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        int row = (int)(i / width), c = (int)(i % width);
        int part = c / dm, cc = c % dm, head = cc / DK, k = cc % DK;
        int b = row / L, l = row % L;
        dst[i] = src[((((size_t)part * B + b) * H + head) * L + l) * DK + k];
    }
}

#ifdef LK
#define KPT ((LK + 31) / 32)

__device__ __forceinline__ bool is_masked(const int* kid, int b, int q, int k,
                                          int pad_id) {
    if (kid[(size_t)b * LK + k] == pad_id) return true;
#if CAUSAL
    if (k > q) return true;
#endif
    return false;
}

// One warp per score row. S: (B*H*LQ, LK) scaled scores.
//   P  = softmax(S with masked keys set to MASK_VALUE)   (saved)
//   Pd = dropout(P)                                       (multiplies V)
// kid: (B, LK) key token ids; a key is masked if it is padding (or in the
// future, when CAUSAL).
extern "C" __global__ void softmax_fwd(
        const T* __restrict__ S, T* __restrict__ P, T* __restrict__ Pd,
        const int* __restrict__ kid, int rows, int LQ, int H, int pad_id,
        unsigned seed, float p) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    int q = row % LQ, b = row / (LQ * H);
    size_t base = (size_t)row * LK;
    float v[KPT];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int k = lane + 32 * j;
        v[j] = -INFINITY;
        if (k < LK) {
            v[j] = is_masked(kid, b, q, k, pad_id) ? MASK_VALUE : tof(S[base + k]);
            mx = fmaxf(mx, v[j]);
        }
    }
    mx = warp_max(mx);
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int k = lane + 32 * j;
        v[j] = (k < LK) ? __expf(v[j] - mx) : 0.f;
        sum += v[j];
    }
    float inv = 1.f / warp_sum(sum);
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int k = lane + 32 * j;
        if (k < LK) {
            float pr = v[j] * inv;
            P[base + k] = fromf(pr);
            Pd[base + k] = fromf(pr * drop_mult(seed, (unsigned)(base + k), p));
        }
    }
}

// dS = P * (dP - sum_k dP*P), with dP = dropout_mask * dPd; masked keys get
// 0 (they were overwritten by a constant). Written in place over dPd.
extern "C" __global__ void softmax_bwd(
        T* __restrict__ dPd, const T* __restrict__ P,
        const int* __restrict__ kid, int rows, int LQ, int H, int pad_id,
        unsigned seed, float p) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    int q = row % LQ, b = row / (LQ * H);
    size_t base = (size_t)row * LK;
    float dp[KPT], pr[KPT];
    float dot = 0.f;
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int k = lane + 32 * j;
        dp[j] = 0.f; pr[j] = 0.f;
        if (k < LK) {
            pr[j] = tof(P[base + k]);
            dp[j] = tof(dPd[base + k]) * drop_mult(seed, (unsigned)(base + k), p);
            dot += dp[j] * pr[j];
        }
    }
    dot = warp_sum(dot);
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int k = lane + 32 * j;
        if (k < LK) {
            float ds = is_masked(kid, b, q, k, pad_id) ? 0.f : pr[j] * (dp[j] - dot);
            dPd[base + k] = fromf(ds);
        }
    }
}
#endif  // LK

// ---------------------------------------------------------------------------
// Loss: label-smoothed cross-entropy, forward and backward in one pass
// ---------------------------------------------------------------------------

// One block per row of logits (N, V). Overwrites the row with
//   dlogits = (softmax - q) * loss_scale / count      (0 on ignored rows)
// and adds loss_row / count to *loss_sum. q is the smoothed target:
//   uniform:    eps/V everywhere, + (1-eps) at the target
//   non_target: (1-eps) at the target, 0 at `pad`, eps/K on the K others
// pad / ignore < 0 mean "none". scalars[0] holds the loss scale.
extern "C" __global__ void cross_entropy(
        T* __restrict__ logits, const int* __restrict__ targets, int V,
        float eps, int non_target, int pad, int ignore,
        const float* __restrict__ scalars, float inv_count,
        float* __restrict__ loss_sum) {
    __shared__ float red[32];
    __shared__ float s_xt, s_xpad;
    int row = blockIdx.x;
    T* x = logits + (size_t)row * V;
    int t = targets[row];
    bool valid = ignore < 0 || t != ignore;
    if (!valid) t = 0;
    float mx = -INFINITY;
    for (int c = threadIdx.x; c < V; c += blockDim.x) mx = fmaxf(mx, tof(x[c]));
    mx = block_max(mx, red);
    float se = 0.f, sx = 0.f;
    for (int c = threadIdx.x; c < V; c += blockDim.x) {
        float v = tof(x[c]);
        se += __expf(v - mx);
        sx += v;
    }
    se = block_sum(se, red);
    sx = block_sum(sx, red);
    if (threadIdx.x == 0) {
        s_xt = tof(x[t]);
        s_xpad = pad >= 0 ? tof(x[pad]) : 0.f;
    }
    __syncthreads();
    float lse = mx + __logf(se);
    float xt = s_xt;
    bool pad_other = pad >= 0 && t != pad;  // pad is a non-target class here
    float w, loss;
    if (non_target) {
        float k = fmaxf((float)(V - 1 - (pad_other ? 1 : 0)), 1.f);
        float others = sx - xt - (pad_other ? s_xpad : 0.f);
        w = eps / k;
        loss = (1.f - eps) * (lse - xt) + eps * (lse - others / k);
    } else {
        w = eps / V;
        loss = (1.f - eps) * (lse - xt) + eps * (lse - sx / V);
    }
    float coef = valid ? scalars[0] * inv_count : 0.f;
    for (int c = threadIdx.x; c < V; c += blockDim.x) {
        float q = w;
        if (non_target) {
            if (c == t) q = 1.f - eps;
            else if (pad_other && c == pad) q = 0.f;
        } else if (c == t) {
            q += 1.f - eps;
        }
        x[c] = fromf((__expf(tof(x[c]) - lse) - q) * coef);
    }
    if (threadIdx.x == 0 && valid) atomicAdd(loss_sum, loss * inv_count);
}

// ---------------------------------------------------------------------------
// Optimizer: gradient norm, loss scaling, AdamW over one flat buffer
// ---------------------------------------------------------------------------
// scalars: [0] loss scale  [1] sum of squared (scaled) grads  [2] loss
//          [3] grad coefficient (unscale * clip)  [4] skip (overflow)
//          [5] step count  [6] 1-b1^t  [7] 1-b2^t  [8] good steps since
//          last scale change  [9] grad norm (unscaled, before clipping)

extern "C" __global__ void reset_scalars(float* s) {
    s[1] = 0.f;
    s[2] = 0.f;
}

extern "C" __global__ void sum_squares(const float* __restrict__ g, int n,
                                       float* __restrict__ out) {
    __shared__ float red[32];
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float v = g[i];
        acc += v * v;
    }
    acc = block_sum(acc, red);
    if (threadIdx.x == 0) atomicAdd(out, acc);
}

// One thread. Decides this step's gradient coefficient, whether to skip
// (non-finite gradients), the bias corrections, and the next loss scale.
extern "C" __global__ void step_prepare(float* s, float b1, float b2,
                                        float max_norm, int dynamic,
                                        int growth_interval) {
    float scale = s[0];
    float inv = 1.f / scale;
    float sq = s[1];
    bool bad = !isfinite(sq);
    float norm = sqrtf(sq) * inv;
    float clip = 1.f;
    if (max_norm > 0.f && norm > max_norm) clip = max_norm / (norm + 1e-6f);
    s[3] = inv * clip;
    s[4] = bad ? 1.f : 0.f;
    s[9] = norm;
    if (!bad) {
        s[5] += 1.f;
        s[6] = 1.f - powf(b1, s[5]);
        s[7] = 1.f - powf(b2, s[5]);
    }
    if (dynamic) {
        if (bad) {
            s[0] = fmaxf(scale * 0.5f, 1.f);
            s[8] = 0.f;
        } else if (s[8] + 1.f >= (float)growth_interval) {
            s[0] = fminf(scale * 2.f, 16777216.f);
            s[8] = 0.f;
        } else {
            s[8] += 1.f;
        }
    }
}

// AdamW (decoupled=1) or Adam with L2 penalty (decoupled=0) over all
// parameters at once. Writes the fp32 master weights and the working copy.
extern "C" __global__ void adamw(
        float* p, const float* __restrict__ g, float* __restrict__ m,
        float* __restrict__ v, T* p_work, int n, const float* __restrict__ s,
        float lr, float b1, float b2, float eps, float wd, int decoupled) {
    if (s[4] != 0.f) return;  // overflow: skip this step
    float coef = s[3], bc1 = s[6], bc2 = s[7];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float pi = p[i];
        float gi = g[i] * coef;
        if (!decoupled) gi += wd * pi;
        float mi = b1 * m[i] + (1.f - b1) * gi;
        float vi = b2 * v[i] + (1.f - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        if (decoupled) pi *= 1.f - lr * wd;
        pi -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
        p[i] = pi;
        p_work[i] = fromf(pi);
    }
}
