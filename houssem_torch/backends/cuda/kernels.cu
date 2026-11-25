// houssem_torch GPU kernels for the compiled training step.
//
// Compiled at startup with NVRTC (driver.compile_kernel). Compile-time
// constants, set per run from the config (each section below is compiled
// only when its constants are defined):
//   USE_HALF   1: activations / working weights are fp16, 0: fp32
//   D, VW      d_model (multiple of 32) and the vector width (elements per
//              load) of the row kernels add_ln_fwd / ln_bwd
//   LN_WARPS   warps per block in ln_bwd
//   MODE       add_ln_fwd: 0 = embedding input, 1 = residual input
//   EVW, ACT   vector width and activation (0 relu, 1 gelu-tanh) of the
//              feed-forward kernels
//   LK, CAUSAL keys per row and causal masking (unfused softmax)
//   CE_V, CE_T, CE_VW  vocabulary, threads, vector width of the
//              register-resident cross-entropy
//   ATT, AQ, AK, AQP, AKP, DK, NW  fused attention: query / key lengths,
//              both padded to 16, head size, warps per block
//
// All reductions (LayerNorm statistics, softmax, loss, norms) and every
// gradient accumulated across rows run in fp32.
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
__device__ __forceinline__ float roundT(float x) { return tof(fromf(x)); }

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

// Vector loads / stores of N consecutive elements (16 bytes for 8 x fp16 or
// 4 x fp32), converted to / from fp32 registers. p must be N-aligned.
template <int N>
__device__ __forceinline__ void load_vec(const T* __restrict__ p, float* f) {
#if USE_HALF
    if constexpr (N == 8) {
        uint4 u = *reinterpret_cast<const uint4*>(p);
        const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 t = __half22float2(h[i]);
            f[2 * i] = t.x; f[2 * i + 1] = t.y;
        }
    } else if constexpr (N == 4) {
        uint2 u = *reinterpret_cast<const uint2*>(p);
        const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float2 t = __half22float2(h[i]);
            f[2 * i] = t.x; f[2 * i + 1] = t.y;
        }
    } else if constexpr (N == 2) {
        float2 t = __half22float2(*reinterpret_cast<const __half2*>(p));
        f[0] = t.x; f[1] = t.y;
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) f[i] = tof(p[i]);
    }
#else
    if constexpr (N == 4) {
        float4 u = *reinterpret_cast<const float4*>(p);
        f[0] = u.x; f[1] = u.y; f[2] = u.z; f[3] = u.w;
    } else if constexpr (N == 2) {
        float2 u = *reinterpret_cast<const float2*>(p);
        f[0] = u.x; f[1] = u.y;
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) f[i] = p[i];
    }
#endif
}

template <int N>
__device__ __forceinline__ void store_vec(T* __restrict__ p, const float* f) {
#if USE_HALF
    if constexpr (N == 8 || N == 4 || N == 2) {
        __half2 h[N / 2];
#pragma unroll
        for (int i = 0; i < N / 2; ++i) h[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
        if constexpr (N == 8) *reinterpret_cast<uint4*>(p) = *reinterpret_cast<uint4*>(h);
        else if constexpr (N == 4) *reinterpret_cast<uint2*>(p) = *reinterpret_cast<uint2*>(h);
        else *reinterpret_cast<__half2*>(p) = h[0];
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) p[i] = fromf(f[i]);
    }
#else
    if constexpr (N == 4) {
        *reinterpret_cast<float4*>(p) = make_float4(f[0], f[1], f[2], f[3]);
    } else if constexpr (N == 2) {
        *reinterpret_cast<float2*>(p) = make_float2(f[0], f[1]);
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) p[i] = f[i];
    }
#endif
}

template <int N>
__device__ __forceinline__ void load_f32(const float* __restrict__ p, float* f) {
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int i = 0; i < N; i += 4) {
            float4 u = *reinterpret_cast<const float4*>(p + i);
            f[i] = u.x; f[i + 1] = u.y; f[i + 2] = u.z; f[i + 3] = u.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) f[i] = p[i];
    }
}

// ---------------------------------------------------------------------------
// Residual stream: (embedding | residual + dropout) fused with LayerNorm
// ---------------------------------------------------------------------------
#if defined(D) && defined(VW)
#define VPT (D / 32)          // values per lane
#define NV (VPT / VW)         // vector chunks per lane
// Lane `lane` owns columns (j*32 + lane)*VW + e, j < NV, e < VW: each warp
// load is 32 consecutive vectors (coalesced, 16 bytes per lane).

#ifdef MODE
// One warp per row of width D.
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
    for (int j = 0; j < NV; ++j) {
        int c = (j * 32 + lane) * VW;
        float a[VW];
#if MODE == 0
        float t[VW], pp[VW];
        load_vec<VW>(trow + c, t);
        load_f32<VW>(prow + c, pp);
#pragma unroll
        for (int e = 0; e < VW; ++e)
            a[e] = (t[e] * emb_scale + pp[e]) * drop_mult(seed, (unsigned)(base + c + e), p);
#else
        float xx[VW], yy[VW], bb[VW];
        load_vec<VW>(x_in + base + c, xx);
        load_vec<VW>(y + base + c, yy);
        if (ybias) load_vec<VW>(ybias + c, bb);
        else {
#pragma unroll
            for (int e = 0; e < VW; ++e) bb[e] = 0.f;
        }
#pragma unroll
        for (int e = 0; e < VW; ++e)
            a[e] = xx[e] + (yy[e] + bb[e]) * drop_mult(seed, (unsigned)(base + c + e), p);
#endif
        store_vec<VW>(x_out + base + c, a);
#pragma unroll
        for (int e = 0; e < VW; ++e) {
            v[j * VW + e] = roundT(a[e]);  // the value backward will read
            sum += v[j * VW + e];
        }
    }
    float mean = warp_sum(sum) * (1.f / D);
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) { float t = v[i] - mean; sq += t * t; }
    float rstd = rsqrtf(warp_sum(sq) * (1.f / D) + eps);
#pragma unroll
    for (int j = 0; j < NV; ++j) {
        int c = (j * 32 + lane) * VW;
        float ww[VW], bb[VW], o[VW];
        load_vec<VW>(w + c, ww);
        load_vec<VW>(b + c, bb);
#pragma unroll
        for (int e = 0; e < VW; ++e) o[e] = (v[j * VW + e] - mean) * rstd * ww[e] + bb[e];
        store_vec<VW>(h + base + c, o);
    }
    if (lane == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
}
#endif  // MODE

#ifdef LN_WARPS
// Sum per-warp column partials over the block, one atomic per column.
__device__ __forceinline__ void block_column_add(float (*s)[D], const float* part,
                                                 float* __restrict__ out) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < NV; ++j)
#pragma unroll
        for (int e = 0; e < VW; ++e) s[warp][(j * 32 + lane) * VW + e] = part[j * VW + e];
    __syncthreads();
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float a = 0.f;
#pragma unroll
        for (int k = 0; k < LN_WARPS; ++k) a += s[k][c];
        atomicAdd(out + c, a);
    }
    __syncthreads();
}

// Backward of one LayerNorm site of the residual stream, fused with the
// dropout of the residual branch that produced its input:
//   g_x  = (accum ? g_x : 0) + LayerNormBackward(g_h)
//   dw  += sum_rows g_h * x_hat,   db += sum_rows g_h       (fp32, atomics)
//   if g_y_prev: g_y_prev = dropout_mask(seed_prev) * g_x  (grad of the
//                previous sublayer's output, before its bias), and
//                db_prev += sum_rows g_y_prev   (that bias's gradient)
// Each warp walks rows with a grid stride, keeping its column partial sums
// in registers; the block combines them in shared memory and issues one
// atomic per column.
extern "C" __global__ void ln_bwd(
        const T* __restrict__ g_h, const T* __restrict__ x,
        const float* __restrict__ mean, const float* __restrict__ rstd,
        const T* __restrict__ w, T* __restrict__ g_x, int accum,
        float* __restrict__ dw, float* __restrict__ db,
        T* __restrict__ g_y_prev, float* __restrict__ db_prev,
        unsigned seed_prev, float p_prev, int rows) {
    __shared__ float s_red[LN_WARPS][D];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float pdw[VPT], pdb[VPT], pdy[VPT], wv[VPT];
#pragma unroll
    for (int j = 0; j < NV; ++j) load_vec<VW>(w + (j * 32 + lane) * VW, wv + j * VW);
#pragma unroll
    for (int i = 0; i < VPT; ++i) { pdw[i] = 0.f; pdb[i] = 0.f; pdy[i] = 0.f; }
    int stride = gridDim.x * LN_WARPS;
    for (int row = blockIdx.x * LN_WARPS + warp; row < rows; row += stride) {
        size_t base = (size_t)row * D;
        float mu = mean[row], rs = rstd[row];
        float xh[VPT], gh[VPT];
        float s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int j = 0; j < NV; ++j) {
            int c = (j * 32 + lane) * VW;
            float gg[VW], xx[VW];
            load_vec<VW>(g_h + base + c, gg);
            load_vec<VW>(x + base + c, xx);
#pragma unroll
            for (int e = 0; e < VW; ++e) {
                int i = j * VW + e;
                xh[i] = (xx[e] - mu) * rs;
                pdb[i] += gg[e];
                pdw[i] += gg[e] * xh[i];
                gh[i] = gg[e] * wv[i];
                s1 += gh[i];
                s2 += gh[i] * xh[i];
            }
        }
        s1 = warp_sum(s1) * (1.f / D);
        s2 = warp_sum(s2) * (1.f / D);
#pragma unroll
        for (int j = 0; j < NV; ++j) {
            int c = (j * 32 + lane) * VW;
            float dx[VW], acc[VW];
            if (accum) load_vec<VW>(g_x + base + c, acc);
#pragma unroll
            for (int e = 0; e < VW; ++e) {
                int i = j * VW + e;
                dx[e] = rs * (gh[i] - s1 - xh[i] * s2) + (accum ? acc[e] : 0.f);
            }
            store_vec<VW>(g_x + base + c, dx);
            if (g_y_prev) {
                float gy[VW];
#pragma unroll
                for (int e = 0; e < VW; ++e) {
                    gy[e] = roundT(roundT(dx[e]) *
                                   drop_mult(seed_prev, (unsigned)(base + c + e), p_prev));
                    pdy[j * VW + e] += gy[e];
                }
                store_vec<VW>(g_y_prev + base + c, gy);
            }
        }
    }
    block_column_add(s_red, pdw, dw);
    block_column_add(s_red, pdb, db);
    if (db_prev) block_column_add(s_red, pdy, db_prev);
}
#endif  // LN_WARPS
#endif  // D && VW

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
#if defined(ACT) && defined(EVW)
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

// Row-blocked: thread i owns columns [i*EVW, i*EVW + EVW) (blockDim.x =
// cols / EVW), block b owns rows [b*rpb, (b+1)*rpb).
//   u += bias (kept for backward);  a = dropout(act(u)).
extern "C" __global__ void bias_act_fwd(
        T* __restrict__ u, const T* __restrict__ bias, T* __restrict__ a,
        int rows, int cols, int rpb, unsigned seed, float p) {
    int c = threadIdx.x * EVW;
    if (c >= cols) return;
    float bb[EVW];
    load_vec<EVW>(bias + c, bb);
    int r1 = min(rows, (blockIdx.x + 1) * rpb);
    for (int r = blockIdx.x * rpb; r < r1; ++r) {
        size_t i = (size_t)r * cols + c;
        float uu[EVW], aa[EVW];
        load_vec<EVW>(u + i, uu);
#pragma unroll
        for (int e = 0; e < EVW; ++e) {
            uu[e] = roundT(uu[e] + bb[e]);
            aa[e] = act_f(uu[e]) * drop_mult(seed, (unsigned)(i + e), p);
        }
        store_vec<EVW>(u + i, uu);
        store_vec<EVW>(a + i, aa);
    }
}

// g (grad of a) -> grad of u, in place: g * dropout_mask * act'(u), and
// dbias += column sums of the result (the bias gradient). Same layout as
// bias_act_fwd.
extern "C" __global__ void act_bwd(
        T* __restrict__ g, const T* __restrict__ u, float* __restrict__ dbias,
        int rows, int cols, int rpb, unsigned seed, float p) {
    int c = threadIdx.x * EVW;
    if (c >= cols) return;
    float part[EVW];
#pragma unroll
    for (int e = 0; e < EVW; ++e) part[e] = 0.f;
    int r1 = min(rows, (blockIdx.x + 1) * rpb);
    for (int r = blockIdx.x * rpb; r < r1; ++r) {
        size_t i = (size_t)r * cols + c;
        float gg[EVW], uu[EVW];
        load_vec<EVW>(g + i, gg);
        load_vec<EVW>(u + i, uu);
#pragma unroll
        for (int e = 0; e < EVW; ++e) {
            gg[e] = roundT(gg[e] * drop_mult(seed, (unsigned)(i + e), p) * act_grad(uu[e]));
            part[e] += gg[e];
        }
        store_vec<EVW>(g + i, gg);
    }
    if (dbias) {
#pragma unroll
        for (int e = 0; e < EVW; ++e) atomicAdd(dbias + c + e, part[e]);
    }
}
#endif  // ACT && EVW

// ---------------------------------------------------------------------------
// Attention, unfused path (fp32, or shapes the fused kernels don't cover)
// ---------------------------------------------------------------------------

// src (B*L, NP*H*DK) [+ bias] -> dst (NP, B, H, L, DK).  NP = 3 for Q|K|V.
extern "C" __global__ void split_heads(
        const T* __restrict__ src, const T* __restrict__ bias,
        T* __restrict__ dst, int B, int L, int H, int DKr, int NP) {
    int dm = H * DKr, width = NP * dm;
    size_t n = (size_t)B * L * width;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        int row = (int)(i / width), c = (int)(i % width);
        int part = c / dm, cc = c % dm, head = cc / DKr, k = cc % DKr;
        int b = row / L, l = row % L;
        float v = tof(src[i]) + (bias ? tof(bias[c]) : 0.f);
        dst[((((size_t)part * B + b) * H + head) * L + l) * DKr + k] = fromf(v);
    }
}

// Inverse of split_heads (no bias): src (NP, B, H, L, DK) -> dst (B*L, NP*H*DK).
extern "C" __global__ void merge_heads(
        const T* __restrict__ src, T* __restrict__ dst,
        int B, int L, int H, int DKr, int NP) {
    int dm = H * DKr, width = NP * dm;
    size_t n = (size_t)B * L * width;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        int row = (int)(i / width), c = (int)(i % width);
        int part = c / dm, cc = c % dm, head = cc / DKr, k = cc % DKr;
        int b = row / L, l = row % L;
        dst[i] = src[((((size_t)part * B + b) * H + head) * L + l) * DKr + k];
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
// Attention, fused (fp16, tensor cores): one block per (batch, head)
// ---------------------------------------------------------------------------
// Q/K/V are read straight from the projection outputs (row stride qs/ks/vs,
// head h at column h*DK, bias added on load), the context is written in
// merged (B*AQ, H*DK) layout. Nothing of size AQ x AK touches global memory:
// forward keeps scores and probabilities in registers and saves the row max
// and log(sum of exp) per row (two floats: as one log-sum-exp, a fully
// masked row's -1e9 + log(AK) would round to -1e9 in fp32); backward
// recomputes the probabilities from them.
//
// Matrix products use mma.sync.m16n8k8 (Turing fp16 tensor cores, fp32
// accumulation). Per thread (lane = 4*g + t): A fragment (16x8, row-major)
// = rows g, g+8 x cols 2t, 2t+1; B fragment (8x8) = rows 2t, 2t+1 x col g;
// C fragment (16x8) = rows g, g+8 x cols 2t, 2t+1. A C fragment of one
// product is directly the A fragment of the next (FlashAttention-2 style).
#ifdef ATT
#define LDS (DK + 8)      // shared row stride (halves) for Q, K, V, dO
#define LDP (AKP + 8)     // shared row stride (halves) for dS, Pd: [query][key]
#define NT_K (AKP / 8)    // 8-wide key tiles
#define NT_D (DK / 8)     // 8-wide head-dim tiles
// The +8 padding makes consecutive rows start 4 banks apart, so the 8 row
// addresses of every ldmatrix 8x8 load hit all 32 banks once.

__device__ __forceinline__ unsigned pack2(__half a, __half b) {
    return (unsigned)__half_as_ushort(a) | ((unsigned)__half_as_ushort(b) << 16);
}

__device__ __forceinline__ unsigned pack2f(float a, float b) {
    return pack2(__float2half(a), __float2half(b));
}

__device__ __forceinline__ void mma16816(float* c, unsigned a0, unsigned a1,
                                         unsigned b0) {
    // Not volatile: a pure register op, so the compiler may interleave
    // independent products instead of serializing every accumulator chain.
    asm(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// ldmatrix: lanes 8i..8i+7 give the row addresses of 8x8 matrix i; lane
// 4g+t receives row g, cols 2t, 2t+1 of each matrix (.trans: rows 2t, 2t+1,
// col g, i.e. of the transposed matrix).
__device__ __forceinline__ unsigned smem_addr(const __half* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

template <bool T>
__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2,
                                      unsigned& r3, const __half* p) {
    if constexpr (T)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}

template <bool T>
__device__ __forceinline__ void ldsm2(unsigned& r0, unsigned& r1, const __half* p) {
    if constexpr (T)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(r0), "=r"(r1) : "r"(smem_addr(p)));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(r0), "=r"(r1) : "r"(smem_addr(p)));
}

template <bool T>
__device__ __forceinline__ void ldsm1(unsigned& r0, const __half* p) {
    if constexpr (T)
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                     : "=r"(r0) : "r"(smem_addr(p)));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                     : "=r"(r0) : "r"(smem_addr(p)));
}

// B fragments of N side-by-side 8x8 blocks: out[i] = block (rows 0..7,
// cols 8i..8i+7) of the shared matrix at p (row stride ld). Without T the
// block is stored [n][k] (K rows for Q K^T); with T it is stored [k][n]
// (V rows for P V).
template <int N, bool T>
__device__ __forceinline__ void ldsm_b(unsigned* out, const __half* p, int ld) {
    int lane = threadIdx.x & 31, r = lane & 7, m = lane >> 3;
#pragma unroll
    for (int i = 0; i + 4 <= N; i += 4)
        ldsm4<T>(out[i], out[i + 1], out[i + 2], out[i + 3], p + r * ld + 8 * (i + m));
    if constexpr (N % 4 >= 2) {
        constexpr int i = N / 4 * 4;
        ldsm2<T>(out[i], out[i + 1], p + r * ld + 8 * (i + (m & 1)));
    }
    if constexpr (N % 2) {
        constexpr int i = N - 1;
        ldsm1<T>(out[i], p + r * ld + 8 * i);
    }
}

// A fragments of a 16-row tile stored row-major at p: a[i] = (rows 0..15,
// cols 8i..8i+7).
template <int N>
__device__ __forceinline__ void ldsm_a(unsigned (*a)[2], const __half* p, int ld) {
    int lane = threadIdx.x & 31, r = (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int i = 0; i + 2 <= N; i += 2)
        ldsm4<false>(a[i][0], a[i][1], a[i + 1][0], a[i + 1][1],
                     p + r * ld + 8 * (i + (lane >> 4)));
    if constexpr (N % 2) {
        constexpr int i = N - 1;
        ldsm2<false>(a[i][0], a[i][1], p + r * ld + 8 * i);
    }
}

// A fragments of S^T for S stored row-major at p ([k][m], 16 columns m):
// a[i] = A rows m 0..15, k-step i (S rows 8i..8i+7).
template <int N>
__device__ __forceinline__ void ldsm_at(unsigned (*a)[2], const __half* p, int ld) {
    int lane = threadIdx.x & 31, c = 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int i = 0; i + 2 <= N; i += 2)
        ldsm4<true>(a[i][0], a[i][1], a[i + 1][0], a[i + 1][1],
                    p + (8 * i + (lane & 7) + 8 * (lane >> 4)) * ld + c);
    if constexpr (N % 2) {
        constexpr int i = N - 1;
        ldsm2<true>(a[i][0], a[i][1], p + (8 * i + (lane & 7)) * ld + c);
    }
}

// Load one head's rows [0, L) into shared dst[LP][LDS] (rows >= L zeroed),
// adding the bias if given.
__device__ __forceinline__ void load_head(__half* dst, const __half* src,
                                          int stride, const __half* bias,
                                          int b, int h, int L, int LP) {
    for (int i = threadIdx.x; i < LP * (DK / 8); i += blockDim.x) {
        int r = i / (DK / 8), c = (i % (DK / 8)) * 8;
        float f[8];
        if (r < L) {
            load_vec<8>(src + (size_t)(b * L + r) * stride + h * DK + c, f);
            if (bias) {
                float bb[8];
                load_vec<8>(bias + h * DK + c, bb);
#pragma unroll
                for (int e = 0; e < 8; ++e) f[e] += bb[e];
            }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) f[e] = 0.f;
        }
        store_vec<8>(dst + r * LDS + c, f);
    }
}

// key_state: 0 = visible, 1 = padding (MASK_VALUE), 2 = beyond AK (-inf)
__device__ __forceinline__ void load_key_state(int* ks, const int* kid, int b,
                                               int pad_id) {
    for (int j = threadIdx.x; j < AKP; j += blockDim.x)
        ks[j] = j >= AK ? 2 : (kid[(size_t)b * AK + j] == pad_id ? 1 : 0);
}

// Scaled, masked scores of a 16-row query tile (A fragments qa) against key
// tile j: element e is row m0+g (+8 if e >= 2), key 8j+2t+(e&1).
__device__ __forceinline__ void tile_score(float* s, const unsigned (*qa)[2],
                                           const __half* sK, const int* kst,
                                           int j, int m0, int g, int t,
                                           float alpha) {
    unsigned kb[NT_D];
    ldsm_b<NT_D, false>(kb, sK + 8 * j * LDS, LDS);
    s[0] = s[1] = s[2] = s[3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < NT_D; ++kk) mma16816(s, qa[kk][0], qa[kk][1], kb[kk]);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        int col = 8 * j + 2 * t + (e & 1);
        int row = m0 + g + (e >= 2 ? 8 : 0);
        int st = kst[col];
        bool masked = st == 1;
#if CAUSAL
        masked = masked || col > row;
#endif
        s[e] = st == 2 ? -INFINITY : (masked ? MASK_VALUE : s[e] * alpha);
    }
}

extern "C" __global__ void attn_fwd(
        const __half* __restrict__ q, int qs, const __half* __restrict__ qb,
        const __half* __restrict__ k, int ks, const __half* __restrict__ kb,
        const __half* __restrict__ v, int vs, const __half* __restrict__ vb,
        const int* __restrict__ kid, __half* __restrict__ out, int os,
        float* __restrict__ lse, int H, int pad_id, unsigned seed, float p,
        float alpha) {
    extern __shared__ __align__(16) unsigned char smem[];
    __half* sQ = reinterpret_cast<__half*>(smem);
    __half* sK = sQ + AQP * LDS;
    __half* sV = sK + AKP * LDS;
    __shared__ int kst[AKP];
    int bh = blockIdx.x, b = bh / H, h = bh % H;
    load_head(sQ, q, qs, qb, b, h, AQ, AQP);
    load_head(sK, k, ks, kb, b, h, AK, AKP);
    load_head(sV, v, vs, vb, b, h, AK, AKP);
    load_key_state(kst, kid, b, pad_id);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    for (int m0 = warp * 16; m0 < AQP; m0 += NW * 16) {
        unsigned qa[NT_D][2];
        ldsm_a<NT_D>(qa, sQ + m0 * LDS, LDS);
        float s[NT_K][4];
#pragma unroll
        for (int j = 0; j < NT_K; ++j) tile_score(s[j], qa, sK, kst, j, m0, g, t, alpha);
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < NT_K; ++j) {
            mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
            mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
        }
        for (int o = 1; o <= 2; o <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
        }
        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int j = 0; j < NT_K; ++j) {
            s[j][0] = __expf(s[j][0] - mx0); s[j][1] = __expf(s[j][1] - mx0);
            s[j][2] = __expf(s[j][2] - mx1); s[j][3] = __expf(s[j][3] - mx1);
            sum0 += s[j][0] + s[j][1];
            sum1 += s[j][2] + s[j][3];
        }
        for (int o = 1; o <= 2; o <<= 1) {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, o);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, o);
        }
        float inv0 = 1.f / sum0, inv1 = 1.f / sum1;
        int r0 = m0 + g, r1 = m0 + g + 8;
        // O = dropout(P) @ V; the score tile is the A operand directly.
        float o[NT_D][4];
#pragma unroll
        for (int nd = 0; nd < NT_D; ++nd) o[nd][0] = o[nd][1] = o[nd][2] = o[nd][3] = 0.f;
#pragma unroll
        for (int j = 0; j < NT_K; ++j) {
            int c0 = 8 * j + 2 * t;
            size_t i0 = ((size_t)bh * AQ + r0) * AK + c0;
            size_t i1 = ((size_t)bh * AQ + r1) * AK + c0;
            bool v0 = r0 < AQ, v1 = r1 < AQ;
            float p00 = s[j][0] * inv0 * (v0 && c0 < AK ? drop_mult(seed, (unsigned)i0, p) : 0.f);
            float p01 = s[j][1] * inv0 * (v0 && c0 + 1 < AK ? drop_mult(seed, (unsigned)(i0 + 1), p) : 0.f);
            float p10 = s[j][2] * inv1 * (v1 && c0 < AK ? drop_mult(seed, (unsigned)i1, p) : 0.f);
            float p11 = s[j][3] * inv1 * (v1 && c0 + 1 < AK ? drop_mult(seed, (unsigned)(i1 + 1), p) : 0.f);
            unsigned a0 = pack2f(p00, p01), a1 = pack2f(p10, p11);
            unsigned vfr[NT_D];
            ldsm_b<NT_D, true>(vfr, sV + 8 * j * LDS, LDS);
#pragma unroll
            for (int nd = 0; nd < NT_D; ++nd) mma16816(o[nd], a0, a1, vfr[nd]);
        }
#pragma unroll
        for (int nd = 0; nd < NT_D; ++nd) {
            int c = h * DK + 8 * nd + 2 * t;
            if (r0 < AQ)
                *reinterpret_cast<__half2*>(out + (size_t)(b * AQ + r0) * os + c) =
                    __floats2half2_rn(o[nd][0], o[nd][1]);
            if (r1 < AQ)
                *reinterpret_cast<__half2*>(out + (size_t)(b * AQ + r1) * os + c) =
                    __floats2half2_rn(o[nd][2], o[nd][3]);
        }
        if (t == 0) {
            if (r0 < AQ) {
                lse[2 * ((size_t)bh * AQ + r0)] = mx0;
                lse[2 * ((size_t)bh * AQ + r0) + 1] = __logf(sum0);
            }
            if (r1 < AQ) {
                lse[2 * ((size_t)bh * AQ + r1)] = mx1;
                lse[2 * ((size_t)bh * AQ + r1) + 1] = __logf(sum1);
            }
        }
    }
}

// Column sums of a C-fragment accumulator (N 8-wide tiles) over the tile's
// valid rows, added to shared s_col (bias gradients).
template <int N>
__device__ __forceinline__ void add_col_sums(float* s_col, const float (*acc)[4],
                                             bool v0, bool v1, int t) {
#pragma unroll
    for (int nd = 0; nd < N; ++nd) {
        float c0 = (v0 ? acc[nd][0] : 0.f) + (v1 ? acc[nd][2] : 0.f);
        float c1 = (v0 ? acc[nd][1] : 0.f) + (v1 ? acc[nd][3] : 0.f);
        for (int o = 4; o <= 16; o <<= 1) {
            c0 += __shfl_xor_sync(0xffffffffu, c0, o);
            c1 += __shfl_xor_sync(0xffffffffu, c1, o);
        }
        if ((threadIdx.x & 31) < 4) {
            atomicAdd(s_col + 8 * nd + 2 * t, c0);
            atomicAdd(s_col + 8 * nd + 2 * t + 1, c1);
        }
    }
}

// Rows r0, r1 of an accumulator (N 8-wide tiles from column c) to global.
template <int N>
__device__ __forceinline__ void store_rows(__half* dst, int stride, int b, int L,
                                           int c, const float (*acc)[4], int r0,
                                           int r1, int t, float scale) {
#pragma unroll
    for (int nd = 0; nd < N; ++nd) {
        int cc = c + 8 * nd + 2 * t;
        if (r0 < L)
            *reinterpret_cast<__half2*>(dst + (size_t)(b * L + r0) * stride + cc) =
                __floats2half2_rn(acc[nd][0] * scale, acc[nd][1] * scale);
        if (r1 < L)
            *reinterpret_cast<__half2*>(dst + (size_t)(b * L + r1) * stride + cc) =
                __floats2half2_rn(acc[nd][2] * scale, acc[nd][3] * scale);
    }
}

// How many ways to split the NT_D head-dim tiles so `want` warps share them.
constexpr int att_dsplit(int want) {
    return want >= 8 && NT_D % 8 == 0 ? 8
         : want >= 4 && NT_D % 4 == 0 ? 4
         : want >= 2 && NT_D % 2 == 0 ? 2 : 1;
}

// How many ways to split the NT_K key tiles (NT_K is even) among `want` warps.
constexpr int att_ksplit(int want) {
    return want >= 8 && NT_K % 8 == 0 ? 8
         : want >= 4 && NT_K % 4 == 0 ? 4
         : want >= 2 ? 2 : 1;
}

// Backward, NWB warps. o: the forward's context (merged layout), dout: its
// gradient. Writes dq / dk / dv (merged layouts, every valid row) and adds
// the bias gradients (column sums) to dqb / dkb / dvb if given.
//   phase 1, warps own (query tile, every KS-th key tile): recompute P from
//            lse; dPd = dO V^T; dS = P (mask*dPd - D) with D = rowsum(dO*O);
//            dS and dropout(P) go to shared memory.
//   phase 2, warps own (query tile, head-dim slice): dQ = a*dS K, and
//            (key tile, head-dim slice): dV = Pd^T dO, dK = a*dS^T Q.
extern "C" __global__ void attn_bwd(
        const __half* __restrict__ q, int qs, const __half* __restrict__ qb,
        const __half* __restrict__ k, int ks, const __half* __restrict__ kb,
        const __half* __restrict__ v, int vs, const __half* __restrict__ vb,
        const int* __restrict__ kid,
        const __half* __restrict__ o, int os, const __half* __restrict__ dout,
        int dos, const float* __restrict__ lse,
        __half* __restrict__ dq, int dqs, __half* __restrict__ dk, int dks,
        __half* __restrict__ dv, int dvs,
        float* __restrict__ dqb, float* __restrict__ dkb, float* __restrict__ dvb,
        int H, int pad_id, unsigned seed, float p, float alpha) {
    constexpr int QT = AQP / 16, KT = AKP / 16;
    constexpr int KS = att_ksplit(NWB / QT);
    constexpr int DSQ = att_dsplit(NWB / QT), NDQ = NT_D / DSQ;
    constexpr int DSK = att_dsplit(NWB / KT), NDK = NT_D / DSK;
    constexpr int NQ = QT * DSQ;
    extern __shared__ __align__(16) unsigned char smem[];
    __half* sQ = reinterpret_cast<__half*>(smem);
    __half* sK = sQ + AQP * LDS;
    __half* sV = sK + AKP * LDS;
    __half* sdO = sV + AKP * LDS;
    __half* sdS = sdO + AQP * LDS;
    __half* sPd = sdS + AQP * LDP;
    __shared__ int kst[AKP];
    __shared__ float sD[AQP];
    __shared__ float s_bias[3][DK];
    int bh = blockIdx.x, b = bh / H, h = bh % H;
    load_head(sQ, q, qs, qb, b, h, AQ, AQP);
    load_head(sK, k, ks, kb, b, h, AK, AKP);
    load_head(sV, v, vs, vb, b, h, AK, AKP);
    load_head(sdO, dout, dos, nullptr, b, h, AQ, AQP);
    load_key_state(kst, kid, b, pad_id);
    for (int i = threadIdx.x; i < 3 * DK; i += blockDim.x) (&s_bias[0][0])[i] = 0.f;
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    // D[r] = dO[r] * O[r] (dO from shared, rounded to fp16 like O)
    for (int r = warp; r < AQP; r += NWB) {
        float acc = 0.f;
        if (r < AQ)
            for (int c = 2 * lane; c < DK; c += 64) {
                float2 of = __half22float2(*reinterpret_cast<const __half2*>(
                    o + (size_t)(b * AQ + r) * os + h * DK + c));
                float2 df = __half22float2(*reinterpret_cast<const __half2*>(
                    sdO + r * LDS + c));
                acc += of.x * df.x + of.y * df.y;
            }
        acc = warp_sum(acc);
        if (lane == 0) sD[r] = acc;
    }
    __syncthreads();

    // ---- phase 1: dS and dropout(P), (query tile, key tiles ks, ks+KS, ...)
    for (int it = warp; it < QT * KS; it += NWB) {
        int m0 = (it % QT) * 16, kq = it / QT;
        int r0 = m0 + g, r1 = m0 + g + 8;
        bool v0 = r0 < AQ, v1 = r1 < AQ;
        float m0r = v0 ? lse[2 * ((size_t)bh * AQ + r0)] : 0.f;
        float m1r = v1 ? lse[2 * ((size_t)bh * AQ + r1)] : 0.f;
        float l0 = v0 ? lse[2 * ((size_t)bh * AQ + r0) + 1] : 0.f;
        float l1 = v1 ? lse[2 * ((size_t)bh * AQ + r1) + 1] : 0.f;
        float D0 = sD[r0], D1 = sD[r1];
        unsigned qa[NT_D][2], da[NT_D][2];
        ldsm_a<NT_D>(qa, sQ + m0 * LDS, LDS);
        ldsm_a<NT_D>(da, sdO + m0 * LDS, LDS);
#pragma unroll
        for (int jj = 0; jj < NT_K / KS; ++jj) {
            int j = kq + KS * jj;
            float s[4];
            tile_score(s, qa, sK, kst, j, m0, g, t, alpha);
            unsigned vfr[NT_D];
            ldsm_b<NT_D, false>(vfr, sV + 8 * j * LDS, LDS);
            float dpd[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < NT_D; ++kk) mma16816(dpd, da[kk][0], da[kk][1], vfr[kk]);
            float ds[4], pd[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int col = 8 * j + 2 * t + (e & 1);
                int row = e >= 2 ? r1 : r0;
                bool valid = (e >= 2 ? v1 : v0) && col < AK;
                float pr = valid ? __expf((s[e] - (e >= 2 ? m1r : m0r)) -
                                          (e >= 2 ? l1 : l0)) : 0.f;
                float m = valid ? drop_mult(seed, (unsigned)(((size_t)bh * AQ + row) * AK + col), p)
                                : 0.f;
                bool masked = kst[col] == 1;
#if CAUSAL
                masked = masked || col > row;
#endif
                ds[e] = (valid && !masked) ? pr * (m * dpd[e] - (e >= 2 ? D1 : D0)) : 0.f;
                pd[e] = pr * m;
            }
            int c0 = 8 * j + 2 * t;
            *reinterpret_cast<__half2*>(sdS + r0 * LDP + c0) = __floats2half2_rn(ds[0], ds[1]);
            *reinterpret_cast<__half2*>(sdS + r1 * LDP + c0) = __floats2half2_rn(ds[2], ds[3]);
            *reinterpret_cast<__half2*>(sPd + r0 * LDP + c0) = __floats2half2_rn(pd[0], pd[1]);
            *reinterpret_cast<__half2*>(sPd + r1 * LDP + c0) = __floats2half2_rn(pd[2], pd[3]);
        }
    }
    __syncthreads();

    // ---- phase 2: dQ (query tile, head-dim slice), dK / dV (key tile, slice)
    for (int it = warp; it < NQ + KT * DSK; it += NWB) {
        if (it < NQ) {
            int m0 = (it % QT) * 16, d0 = (it / QT) * NDQ;
            int r0 = m0 + g, r1 = m0 + g + 8;
            float acc[NDQ][4];
#pragma unroll
            for (int i = 0; i < NDQ; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
            for (int k0 = 0; k0 < AKP; k0 += 16) {      // keys, two 8-steps
                unsigned a[2][2];
                ldsm_a<2>(a, sdS + m0 * LDP + k0, LDP);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    unsigned bf[NDQ];
                    ldsm_b<NDQ, true>(bf, sK + (k0 + 8 * kk) * LDS + 8 * d0, LDS);
#pragma unroll
                    for (int i = 0; i < NDQ; ++i) mma16816(acc[i], a[kk][0], a[kk][1], bf[i]);
                }
            }
#pragma unroll
            for (int i = 0; i < NDQ; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][e] *= alpha;
            store_rows<NDQ>(dq, dqs, b, AQ, h * DK + 8 * d0, acc, r0, r1, t, 1.f);
            if (dqb) add_col_sums<NDQ>(s_bias[0] + 8 * d0, acc, r0 < AQ, r1 < AQ, t);
        } else {
            int i2 = it - NQ, m0 = (i2 % KT) * 16, d0 = (i2 / KT) * NDK;
            int r0 = m0 + g, r1 = m0 + g + 8;           // key rows
            float dv_acc[NDK][4], dk_acc[NDK][4];
#pragma unroll
            for (int i = 0; i < NDK; ++i) {
                dv_acc[i][0] = dv_acc[i][1] = dv_acc[i][2] = dv_acc[i][3] = 0.f;
                dk_acc[i][0] = dk_acc[i][1] = dk_acc[i][2] = dk_acc[i][3] = 0.f;
            }
            for (int k0 = 0; k0 < AQP; k0 += 16) {      // queries, two 8-steps
                unsigned ap[2][2], as[2][2];
                ldsm_at<2>(ap, sPd + k0 * LDP + m0, LDP);
                ldsm_at<2>(as, sdS + k0 * LDP + m0, LDP);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    unsigned bo[NDK], bq[NDK];
                    ldsm_b<NDK, true>(bo, sdO + (k0 + 8 * kk) * LDS + 8 * d0, LDS);
                    ldsm_b<NDK, true>(bq, sQ + (k0 + 8 * kk) * LDS + 8 * d0, LDS);
#pragma unroll
                    for (int i = 0; i < NDK; ++i) {
                        mma16816(dv_acc[i], ap[kk][0], ap[kk][1], bo[i]);
                        mma16816(dk_acc[i], as[kk][0], as[kk][1], bq[i]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < NDK; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) dk_acc[i][e] *= alpha;
            store_rows<NDK>(dv, dvs, b, AK, h * DK + 8 * d0, dv_acc, r0, r1, t, 1.f);
            store_rows<NDK>(dk, dks, b, AK, h * DK + 8 * d0, dk_acc, r0, r1, t, 1.f);
            bool v0 = r0 < AK, v1 = r1 < AK;
            if (dvb) add_col_sums<NDK>(s_bias[2] + 8 * d0, dv_acc, v0, v1, t);
            if (dkb) add_col_sums<NDK>(s_bias[1] + 8 * d0, dk_acc, v0, v1, t);
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < DK; c += blockDim.x) {
        if (dqb) atomicAdd(dqb + h * DK + c, s_bias[0][c]);
        if (dkb) atomicAdd(dkb + h * DK + c, s_bias[1][c]);
        if (dvb) atomicAdd(dvb + h * DK + c, s_bias[2][c]);
    }
}
#endif  // ATT

// ---------------------------------------------------------------------------
// Loss: label-smoothed cross-entropy, forward and backward in one pass
// ---------------------------------------------------------------------------
// One block per row of logits (N, V). Overwrites the row with
//   dlogits = (softmax - q) * loss_scale / count      (0 on ignored rows)
// and adds loss_row / count to *loss_sum. q is the smoothed target:
//   uniform:    eps/V everywhere, + (1-eps) at the target
//   non_target: (1-eps) at the target, 0 at `pad`, eps/K on the K others
// pad / ignore < 0 mean "none". scalars[0] holds the loss scale.

__device__ __forceinline__ void ce_row_terms(float lse, float sx, float xt, float xpad,
                                             int V, int t, float eps, int non_target,
                                             int pad, float* w, float* loss,
                                             bool* pad_other) {
    *pad_other = pad >= 0 && t != pad;
    if (non_target) {
        float k = fmaxf((float)(V - 1 - (*pad_other ? 1 : 0)), 1.f);
        float others = sx - xt - (*pad_other ? xpad : 0.f);
        *w = eps / k;
        *loss = (1.f - eps) * (lse - xt) + eps * (lse - others / k);
    } else {
        *w = eps / V;
        *loss = (1.f - eps) * (lse - xt) + eps * (lse - sx / V);
    }
}

__device__ __forceinline__ float ce_target(int c, int t, float w, float eps,
                                           int non_target, bool pad_other, int pad) {
    if (non_target) {
        if (c == t) return 1.f - eps;
        if (pad_other && c == pad) return 0.f;
        return w;
    }
    return c == t ? w + 1.f - eps : w;
}

// General version: reads the row from global memory three times.
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
    float w, loss;
    bool pad_other;
    ce_row_terms(lse, sx, s_xt, s_xpad, V, t, eps, non_target, pad, &w, &loss, &pad_other);
    float coef = valid ? scalars[0] * inv_count : 0.f;
    for (int c = threadIdx.x; c < V; c += blockDim.x)
        x[c] = fromf((__expf(tof(x[c]) - lse) -
                      ce_target(c, t, w, eps, non_target, pad_other, pad)) * coef);
    if (threadIdx.x == 0 && valid) atomicAdd(loss_sum, loss * inv_count);
}

#if defined(CE_V) && defined(CE_T) && defined(CE_VW)
#define CE_NPT ((CE_V + CE_T * CE_VW - 1) / (CE_T * CE_VW))
// Register-resident version: the row is read once into registers and the
// gradient written once (blockDim.x == CE_T).
extern "C" __global__ void cross_entropy_reg(
        T* __restrict__ logits, const int* __restrict__ targets,
        float eps, int non_target, int pad, int ignore,
        const float* __restrict__ scalars, float inv_count,
        float* __restrict__ loss_sum) {
    __shared__ float red[32];
    __shared__ float s_xt, s_xpad;
    int row = blockIdx.x;
    T* x = logits + (size_t)row * CE_V;
    int t = targets[row];
    bool valid = ignore < 0 || t != ignore;
    if (!valid) t = 0;
    float v[CE_NPT * CE_VW];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < CE_NPT; ++j) {
        int c = (j * CE_T + threadIdx.x) * CE_VW;
        if (c < CE_V) {
            load_vec<CE_VW>(x + c, v + j * CE_VW);
#pragma unroll
            for (int e = 0; e < CE_VW; ++e) mx = fmaxf(mx, v[j * CE_VW + e]);
        }
    }
    if (threadIdx.x == 0) {
        s_xt = tof(x[t]);
        s_xpad = pad >= 0 ? tof(x[pad]) : 0.f;
    }
    mx = block_max(mx, red);
    float se = 0.f, sx = 0.f;
#pragma unroll
    for (int j = 0; j < CE_NPT; ++j) {
        int c = (j * CE_T + threadIdx.x) * CE_VW;
        if (c < CE_V) {
#pragma unroll
            for (int e = 0; e < CE_VW; ++e) {
                se += __expf(v[j * CE_VW + e] - mx);
                sx += v[j * CE_VW + e];
            }
        }
    }
    se = block_sum(se, red);
    sx = block_sum(sx, red);
    float lse = mx + __logf(se);
    float w, loss;
    bool pad_other;
    ce_row_terms(lse, sx, s_xt, s_xpad, CE_V, t, eps, non_target, pad, &w, &loss,
                 &pad_other);
    float coef = valid ? scalars[0] * inv_count : 0.f;
#pragma unroll
    for (int j = 0; j < CE_NPT; ++j) {
        int c = (j * CE_T + threadIdx.x) * CE_VW;
        if (c < CE_V) {
            float gr[CE_VW];
#pragma unroll
            for (int e = 0; e < CE_VW; ++e)
                gr[e] = (__expf(v[j * CE_VW + e] - lse) -
                         ce_target(c + e, t, w, eps, non_target, pad_other, pad)) * coef;
            store_vec<CE_VW>(x + c, gr);
        }
    }
    if (threadIdx.x == 0 && valid) atomicAdd(loss_sum, loss * inv_count);
}
#endif  // CE_V

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

// n4 = number of float4 groups (the flat buffers are padded to 64).
extern "C" __global__ void sum_squares(const float4* __restrict__ g, int n4,
                                       float* __restrict__ out) {
    __shared__ float red[32];
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * blockDim.x) {
        float4 v = g[i];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
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
// parameters at once, four per thread per iteration. Writes the fp32 master
// weights and the working copy (which is the master itself in fp32 mode).
extern "C" __global__ void adamw(
        float* p, const float* __restrict__ g, float* __restrict__ m,
        float* __restrict__ v, T* p_work, int n4, const float* __restrict__ s,
        float lr, float b1, float b2, float eps, float wd, int decoupled) {
    if (s[4] != 0.f) return;  // overflow: skip this step
    float coef = s[3], bc1 = s[6], bc2 = s[7];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * blockDim.x) {
        float4 P = reinterpret_cast<float4*>(p)[i];
        float4 G = reinterpret_cast<const float4*>(g)[i];
        float4 M = reinterpret_cast<float4*>(m)[i];
        float4 W = reinterpret_cast<float4*>(v)[i];
        float pp[4] = {P.x, P.y, P.z, P.w}, gg[4] = {G.x, G.y, G.z, G.w};
        float mm[4] = {M.x, M.y, M.z, M.w}, vv[4] = {W.x, W.y, W.z, W.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float gi = gg[e] * coef;
            if (!decoupled) gi += wd * pp[e];
            mm[e] = b1 * mm[e] + (1.f - b1) * gi;
            vv[e] = b2 * vv[e] + (1.f - b2) * gi * gi;
            if (decoupled) pp[e] *= 1.f - lr * wd;
            pp[e] -= lr * (mm[e] / bc1) / (sqrtf(vv[e] / bc2) + eps);
        }
        reinterpret_cast<float4*>(m)[i] = make_float4(mm[0], mm[1], mm[2], mm[3]);
        reinterpret_cast<float4*>(v)[i] = make_float4(vv[0], vv[1], vv[2], vv[3]);
        reinterpret_cast<float4*>(p)[i] = make_float4(pp[0], pp[1], pp[2], pp[3]);
#if USE_HALF
        store_vec<4>(p_work + 4 * (size_t)i, pp);
#endif
    }
}
