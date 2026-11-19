#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define FUSION_EXPORT __declspec(dllexport)
#else
#define FUSION_EXPORT
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define FUSION_HAS_AVX2_IMPL 1
#define FUSION_AVX2 __attribute__((target("avx2")))
#else
#define FUSION_HAS_AVX2_IMPL 0
#define FUSION_AVX2
#endif

namespace {

inline float preactivation(float a, float b) {
    return (a * b + 0.25f) / 2.0f;
}

inline float forward_value(float a, float b) {
    const float z = preactivation(a, b);
    const float relu_value = z > 0.0f ? z : 0.0f;
    return std::log(std::exp(relu_value) + 1.0f);
}

inline float derivative_value(float a, float b) {
    const float z = preactivation(a, b);
    if (!(z > 0.0f)) {
        return 0.0f;
    }
    const float exp_value = std::exp(z);
    return (exp_value / (exp_value + 1.0f)) * 0.5f;
}

inline float evaluate(float a, float b, float& derivative) {
    const float z = preactivation(a, b);
    const float relu_value = z > 0.0f ? z : 0.0f;
    const float exp_value = std::exp(relu_value);
    derivative = z > 0.0f ? (exp_value / (exp_value + 1.0f)) * 0.5f : 0.0f;
    return std::log(exp_value + 1.0f);
}

#if FUSION_HAS_AVX2_IMPL
FUSION_AVX2 inline __m256 exp_approx_avx2(__m256 x) {
    const __m256 lower = _mm256_set1_ps(-80.0f);
    const __m256 upper = _mm256_set1_ps(80.0f);
    x = _mm256_min_ps(_mm256_max_ps(x, lower), upper);

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 ln2_high = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_low = _mm256_set1_ps(-0.00021219444005469058f);
    const __m256 rounded =
        _mm256_round_ps(_mm256_mul_ps(x, log2e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256i exponent = _mm256_cvttps_epi32(rounded);
    const __m256 exponent_f = _mm256_cvtepi32_ps(exponent);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(exponent_f, ln2_high));
    r = _mm256_sub_ps(r, _mm256_mul_ps(exponent_f, ln2_low));

    __m256 polynomial = _mm256_set1_ps(1.0f / 5040.0f);
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f / 720.0f), _mm256_mul_ps(r, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f / 120.0f), _mm256_mul_ps(r, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f / 24.0f), _mm256_mul_ps(r, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f / 6.0f), _mm256_mul_ps(r, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(0.5f), _mm256_mul_ps(r, polynomial));
    polynomial = _mm256_add_ps(_mm256_set1_ps(1.0f),
                               _mm256_mul_ps(r, polynomial));
    polynomial = _mm256_add_ps(_mm256_set1_ps(1.0f),
                               _mm256_mul_ps(r, polynomial));

    const __m256i biased = _mm256_add_epi32(exponent, _mm256_set1_epi32(127));
    const __m256 scale = _mm256_castsi256_ps(_mm256_slli_epi32(biased, 23));
    return _mm256_mul_ps(polynomial, scale);
}

FUSION_AVX2 inline __m256 log1p_exp_neg_avx2(__m256 positive) {
    const __m256 exp_negative =
        exp_approx_avx2(_mm256_sub_ps(_mm256_setzero_ps(), positive));
    const __m256 u = _mm256_div_ps(
        exp_negative,
        _mm256_add_ps(_mm256_set1_ps(2.0f), exp_negative));
    const __m256 u_squared = _mm256_mul_ps(u, u);

    __m256 polynomial = _mm256_set1_ps(1.0f / 13.0f);
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f / 11.0f), _mm256_mul_ps(u_squared, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f / 9.0f), _mm256_mul_ps(u_squared, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f / 7.0f), _mm256_mul_ps(u_squared, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f / 5.0f), _mm256_mul_ps(u_squared, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f / 3.0f), _mm256_mul_ps(u_squared, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(1.0f), _mm256_mul_ps(u_squared, polynomial));
    return _mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(u, polynomial));
}

FUSION_AVX2 void avx2_forward(
    const float* a, const float* b, float* output, std::size_t size) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 quarter = _mm256_set1_ps(0.25f);
    const __m256 zero = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 z = _mm256_mul_ps(
            _mm256_add_ps(_mm256_mul_ps(va, vb), quarter), half);
        const __m256 active = _mm256_cmp_ps(z, zero, _CMP_GT_OQ);
        const __m256 positive = _mm256_blendv_ps(zero, z, active);
        const __m256 softplus =
            _mm256_add_ps(positive, log1p_exp_neg_avx2(positive));
        _mm256_storeu_ps(output + i, softplus);
    }
    for (; i < size; ++i) {
        output[i] = forward_value(a[i], b[i]);
    }
}

FUSION_AVX2 void avx2_backward(
    const float* a, const float* b, const float* grad_output,
    float* grad_a, float* grad_b, std::size_t size) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 quarter = _mm256_set1_ps(0.25f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 z = _mm256_mul_ps(
            _mm256_add_ps(_mm256_mul_ps(va, vb), quarter), half);
        const __m256 active = _mm256_cmp_ps(z, zero, _CMP_GT_OQ);
        const __m256 positive = _mm256_blendv_ps(zero, z, active);
        const __m256 exp_negative =
            exp_approx_avx2(_mm256_sub_ps(zero, positive));
        __m256 derivative =
            _mm256_div_ps(half, _mm256_add_ps(one, exp_negative));
        derivative = _mm256_and_ps(derivative, active);
        const __m256 upstream = _mm256_loadu_ps(grad_output + i);
        _mm256_storeu_ps(
            grad_a + i, _mm256_mul_ps(_mm256_mul_ps(upstream, derivative), vb));
        _mm256_storeu_ps(
            grad_b + i, _mm256_mul_ps(_mm256_mul_ps(upstream, derivative), va));
    }
    for (; i < size; ++i) {
        const float derivative = derivative_value(a[i], b[i]);
        grad_a[i] = grad_output[i] * derivative * b[i];
        grad_b[i] = grad_output[i] * derivative * a[i];
    }
}

FUSION_AVX2 void avx2_forward_backward(
    const float* a, const float* b, float* output,
    float* grad_a, float* grad_b, std::size_t size) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 quarter = _mm256_set1_ps(0.25f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 z = _mm256_mul_ps(
            _mm256_add_ps(_mm256_mul_ps(va, vb), quarter), half);
        const __m256 active = _mm256_cmp_ps(z, zero, _CMP_GT_OQ);
        const __m256 positive = _mm256_blendv_ps(zero, z, active);
        const __m256 exp_negative =
            exp_approx_avx2(_mm256_sub_ps(zero, positive));
        const __m256 softplus =
            _mm256_add_ps(positive, log1p_exp_neg_avx2(positive));
        __m256 derivative =
            _mm256_div_ps(half, _mm256_add_ps(one, exp_negative));
        derivative = _mm256_and_ps(derivative, active);
        _mm256_storeu_ps(output + i, softplus);
        _mm256_storeu_ps(grad_a + i, _mm256_mul_ps(derivative, vb));
        _mm256_storeu_ps(grad_b + i, _mm256_mul_ps(derivative, va));
    }
    for (; i < size; ++i) {
        float derivative;
        output[i] = evaluate(a[i], b[i], derivative);
        grad_a[i] = derivative * b[i];
        grad_b[i] = derivative * a[i];
    }
}
#endif

}  // namespace

extern "C" {

FUSION_EXPORT void fusion_cpp_forward(
    const float* a, const float* b, float* output, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = forward_value(a[i], b[i]);
    }
}

FUSION_EXPORT void fusion_cpp_backward(
    const float* a, const float* b, const float* grad_output,
    float* grad_a, float* grad_b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        const float derivative = derivative_value(a[i], b[i]);
        const float upstream = grad_output[i];
        grad_a[i] = upstream * derivative * b[i];
        grad_b[i] = upstream * derivative * a[i];
    }
}

FUSION_EXPORT void fusion_cpp_forward_backward(
    const float* a, const float* b, float* output,
    float* grad_a, float* grad_b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        float derivative;
        output[i] = evaluate(a[i], b[i], derivative);
        grad_a[i] = derivative * b[i];
        grad_b[i] = derivative * a[i];
    }
}

FUSION_EXPORT int fusion_cpp_avx2_supported() {
#if FUSION_HAS_AVX2_IMPL
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
}

FUSION_EXPORT int fusion_cpp_openmp_enabled() {
#if defined(_OPENMP)
    return 1;
#else
    return 0;
#endif
}

FUSION_EXPORT void fusion_cpp_set_threads(int threads) {
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(threads > 0 ? threads : 1);
#else
    (void)threads;
#endif
}

FUSION_EXPORT void fusion_cpp_forward_omp(
    const float* a, const float* b, float* output, std::size_t size) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(size >= 8192)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
        output[i] = forward_value(a[i], b[i]);
    }
}

FUSION_EXPORT void fusion_cpp_backward_omp(
    const float* a, const float* b, const float* grad_output,
    float* grad_a, float* grad_b, std::size_t size) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(size >= 8192)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
        const float derivative = derivative_value(a[i], b[i]);
        const float upstream = grad_output[i];
        grad_a[i] = upstream * derivative * b[i];
        grad_b[i] = upstream * derivative * a[i];
    }
}

FUSION_EXPORT void fusion_cpp_forward_backward_omp(
    const float* a, const float* b, float* output,
    float* grad_a, float* grad_b, std::size_t size) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(size >= 8192)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
        float derivative;
        output[i] = evaluate(a[i], b[i], derivative);
        grad_a[i] = derivative * b[i];
        grad_b[i] = derivative * a[i];
    }
}

#if FUSION_HAS_AVX2_IMPL
FUSION_EXPORT FUSION_AVX2 void fusion_cpp_forward_avx2(
    const float* a, const float* b, float* output, std::size_t size) {
    avx2_forward(a, b, output, size);
}

FUSION_EXPORT FUSION_AVX2 void fusion_cpp_backward_avx2(
    const float* a, const float* b, const float* grad_output,
    float* grad_a, float* grad_b, std::size_t size) {
    avx2_backward(a, b, grad_output, grad_a, grad_b, size);
}

FUSION_EXPORT FUSION_AVX2 void fusion_cpp_forward_backward_avx2(
    const float* a, const float* b, float* output,
    float* grad_a, float* grad_b, std::size_t size) {
    avx2_forward_backward(a, b, output, grad_a, grad_b, size);
}
#else
FUSION_EXPORT void fusion_cpp_forward_avx2(
    const float* a, const float* b, float* output, std::size_t size) {
    fusion_cpp_forward(a, b, output, size);
}

FUSION_EXPORT void fusion_cpp_backward_avx2(
    const float* a, const float* b, const float* grad_output,
    float* grad_a, float* grad_b, std::size_t size) {
    fusion_cpp_backward(a, b, grad_output, grad_a, grad_b, size);
}

FUSION_EXPORT void fusion_cpp_forward_backward_avx2(
    const float* a, const float* b, float* output,
    float* grad_a, float* grad_b, std::size_t size) {
    fusion_cpp_forward_backward(a, b, output, grad_a, grad_b, size);
}
#endif

}  // extern "C"
