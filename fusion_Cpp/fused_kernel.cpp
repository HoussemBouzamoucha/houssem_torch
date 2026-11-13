#include <cmath>
#include <cstddef>

#if defined(_WIN32)
#define FUSION_EXPORT __declspec(dllexport)
#else
#define FUSION_EXPORT
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

inline float evaluate(float a, float b, float& derivative) {
    const float z = preactivation(a, b);
    const float relu_value = z > 0.0f ? z : 0.0f;
    const float exp_value = std::exp(relu_value);
    derivative = z > 0.0f ? (exp_value / (exp_value + 1.0f)) * 0.5f : 0.0f;
    return std::log(exp_value + 1.0f);
}

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
        const float z = preactivation(a[i], b[i]);
        const float relu_value = z > 0.0f ? z : 0.0f;
        const float exp_value = std::exp(relu_value);
        const float derivative =
            z > 0.0f ? (exp_value / (exp_value + 1.0f)) * 0.5f : 0.0f;
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

}  // extern "C"
