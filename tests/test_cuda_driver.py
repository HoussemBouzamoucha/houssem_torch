"""GPU driver layer: memory, NVRTC kernels, cuBLAS matmuls.

Skipped when CuPy or a CUDA GPU is missing. Every result is checked against
NumPy on the host.
"""
import numpy as np
import pytest

from houssem_torch.backends.cuda import driver

pytestmark = pytest.mark.skipif(not driver.is_available(),
                                reason="needs CuPy and a CUDA GPU")

rng = np.random.default_rng(0)


def test_device_info():
    info = driver.device_info()
    assert info.total_memory > 0
    assert info.compute_capability >= (5, 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_buffer_round_trip(dtype):
    host = (rng.standard_normal((3, 5)) * 10).astype(dtype)
    buf = driver.from_numpy(host)
    assert buf.shape == (3, 5) and buf.dtype == dtype and buf.ptr != 0
    np.testing.assert_array_equal(buf.numpy(), host)

    other = driver.empty((3, 5), dtype)
    other.copy_(buf)
    np.testing.assert_array_equal(other.numpy(), host)
    other.fill_(0)
    assert not other.numpy().any()
    assert buf.view(15).shape == (15,)


def test_buffer_rejects_bad_input():
    with pytest.raises(TypeError):
        driver.empty((2,), np.float64)
    buf = driver.zeros((2, 2))
    with pytest.raises(ValueError):
        buf.copy_from(np.zeros((3, 2)))


SCALE_ADD = r"""
extern "C" __global__ void scale_add(const float* x, const float* y,
                                     float* out, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i] + BIAS;
}
"""


def test_compile_and_launch_with_baked_constant():
    n = 1000
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    dx, dy, out = driver.from_numpy(x), driver.from_numpy(y), driver.empty(n)
    k = driver.compile_kernel(SCALE_ADD, "scale_add", defines={"BIAS": 0.5})
    k((n + 255) // 256, 256, dx, dy, out, 2.0, n)
    np.testing.assert_allclose(out.numpy(), 2 * x + y + 0.5, rtol=1e-6)
    # Same source + constants comes from the cache; new constants recompile.
    assert driver.compile_kernel(SCALE_ADD, "scale_add", {"BIAS": 0.5}) is k
    assert driver.compile_kernel(SCALE_ADD, "scale_add", {"BIAS": 1.0}) is not k


@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
def test_gemm_fp32(trans_a, trans_b):
    m, n, k = 37, 23, 19  # odd sizes catch leading-dimension mistakes
    a = rng.standard_normal((k, m) if trans_a else (m, k)).astype(np.float32)
    b = rng.standard_normal((n, k) if trans_b else (k, n)).astype(np.float32)
    c0 = rng.standard_normal((m, n)).astype(np.float32)
    c = driver.from_numpy(c0)
    driver.gemm(driver.from_numpy(a), driver.from_numpy(b), c,
                trans_a=trans_a, trans_b=trans_b, alpha=0.5, beta=2.0)
    opa = a.T if trans_a else a
    opb = b.T if trans_b else b
    np.testing.assert_allclose(c.numpy(), 0.5 * opa @ opb + 2.0 * c0,
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("out_dtype", [np.float16, np.float32])
def test_gemm_fp16_inputs(out_dtype):
    m, n, k = 64, 48, 512
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float16)
    c = driver.empty((m, n), out_dtype)
    driver.gemm(driver.from_numpy(a), driver.from_numpy(b), c)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(c.numpy().astype(np.float32), ref,
                               rtol=1e-2, atol=1e-2)


def test_gemm_shape_errors():
    a, b = driver.zeros((4, 3)), driver.zeros((5, 2))
    with pytest.raises(ValueError):
        driver.gemm(a, b, driver.zeros((4, 2)))
    with pytest.raises(TypeError):
        driver.gemm(a, driver.zeros((3, 2)), driver.zeros((4, 2), np.float16))


@pytest.mark.parametrize("trans_b", [False, True])
def test_gemm_batched(trans_b):
    # Shapes of attention scores: (batch·heads, T, d_head) @ (.., d_head, T)
    bh, t, dh = 6, 17, 8
    q = rng.standard_normal((bh, t, dh)).astype(np.float32)
    kk = rng.standard_normal((bh, t, dh) if trans_b else (bh, dh, t))
    kk = kk.astype(np.float32)
    s = driver.empty((bh, t, t))
    driver.gemm_batched(driver.from_numpy(q), driver.from_numpy(kk), s,
                        trans_b=trans_b, alpha=dh ** -0.5)
    ref = q @ (kk.transpose(0, 2, 1) if trans_b else kk) * dh ** -0.5
    np.testing.assert_allclose(s.numpy(), ref, rtol=1e-4, atol=1e-4)


def test_timer_and_stream():
    a = driver.from_numpy(rng.standard_normal((256, 256)).astype(np.float32))
    c = driver.empty((256, 256))
    with driver.Stream() as s:
        t = driver.Timer()
        t.start()
        driver.gemm(a, a, c)
        ms = t.stop()
        s.synchronize()
    assert ms >= 0.0
    np.testing.assert_allclose(c.numpy(), a.numpy() @ a.numpy(),
                               rtol=1e-3, atol=1e-3)
