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


def test_gemm_tuning_picks_a_working_algorithm():
    driver.reset_gemm_algorithms()
    m, n, k = 96, 80, 64
    a = driver.from_numpy(rng.standard_normal((m, k)).astype(np.float16))
    b = driver.from_numpy((rng.standard_normal((k, n)) / 8).astype(np.float16))
    ab = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
    c0 = rng.standard_normal((m, n)).astype(np.float32)
    c = driver.from_numpy(c0)
    with driver.gemm_tuning():
        driver.gemm(a, b, c, beta=1.0)
    # Candidates are timed on a scratch output: c got exactly one multiply.
    np.testing.assert_allclose(c.numpy(), ab + c0, rtol=1e-2, atol=1e-2)
    (label, info), = driver.gemm_algorithms().items()
    assert label == "96x80x64 float16->float32 +C"
    assert info["algo"] in info["us"]
    driver.gemm(a, b, c, beta=1.0)                      # reuses the choice
    np.testing.assert_allclose(c.numpy(), 2 * ab + c0, rtol=1e-2, atol=1e-2)
    driver.reset_gemm_algorithms()
    assert driver.gemm_algorithms() == {}


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


def test_arena_measures_and_shares_one_block():
    meter = driver.Arena()
    a = meter.take((10, 3), np.float16)        # 60 bytes
    b = meter.take(5, np.float32)              # aligned to 256, +20
    assert a.shape == (10, 3) and b.shape == (5,)
    assert not a.numpy().any()
    assert meter.used == 256 + 20
    arena = driver.Arena(meter.used)
    one, two = arena.cursor(), arena.cursor()
    x = one.take((10, 3), np.float16)
    y = two.take(15, np.float32)               # same front of the block
    x.copy_from(np.ones((10, 3), np.float16))
    assert y.numpy()[0] != 0                   # overlaps x
    one.take(5, np.float32)
    with pytest.raises(MemoryError):
        one.take(1, np.float32)
