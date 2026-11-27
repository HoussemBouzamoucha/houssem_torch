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


# cuBLASLt fused epilogues -------------------------------------------------

needs_lt = pytest.mark.skipif(not driver.lt_available(),
                              reason="cuBLASLt not loadable")


def pack_mask(on):
    """Row-major, least-significant-bit-first bit mask as int32 words."""
    bits = np.packbits(on.astype(np.uint8).reshape(-1), bitorder="little")
    return bits.view(np.int32)


def fp16_operands(m, n, k, trans_b=False):
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = (rng.standard_normal((n, k) if trans_b else (k, n))
         / np.sqrt(k)).astype(np.float16)
    ref = a.astype(np.float32) @ (b.T if trans_b else b).astype(np.float32)
    return a, b, ref


@needs_lt
def test_lt_bias_epilogues():
    m, n, k = 40, 256, 64
    a, b, ref = fp16_operands(m, n, k)
    bias = rng.standard_normal(n).astype(np.float16)
    for epi, want in (("default", ref), ("bias", ref + bias),
                      ("relu_bias", np.maximum(ref + bias, 0))):
        c = driver.empty((m, n), np.float16)
        driver.gemm_epilogue(driver.from_numpy(a), driver.from_numpy(b), c,
                             epilogue=epi,
                             bias=driver.from_numpy(bias) if epi != "default"
                             else None)
        np.testing.assert_allclose(c.numpy().astype(np.float32), want,
                                   rtol=1e-2, atol=1e-2, err_msg=epi)


@needs_lt
def test_lt_relu_mask_layout_is_row_major_lsb_first():
    """The bit order our forward kernel writes must be the one cuBLASLt
    uses: let cuBLASLt write a ReLU mask and compare."""
    m, n, k = 24, 384, 64
    a, b, ref = fp16_operands(m, n, k)
    bias = rng.standard_normal(n).astype(np.float16)
    c = driver.empty((m, n), np.float16)
    aux = driver.zeros(m * n // 32, np.int32)
    driver.gemm_epilogue(driver.from_numpy(a), driver.from_numpy(b), c,
                         epilogue="relu_aux_bias",
                         bias=driver.from_numpy(bias), aux=aux)
    got = aux.numpy()
    on = c.numpy() > 0
    candidates = {
        "row-major lsb-first": pack_mask(on),
        "row-major msb-first": np.packbits(on.reshape(-1)).view(np.int32),
        "column-major lsb-first": pack_mask(on.T.copy()),
    }
    matches = [k for k, v in candidates.items() if np.array_equal(v, got)]
    assert matches == ["row-major lsb-first"], (
        f"mask layout matches {matches}; first words "
        f"{[hex(x & 0xffffffff) for x in got[:4]]} vs on[0,:32]="
        f"{on[0, :32].astype(int).tolist()}")


@needs_lt
@pytest.mark.parametrize("trans_b", [False, True])
def test_lt_drelu_bgrad(trans_b):
    """c = alpha·(a@b) ⊙ mask, bias = column sums of c."""
    m, n, k = 48, 256, 64
    a, b, ref = fp16_operands(m, n, k, trans_b)
    on = rng.random((m, n)) < 0.6
    c = driver.empty((m, n), np.float16)
    db = driver.empty(n, np.float16)
    alpha = 1.0 / 0.9
    driver.gemm_epilogue(driver.from_numpy(a), driver.from_numpy(b), c,
                         epilogue="drelu_bgrad", bias=db,
                         aux=driver.from_numpy(pack_mask(on)),
                         trans_b=trans_b, alpha=alpha)
    want = alpha * ref * on
    np.testing.assert_allclose(c.numpy().astype(np.float32), want,
                               rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(db.numpy().astype(np.float32), want.sum(0),
                               rtol=2e-2, atol=5e-2)


@needs_lt
def test_lt_epilogue_argument_errors():
    a, b = driver.zeros((8, 16), np.float16), driver.zeros((16, 128), np.float16)
    c = driver.empty((8, 128), np.float16)
    with pytest.raises(ValueError):
        driver.gemm_epilogue(a, b, c, epilogue="bias")          # no bias
    with pytest.raises(ValueError):
        driver.gemm_epilogue(a, b, c, epilogue="drelu_bgrad",
                             bias=driver.empty(128, np.float16))  # no mask
    with pytest.raises(ValueError):
        driver.gemm_epilogue(a, b, c, epilogue="nope")
    with pytest.raises(TypeError):
        driver.gemm_epilogue(a, b, c, epilogue="drelu_bgrad",
                             bias=driver.empty(128, np.float32),
                             aux=driver.zeros(8 * 128 // 32, np.int32))
