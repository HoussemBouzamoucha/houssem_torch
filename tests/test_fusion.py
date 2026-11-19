import numpy as np
import pytest
import torch

import houssem_torch as ht
from houssem_torch.backends.cpu.kernels import CPUStorage
from houssem_torch.fusion.engine import FusedCPUStorage


def _function(x, y):
    return ht.log(ht.exp(ht.relu((x * y + 0.25) / 2.0)) + 1.0)


def _run_houssem(x_data, y_data, enabled, dtype=ht.float32):
    x = ht.tensor(x_data, dtype=dtype, requires_grad=True)
    y = ht.tensor(y_data, dtype=dtype, requires_grad=True)
    with ht.fusion_mode(enabled):
        output = _function(x, y)
    output.sum().backward()
    return output.numpy().copy(), x.grad.numpy().copy(), y.grad.numpy().copy()


def test_fusion_is_opt_in_and_can_be_disabled():
    x = ht.tensor([-1.0, 0.0, 2.0])
    assert isinstance((x + 1.0)._storage, CPUStorage)
    with ht.fusion_mode():
        fused = x + 1.0
    with ht.fusion_mode(enabled=False):
        eager = x + 1.0
    assert isinstance(fused._storage, FusedCPUStorage)
    assert isinstance(eager._storage, CPUStorage)


def test_fused_region_defers_intermediates(numba):
    x = ht.tensor([-2.0, 0.0, 3.0])
    with ht.fusion_mode():
        first = x * 2.0
        second = first + 3.0
        output = ht.relu(second)

    assert not first._storage.materialized
    assert not second._storage.materialized
    np.testing.assert_allclose(output.numpy(), [0.0, 3.0, 9.0])
    assert not first._storage.materialized
    assert not second._storage.materialized


def test_fusion_snapshots_inputs_at_operation_time(numba):
    x = ht.tensor([1.0, 2.0])
    with ht.fusion_mode():
        output = x * 2.0 + 1.0
    x.numpy()[:] = 10.0
    np.testing.assert_allclose(output.numpy(), [3.0, 5.0])


def test_fused_forward_and_gradients_match_unfused_and_torch(numba):
    x_data = np.array([[-1.5], [0.0], [1.25]], dtype=np.float32)
    y_data = np.array([[0.5, -2.0, 1.0]], dtype=np.float32)
    eager = _run_houssem(x_data, y_data, enabled=False)
    fused = _run_houssem(x_data, y_data, enabled=True)

    torch_x = torch.tensor(x_data, requires_grad=True)
    torch_y = torch.tensor(y_data, requires_grad=True)
    torch_output = torch.log(
        torch.exp(torch.relu((torch_x * torch_y + 0.25) / 2.0)) + 1.0
    )
    torch_output.sum().backward()

    for fused_value, eager_value, torch_value in zip(
        fused, eager, (torch_output.detach().numpy(),
                       torch_x.grad.numpy(), torch_y.grad.numpy())
    ):
        np.testing.assert_allclose(fused_value, eager_value, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(fused_value, torch_value, rtol=1e-5, atol=1e-6)


def test_fused_int64_pointwise_operations(numba):
    values = np.array([-3, 0, 4], dtype=np.int64)
    x = ht.tensor(values, dtype=ht.int64)
    with ht.fusion_mode():
        output = ht.relu((x * 2) + 1)
    np.testing.assert_array_equal(output.numpy(), [0, 1, 9])
    assert output.numpy().dtype == np.int64


def test_no_grad_and_reduction_boundary(numba):
    x = ht.tensor([0.0, 1.0, 2.0])
    with ht.fusion_mode():
        output = ht.exp(x * 2.0 + 1.0)
        total = output.sum()
    assert not output.requires_grad
    assert output.is_leaf
    assert np.allclose(total.numpy(), np.exp([1.0, 3.0, 5.0]).sum())


def test_repeated_backward_accumulates_with_fusion(numba):
    x = ht.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    with ht.fusion_mode():
        loss = (x * 2.0 + 3.0).sum()
        loss.backward()
        loss.backward()
    np.testing.assert_allclose(x.grad.numpy(), [4.0, 4.0, 4.0])
    assert isinstance(x.grad._storage, CPUStorage)


def test_deep_fused_chain_matches_eager(numba):
    values = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    x_fused = ht.tensor(values, requires_grad=True)
    x_eager = ht.tensor(values, requires_grad=True)
    with ht.fusion_mode():
        fused = x_fused
        for _ in range(40):
            fused = ht.relu(fused * 1.01 + 0.01)
    eager = x_eager
    for _ in range(40):
        eager = ht.relu(eager * 1.01 + 0.01)

    np.testing.assert_allclose(fused.numpy(), eager.numpy(), rtol=1e-5, atol=1e-6)
    fused.sum().backward()
    eager.sum().backward()
    np.testing.assert_allclose(
        x_fused.grad.numpy(), x_eager.grad.numpy(), rtol=1e-5, atol=1e-6
    )


def test_noncontiguous_input_falls_back_to_eager(numba):
    x = ht.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose()
    with ht.fusion_mode():
        output = x + 1.0
    assert isinstance(output._storage, CPUStorage)
    np.testing.assert_allclose(output.numpy(), [[2.0, 4.0], [3.0, 5.0]])


def test_dtype_changing_operation_falls_back(numba):
    x = ht.tensor([0, 1, 2], dtype=ht.int64)
    with ht.fusion_mode():
        output = ht.exp(x)
    assert isinstance(output._storage, CPUStorage)
    np.testing.assert_allclose(output.numpy(), np.exp(x.numpy()))


def test_incompatible_broadcast_uses_eager_error():
    x = ht.tensor(np.ones((2, 3), dtype=np.float32))
    y = ht.tensor(np.ones((4,), dtype=np.float32))
    with ht.fusion_mode(), pytest.raises(ValueError):
        x + y


@pytest.fixture
def numba():
    return pytest.importorskip("numba")
