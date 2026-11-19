"""Weight initializers. All return float32 NumPy arrays drawn from the
framework's seeded generator (`ht.manual_seed` makes them reproducible)."""
from __future__ import annotations
import numpy as np
from houssem_torch.random import get_rng


def xavier_uniform(shape: tuple[int, int]) -> np.ndarray:
    """Uniform in ±sqrt(6 / (fan_in + fan_out)) (Glorot & Bengio).

    Keeps activations at a similar scale from layer to layer, which is why
    transformer implementations use it for linear projections.
    shape: (fan_in, fan_out), the (in, out) layout used by `Linear`.
    """
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return get_rng().uniform(-limit, limit, size=shape).astype(np.float32)


def normal(shape, std: float) -> np.ndarray:
    """Normal(0, std²). Used for the embedding table."""
    return (get_rng().standard_normal(shape) * std).astype(np.float32)


def zeros(shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def ones(shape) -> np.ndarray:
    return np.ones(shape, dtype=np.float32)
