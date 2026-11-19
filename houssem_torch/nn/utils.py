from __future__ import annotations
import numpy as np


def clip_grad_norm(parameters, max_norm: float) -> float:
    """Scale all gradients down together if their combined size is too big.

    The combined size is the global L2 norm: sqrt of the sum of every squared
    gradient value across all parameters. If it exceeds `max_norm`, every
    gradient is multiplied by max_norm / norm, which keeps the update's
    direction but caps its length. This prevents a single bad batch from
    blowing up training.

    Returns the norm measured BEFORE clipping (useful to log).
    """
    grads = [p for p in parameters if p._grad is not None]
    total = float(np.sqrt(sum(float(np.sum(np.square(p._grad, dtype=np.float64)))
                              for p in grads)))
    if total > max_norm:
        scale = np.float32(max_norm / (total + 1e-6))
        for p in grads:
            p._grad = p._grad * scale
    return total
