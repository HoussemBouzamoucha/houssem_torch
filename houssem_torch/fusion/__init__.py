from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_enabled = ContextVar("houssem_torch_fusion_enabled", default=False)
_backward = ContextVar("houssem_torch_fusion_backward", default=False)


def is_fusion_enabled() -> bool:
    return _enabled.get()


def is_fusion_backward() -> bool:
    return _backward.get()


@contextmanager
def fusion_mode(enabled: bool = True) -> Iterator[None]:
    """Enable or disable CPU pointwise fusion in this context."""
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


@contextmanager
def fusion_backward_mode() -> Iterator[None]:
    token = _backward.set(True)
    try:
        yield
    finally:
        _backward.reset(token)


__all__ = ["fusion_mode"]
