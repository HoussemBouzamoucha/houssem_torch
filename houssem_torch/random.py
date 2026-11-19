"""One seeded random generator for the whole framework (dropout, etc.).

A single generator makes runs reproducible and its state easy to save in a
checkpoint (`get_state` / `set_state`).
"""
from __future__ import annotations
import numpy as np

_rng = np.random.default_rng()


def manual_seed(seed: int) -> None:
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    return _rng


def get_state() -> dict:
    return _rng.bit_generator.state


def set_state(state: dict) -> None:
    _rng.bit_generator.state = state
