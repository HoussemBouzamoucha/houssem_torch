from __future__ import annotations
import numpy as np


class Optimizer:
    """Base class: holds the parameters and updates them from their gradients.

    Usage, once per training step:
        optimizer.zero_grad()   # clear last step's gradients
        loss.backward()         # compute new gradients
        optimizer.step()        # update every parameter in place

    Parameters are de-duplicated (a shared parameter is updated once) and
    only those with requires_grad are kept. Updates write into each
    parameter's existing array, so modules keep pointing at the same weights.
    """

    def __init__(self, params, lr: float):
        unique, seen = [], set()
        for p in params:
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        if not unique:
            raise ValueError("optimizer got no trainable parameters.")
        self.params = unique
        self.lr = lr

    def zero_grad(self) -> None:
        """Clear every parameter's gradient."""
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        """Apply one update using the current gradients."""
        raise NotImplementedError

    # Checkpointing: subclasses list their per-parameter state arrays here.
    _state_names: tuple[str, ...] = ()

    def state_dict(self) -> dict:
        """Everything needed to resume: lr, step counts and per-parameter state
        (e.g. Adam's moving averages), in parameter order."""
        state = {"lr": self.lr}
        for name in self._state_names:
            value = getattr(self, name)
            state[name] = [v.copy() if isinstance(v, np.ndarray) else v
                           for v in value]
        return state

    def load_state_dict(self, state: dict) -> None:
        self.lr = state["lr"]
        for name in self._state_names:
            current, saved = getattr(self, name), state[name]
            if len(saved) != len(current):
                raise ValueError(f"optimizer state '{name}' has {len(saved)} "
                                 f"entries, expected {len(current)}.")
            for i, value in enumerate(saved):
                if isinstance(current[i], np.ndarray):
                    if np.shape(value) != current[i].shape:
                        raise ValueError(f"optimizer state '{name}[{i}]' shape "
                                         f"{np.shape(value)} != {current[i].shape}")
                    current[i][...] = value
                else:
                    current[i] = value
