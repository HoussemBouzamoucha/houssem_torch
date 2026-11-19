"""`Module`: the base class every layer and model is built on.

A Module is a Python object that owns Parameters (trainable tensors) and/or
other Modules (sub-layers). Assigning one as an attribute is all it takes to
register it:

    class MyLayer(Module):
        def __init__(self):
            self.proj = Linear(8, 8)             # sub-module
            self.scale = Parameter(np.ones(8))   # parameter

        def forward(self, x):
            return self.proj(x) * self.scale

From that, Module provides parameter discovery (for the optimizer),
train/eval switching (for dropout) and saving/loading weights by name (for
checkpoints).
"""
from __future__ import annotations
from typing import Iterator
import numpy as np
from houssem_torch.parameter import Parameter


class Module:
    # Class-level default so subclasses don't need to call super().__init__().
    training: bool = True

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """Compute the module's output. Every subclass implements this."""
        raise NotImplementedError(f"{type(self).__name__} has no forward().")

    def __call__(self, *args, **kwargs):
        """`layer(x)` is shorthand for `layer.forward(x)`."""
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Discovering parameters and sub-modules
    # ------------------------------------------------------------------

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        """Yield (dotted name, module) for this module and every descendant.

        The root module has name "" (or `prefix`).
        """
        yield prefix.rstrip("."), self
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield from value.named_modules(f"{prefix}{name}.")

    def modules(self) -> Iterator["Module"]:
        """Yield this module and every descendant module."""
        for _, module in self.named_modules():
            yield module

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        """Yield (dotted name, parameter) for every parameter, each exactly once.

        Names follow the attribute path, e.g. "encoder.0.self_attn.w_o.weight".
        A parameter shared by several modules (like the transformer's shared
        embedding table) is yielded once, under the first name it is found
        at, so an optimizer never updates it twice.
        """
        seen: set[int] = set()

        def walk(module, prefix):
            # Depth-first in attribute (definition) order.
            for name, value in vars(module).items():
                if isinstance(value, Parameter):
                    if id(value) not in seen:
                        seen.add(id(value))
                        yield prefix + name, value
                elif isinstance(value, Module):
                    yield from walk(value, f"{prefix}{name}.")

        yield from walk(self, "")

    def parameters(self) -> list[Parameter]:
        """All parameters, each once. Pass this to an optimizer."""
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        """Total number of trainable values."""
        return sum(p.numel for p in self.parameters())

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "Module":
        """Set training mode on this module and all descendants.

        Training mode turns dropout on; `eval()` turns it off. Returns self,
        so `model.eval()` can be chained.
        """
        for module in self.modules():
            module.training = mode
        return self

    def eval(self) -> "Module":
        """Evaluation mode: `train(False)`. Use for validation and translating."""
        return self.train(False)

    def zero_grad(self) -> None:
        """Clear every parameter's gradient (call once per optimizer step)."""
        for p in self.parameters():
            p.grad = None

    # ------------------------------------------------------------------
    # Saving and loading weights
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        """Copy every parameter into {name: NumPy array}.

        The dict can be saved with `np.savez(path, **model.state_dict())`.
        """
        return {name: p.numpy().copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        """Copy arrays from `state` into the parameters, in place.

        Shapes must match exactly. With strict=True, missing or unexpected
        names raise an error, which catches loading the wrong checkpoint.
        """
        params = dict(self.named_parameters())
        if strict:
            missing = sorted(set(params) - set(state))
            unexpected = sorted(set(state) - set(params))
            if missing or unexpected:
                raise KeyError(f"state_dict mismatch: missing {missing}, "
                               f"unexpected {unexpected}")
        for name, p in params.items():
            if name not in state:
                continue
            value = np.asarray(state[name])
            if value.shape != p.shape:
                raise ValueError(f"shape mismatch for '{name}': checkpoint "
                                 f"{value.shape} vs parameter {p.shape}")
            p.numpy()[...] = value

    def __repr__(self) -> str:
        children = [f"  {n}: {type(m).__name__}" for n, m in vars(self).items()
                    if isinstance(m, Module)]
        body = ("\n" + "\n".join(children) + "\n") if children else ""
        return f"{type(self).__name__}({body})"


class ModuleList(Module):
    """A list of sub-modules, e.g. the N stacked encoder layers.

    Iterate, index and len() it like a list. Each item is registered under its
    index, so parameters get names like "encoder.0.ffn.up.weight".
    """

    def __init__(self, modules=()):
        self._items: list[Module] = []
        for module in modules:
            self.append(module)

    def append(self, module: Module) -> None:
        setattr(self, str(len(self._items)), module)
        self._items.append(module)

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> Module:
        return self._items[i]
