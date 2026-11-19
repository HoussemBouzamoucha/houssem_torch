from __future__ import annotations


def noam_lr(step: int, d_model: int, warmup_steps: int, factor: float = 1.0) -> float:
    """The transformer paper's learning-rate formula ("Noam" schedule):

        lr = factor · d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)

    It rises linearly for `warmup_steps` steps (large early updates are
    unstable while Adam's statistics are still poor), peaks at step = warmup,
    then decays like 1/sqrt(step). The paper used warmup = 4000.
    `step` starts at 1.
    """
    step = max(step, 1)
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


class NoamSchedule:
    """Sets `optimizer.lr` from `noam_lr` each step.

    Call `scheduler.step()` once per training step, before `optimizer.step()`.
    Its counter is part of `state_dict()` so a resumed run continues the
    schedule where it stopped.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000,
                 factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        """Advance one step, set and return the new learning rate."""
        self.step_num += 1
        lr = noam_lr(self.step_num, self.d_model, self.warmup_steps, self.factor)
        self.optimizer.lr = lr
        return lr

    def state_dict(self) -> dict:
        return {"step_num": self.step_num}

    def load_state_dict(self, state: dict) -> None:
        self.step_num = state["step_num"]
