_enabled = True


def is_grad_enabled() -> bool:
    return _enabled


class no_grad:
    """Context manager that disables graph recording."""

    def __enter__(self):
        global _enabled
        self._prev = _enabled
        _enabled = False

    def __exit__(self, *exc):
        global _enabled
        _enabled = self._prev