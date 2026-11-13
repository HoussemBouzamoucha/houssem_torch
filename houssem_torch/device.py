class Device:
    VALID = {"cpu", "cuda"}

    def __init__(self, device_str: str = "cpu"):
        parts = device_str.lower().split(":")
        name = parts[0]
        if name not in self.VALID:
            raise ValueError(f"Unknown device '{device_str}'. Valid: {self.VALID}")
        self.type = name
        self.index = int(parts[1]) if len(parts) > 1 else 0

    @property
    def is_cpu(self) -> bool:
        return self.type == "cpu"

    @property
    def is_cuda(self) -> bool:
        return self.type == "cuda"

    def __repr__(self) -> str:
        if self.type == "cpu":
            return "device(cpu)"
        return f"device(cuda:{self.index})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        return False