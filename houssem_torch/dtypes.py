class DType:
    def __init__(self, name: str, numpy_name: str, itemsize: int):
        self.name = name
        self.numpy_name = numpy_name
        self.itemsize = itemsize

    def __repr__(self) -> str:
        return f"houssem_torch.{self.name}"

    def __eq__(self, other) -> bool:
        return isinstance(other, DType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


# Supported dtypes for now — extend later
float32 = DType("float32", "float32", 4)
int64   = DType("int64",   "int64",   8)

DTYPE_MAP = {
    "float32": float32,
    "int64":   int64,
}