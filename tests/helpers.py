from io import BytesIO
from itertools import product
import random
from typing import Any, List

import torch

test_dims_rng = random.Random(42)


TRUE_FALSE = (True, False)
BOOLEAN_TRIPLES = list(product(TRUE_FALSE, repeat=3))  # all combinations of (bool, bool, bool)
BOOLEAN_TUPLES = list(product(TRUE_FALSE, repeat=2))  # all combinations of (bool, bool)


def torch_save_to_buffer(obj):
    buffer = BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return buffer


def torch_load_from_buffer(buffer):
    buffer.seek(0)
    obj = torch.load(buffer)
    buffer.seek(0)
    return obj


def get_test_dims(min: int, max: int, *, n: int) -> List[int]:
    return [test_dims_rng.randint(min, max) for _ in range(n)]


def format_with_label(label: str, value: Any) -> str:
    if isinstance(value, bool):
        formatted = "T" if value else "F"
    elif isinstance(value, (list, tuple)) and all(isinstance(v, bool) for v in value):
        formatted = "".join("T" if b else "F" for b in value)
    else:
        formatted = str(value)
    return f"{label}={formatted}"


def id_formatter(label: str):
    """
    Return a function that formats the value given to it with the given label.
    """
    return lambda value: format_with_label(label, value)


DTYPE_NAMES = {
    torch.bfloat16: "bf16",
    torch.bool: "bool",
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int8: "int8",
}


def describe_dtype(dtype: torch.dtype) -> str:
    return DTYPE_NAMES.get(dtype) or str(dtype).rpartition(".")[2]


def get_blocksizes(hip_env: bool) -> List[int]:
    if not hip_env:
        return [4096, 2048, 1024, 512, 256, 128, 64]
    else:
        return [4096, 2048, 1024, 512, 256, 128]
