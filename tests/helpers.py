import functools
from io import BytesIO
from itertools import product
import random
import sys
import time
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


# Copied from: https://github.com/huggingface/transformers/blob/2d92db8458f7143f64f9b13cbcfee5eb8d0cab90/src/transformers/testing_utils.py#L2131
def is_flaky(max_attempts=5, wait_before_retry=None, description=None):
    """
    To decorate flaky tests. They will be retried on failures.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
        description (`str`, *optional*):
            A string to describe the situation (what / where / why is flaky, link to GH issue/PR comments, errors,
            etc.)
    """

    def decorator(test_func_ref):
        @functools.wraps(test_func_ref)
        def wrapper(*args, **kwargs):
            retry_count = 1

            while retry_count < max_attempts:
                try:
                    return test_func_ref(*args, **kwargs)

                except Exception as err:
                    print(f"Test failed with {err} at try {retry_count}/{max_attempts}.", file=sys.stderr)
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    retry_count += 1

            return test_func_ref(*args, **kwargs)

        return wrapper

    return decorator
