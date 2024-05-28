import dataclasses
import logging
import re
import subprocess
from typing import List, Optional, Tuple

import torch


@dataclasses.dataclass(frozen=True)
class CUDASpecs:
    highest_compute_capability: Tuple[int, int]
    cuda_version_string: str
    cuda_version_tuple: Tuple[int, int]

    @property
    def has_cublaslt(self) -> bool:
        return self.highest_compute_capability >= (7, 5)


def get_compute_capabilities() -> List[Tuple[int, int]]:
    return sorted(torch.cuda.get_device_capability(torch.cuda.device(i)) for i in range(torch.cuda.device_count()))


def get_cuda_version_tuple() -> Tuple[int, int]:
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION
    if torch.version.cuda:
        major, minor = map(int, torch.version.cuda.split("."))
    elif torch.version.hip:
        major, minor = map(int, torch.version.hip.split(".")[0:2])
    return major, minor


def get_cuda_version_string() -> str:
    major, minor = get_cuda_version_tuple()
    return f"{major}{minor}"


def get_cuda_specs() -> Optional[CUDASpecs]:
    if not torch.cuda.is_available():
        return None

    return CUDASpecs(
        highest_compute_capability=(get_compute_capabilities()[-1]),
        cuda_version_string=(get_cuda_version_string()),
        cuda_version_tuple=get_cuda_version_tuple(),
    )


def get_rocm_gpu_arch() -> str:
    logger = logging.getLogger(__name__)
    try:
        if torch.version.hip:
            result = subprocess.run(["rocminfo"], capture_output=True, text=True)
            match = re.search(r"Name:\s+gfx([a-zA-Z\d]+)", result.stdout)
            if match:
                return "gfx" + match.group(1)
            else:
                return "unknown"
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"Could not detect ROCm GPU architecture: {e}")
        if torch.cuda.is_available():
            logger.warning(
                """
ROCm GPU architecture detection failed despite ROCm being available.
                """,
            )
        return "unknown"
