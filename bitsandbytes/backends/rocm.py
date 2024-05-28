from .cuda import CUDABackend


class ROCmBackend(CUDABackend):
    """
    Backend for AMD ROCm implementation.

    The interface is largely the same as the CUDA implementation, so only any
    differences need to be implemented here.
    """

    pass
