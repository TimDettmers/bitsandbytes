# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from bitsandbytes.backends import backends, ensure_backend_is_available
from bitsandbytes.utils import QuantState

from .cextension import HIP_ENVIRONMENT, lib


# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


name2qmap = {}

if lib and lib.compiled_with_cuda:
    """C FUNCTIONS FOR OPTIMIZERS"""
    str2optimizer8bit = {
        "adam": (
            lib.cadam_static_8bit_grad_32,
            lib.cadam_static_8bit_grad_16,
        ),
        "momentum": (
            lib.cmomentum_static_8bit_grad_32,
            lib.cmomentum_static_8bit_grad_16,
        ),
        "rmsprop": (
            lib.crmsprop_static_8bit_grad_32,
            lib.crmsprop_static_8bit_grad_16,
        ),
        "lion": (
            lib.clion_static_8bit_grad_32,
            lib.clion_static_8bit_grad_16,
        ),
        "lamb": (
            lib.cadam_static_8bit_grad_32,
            lib.cadam_static_8bit_grad_16,
        ),
        "lars": (
            lib.cmomentum_static_8bit_grad_32,
            lib.cmomentum_static_8bit_grad_16,
        ),
    }


class GlobalPageManager:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.paged_tensors = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def prefetch_all(self, to_cpu=False):
        # assume the first added, will be the
        # ones that are used first, so swap them in last
        # in the case they are evicted again
        for t in self.paged_tensors[::-1]:
            prefetch_tensor(t, to_cpu)


class CUBLAS_Context:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.context = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def get_context(self, device):
        if device.index not in self.context:
            prev_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            self.context[device.index] = ct.c_void_p(lib.get_context())
            torch.cuda.set_device(prev_device)
        return self.context[device.index]


class Cusparse_Context:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        # self.context = ct.c_void_p(lib.get_cusparse())
        if torch.version.cuda:
            self.context = ct.c_void_p(lib.get_cusparse())
        elif torch.version.hip:
            self.context = ct.c_void_p(lib.get_hipsparse())

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


dtype2bytes = {}
dtype2bytes[torch.float32] = 4
dtype2bytes[torch.float16] = 2
dtype2bytes[torch.bfloat16] = 2
dtype2bytes[torch.uint8] = 1
dtype2bytes[torch.int8] = 1

FIRST_CUDA_DEVICE = torch.device("cuda", index=0)


def get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE):
    num_bytes = dtype2bytes[dtype] * prod(shape)
    cuda_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
    c_ptr = ct.cast(cuda_ptr, ct.POINTER(ct.c_int))
    new_array = np.ctypeslib.as_array(c_ptr, shape=shape)
    out = torch.frombuffer(new_array, dtype=dtype, count=prod(shape)).view(shape)
    out.is_paged = True
    out.page_deviceid = device.index
    return out


def prefetch_tensor(A, to_cpu=False):
    assert A.is_paged, "Only paged tensors can be prefetched!"
    if to_cpu:
        deviceid = -1
    else:
        deviceid = A.page_deviceid

    num_bytes = dtype2bytes[A.dtype] * A.numel()
    lib.cprefetch(get_ptr(A), ct.c_size_t(num_bytes), ct.c_int32(deviceid))


def elementwise_func(func_name, A, B, value, prefetch=True):
    func = None
    if A.dtype == torch.float32:
        func = getattr(lib, f"c{func_name}_fp32", None)
        cvalue = ct.c_float(value)
    elif A.dtype == torch.uint8:
        func = getattr(lib, f"c{func_name}_uint8", None)
        cvalue = ct.c_uint8(value)

    if func is None:
        raise NotImplementedError(f"Function not implemented: {func_name}")

    is_managed = getattr(A, "is_managed", False)
    if is_managed and prefetch:
        prefetch_tensor(A)
        if B is not None:
            prefetch_tensor(B)

    func(get_ptr(A), get_ptr(B), cvalue, ct.c_int64(A.numel()))
    if A.is_paged or B.is_paged:
        # paged function are fully asynchronous
        # if we return from this function, we want to the tensor
        # to be in the correct state, that is the final state after the
        # operation occurred. So we synchronize.
        torch.cuda.synchronize()


def fill(A, value, device=None, prefetch=True):
    elementwise_func("fill", A, None, value)


def arange(A, device=None):
    elementwise_func("arange", A, None, 0)


def _mul(A, B, device=None):
    elementwise_func("_mul", A, B, 0)


def create_linear_map(signed=True, total_bits=8, add_zero=True):
    sign = -1.0 if signed else 0.0
    total_values = 2**total_bits
    if add_zero or total_bits < 8:
        # add a zero
        # since we simulate less bits by having zeros in the data type, we
        # we need to center the quantization around zero and as such lose
        # a single value
        total_values = 2**total_bits if not signed else 2**total_bits - 1

    values = torch.linspace(sign, 1.0, total_values)
    gap = 256 - values.numel()
    if gap == 0:
        return values
    else:
        l = values.numel() // 2  # noqa: E741
        return torch.Tensor(values[:l].tolist() + [0] * gap + values[l:].tolist())


def create_normal_map(offset=0.9677083, use_extra_value=True):
    try:
        from scipy.stats import norm
    except ImportError as ie:
        raise ImportError(
            "Scipy is required for `create_normal_map`. Install `bitsandbytes` with the `[test]` extra.",
        ) from ie

    if use_extra_value:
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1]).tolist()
        v2 = [0] * (256 - 15)  ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, 8)[:-1]).tolist()
        v2 = [0] * (256 - 14)  ## we have 14 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()

    v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 256

    return values


def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e + p == total_bits - has_sign
    # the exponent is biased to 2^(e-1) -1 == 0
    evalues = []
    pvalues = []
    for i, val in enumerate(range(-(2 ** (exponent_bits - has_sign)), 2 ** (exponent_bits - has_sign), 1)):
        evalues.append(2**val)

    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    # for ev in evalues:
    bias = 2 ** (exponent_bits - 1)
    for evalue in range(2 ** (exponent_bits)):
        for bit_pattern in lst:
            value = 1 if evalue != 0 else 0
            for i, pval in enumerate(list(bit_pattern)):
                value += pval * (2 ** -(i + 1))
            if evalue == 0:
                # subnormals
                value = value * 2**-(bias)
            else:
                # normals
                value = value * 2 ** -(evalue - bias - 1)
            values.append(value)
            if signed:
                values.append(-value)

    assert len(values) == 2**total_bits
    values.sort()
    if total_bits < 8:
        gap = 256 - len(values)
        for i in range(gap):
            values.append(0)
    values.sort()
    code = torch.Tensor(values)
    code /= code.max()

    return code


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return Tensor(data)


def create_quantile_map(A, total_bits=8):
    q = estimate_quantiles(A, num_quantiles=2**total_bits - 1)
    q = q.tolist()
    q.append(0)

    gap = 256 - len(q)
    for i in range(gap):
        q.append(0)

    q.sort()

    q = Tensor(q)
    q = q / q.abs().max()
    return q


def get_special_format_str():
    if not torch.cuda.is_available():
        return "col_turing"
    major, _minor = torch.cuda.get_device_capability()
    if major <= 7:
        return "col_turing"
    if major == 8:
        return "col_ampere"
    return "col_turing"


def is_on_gpu(tensors):
    on_gpu = True
    gpu_ids = set()
    for t in tensors:
        if t is None:
            continue  # NULL pointers are fine
        is_paged = getattr(t, "is_paged", False)
        on_gpu &= t.device.type == "cuda" or is_paged
        if not is_paged:
            gpu_ids.add(t.device.index)
    if not on_gpu:
        raise TypeError(
            f"All input tensors need to be on the same GPU, but found some tensors to not be on a GPU:\n {[(t.shape, t.device) for t in tensors]}",
        )
    if len(gpu_ids) > 1:
        raise TypeError(
            f"Input tensors need to be on the same GPU, but found the following tensor and device combinations:\n {[(t.shape, t.device) for t in tensors]}",
        )
    return on_gpu


def get_ptr(A: Optional[Tensor]) -> Optional[ct.c_void_p]:
    """
    Get the ctypes pointer from a PyTorch Tensor.

    Parameters
    ----------
    A : torch.tensor
        The PyTorch tensor.

    Returns
    -------
    ctypes.c_void_p
    """
    if A is None:
        return None
    else:
        return ct.c_void_p(A.data.data_ptr())


def pre_call(device):
    prev_device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    return prev_device


def post_call(prev_device):
    torch.cuda.set_device(prev_device)


def get_transform_func(dtype, orderA, orderOut, transpose=False):
    name = f'ctransform_{(8 if dtype == torch.int8 else 32)}_{orderA}_to_{orderOut}_{"t" if transpose else "n"}'
    if not hasattr(lib, name):
        print(name)
        raise ValueError(
            f"Transform function not supported: {orderA} to {orderOut} for data type {dtype} and transpose={transpose}",
        )
    else:
        return getattr(lib, name)


def get_transform_buffer(shape, dtype, device, to_order, from_order="row", transpose=False):
    # init_func = torch.empty
    init_func = torch.zeros
    dims = len(shape)

    if dims == 2:
        rows = shape[0]
    elif dims == 3:
        rows = shape[0] * shape[1]
    cols = shape[-1]

    if transpose:
        # swap dims
        tmp = rows
        rows = cols
        cols = tmp
        shape = shape[::-1]
    state = (shape, to_order)

    if to_order == "row" or to_order == "col":
        return init_func(shape, dtype=dtype, device=device), state
    elif to_order == "col32":
        # blocks of 32 columns (padded)
        cols = 32 * ((cols + 31) // 32)
        return init_func((rows, cols), dtype=dtype, device=device), state
    elif to_order == "col_turing":
        # blocks of 32 columns and 8 rows
        cols = 32 * ((cols + 31) // 32)
        rows = 8 * ((rows + 7) // 8)
        return init_func((rows, cols), dtype=dtype, device=device), state
    elif to_order == "col_ampere":
        # blocks of 32 columns and 32 rows
        cols = 32 * ((cols + 31) // 32)
        rows = 32 * ((rows + 31) // 32)
        return init_func((rows, cols), dtype=dtype, device=device), state
    else:
        raise NotImplementedError(f"To_order not supported: {to_order}")


def nvidia_transform(
    A,
    to_order,
    from_order="row",
    out=None,
    transpose=False,
    state=None,
    ld=None,
):
    if HIP_ENVIRONMENT:
        # col32/col_turing/col_ampere are not applicable to ROCm
        # Use col format instead
        to_order = "col" if to_order in ["col32", "col_turing", "col_ampere"] else to_order
        from_order = "col" if from_order in ["col32", "col_turing", "col_ampere"] else from_order

    if state is None:
        state = (A.shape, from_order)
    else:
        from_order = state[1]
    if out is None:
        out, new_state = get_transform_buffer(state[0], A.dtype, A.device, to_order, state[1], transpose)
    else:
        new_state = (state[1], to_order)
    func = get_transform_func(A.dtype, from_order, to_order, transpose)

    shape = state[0]
    if len(shape) == 2:
        dim1 = ct.c_int32(shape[0])
        dim2 = ct.c_int32(shape[1])
    elif ld is not None:
        n = prod(shape)
        dim1 = prod([shape[i] for i in ld])
        dim2 = ct.c_int32(n // dim1)
        dim1 = ct.c_int32(dim1)
    else:
        dim1 = ct.c_int32(shape[0] * shape[1])
        dim2 = ct.c_int32(shape[2])

    ptr = CUBLAS_Context.get_instance().get_context(A.device)
    func(ptr, get_ptr(A), get_ptr(out), dim1, dim2)

    return out, new_state


def estimate_quantiles(
    A: Tensor,
    out: Optional[torch.Tensor] = None,
    offset: float = 1 / 512,
    num_quantiles=256,
) -> Tensor:
    """
    Estimates 256 equidistant quantiles on the input tensor eCDF.

    Uses SRAM-Quantiles algorithm to quickly estimate 256 equidistant quantiles
    via the eCDF of the input tensor `A`. This is a fast but approximate algorithm
    and the extreme quantiles close to 0 and 1 have high variance / large estimation
    errors. These large errors can be avoided by using the offset variable which trims
    the distribution. The default offset value of 1/512 ensures minimum entropy encoding -- it
    trims 1/512 = 0.2% from each side of the distrivution. An offset value of 0.01 to 0.02
    usually has a much lower error but is not a minimum entropy encoding. Given an offset
    of 0.02 equidistance points in the range [0.02, 0.98] are used for the quantiles.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor. Any shape.
    out : torch.Tensor
        Tensor with the 256 estimated quantiles.
    offset : float
        The offset for the first and last quantile from 0 and 1. Default: 1/(2*num_quantiles)
    num_quantiles : int
        The number of equally spaced quantiles.

    Returns
    -------
    torch.Tensor:
        The 256 quantiles in float32 datatype.
    """
    if A.numel() < 256:
        raise NotImplementedError(
            f"Quantile estimation needs at least 256 values in the Tensor, but Tensor had only {A.numel()} values.",
        )
    if num_quantiles > 256:
        raise NotImplementedError(
            f"Currently only a maximum of 256 equally spaced quantiles are supported, but the argument num_quantiles={num_quantiles}",
        )
    if num_quantiles < 256 and offset == 1 / (512):
        # override default arguments
        offset = 1 / (2 * num_quantiles)

    if out is None:
        out = torch.zeros((256,), dtype=torch.float32, device=A.device)
    is_on_gpu([A, out])
    device = pre_call(A.device)
    if A.dtype == torch.float32:
        lib.cestimate_quantiles_fp32(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    elif A.dtype == torch.float16:
        lib.cestimate_quantiles_fp16(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
    else:
        raise NotImplementedError(f"Not supported data type {A.dtype}")
    post_call(device)

    if num_quantiles < 256:
        step = round(256 / num_quantiles)
        idx = torch.linspace(0, 255, num_quantiles).long().to(A.device)
        out = out[idx]

    return out


# maintain the compatibility as F.QuantState
QuantState = QuantState


def quantize_blockwise(
    A: Tensor,
    code: Optional[torch.Tensor] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=4096,
    nested=False,
) -> Tuple[Tensor, QuantState]:
    """
    Quantize tensor A in blocks of size 4096 values.

    Quantizes tensor A by dividing it into blocks of 4096 values.
    Then the absolute maximum value within these blocks is calculated
    for the non-linear quantization.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        The output tensor (8-bit).

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor.
    tuple(torch.Tensor, torch.Tensor):
        The quantization state to undo the quantization.
    """

    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    if absmax is None:
        n = A.numel()
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)

    if out is None:
        out = torch.zeros_like(A, dtype=torch.uint8)

    if A.device.type != "cpu":
        # Some AMD GPUs have warpsize 64
        # Set min blocksize to 128 (~warpsize 64 in kernel) for HIP
        if not HIP_ENVIRONMENT:
            assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
        else:
            assert blocksize in [4096, 2048, 1024, 512, 256, 128]
        cblocksize = ct.c_int32(blocksize)
        prev_device = pre_call(A.device)
        code = code.to(A.device)
        is_on_gpu([code, A, out, absmax])
        if A.dtype == torch.float32:
            lib.cquantize_blockwise_fp32(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                cblocksize,
                ct.c_int(A.numel()),
            )
        elif A.dtype == torch.float16:
            lib.cquantize_blockwise_fp16(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                cblocksize,
                ct.c_int(A.numel()),
            )
        elif A.dtype == torch.bfloat16:
            lib.cquantize_blockwise_bf16(
                get_ptr(code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                cblocksize,
                ct.c_int(A.numel()),
            )
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        post_call(A.device)
    else:
        # cpu
        code = code.cpu()
        lib.cquantize_blockwise_cpu_fp32(
            get_ptr(code),
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_longlong(blocksize),
            ct.c_longlong(A.numel()),
        )

    if nested:
        offset = absmax.mean()
        absmax -= offset
        qabsmax, state2 = quantize_blockwise(absmax, blocksize=blocksize, nested=False)
        quant_state = QuantState(
            absmax=qabsmax,
            code=code,
            blocksize=blocksize,
            dtype=A.dtype,
            offset=offset,
            state2=state2,
        )
    else:
        quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=A.dtype)

    return out, quant_state


def dequantize_blockwise(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
    nested=False,
) -> Tensor:
    """
    Dequantizes blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in
    blocks of size 4096.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor.
    quant_state : QuantState
        Object with code, absmax and other quantization state components.
    absmax : torch.Tensor
        The absmax values.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        Dequantized output tensor (default: float32)


    Returns
    -------
    torch.Tensor:
        Dequantized tensor (default: float32)
    """
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    if quant_state is None:
        quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=torch.float32)

    absmax = quant_state.absmax
    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()

    if out is None:
        out = torch.empty(A.shape, dtype=quant_state.dtype, device=A.device)

    if A.device.type != "cpu":
        device = pre_call(A.device)
        code = quant_state.code.to(A.device)
        supported_blocksizes = [2048, 4096, 1024, 512, 256, 128, 64]
        # Some AMD GPUs have warpsize 64
        # Set min blocksize to 128 (~warpsize 64 in kernel) for HIP
        if HIP_ENVIRONMENT:
            supported_blocksizes = supported_blocksizes[:-1]
        if quant_state.blocksize not in supported_blocksizes:
            raise ValueError(
                f"The blockwise of {quant_state.blocksize} is not supported. Supported values: {supported_blocksizes}",
            )
        is_on_gpu([A, absmax, out])
        if out.dtype == torch.float32:
            lib.cdequantize_blockwise_fp32(
                get_ptr(quant_state.code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_int(quant_state.blocksize),
                ct.c_int(A.numel()),
            )
        elif out.dtype == torch.float16:
            lib.cdequantize_blockwise_fp16(
                get_ptr(quant_state.code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_int(quant_state.blocksize),
                ct.c_int(A.numel()),
            )
        elif out.dtype == torch.bfloat16:
            lib.cdequantize_blockwise_bf16(
                get_ptr(quant_state.code),
                get_ptr(A),
                get_ptr(absmax),
                get_ptr(out),
                ct.c_int(quant_state.blocksize),
                ct.c_int(A.numel()),
            )
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
        post_call(A.device)
    else:
        code = quant_state.code.cpu()
        lib.cdequantize_blockwise_cpu_fp32(
            get_ptr(code),
            get_ptr(A),
            get_ptr(quant_state.absmax),
            get_ptr(out),
            ct.c_longlong(quant_state.blocksize),
            ct.c_longlong(A.numel()),
        )

    return out


def get_4bit_type(typename, device=None, blocksize=64):
    if device is None:
        device = "cuda"
    data = None
    if typename == "nf4":
        """ Implements the NF4 data type.

            Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
            is normalized into the range [-1, 1].

            For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

            Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
            the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
        """
        data = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ]
    elif typename == "fp4":
        # 0b000 = 0
        # 0b001 = 0.0625
        # 0b010 = 8
        # 0b011 = 12
        # 0b100 = 4
        # 0b101 = 6
        # 0b110 = 2
        # 0b111 = 3
        # can also be created with bnb.functional.create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4)
        data = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
    elif typename == "int4":
        data = [7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7]
    elif typename == "af4":
        # Taken from: NF4 Isn't Information Theoretically Optimal (and that's Good)
        # https://arxiv.org/abs/2306.06965
        if blocksize == 64:
            data = [
                -1.0,
                -0.69441008,
                -0.51243739,
                -0.3736951,
                -0.25607552,
                -0.14982478,
                -0.04934812,
                0.0,
                0.04273164,
                0.12934483,
                0.21961274,
                0.31675666,
                0.42563882,
                0.55496234,
                0.72424863,
                1.0,
            ][::-1]
        else:
            raise NotImplementedError("4-bit AbnormalFloats currently only support blocksize 64.")

    if data is None:
        raise NotImplementedError(f"Typename {typename} not supported")

    data = torch.tensor(data, device=device)
    data.div_(data.abs().max())

    assert data.numel() == 16

    return data


def quantize_fp4(
    A: Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=None,
    compress_statistics=False,
    quant_storage=torch.uint8,
):
    if blocksize is None:
        # Some AMD GPUs have warpsize 64
        # Set default blocksize to 128 (~warpsize 64 in kernel) for HIP
        blocksize = 64 if not HIP_ENVIRONMENT else 128
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "fp4", quant_storage)


def quantize_nf4(
    A: Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=None,
    compress_statistics=False,
    quant_storage=torch.uint8,
):
    if blocksize is None:
        # Some AMD GPUs have warpsize 64
        # Set default blocksize to 128 (~warpsize 64 in kernel) for HIP
        blocksize = 64 if not HIP_ENVIRONMENT else 128
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "nf4", quant_storage)


def quantize_4bit(
    A: Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=None,
    compress_statistics=False,
    quant_type="fp4",
    quant_storage=torch.uint8,
) -> Tuple[Tensor, QuantState]:
    """
    Quantize tensor A in blocks of 4-bit values.

    Quantizes tensor A by dividing it into blocks which are independently quantized to FP4.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        The output tensor.
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}

    Returns
    -------
    torch.Tensor:
        Tensor with packed 4-bit values.
    tuple(torch.Tensor, torch.Size, torch.dtype, int):
        The quantization state to undo the quantization.
    """
    ensure_backend_is_available(A.device.type)
    return backends[A.device.type].quantize_4bit(
        A,
        absmax=absmax,
        out=out,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        quant_storage=quant_storage,
    )


def dequantize_fp4(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
) -> Tensor:
    if blocksize is None:
        # Some AMD GPUs have warpsize 64
        # Set default blocksize to 128 (~warpsize 64 in kernel) for HIP
        blocksize = 64 if not HIP_ENVIRONMENT else 128

    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "fp4")


def dequantize_nf4(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
) -> Tensor:
    if blocksize is None:
        # Some AMD GPUs have warpsize 64
        # Set default blocksize to 128 (~warpsize 64 in kernel) for HIP
        blocksize = 64 if not HIP_ENVIRONMENT else 128

    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "nf4")


def dequantize_4bit(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type="fp4",
) -> Tensor:
    """
    Dequantizes FP4 blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in blocks of size blocksize.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor (packed 4-bit values).
    quant_state : QuantState
        object with quantisation stats, incl. absmax values, original tensor shape and original dtype.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        Dequantized output tensor.
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}


    Returns
    -------
    torch.Tensor:
        Dequantized tensor.
    """
    ensure_backend_is_available(A.device.type)
    return backends[A.device.type].dequantize_4bit(
        A, quant_state=quant_state, absmax=absmax, out=out, blocksize=blocksize, quant_type=quant_type
    )


def quantize(
    A: Tensor,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    absmax = torch.abs(A).max()
    if absmax.dtype != torch.float32:
        absmax = absmax.float()
    inp = A / absmax
    out = quantize_no_absmax(inp, code, out)
    return out, (absmax, code)


def dequantize(
    A: Tensor,
    state: Optional[Tuple[Tensor, Tensor]] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    assert state is not None or absmax is not None
    if code is None and state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    if state is None:
        state = (absmax, code)
    out = dequantize_no_absmax(A, state[1], out)
    return out * state[0]


def quantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    """
    Quantizes input tensor to 8-bit.

    Quantizes the 32-bit input tensor `A` to the 8-bit output tensor
    `out` using the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor, optional
        The output tensor. Needs to be of type byte.

    Returns
    -------
    torch.Tensor:
        Quantized 8-bit tensor.
    """
    prev_device = pre_call(A.device)
    if out is None:
        out = torch.zeros_like(A, dtype=torch.uint8)
    is_on_gpu([A, out])
    lib.cquantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    post_call(prev_device)
    return out


def dequantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    """
    Dequantizes the 8-bit tensor to 32-bit.

    Dequantizes the 8-bit tensor `A` to the 32-bit tensor `out` via
    the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The 8-bit input tensor.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        The 32-bit output tensor.

    Returns
    -------
    torch.Tensor:
        32-bit output tensor.
    """
    prev_device = pre_call(A.device)
    if out is None:
        out = torch.zeros_like(A, dtype=torch.float32)
    is_on_gpu([code, A, out])
    lib.cdequantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    post_call(prev_device)
    return out


def optimizer_update_32bit(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    beta1: float,
    eps: float,
    step: int,
    lr: float,
    state2: Optional[torch.Tensor] = None,
    beta2: float = 0.0,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None,
    max_unorm: float = 0.0,
    skip_zeros=False,
) -> None:
    ensure_backend_is_available(g.device.type)
    return backends[g.device.type].optimizer_update_32bit(
        optimizer_name=optimizer_name,
        g=g,
        p=p,
        state1=state1,
        beta1=beta1,
        eps=eps,
        step=step,
        lr=lr,
        state2=state2,
        beta2=beta2,
        weight_decay=weight_decay,
        gnorm_scale=gnorm_scale,
        unorm_vec=unorm_vec,
        max_unorm=max_unorm,
        skip_zeros=skip_zeros,
    )


def optimizer_update_8bit(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: Tensor,
    qmap2: Optional[torch.Tensor],
    max1: Tensor,
    max2: Optional[torch.Tensor],
    new_max1: Tensor,
    new_max2: Optional[torch.Tensor],
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None,
    max_unorm: float = 0.0,
) -> None:
    """
    Performs an inplace Adam update.

    Universal Adam update for 32/8-bit state and 32/16-bit gradients/weights.
    Uses AdamW formulation if weight decay > 0.0.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer. Choices {adam, momentum}
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Adam state 1.
    state2 : torch.Tensor
        Adam state 2.
    beta1 : float
        Adam beta1.
    beta2 : float
        Adam beta2.
    eps : float
        Adam epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    qmap1 : torch.Tensor
        Quantization map for first Adam state.
    qmap2 : torch.Tensor
        Quantization map for second Adam state.
    max1 : torch.Tensor
        Max value for first Adam state update.
    max2 : torch.Tensor
        Max value for second Adam state update.
    new_max1 : torch.Tensor
        Max value for the next Adam update of the first state.
    new_max2 : torch.Tensor
        Max value for the next Adam update of the second state.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    """

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    prev_device = pre_call(g.device)
    is_on_gpu([g, p, state1, state2, unorm_vec, qmap1, qmap2, max1, max2, new_max1, new_max2])
    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][0](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(max1),
            get_ptr(max2),
            get_ptr(new_max1),
            get_ptr(new_max2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_int32(g.numel()),
        )
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][1](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(max1),
            get_ptr(max2),
            get_ptr(new_max1),
            get_ptr(new_max2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_int32(g.numel()),
        )
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
        )
    post_call(prev_device)


def optimizer_update_8bit_blockwise(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: Tensor,
    qmap2: Optional[torch.Tensor],
    absmax1: Tensor,
    absmax2: Optional[torch.Tensor],
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    skip_zeros=False,
) -> None:
    ensure_backend_is_available(g.device.type)
    return backends[g.device.type].optimizer_update_8bit_blockwise(
        optimizer_name=optimizer_name,
        g=g,
        p=p,
        state1=state1,
        state2=state2,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        step=step,
        lr=lr,
        qmap1=qmap1,
        qmap2=qmap2,
        absmax1=absmax1,
        absmax2=absmax2,
        weight_decay=weight_decay,
        gnorm_scale=gnorm_scale,
        skip_zeros=skip_zeros,
    )


def percentile_clipping(grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int = 5):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    step: int
        The current optimiation steps (number of past gradient norms).

    """
    prev_device = pre_call(grad.device)
    is_on_gpu([grad, gnorm_vec])
    if grad.dtype == torch.float32:
        lib.cpercentile_clipping_g32(
            get_ptr(grad),
            get_ptr(gnorm_vec),
            ct.c_int32(step),
            ct.c_int32(grad.numel()),
        )
    elif grad.dtype == torch.float16:
        lib.cpercentile_clipping_g16(
            get_ptr(grad),
            get_ptr(gnorm_vec),
            ct.c_int32(step),
            ct.c_int32(grad.numel()),
        )
    else:
        raise ValueError(f"Gradient type {grad.dtype} not supported!")
    post_call(prev_device)

    current_gnorm = torch.sqrt(gnorm_vec[step % 100])
    vals, idx = torch.sort(gnorm_vec)
    clip_value = torch.sqrt(vals[percentile])
    gnorm_scale = 1.0

    if current_gnorm > clip_value:
        gnorm_scale = clip_value / current_gnorm

    return current_gnorm, clip_value, gnorm_scale


def histogram_scatter_add_2d(histogram: Tensor, index1: Tensor, index2: Tensor, source: Tensor):
    assert len(histogram.shape) == 2
    assert histogram.dtype == torch.float32
    assert source.dtype == torch.float32
    assert index1.dtype == torch.int32
    assert index2.dtype == torch.int32

    assert histogram.device.type == "cuda"
    assert index1.device.type == "cuda"
    assert index2.device.type == "cuda"
    assert source.device.type == "cuda"

    maxdim1 = ct.c_int32(histogram.shape[0])
    n = ct.c_int32(index1.numel())
    is_on_gpu([histogram, index1, index2, source])
    lib.chistogram_scatter_add_2d(get_ptr(histogram), get_ptr(index1), get_ptr(index2), get_ptr(source), maxdim1, n)


def check_matmul(A, B, out, transposed_A, transposed_B, expected_type=torch.int8):
    if not torch.cuda.is_initialized():
        torch.cuda.init()
    if A.dtype != expected_type or B.dtype != expected_type:
        raise TypeError(f"Expected torch.int8 input tensors A and B, but got {A.dtype} and {B.dtype}")

    sA = A.shape
    sB = B.shape
    tA = transposed_A
    tB = transposed_B

    correct = True

    if len(sA) == 2 and len(sB) == 2:
        if not tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[0] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[0] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[1] != B.shape[1]:
            correct = False
    elif len(sA) == 3 and len(sB) == 2:
        if not tA and not tB and A.shape[2] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[1] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[2] != B.shape[1]:
            correct = False
    elif len(sA) == 3 and len(sB) == 3:
        if not tA and not tB and A.shape[2] != B.shape[1]:
            correct = False
        elif tA and not tB and A.shape[1] != B.shape[1]:
            correct = False
        elif tA and tB and A.shape[1] != B.shape[2]:
            correct = False
        elif not tA and tB and A.shape[2] != B.shape[2]:
            correct = False

    if out is not None:
        sout = out.shape
        # special case common in backprop
        if not correct and len(sA) == 3 and len(sB) == 3:
            if sout[0] == sA[2] and sout[1] == sB[2] and sA[0] == sB[0] and sA[1] == sB[1]:
                correct = True
    else:
        if len(sA) == 2 and len(sB) == 2:
            if not tA and not tB:
                sout = (sA[0], sB[1])
            elif tA and tB:
                sout = (sA[1], sB[0])
            elif tA and not tB:
                sout = (sA[1], sB[1])
            elif not tA and tB:
                sout = (sA[0], sB[0])
        elif len(sA) == 3 and len(sB) == 2:
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[1])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[0])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[1])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[0])
        elif len(sA) == 3 and len(sB) == 3:
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[2])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[1])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[2])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[1])

    if not correct:
        raise ValueError(
            f"Tensor dimensions incorrect for matrix mulitiplication: A x B: {sA} x {sB} with transpose for A x B: {tA} x {tB}.",
        )

    return sout


def gemv_4bit(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
    state=None,
):
    ensure_backend_is_available(A.device.type)
    return backends[A.device.type].gemv_4bit(
        A,
        B,
        out=out,
        transposed_A=transposed_A,
        transposed_B=transposed_B,
        state=state,
    )


def igemm(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
):
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]:
            return batched_igemm(A, B, out)

    sA = A.shape
    sB = B.shape
    if transposed_A and len(sA) == 2:
        sA = (sA[1], sA[0])
    elif transposed_A and len(sA) == 3:
        sA = (sA[0], sA[2], sA[0])
    if transposed_B and len(sB) == 2:
        sB = (sB[1], sB[0])
    elif transposed_B and len(sB) == 3:
        sB = (sB[0], sB[2], sB[0])
    # this is a mess: cuBLAS expect column major, but PyTorch is row major.
    # So to perform the matrix multiplication, we have to treat A, B, and C matrices
    # (transpose of row major is column major)
    # This means we compute B^T A^T = C^T and we explicitly switch the dimensions of each of these

    # matrices in the input arguments for cuBLAS
    # column major: A @ B = C: [m, k] @ [k, n] = [m, n]
    # row major: B^T @ A^T = C^T: [m, k] @ [k, n] = [m, n]
    # column major with row major layout: B^T @ A^T = C^T: [k, m] @ [n, k] = [n, m]
    if len(sB) == 2:
        if B.stride()[0] == B.shape[1]:
            transposed_B = False
        elif B.stride()[1] == B.shape[0]:
            transposed_B = True
        if len(A.shape) == 2:
            if A.stride()[0] == A.shape[1]:
                transposed_A = False
            elif A.stride()[1] == A.shape[0]:
                transposed_A = True
        else:
            if A.stride()[1] == A.shape[2]:
                transposed_A = False
            elif A.stride()[2] == A.shape[1]:
                transposed_A = True

        if len(sA) == 2:
            n = sA[0]
            ldb = A.stride()[1 if transposed_A else 0]
        elif len(sA) == 3 and len(sB) == 2:
            n = sA[0] * sA[1]
            ldb = sA[2]

        m = sB[1]
        k = sB[0]
        lda = B.stride()[(1 if transposed_B else 0)]
        ldc = sB[1]
    elif len(sB) == 3:
        # special case
        assert len(sA) == 3
        if not (sA[0] == sB[0] and sA[1] == sB[1]):
            raise ValueError(
                f"Only bsi,bso->io supported for tensor contractions, but dims for A x B were: {sA} x {sB}",
            )

        transposed_A = True
        transposed_B = False

        m = sB[2]
        n = sA[2]
        k = sB[0] * sB[1]

        lda = m
        ldb = sA[2]
        ldc = m

    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    # B^T @ A^T = C^T
    # [km, nk -> mn]
    is_on_gpu([B, A, out])
    lib.cigemm(
        ptr,
        ct.c_bool(transposed_B),
        ct.c_bool(transposed_A),
        ct.c_int32(m),
        ct.c_int32(n),
        ct.c_int32(k),
        get_ptr(B),
        get_ptr(A),
        get_ptr(out),
        ct.c_int32(lda),
        ct.c_int32(ldb),
        ct.c_int32(ldc),
    )
    return out


def batched_igemm(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
):
    if not len(A.shape) == 3 or not len(B.shape) == 3:
        raise ValueError(f"Expected 3-dimensional tensors for bmm, but got shapes A and B: {A.shape} and {B.shape}")
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)

    if B.is_contiguous():
        lda = B.stride()[1]
        transposed_A = False
    else:
        s = B.stride()
        if s[0] != B.shape[0]:
            B = B.contiguous()
            lda = B.stride()[1]
        elif s[2] == B.shape[1]:
            transposed_A = True
            lda = B.stride()[2]
        else:
            if s[2] == 1:
                B = B.contiguous()
                lda = B.stride()[1]
            elif s[1] == 1:
                B = B.contiguous()
                lda = B.stride()[1]
            else:
                B = B.contiguous()
                lda = B.stride()[1]

    if A.is_contiguous():
        ldb = A.stride()[1]
        transposed_B = False
    else:
        s = A.stride()
        if s[0] != A.shape[0]:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False
        elif s[2] == A.shape[1]:
            ldb = A.stride()[2]
            transposed_B = True
        else:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False

    # this is a mess: cuBLAS expect column major, but PyTorch is row major.
    # So to perform the matrix multiplication, we have to treat A, B, and C matrices
    # (transpose of row major is column major)
    # This means we compute B^T A^T = C^T and we explicitly switch the dimensions of each of these
    # matrices in the input arguments for cuBLAS

    # column major: A @ B = C: [batch, m, k] @ [batch, k, n] = [batch, m, n]
    # row major: B^T @ A^T = C^T: [batch, m, k] @ [batch, k, n] = [batch, m, n]
    # column major with row major layout: B^T @ A^T = C^T: [batch, k, m] @ [batch, n, k] = [batch, n, m]
    num_batch = A.shape[0]
    n = A.shape[1]
    m = B.shape[2]
    k = B.shape[1]

    ldc = m

    strideA = B.shape[1] * B.shape[2]
    strideB = A.shape[1] * A.shape[2]
    strideC = A.shape[1] * B.shape[2]

    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    is_on_gpu([B, A, out])
    lib.cbatched_igemm(
        ptr,
        ct.c_bool(transposed_B),
        ct.c_bool(transposed_A),
        ct.c_int32(m),
        ct.c_int32(n),
        ct.c_int32(k),
        get_ptr(B),
        get_ptr(A),
        get_ptr(out),
        ct.c_int32(lda),
        ct.c_int32(ldb),
        ct.c_int32(ldc),
        ct.c_long(strideA),
        ct.c_long(strideB),
        ct.c_long(strideC),
        ct.c_uint32(num_batch),
    )
    return out


def igemmlt(A, B, SA, SB, out=None, Sout=None, dtype=torch.int32):
    ensure_backend_is_available(A.device.type)
    return backends[A.device.type].igemmlt(A, B, SA, SB, out=out, Sout=Sout, dtype=dtype)


def mm_dequant(A, quant_state, row_stats, col_stats, out=None, new_row_stats=None, new_col_stats=None, bias=None):
    ensure_backend_is_available(A.device.type)
    return backends[A.device.type].mm_dequant(
        A,
        quant_state,
        row_stats,
        col_stats,
        out=out,
        new_row_stats=new_row_stats,
        new_col_stats=new_col_stats,
        bias=bias,
    )


def get_colrow_absmax(A, row_stats=None, col_stats=None, nnz_block_ptr=None, threshold=0.0):
    assert A.dtype == torch.float16
    device = A.device

    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    col_tiles = (cols + 255) // 256
    tiled_rows = ((rows + 15) // 16) * 16
    if row_stats is None:
        row_stats = torch.empty((rows,), dtype=torch.float32, device=device).fill_(-50000.0)
    if col_stats is None:
        col_stats = torch.empty((cols,), dtype=torch.float32, device=device).fill_(-50000.0)

    if nnz_block_ptr is None and threshold > 0.0:
        nnz_block_ptr = torch.zeros(((tiled_rows * col_tiles) + 1,), dtype=torch.int32, device=device)

    ptrA = get_ptr(A)
    ptrRowStats = get_ptr(row_stats)
    ptrColStats = get_ptr(col_stats)
    ptrNnzrows = get_ptr(nnz_block_ptr)
    rows = ct.c_int32(rows)
    cols = ct.c_int32(cols)

    prev_device = pre_call(A.device)
    is_on_gpu([A, row_stats, col_stats, nnz_block_ptr])
    lib.cget_col_row_stats(ptrA, ptrRowStats, ptrColStats, ptrNnzrows, ct.c_float(threshold), rows, cols)
    post_call(prev_device)

    if threshold > 0.0:
        nnz_block_ptr.cumsum_(0)

    return row_stats, col_stats, nnz_block_ptr


class COOSparseTensor:
    def __init__(self, rows, cols, nnz, rowidx, colidx, values):
        assert rowidx.dtype == torch.int32
        assert colidx.dtype == torch.int32
        if values.device == torch.device("cpu") or torch.device("xpu"):
            assert values.dtype in [torch.bfloat16, torch.half, torch.float]
        else:
            assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colidx.numel() == nnz

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowidx = rowidx
        self.colidx = colidx
        self.values = values


class CSRSparseTensor:
    def __init__(self, rows, cols, nnz, rowptr, colidx, values):
        assert rowptr.dtype == torch.int32
        assert colidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert colidx.numel() == nnz
        assert rowptr.numel() == rows + 1

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowptr = rowptr
        self.colidx = colidx
        self.values = values


class CSCSparseTensor:
    def __init__(self, rows, cols, nnz, colptr, rowidx, values):
        assert colptr.dtype == torch.int32
        assert rowidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colptr.numel() == cols + 1

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.colptr = colptr
        self.rowidx = rowidx
        self.values = values


def coo2csr(cooA):
    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    values.add_(1)
    rowptr = torch.zeros((cooA.rows + 1,), dtype=torch.int32, device=cooA.rowidx.device)
    rowptr.scatter_(index=values.long(), src=counts.int(), dim=0)
    rowptr.cumsum_(0)
    return CSRSparseTensor(cooA.rows, cooA.cols, cooA.nnz, rowptr, cooA.colidx, cooA.values)


def coo2csc(cooA):
    val, col2rowidx = torch.sort(cooA.colidx)
    rowidx = cooA.rowidx[col2rowidx]
    values = cooA.values[col2rowidx]
    colvalues, counts = torch.unique(val, return_counts=True)
    colvalues.add_(1)
    colptr = torch.zeros((cooA.cols + 1,), dtype=torch.int32, device=cooA.colidx.device)
    colptr.scatter_(index=colvalues.long(), src=counts.int(), dim=0)
    colptr.cumsum_(0)
    return CSCSparseTensor(cooA.rows, cooA.cols, cooA.nnz, colptr, rowidx, values)


def coo_zeros(rows, cols, nnz, device, dtype=torch.half):
    rowidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    colidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    values = torch.zeros((nnz,), dtype=dtype, device=device)
    return COOSparseTensor(rows, cols, nnz, rowidx, colidx, values)


def double_quant(A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0):
    ensure_backend_is_available(A.device.type)
    return backends[A.device.type].double_quant(
        A, col_stats=col_stats, row_stats=row_stats, out_col=out_col, out_row=out_row, threshold=threshold
    )


def transform(A, to_order, from_order="row", out=None, transpose=False, state=None, ld=None):
    ensure_backend_is_available(A.device.type)
    return backends[A.device.type].transform(
        A, to_order, from_order=from_order, out=out, transpose=transpose, state=state, ld=ld
    )


def spmm_coo(cooA, B, out=None):
    if out is None:
        out = torch.empty((cooA.rows, B.shape[1]), device=B.device, dtype=B.dtype)
    nnz = cooA.nnz
    assert cooA.rowidx.numel() == nnz
    assert cooA.colidx.numel() == nnz
    assert cooA.values.numel() == nnz
    assert cooA.cols == B.shape[0]

    transposed_B = False if B.is_contiguous() else True

    ldb = B.stride()[(1 if transposed_B else 0)]
    ldc = B.shape[1]

    ptr = Cusparse_Context.get_instance().context

    ptrRowidx = get_ptr(cooA.rowidx)
    ptrColidx = get_ptr(cooA.colidx)
    ptrValues = get_ptr(cooA.values)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    cnnz = ct.c_int32(cooA.nnz)
    crowsA = ct.c_int32(cooA.rows)
    ccolsA = ct.c_int32(cooA.cols)
    ccolsB = ct.c_int32(B.shape[1])
    cldb = ct.c_int32(ldb)
    cldc = ct.c_int32(ldc)

    is_on_gpu([cooA.rowidx, cooA.colidx, cooA.values, B, out])
    lib.cspmm_coo(
        ptr,
        ptrRowidx,
        ptrColidx,
        ptrValues,
        cnnz,
        crowsA,
        ccolsA,
        ccolsB,
        cldb,
        ptrB,
        cldc,
        ptrC,
        ct.c_bool(transposed_B),
    )

    return out


def spmm_coo_very_sparse(cooA, B, dequant_stats=None, out=None):
    if out is None:
        out = torch.zeros((cooA.rows, B.shape[1]), device=B.device, dtype=cooA.values.dtype)
    nnz = cooA.nnz
    prev_device = pre_call(B.device)
    assert cooA.rowidx.numel() == nnz
    assert cooA.colidx.numel() == nnz
    assert cooA.values.numel() == nnz
    assert cooA.cols == B.shape[0], f"{cooA.cols} vs {B.shape}"

    transposed_B = False if B.is_contiguous() else True

    ldb = B.stride()[(1 if transposed_B else 0)]
    ldc = B.shape[1]

    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    offset = counts.cumsum(0).int()
    max_count, max_idx = torch.sort(counts, descending=True)
    max_idx = max_idx.int()
    max_count = max_count.int()
    assert max_count[0] <= 32, f"Current max count per row is 8 but found {max_count[0]}."
    assert B.dtype in [torch.float16, torch.int8]
    ptrOffset = get_ptr(offset)
    ptrMaxCount = get_ptr(max_count)
    ptrMaxIdx = get_ptr(max_idx)

    ptrRowidx = get_ptr(cooA.rowidx)
    ptrColidx = get_ptr(cooA.colidx)
    ptrValues = get_ptr(cooA.values)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    ptrDequantStats = get_ptr(dequant_stats)
    cnnz_rows = ct.c_int32(counts.numel())
    cnnz = ct.c_int32(cooA.nnz)
    crowsA = ct.c_int32(cooA.rows)
    ccolsA = ct.c_int32(cooA.cols)
    crowsB = ct.c_int32(B.shape[1])
    ccolsB = ct.c_int32(B.shape[1])
    cldb = ct.c_int32(ldb)
    cldc = ct.c_int32(ldc)

    is_on_gpu([cooA.rowidx, cooA.colidx, cooA.values, B, out, dequant_stats])
    if B.dtype == torch.float16:
        lib.cspmm_coo_very_sparse_naive_fp16(
            ptrMaxCount,
            ptrMaxIdx,
            ptrOffset,
            ptrRowidx,
            ptrColidx,
            ptrValues,
            ptrB,
            ptrC,
            ptrDequantStats,
            cnnz_rows,
            cnnz,
            crowsA,
            crowsB,
            ccolsB,
        )
    elif B.dtype == torch.int8:
        lib.cspmm_coo_very_sparse_naive_int8(
            ptrMaxCount,
            ptrMaxIdx,
            ptrOffset,
            ptrRowidx,
            ptrColidx,
            ptrValues,
            ptrB,
            ptrC,
            ptrDequantStats,
            cnnz_rows,
            cnnz,
            crowsA,
            crowsB,
            ccolsB,
        )
    # else: assertion error
    post_call(prev_device)

    return out


C = 127.0


def vectorwise_quant(x, dim=1, quant_type="vector"):
    if quant_type == "linear":
        max1 = torch.abs(x).max().float()
        xq = torch.round(x / max1 * 127).to(torch.int8)
        return xq, max1
    elif quant_type in ["vector", "row"]:
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        xq = torch.clamp(torch.round(x * (C / max1)), -128, 127).to(torch.int8)
        return xq, max1
    elif quant_type == "zeropoint":
        dtype = x.dtype
        x = x.float()
        dyna = x.max() - x.min()
        if dyna == 0:
            dyna = 1
        qx = 255.0 / dyna
        minx = x.min()
        zpx = torch.round(minx * qx)
        x = torch.round(qx * x - zpx) + zpx
        return x, qx
    elif quant_type in ["vector-zeropoint", "row-zeropoint"]:
        dtype = x.dtype
        x = x.float()
        dyna = torch.amax(x, dim=dim, keepdim=True) - torch.amin(x, dim=dim, keepdim=True)
        dyna[dyna == 0] = 1
        qx = 255.0 / dyna
        minx = torch.amin(x, dim=dim, keepdim=True)
        zpx = torch.round(minx * qx)
        x = torch.round(qx * x - zpx) + zpx
        return x, qx
    elif quant_type == "truncated-vector":
        with torch.no_grad():
            absx = torch.abs(x)
            max1 = torch.amax(absx, dim=dim, keepdim=True)
            max1 = max1 * 0.7
            idx = absx > max1.expand_as(absx)
            sign = torch.sign(x[idx])
            x[idx] = max1.expand_as(absx)[idx] * sign
            xq = torch.round(x / max1 * C).to(torch.int8)
        return xq, max1
    else:
        return None


def vectorwise_dequant(xq, max1, quant_type="vector"):
    if quant_type == "vector":
        x = (xq / C * max1).to(torch.float32)
        return x
    else:
        return None


def vectorwise_mm_dequant(xq, S1, S2, dtype=torch.half, quant_type="vector"):
    if quant_type == "linear":
        norm = S1 * S2 / (C * C)
        # double cast needed to prevent overflows
        return (xq.float() * norm).to(dtype)
    elif quant_type == "zeropoint":
        norm = 1.0 / (S1 * S2)
        return (xq.float() * norm).to(dtype)
    elif quant_type == "row-zeropoint":
        norm = 1.0 / (S1 * S2)
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= norm
        else:
            x *= norm
        return x.to(dtype)
    elif quant_type == "vector-zeropoint":
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= 1.0 / S1
        else:
            x *= 1.0 / S1
        x *= 1.0 / S2.t()
        return x.to(dtype)
    elif quant_type == "row":
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1 * S2 / (C * C)
        else:
            x *= S1 * S2 / (C * C)
        return x.to(dtype)
    elif quant_type in ["truncated-vector", "vector"]:
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1 / C
        else:
            x *= S1 / C
        x *= S2 / C
        return x.to(dtype)
    else:
        return None


def dequant_min_max(xq, A, B, SA, SB, dtype=torch.half):
    offset = B.float().t().sum(0) * (SA[0] + SA[1])
    x = xq.float()
    if len(xq.shape) == 2 and len(SB.shape) == 3:
        SB = SB.squeeze(0)
    if len(SB.shape) == 2:
        x *= SB.t() / 127
    else:
        x *= SB / 127
    x *= SA[1] / 127
    x += offset
    return x.to(dtype)


def extract_outliers(A, SA, idx):
    ensure_backend_is_available(A.device.type)
    return backends[A.device.type].extract_outliers(A, SA, idx)


def pipeline_test(A, batch_size):
    out = torch.zeros_like(A)
    lib.cpipeline_test(get_ptr(A), get_ptr(out), ct.c_size_t(A.numel()), ct.c_size_t(batch_size))
    return out
