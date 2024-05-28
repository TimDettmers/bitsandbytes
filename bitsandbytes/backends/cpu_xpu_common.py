from typing import Optional
import warnings

import torch

from bitsandbytes.functional import (
    QuantState,
    get_4bit_type,
)

try:
    # to support Intel CPU/GPU (XPU) backend
    import intel_extension_for_pytorch as ipex

    ipex_cpu = ipex if ipex._C._has_cpu() else None
    ipex_xpu = ipex if ipex._C._has_xpu() else None
except BaseException:
    ipex_cpu = None
    ipex_xpu = None


Tensor = torch.Tensor


def _torch_version_prereq(major, minor):
    ver_major = int(torch.__version__.split(".")[0])
    ver_minor = int(torch.__version__.split(".")[1])
    return ver_major * 32 + ver_minor >= major * 32 + minor


def _ipex_cpu_version_prereq(major, minor):
    if ipex_cpu is not None:
        ver_major = ipex_cpu.__version__.split(".")[0]
        ver_minor = ipex_cpu.__version__.split(".")[1]
        return int(ver_major) * 32 + int(ver_minor) >= major * 32 + minor
    return False


def _ipex_xpu_version_prereq(major, minor):
    if ipex_xpu is not None:
        ver_major = ipex_xpu.__version__.split(".")[0]
        ver_minor = ipex_xpu.__version__.split(".")[1]
        return int(ver_major) * 32 + int(ver_minor) >= major * 32 + minor
    return False


def _maybe_torch_compile(func):
    # torch.compile requires pytorch >= 2.0
    if _torch_version_prereq(2, 0):
        options = {}
        # fx_graph_cache requires pytorch >= 2.2
        if _torch_version_prereq(2, 2):
            options.update({"fx_graph_cache": True})
        return torch.compile(func, dynamic=True, options=options)
    return func


# Don't use torch.compile for now due to PyTorch issue https://github.com/pytorch/pytorch/issues/124382
def double_quant_impl(A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0):
    """
    Find absolute max values of each row/column of a tensor, and symmetrically quantize it to int8.
    If threshold > 0.0, only values <= threshold are counted. All outliers are zeroed out in
    the original tensor and they are kept in COO format: (rows, cols, values)
    If threshold == 0.0, there are no outliers.
    Args:
        A The tensor to be analyzed and quantized.
        col_stats Absolute max values of each column of A. If it is not None, use the values directly.
            Otherwise, find the values.
        row_stats Absolute max values of each row of A. If it is not None, use the values directly.
            Otherwise, find the values.
        out_col Output buffer for the result quantized per column if it is not None
        out_row Output buffer for the result quantized per row if it is not None
        threshold The threshold for finding outliers if it is > 0.0. Otherwise it has no effect.
    Return:
        A tuple of output quantized per row, output quantized per column, absolute max values of
        each row of A, absolute max values of each column of A, outliers in COO format
    """
    from ..functional import COOSparseTensor

    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        assert A.dim() == 2, f"double_quant: Input tensor should be 2d or 3d but got {A.dim()}d"
        rows = A.shape[0]
    A = A.reshape(rows, cols)

    coo_tensor = None

    def get_row_col_stats(A):
        row_stats = torch.max(torch.abs(A), 1).values  # absolute max of each row
        col_stats = torch.max(torch.abs(A), 0).values  # absolute max of each col
        return row_stats, col_stats

    def quant_to_int8(A, stats):
        return torch.clamp(torch.round(A * (127.0 / stats)), -128, 127).to(torch.int8)

    if threshold == 0.0:
        if row_stats is None or col_stats is None:
            row_stats, col_stats = get_row_col_stats(A)
    else:
        outlier_indices = torch.abs(A) >= threshold  # find outliers
        outlier_coord = outlier_indices.nonzero()  # get outlier coordinates
        outlier_rows = outlier_coord[:, 0]  # outlier row for COO sparse tensor
        outlier_cols = outlier_coord[:, 1]  # outlier column for COO sparse tensor
        outlier_values = A[outlier_indices]  # outlier values for COO sparse tensor
        coo_tensor = COOSparseTensor(
            A.shape[0], A.shape[1], outlier_values.numel(), outlier_rows.int(), outlier_cols.int(), outlier_values
        )
        if row_stats is None or col_stats is None:
            A[outlier_indices] = 0  # zero out outliers
            row_stats, col_stats = get_row_col_stats(A)

    quant_by_row = quant_to_int8(A, row_stats.unsqueeze(-1))
    quant_by_col = quant_to_int8(A, col_stats.unsqueeze(0))

    if coo_tensor is not None:
        A[outlier_indices] = outlier_values  # restore outliers for later use

    if out_row is not None:
        out_row.copy_(quant_by_row)
    else:
        out_row = quant_by_row
    if out_col is not None:
        out_col.copy_(quant_by_col)
    else:
        out_col = quant_by_col
    # Return float stats to align with CUDA impl
    return out_row, out_col, row_stats.float(), col_stats.float(), coo_tensor


def igemmlt_impl(A, B, SA=None, SB=None, out=None, Sout=None, dtype=torch.int32):
    """
    Do GEMMM computation. Data type: int8 * int8 -> int32.
    Args:
        A Activation of linear, data type is int8
        B Weight of linear, data type is int8
        SA Not used for CPU/XPU
        SB Not used for CPU/XPU
        out Specified output tensor if it is not None
        Sout Not used for CPU/XPU but returned as is
        dtype Data type of output
    Return:
        A tuple of GEMM result in dtype and Sout
    """
    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    if out is not None:
        assert out.dtype == dtype

    dimsA = A.ndim
    dimsB = B.ndim
    shapeA = A.shape
    shapeB = B.shape
    assert dimsA in [2, 3], "Only two or three dimensional matrices are supported for argument A"
    assert dimsB == 2, "Only two dimensional matrices are supported for argument B"

    if dimsA == 2:
        m = shapeA[0]
    elif dimsA == 3:
        m = shapeA[0] * shapeA[1]
    n = shapeB[0]
    k = shapeA[-1]
    assert shapeA[-1] == shapeB[-1], f"Shapes of A and B do not match, got {shapeA} and {shapeB}"

    # if the tensor is empty, return a transformed empty tensor with the right dimensions
    if shapeA[0] == 0 and dimsA == 2:
        return torch.empty((0, n), device=A.device, dtype=A.dtype)
    elif shapeA[1] == 0 and dimsA == 3:
        return torch.empty(tuple(shapeA[:2] + [n]), device=A.device, dtype=A.dtype)

    A_reshaped = A.reshape(m, k)

    # torch._int_mm is available on CPU since torch 2.4
    if _torch_version_prereq(2, 4):
        C = torch._int_mm(A_reshaped, B.T).to(dtype)
    else:
        C = torch.matmul(A_reshaped.float(), B.t().float()).to(dtype)
    if C.ndim != dimsA:
        assert dimsA == 3
        shapeOut = (shapeA[0], m // shapeA[0], C.shape[-1])
        C = C.reshape(shapeOut)
    if out is not None:
        out.copy_(C)
    else:
        out = C

    return out, Sout


@_maybe_torch_compile
def mm_dequant_impl(
    A,
    quant_state,
    row_stats,
    col_stats,
    out=None,
    new_row_stats=None,
    new_col_stats=None,
    bias=None,
    compute_dtype=torch.float32,
    output_dtype=torch.float32,
):
    """
    Dequant and add bias
    out = A_int32 * (abs_max_A * abs_max_B) / 127 * 127 + bias
    Args:
        A The output of int8 gemm, whose dtype is int32
        quant_state Not used for CPU
        row_stats Absolute max value of each row of input (A) of gemm
        col_stats Absolute max value of each row of weight (B) of gemm
        out Output buffer
        new_row_stats Not used for CPU/XPU
        new_col_stats Not used for CPU/XPU
        bias Bias of linear
        compute_dtype Data type for computation
        output_dtype Data type for output
    Return:
        The result
    """
    assert A.dtype == torch.int32
    out_shape = A.shape
    if len(out_shape) == 3:
        out_shape = (out_shape[0] * out_shape[1], out_shape[2])

    if compute_dtype not in [torch.float32, torch.bfloat16]:
        warnings.warn(f"mm_dequant_{A.device}: compute_dtype {compute_dtype} is not supported, will use float instead")
        compute_dtype = torch.float32
    A_reshaped = A.reshape(out_shape).to(compute_dtype)
    row_stats = row_stats.reshape(-1).unsqueeze(-1).to(compute_dtype)
    col_stats = col_stats.reshape(-1).unsqueeze(0).to(compute_dtype)
    out = A_reshaped * row_stats * col_stats / (127 * 127)
    if bias is not None:
        out = out + bias.to(compute_dtype)
    out = out.to(output_dtype)
    return out


NF4_QUANT_TABLE = [
    -1.0 - 1e-2,  # 0b0000
    -0.8480964004993439,  # 0b0001
    -0.6106329262256622,  # 0b0010
    -0.4599952697753906,  # 0b0011
    -0.33967943489551544,  # 0b0100
    -0.23460740596055984,  # 0b0101
    -0.13791173323988914,  # 0b0110
    -0.045525018125772476,  # 0b0111
    0.03979014977812767,  # 0b1000
    0.1202552504837513,  # 0b1001
    0.2035212516784668,  # 0b1010
    0.2920137718319893,  # 0b1011
    0.3893125355243683,  # 0b1100
    0.5016634166240692,  # 0b1101
    0.6427869200706482,  # 0b1110
    0.8614784181118011,  # 0b1111
]


FP4_QUANT_TABLE = {
    0 - 1e-2: 0,  # 0b0000
    0.00260417: 1,  # 0b0001
    0.0859375: 6,  # 0b0110
    0.20833333: 7,  # 0b0111
    0.29166667: 4,  # 0b0100
    0.4166667: 5,  # 0b0101
    0.583333: 2,  # 0b0010
    0.8333333: 3,  # 0b0011
}


# It's faster not to use torch.compile
def quantize_4bit_impl(
    A: Tensor,
    absmax: Tensor = None,
    out: Tensor = None,
    blocksize=64,
    compress_statistics=False,
    quant_type="nf4",
) -> Tensor:
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
        The output tensor (8-bit).
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}, only nf4 is supported now

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor with packed 4-bit values.
    tuple(torch.Tensor, torch.Size, torch.dtype, int):
        The quantization state to undo the quantization.
    """
    if quant_type not in ["nf4", "fp4"]:
        raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented for CPU/XPU.")
    if quant_type == "fp4":
        warnings.warn("fp4 quantization is currently slow on CPU/XPU. Please Use nf4 instead for better performance.")
    assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
    n = A.numel()
    input_shape = A.shape
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0

    if absmax is None:
        absmax = torch.zeros((blocks,), device=A.device, dtype=A.dtype)

    if out is None:
        out = torch.zeros(((n + 1) // 2), dtype=torch.uint8, device=A.device)

    rem = n % blocksize
    has_rem = rem > 0

    # Scale tensor to [-1, 1]
    A_reshaped = A.reshape(n)
    A_com = A_reshaped[: n - rem]
    A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
    absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
    scaled_A = torch.clamp(A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1)
    scaled_A = scaled_A.reshape(-1)
    if has_rem:
        absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
        scaled_A_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax[-1]), -1, 1)
        scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)
    # map [-1, 1] to nf4/fp4
    out_uint8 = torch.empty(scaled_A.shape, dtype=torch.uint8)
    if quant_type == "nf4":
        for i in range(len(NF4_QUANT_TABLE)):
            out_uint8[scaled_A > NF4_QUANT_TABLE[i]] = i
    elif quant_type == "fp4":
        sign = scaled_A < 0
        abs_scaled_A = torch.abs(scaled_A)
        for key, val in FP4_QUANT_TABLE.items():
            out_uint8[abs_scaled_A > key] = val
        out_uint8 += sign.to(torch.uint8) * 8
    if out_uint8.size(-1) % 2:
        out_uint8 = torch.nn.functional.pad(out_uint8, (0, 1), value=0)
    out[:] = out_uint8[1::2].bitwise_left_shift(4).bitwise_or_(out_uint8[::2])

    code = get_4bit_type(quant_type, device=A.device)

    if compress_statistics:
        raise NotImplementedError("bnb_4bit_use_double_quant is not supported yet for CPU/XPU")
    else:
        state = QuantState(
            absmax=absmax,
            shape=input_shape,
            dtype=A.dtype,
            blocksize=blocksize,
            code=code,
            quant_type=quant_type,
        )

    if ipex_cpu and _ipex_cpu_version_prereq(2, 3) and input_shape[1] % blocksize == 0 and quant_type == "nf4":
        # lowp_mode: lowest precision for computation
        lowp_mode = ipex_cpu.quantization.WoqLowpMode.BF16
        state.op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
            out.reshape([input_shape[0], input_shape[1] // 2]),
            ipex_cpu.quantization.WoqWeightDtype.NF4,
            input_shape,  # weight shape
            absmax.view(input_shape[0], input_shape[1] // blocksize),  # scales
            None,  # zero_points
            None,  # bias
            None,  # g_idx
            None,  # batch_size
            blocksize,
            int(lowp_mode),
            -1,  # act_quant_mode. -1 means don't quant activation
        )
        state.absmax = torch.Tensor()
        return torch.Tensor(), state

    return out, state


@_maybe_torch_compile
def dequantize_4bit_impl(
    A: Tensor,
    quant_state=None,
    absmax: Tensor = None,
    out: Tensor = None,
    blocksize: int = 64,
    quant_type="nf4",
) -> Tensor:
    """
    Dequantizes FP4 blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in blocks of size blocksize.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor (packed 4-bit values).
    quant_state : QuantState
        object with quantisation stats, incl. absmax values, original tensor shape and original dtype.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        Dequantized output tensor.
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}, only nf4 is supported now


    Returns
    -------
    torch.Tensor:
        Dequantized tensor.
    """

    if quant_state is None:
        assert absmax is not None and out is not None

        quant_state = QuantState(
            absmax=absmax,
            shape=out.shape,
            dtype=out.dtype,
            blocksize=blocksize,
            quant_type=quant_type,
        )

    else:
        absmax = quant_state.absmax

    if quant_type not in ["nf4", "fp4"]:
        raise NotImplementedError(
            f"4-bit quantization data type {quant_state.quant_type} is not implemented for CPU/XPU."
        )

    if quant_state.nested:
        raise NotImplementedError("bnb_4bit_use_double_quant is not supported yet for CPU/XPU")

    if ipex_cpu and _ipex_cpu_version_prereq(2, 3) and hasattr(quant_state, "op_context"):
        assert quant_state.op_context is not None
        A = quant_state.op_context.to_public(quant_state.op_context.get_weight())
        A = A.reshape(-1)
        absmax = quant_state.op_context.get_scales().reshape(-1)

    if out is None:
        out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)

    n = out.numel()
    # Map nf4 to [-1, 1]
    out_uint8 = torch.empty(A.size(0) * 2, dtype=torch.uint8, device=A.device)
    out_uint8[::2] = A.bitwise_and(0xF)
    out_uint8[1::2] = A.bitwise_right_shift(4)
    out_dq = torch.empty(out_uint8.shape).to(quant_state.dtype)
    for i in range(len(quant_state.code)):
        out_dq[out_uint8 == i] = quant_state.code[i]

    # Apply scales
    if out_dq.numel() != n:
        assert out_dq.numel() == n + 1
        out_dq = torch.narrow(out_dq, 0, 0, n)
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    rem = n % blocksize
    has_rem = rem > 0
    out_reshaped = out.reshape(-1)
    out_reshaped[: n - rem] = (out_dq[: n - rem].view(-1, blocksize) * absmax[: blocks - has_rem].view(-1, 1)).reshape(
        -1
    )
    if has_rem:
        out_reshaped[n - rem :] = out_dq[n - rem :] * absmax[-1]

    # take transpose here because weight is transposed (again) for computation
    return out.t()


# Do not need torch.compile here as we are calling torch/ipex kernel
def gemm_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
    state: QuantState = None,
) -> torch.Tensor:
    """
    Matrix-matrix multiplication with 4-bit quantization.

    Parameters
    ----------
    A : torch.Tensor
        The first input tensor. Usually the activation tensor.
    B : torch.Tensor
        The second input tensor. Usually the weight tensor.
    out : torch.Tensor
        The output tensor.
    transposed_A : bool
        Whether A is transposed
    transposed_B : bool
        Whether B is transposed
    state : QuantState
        Contains quantization info, such as blocksize and dtype

    Returns
    -------
    torch.Tensor:
        GEMM output tensor.
    """
    if ipex_cpu and _ipex_cpu_version_prereq(2, 3) and hasattr(state, "op_context"):
        assert state.op_context is not None
        output = torch.ops.torch_ipex.ipex_woq_linear(A, state.op_context.get_data_handle())
    else:
        dqB = dequantize_4bit_impl(B, state, blocksize=state.blocksize)
        output = torch.matmul(A, dqB)
    if out is not None:
        out.copy_(output)
    else:
        out = output
    return out
