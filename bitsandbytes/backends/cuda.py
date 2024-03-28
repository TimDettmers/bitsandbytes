import ctypes as ct
from typing import Optional, Tuple

import torch

from bitsandbytes.cextension import lib
from bitsandbytes.functional import (
    CUBLAS_Context,
    coo_zeros,
    dequantize_blockwise,
    dtype2bytes,
    get_4bit_type,
    get_colrow_absmax,
    get_ptr,
    get_transform_buffer,
    is_on_gpu,
    post_call,
    pre_call,
    prod,
    quantize_blockwise,
)
from bitsandbytes.utils import QuantState

from .base import Backend


class CUDABackend(Backend):
    def double_quant(
        self, A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
    ):
        device = A.device
        assert A.dtype == torch.half
        assert device.type == "cuda"
        prev_device = pre_call(A.device)

        cols = A.shape[-1]
        if len(A.shape) == 3:
            rows = A.shape[0] * A.shape[1]
        else:
            rows = A.shape[0]

        if row_stats is None or col_stats is None:
            row_stats, col_stats, nnz_row_ptr = get_colrow_absmax(
                A, threshold=threshold
            )

        if out_col is None:
            out_col = torch.zeros(A.shape, device=device, dtype=torch.int8)
        if out_row is None:
            out_row = torch.zeros(A.shape, device=device, dtype=torch.int8)

        coo_tensor = None
        ptrA = get_ptr(A)
        ptrColStats = get_ptr(col_stats)
        ptrRowStats = get_ptr(row_stats)
        ptrOutCol = get_ptr(out_col)
        ptrOutRow = get_ptr(out_row)

        is_on_gpu([A, col_stats, row_stats, out_col, out_row])
        if threshold > 0.0:
            nnz = nnz_row_ptr[-1].item()
            if nnz > 0:
                coo_tensor = coo_zeros(
                    A.shape[0], A.shape[1], nnz_row_ptr[-1].item(), device
                )
                ptrRowIdx = get_ptr(coo_tensor.rowidx)
                ptrColIdx = get_ptr(coo_tensor.colidx)
                ptrVal = get_ptr(coo_tensor.values)
                ptrRowPtr = get_ptr(nnz_row_ptr)

                lib.cdouble_rowcol_quant(
                    ptrA,
                    ptrRowStats,
                    ptrColStats,
                    ptrOutCol,
                    ptrOutRow,
                    ptrRowIdx,
                    ptrColIdx,
                    ptrVal,
                    ptrRowPtr,
                    ct.c_float(threshold),
                    ct.c_int32(rows),
                    ct.c_int32(cols),
                )
                val, idx = torch.sort(coo_tensor.rowidx)
                coo_tensor.rowidx = val
                coo_tensor.colidx = coo_tensor.colidx[idx]
                coo_tensor.values = coo_tensor.values[idx]
            else:
                lib.cdouble_rowcol_quant(
                    ptrA,
                    ptrRowStats,
                    ptrColStats,
                    ptrOutCol,
                    ptrOutRow,
                    None,
                    None,
                    None,
                    None,
                    ct.c_float(0.0),
                    ct.c_int32(rows),
                    ct.c_int32(cols),
                )
        else:
            lib.cdouble_rowcol_quant(
                ptrA,
                ptrRowStats,
                ptrColStats,
                ptrOutCol,
                ptrOutRow,
                None,
                None,
                None,
                None,
                ct.c_float(threshold),
                ct.c_int32(rows),
                ct.c_int32(cols),
            )
        post_call(prev_device)

        return out_row, out_col, row_stats, col_stats, coo_tensor

    def transform(self, A, to_order, from_order='row', out=None, transpose=False, state=None, ld=None):
        prev_device = pre_call(A.device)
        if state is None:
            state = (A.shape, from_order)
        else:
            from_order = state[1]

        if out is None:
            out, new_state = get_transform_buffer(state[0], A.dtype, A.device, to_order, state[1], transpose)
        else:
            new_state = (state[0], to_order) # (shape, order)

        shape = state[0]
        if len(shape) == 2:
            dim1 = ct.c_int32(shape[0])
            dim2 = ct.c_int32(shape[1])
        else:
            dim1 = ct.c_int32(shape[0] * shape[1])
            dim2 = ct.c_int32(shape[2])

        is_on_gpu([A, out])
        if to_order == 'col32':
            if transpose:
                lib.ctransform_row2col32T(get_ptr(A), get_ptr(out), dim1, dim2)
            else:
                lib.ctransform_row2col32(get_ptr(A), get_ptr(out), dim1, dim2)

        elif to_order == "col_turing":
            if transpose:
                lib.ctransform_row2turingT(get_ptr(A), get_ptr(out), dim1, dim2)
            else:
                lib.ctransform_row2turing(get_ptr(A), get_ptr(out), dim1, dim2)

        elif to_order == "col_ampere":
            if transpose:
                lib.ctransform_row2ampereT(get_ptr(A), get_ptr(out), dim1, dim2)
            else:
                lib.ctransform_row2ampere(get_ptr(A), get_ptr(out), dim1, dim2)

        elif to_order == "row":
            if from_order == "col_turing":
                lib.ctransform_turing2row(get_ptr(A), get_ptr(out), dim1, dim2)
            elif from_order == "col_ampere":
                lib.ctransform_ampere2row(get_ptr(A), get_ptr(out), dim1, dim2)

        else:
            raise NotImplementedError(f'Transform function not implemented: From {from_order} to {to_order}')

        post_call(prev_device)

        return out, new_state

    def igemmlt(self, A, B, SA, SB, out=None, Sout=None, dtype=torch.int32):
        shapeA = SA[0]
        shapeB = SB[0]
        dimsA = len(shapeA)
        dimsB = len(shapeB)

        assert dimsB == 2, 'Only two dimensional matrices are supported for argument B'
        if dimsA == 2:
            m = shapeA[0]
        elif dimsA == 3:
            m = shapeA[0] * shapeA[1]

        rows = n = shapeB[0]
        assert prod(list(shapeA)) > 0, f'Input tensor dimensions need to be > 0: {shapeA}'

        # if the tensor is empty, return a transformed empty tensor with the right dimensions
        if shapeA[0] == 0 and dimsA == 2:
            return torch.empty((0, shapeB[0]), device=A.device, dtype=torch.float16)
        elif shapeA[1] == 0 and dimsA == 3:
            return torch.empty(tuple(shapeA[:2] + [shapeB[0]]), device=A.device, dtype=torch.float16)

        if dimsA == 2 and out is None:
            out, Sout = get_transform_buffer(
                (shapeA[0], shapeB[0]), dtype, A.device, "col32", "row"
            )
        elif dimsA == 3 and out is None:
            out, Sout = get_transform_buffer(
                (shapeA[0], shapeA[1], shapeB[0]), dtype, A.device, "col32", "row"
            )

        assert dimsB != 3, "len(B.shape)==3 not supported"
        assert A.device.type == "cuda"
        assert B.device.type == "cuda"
        assert A.dtype == torch.int8
        assert B.dtype == torch.int8
        assert out.dtype == dtype
        assert SA[1] == "col32"
        assert SB[1] in ["col_turing", "col_ampere"]
        assert Sout[1] == "col32"
        assert (
            shapeA[-1] == shapeB[-1]
        ), f"Matmullt only supports A @ B^T. Inner matrix dimensions do not match: A @ B = {shapeA} @ {shapeB}"

        formatB = SB[1]
        prev_device = A.device
        torch.cuda.set_device(A.device)

        ptr = CUBLAS_Context.get_instance().get_context(A.device)
        ptrA = get_ptr(A)
        ptrB = get_ptr(B)
        ptrC = get_ptr(out)

        k = shapeA[-1]
        lda = ct.c_int32(m * 32)
        if formatB == "col_turing":
            # turing: tiles with rows filled up to multiple of 8 rows by 32 columns
            # n = rows
            ldb = ct.c_int32(((rows + 7) // 8) * 8 * 32)
        else:
            # ampere: tiles with rows filled up to multiple of 32 rows by 32 columns
            # n = rows
            ldb = ct.c_int32(((rows + 31) // 32) * 32 * 32)

        ldc = ct.c_int32(m * 32)
        m = ct.c_int32(m)
        n = ct.c_int32(n)
        k = ct.c_int32(k)

        has_error = 0
        ptrRowScale = get_ptr(None)
        is_on_gpu([A, B, out])

        if formatB == 'col_turing':
            if dtype == torch.int32:
                has_error = lib.cigemmlt_turing_32(
                    ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
                )
            else:
                has_error = lib.cigemmlt_turing_8(
                    ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
                )

        elif formatB == "col_ampere":
            if dtype == torch.int32:
                has_error = lib.cigemmlt_ampere_32(
                    ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
                )
            else:
                has_error = lib.cigemmlt_ampere_8(
                    ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc
                )

        if has_error == 100:  # `ERR_NOT_IMPLEMENTED` is defined as 100 in `ops.cu`
            raise NotImplementedError("igemmlt not available (probably built with NO_CUBLASLT)")

        if has_error:
            print(f'A: {shapeA}, B: {shapeB}, C: {Sout[0]}; (lda, ldb, ldc): {(lda, ldb, ldc)}; (m, n, k): {(m, n, k)}')
            raise Exception('cublasLt ran into an error!')

        torch.cuda.set_device(prev_device)

        return out, Sout

    def mm_dequant(
        self,
        A,
        quant_state,
        row_stats,
        col_stats,
        out=None,
        new_row_stats=None,
        new_col_stats=None,
        bias=None
    ):
        assert A.dtype == torch.int32
        if bias is not None: assert bias.dtype == torch.float16
        out_shape = quant_state[0]
        if len(out_shape) == 3:
            out_shape = (out_shape[0] * out_shape[1], out_shape[2])

        if out is None:
            out = torch.empty(out_shape, dtype=torch.float16, device=A.device)
        if new_row_stats is None:
            new_row_stats = torch.empty(
                out_shape[0], dtype=torch.float32, device=A.device
            )
        if new_col_stats is None:
            new_col_stats = torch.empty(
                out_shape[1], dtype=torch.float32, device=A.device
            )
        assert (
            new_row_stats.shape[0] == row_stats.shape[0]
        ), f"{new_row_stats.shape} vs {row_stats.shape}"
        assert (
            new_col_stats.shape[0] == col_stats.shape[0]
        ), f"{new_col_stats.shape} vs {col_stats.shape}"

        prev_device = pre_call(A.device)
        ptrA = get_ptr(A)
        ptrOut = get_ptr(out)
        ptrRowStats = get_ptr(row_stats)
        ptrColStats = get_ptr(col_stats)
        ptrNewRowStats = get_ptr(new_row_stats)
        ptrNewColStats = get_ptr(new_col_stats)
        ptrBias = get_ptr(bias)
        numRows = ct.c_int32(out_shape[0])
        numCols = ct.c_int32(out_shape[1])

        is_on_gpu([A, row_stats, col_stats, out, new_row_stats, new_col_stats, bias])
        lib.cdequant_mm_int32_fp16(ptrA, ptrRowStats, ptrColStats, ptrOut, ptrNewRowStats, ptrNewColStats, ptrBias, numRows, numCols)
        post_call(prev_device)

        return out

    def extract_outliers(self, A, SA, idx):
        shapeA = SA[0]
        formatA = SA[1]
        assert formatA in ["col_turing", "col_ampere"]
        assert A.device.type == "cuda"

        out = torch.zeros(
            (shapeA[0], idx.numel()), dtype=torch.int8, device=A.device
        )

        idx_size = ct.c_int32(idx.numel())
        rows = ct.c_int32(shapeA[0])
        cols = ct.c_int32(shapeA[1])
        ptrA = get_ptr(A)
        ptrIdx = get_ptr(idx)
        ptrOut = get_ptr(out)

        prev_device = pre_call(A.device)

        if formatA == 'col_turing':
            lib.cextractOutliers_turing(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)
        elif formatA == "col_ampere":
            lib.cextractOutliers_ampere(ptrA, ptrIdx, ptrOut, idx_size, rows, cols)

        post_call(prev_device)

        return out

    def quantize_4bit(
        self,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=64,
        compress_statistics=False,
        quant_type='fp4',
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        if A.device.type != 'cuda':
            raise NotImplementedError(f'Device type not supported for FP4 quantization: {A.device.type}')
        if quant_type not in ['fp4', 'nf4']:
            raise NotImplementedError(f'4-bit quantization data type {quant_type} is not implemented.')

        n = A.numel()
        input_shape = A.shape

        if absmax is None:
            blocks = n // blocksize
            blocks += 1 if n % blocksize > 0 else 0
            absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)

        if out is None:
            mod = dtype2bytes[quant_storage] * 2
            out = torch.zeros(((n+1)//mod, 1), dtype=quant_storage, device=A.device)

        assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]

        prev_device = pre_call(A.device)
        is_on_gpu([A, out, absmax])

        if A.dtype == torch.float32:
            if quant_type == 'fp4':
                lib.cquantize_blockwise_fp32_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
            else:
                lib.cquantize_blockwise_fp32_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))

        elif A.dtype == torch.float16:
            if quant_type == 'fp4':
                lib.cquantize_blockwise_fp16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
            else:
                lib.cquantize_blockwise_fp16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))

        elif A.dtype == torch.bfloat16:
            if quant_type == 'fp4':
                lib.cquantize_blockwise_bf16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
            else:
                lib.cquantize_blockwise_bf16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))

        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")

        post_call(A.device)

        code = get_4bit_type(quant_type, device=A.device)

        if compress_statistics:
            offset = absmax.mean()
            absmax -= offset
            qabsmax, state2 = quantize_blockwise(absmax, blocksize=256)
            del absmax
            state = QuantState(absmax=qabsmax, shape=input_shape, dtype=A.dtype, blocksize=blocksize, code=code, quant_type=quant_type, offset=offset, state2=state2)

        else:
            state = QuantState(absmax=absmax, shape=input_shape, dtype=A.dtype, blocksize=blocksize, code=code, quant_type=quant_type)

        return out, state

    def dequantize_4bit(self, A: torch.Tensor, quant_state: Optional[QuantState] = None, absmax: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, blocksize: int = 64, quant_type='fp4') -> torch.Tensor:
        if blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
            raise ValueError(f"The blockwise of {blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]")

        if quant_type not in ['fp4', 'nf4']:
            raise NotImplementedError(f'4-bit quantization data type {quant_type} is not implemented.')

        if quant_state is None:
            assert absmax is not None and out is not None

            quant_state = QuantState(absmax=absmax, shape=out.shape, dtype=out.dtype, blocksize=blocksize, quant_type=quant_type)
        else:
            absmax = quant_state.absmax

        if quant_state.nested:
            absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax += quant_state.offset
            if absmax.dtype != torch.float32: absmax = absmax.float()

        if out is None:
            out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)

        n = out.numel()

        device = pre_call(A.device)
        is_on_gpu([A, absmax, out])

        if out.dtype == torch.float32:
            if quant_state.quant_type == 'fp4':
                lib.cdequantize_blockwise_fp32_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
            else:
                lib.cdequantize_blockwise_fp32_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))

        elif out.dtype == torch.float16:
            if quant_state.quant_type == 'fp4':
                lib.cdequantize_blockwise_fp16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
            else:
                lib.cdequantize_blockwise_fp16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))

        elif out.dtype == torch.bfloat16:
            if quant_state.quant_type == 'fp4':
                lib.cdequantize_blockwise_bf16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
            else:
                lib.cdequantize_blockwise_bf16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))

        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")

        post_call(A.device)

        is_transposed = (True if A.shape[0] == 1 else False)

        if is_transposed: return out.t()
        else: return out
