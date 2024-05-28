from typing import Literal, Optional, Tuple, Union

import torch

from bitsandbytes.utils import QuantState

from .base import Backend
from .cpu_xpu_common import (
    dequantize_4bit_impl,
    double_quant_impl,
    gemm_4bit_impl,
    igemmlt_impl,
    mm_dequant_impl,
    quantize_4bit_impl,
)

Tensor = torch.Tensor


def assert_on_cpu(tensors):
    on_cpu = True
    for t in tensors:
        if t is None:
            continue  # NULL pointers are fine
        on_cpu &= t.device.type == "cpu"
    if not on_cpu:
        raise TypeError(
            "All input tensors need to be on CPU, but found some tensors to not be on CPU:\n"
            f" {[(t.shape, t.device) if isinstance(t, Tensor) else None for t in tensors]}"
        )
    return on_cpu


class CPUBackend(Backend):
    mm_dequant_compute_dtype = torch.bfloat16
    mm_dequant_output_dtype = torch.bfloat16

    def double_quant(
        self,
        A: torch.Tensor,
        col_stats: Optional[torch.Tensor] = None,
        row_stats: Optional[torch.Tensor] = None,
        out_col: Optional[torch.Tensor] = None,
        out_row: Optional[torch.Tensor] = None,
        threshold=0.0,
    ):
        assert_on_cpu([A, col_stats, row_stats, out_col, out_row])
        return double_quant_impl(A, col_stats, row_stats, out_col, out_row, threshold)

    def transform(
        self,
        A: torch.Tensor,
        to_order: str,
        from_order="row",
        out: Optional[torch.Tensor] = None,
        transpose=False,
        state: Optional[Tuple[torch.Size, str]] = None,
        ld=None,
    ):
        """
        Transform tensor A to to_order. It is originally designed for CUDA.
        For CPU, it returns the original tensor if transpose=False.
        Otherwise, it returns the transpose of A
        """
        assert_on_cpu([A, out])
        if transpose:
            if out is not None:
                out.copy_(A.T)
            else:
                out = A.T
        else:
            if out is not None:
                out.copy_(A)
            else:
                out = A
        return out, state

    def igemmlt(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        SA: Tuple[torch.Size, str],
        SB: Tuple[torch.Size, str],
        out: Optional[torch.Tensor] = None,
        Sout: Optional[Tuple[torch.Size, str]] = None,
        dtype=torch.int32,
    ) -> Union[torch.Tensor, Tuple[Optional[Tuple[torch.Tensor, Tuple[torch.Size, str]]]]]:
        assert_on_cpu([A, B])
        return igemmlt_impl(A, B, SA, SB, out, Sout, dtype)

    def mm_dequant(
        self,
        A: torch.Tensor,
        quant_state: Tuple[torch.Size, str],
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        new_row_stats: Optional[torch.Tensor] = None,
        new_col_stats: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert_on_cpu([A, row_stats, col_stats, out, bias])
        return mm_dequant_impl(
            A,
            quant_state,
            row_stats,
            col_stats,
            out,
            new_row_stats,
            new_col_stats,
            bias,
            self.mm_dequant_compute_dtype,
            self.mm_dequant_output_dtype,
        )

    def extract_outliers(
        self,
        A: torch.Tensor,
        SA: Tuple[torch.Size, str],
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract columns of A by idx
        """
        assert_on_cpu([A])
        return A[:, idx].contiguous()

    def quantize_4bit(
        self,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=64,
        compress_statistics=False,
        quant_type: Literal["fp4", "nf4"] = "fp4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        assert_on_cpu([A, absmax, out])
        assert quant_storage == torch.uint8, "CPU backend only supports uint8 quant_storage"
        return quantize_4bit_impl(A, absmax, out, blocksize, compress_statistics, quant_type)

    def dequantize_4bit(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 64,
        quant_type: Literal["fp4", "nf4"] = "fp4",
    ) -> torch.Tensor:
        assert_on_cpu([A, absmax, out])
        return dequantize_4bit_impl(A, quant_state, absmax, out, blocksize, quant_type)

    def gemv_4bit(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transposed_A=False,
        transposed_B=False,
        state: QuantState = None,
    ) -> torch.Tensor:
        assert_on_cpu([A, B, out])
        if state is None:
            raise ValueError("state cannot be None. gemv_4bit() requires the state from quantize_4bit()")

        return gemm_4bit_impl(A, B, out, transposed_A, transposed_B, state)

    def dequantize_blockwise(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        code: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 4096,
        nested=False,
    ) -> torch.Tensor:
        raise NotImplementedError("Not yet implemented for CPU backend")

    def quantize_blockwise(
        self,
        A: torch.Tensor,
        code: Optional[torch.Tensor] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=4096,
        nested=False,
    ) -> Tuple[torch.Tensor, QuantState]:
        raise NotImplementedError("Not yet implemented for CPU backend")

    def optimizer_update_8bit_blockwise(
        self,
        optimizer_name: str,
        g: torch.Tensor,
        p: torch.Tensor,
        state1: torch.Tensor,
        state2: Optional[torch.Tensor],
        beta1: float,
        beta2: float,
        eps: float,
        step: int,
        lr: float,
        qmap1: torch.Tensor,
        qmap2: Optional[torch.Tensor],
        absmax1: torch.Tensor,
        absmax2: Optional[torch.Tensor],
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        skip_zeros=False,
    ) -> None:
        raise NotImplementedError("Not yet implemented for CPU backend")

    def optimizer_update_32bit(
        self,
        optimizer_name: str,
        g: torch.Tensor,
        p: torch.Tensor,
        state1: torch.Tensor,
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
        raise NotImplementedError("Not yet implemented for CPU backend")
