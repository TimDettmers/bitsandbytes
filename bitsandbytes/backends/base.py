from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from bitsandbytes.utils import QuantState


class Backend(ABC):
    """Base class for devices backends that will implement their own 8bits and 4bits functions."""

    @abstractmethod
    def double_quant(
        self,
        A,
        col_stats=None,
        row_stats=None,
        out_col=None,
        out_row=None,
        threshold=0.0,
    ):
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        A,
        to_order,
        from_order="row",
        out=None,
        transpose=False,
        state=None,
        ld=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def igemmlt(self, A, B, SA, SB, out=None, Sout=None, dtype=torch.int32):
        raise NotImplementedError

    @abstractmethod
    def mm_dequant(
        self,
        A,
        quant_state,
        row_stats,
        col_stats,
        out=None,
        new_row_stats=None,
        new_col_stats=None,
        bias=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def extract_outliers(self, A, SA, idx):
        raise NotImplementedError

    @abstractmethod
    def quantize_4bit(
        self,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=64,
        compress_statistics=False,
        quant_type="fp4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
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
        raise NotImplementedError

    @abstractmethod
    def dequantize_4bit(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 64,
        quant_type="fp4",
    ) -> torch.Tensor:
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
        raise NotImplementedError
