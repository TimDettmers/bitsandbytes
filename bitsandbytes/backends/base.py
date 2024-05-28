from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple, Union

import torch

from bitsandbytes.utils import QuantState


class Backend(ABC):
    """Base class for devices backends that will implement their own 8bits and 4bits functions."""

    @abstractmethod
    def double_quant(
        self,
        A: torch.Tensor,
        col_stats: Optional[torch.Tensor] = None,
        row_stats: Optional[torch.Tensor] = None,
        out_col: Optional[torch.Tensor] = None,
        out_row: Optional[torch.Tensor] = None,
        threshold=0.0,
    ):
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def extract_outliers(
        self,
        A: torch.Tensor,
        SA: Tuple[torch.Size, str],
        idx: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
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
        quant_type: Literal["fp4", "nf4"] = "fp4",
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

    @abstractmethod
    def gemv_4bit(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transposed_A=False,
        transposed_B=False,
        state: QuantState = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def quantize_blockwise(
        self,
        A: torch.Tensor,
        code: Optional[torch.Tensor] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=4096,
        nested=False,
    ) -> Tuple[torch.Tensor, QuantState]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        """
        Performs an in-place optimizer update with one or two optimizer states.

        Args:
            optimizer_name (`str`): The name of the optimizer, e.g. `adam`
            g (`torch.Tensor`): Gradient tensor.
            p (`torch.Tensor`): Parameter tensor.
            state1 (`torch.Tensor`): Optimizer state 1.
            state2 (`torch.Tensor`, optional): Optimizer state 2.
            beta1 (`float`): Optimizer beta1.
            beta2 (`float`): Optimizer beta2.
            eps (`float`): Optimizer epsilon.
            step (`int`): Current optimizer step.
            lr (`float`): The learning rate.
            qmap1 (`torch.Tensor`): Quantization map for the first state.
            qmap2 (`torch.Tensor`, optional): Quantization map for the second state.
            absmax1 (`torch.Tensor`): Max value for the first state update.
            absmax2 (`torch.Tensor`, optional): Max value for the second state update.
            weight_decay (`float`, optional): Weight decay. Defaults to 0.0.
            gnorm_scale (`float`, optional): The factor to rescale the gradient to the max clip value. Defaults to 1.0.
            skip_zeros (`bool`, optional): Whether to skip zero-valued gradients or not. Defaults to False.
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        Performs an in-place optimizer update with one or two optimizer states.

        Universal optimizer update for 32-bit state and 32/16-bit gradients/weights

        Args:
            optimizer_name (`str`): The name of the optimizer, e.g. `adam`
            g (`torch.Tensor`): Gradient tensor.
            p (`torch.Tensor`): Parameter tensor.
            state1 (`torch.Tensor`): Optimizer state 1.
            beta1 (`float`): Optimizer beta1.
            eps (`float`): Optimizer epsilon.
            step (`int`): Current optimizer step.
            lr (`float`): The learning rate.
            state2 (`torch.Tensor`, optional): Optimizer state 2. Defaults to None.
            beta2 (`float`, optional): Optimizer beta2. Defaults to 0.0.
            weight_decay (`float`, optional): Defaults to 0.0.
            gnorm_scale (`float`, optional): The factor to rescale the gradient to the max clip value. Defaults to 1.0.
            unorm_vec (`torch.Tensor`, optional): The tensor for the update norm. Defaults to None.
            max_unorm (`float`, optional): The maximum update norm relative to the weight norm.. Defaults to 0.0.
            skip_zeros (`bool`, optional): Whether to skip zero-valued gradients or not. Defaults to False.
        """
        raise NotImplementedError
