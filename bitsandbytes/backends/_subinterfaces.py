from abc import ABC, abstractmethod


class RequiredUtilities(ABC):
    @abstractmethod
    def check_matmul():
        raise NotImplementedError


class FourBitMatmul(ABC):
    @abstractmethod
    def quantize_4bit():
        raise NotImplementedError

    @abstractmethod
    def dequantize_4bit():
        raise NotImplementedError

    @abstractmethod
    def gemv_4bit():
        raise NotImplementedError


class EightBitMatMul(ABC):
    @abstractmethod
    def mm_dequant():
        raise NotImplementedError

    @abstractmethod
    def double_quant():
        raise NotImplementedError

    @abstractmethod
    def extract_outliers():
        raise NotImplementedError

    @abstractmethod
    def igemmlt():
        raise NotImplementedError

    @abstractmethod
    def get_col_row_absmax():
        raise NotImplementedError


class KBitQuantization(ABC):
    @abstractmethod
    def quantize_blockwise():
        raise NotImplementedError

    @abstractmethod
    def dequantize_blockwise():
        raise NotImplementedError

    @abstractmethod
    def estimate_quantiles():
        raise NotImplementedError

    @abstractmethod
    def create_quant_map():
        raise NotImplementedError


class EightBitOptimizer(ABC):
    @abstractmethod
    def optimizer_update_32bit():
        """Needed only for testing purposes."""
        raise NotImplementedError

    @abstractmethod
    def optimizer_update_8bit_blockwise():
        raise NotImplementedError


class CompleteBnbAlgorithmsInterface(
    RequiredUtilities, FourBitMatmul, EightBitMatMul, KBitQuantization, EightBitOptimizer
):
    pass
