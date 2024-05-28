import pytest
import torch

from bitsandbytes.cextension import HIP_ENVIRONMENT
from bitsandbytes.nn import Linear8bitLt
from bitsandbytes.nn.triton_based_modules import SwitchBackLinear
from bitsandbytes.triton.triton_utils import is_triton_available
from tests.helpers import TRUE_FALSE


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.skipif(
    not is_triton_available() or not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 8,
    reason="This test requires triton and a GPU with compute capability 8.0 or higher.",
)
@pytest.mark.parametrize("vector_wise_quantization", TRUE_FALSE)
def test_switchback(vector_wise_quantization):
    for dim in [83]:
        for batch in [13]:
            standard = torch.nn.Linear(dim, 4 * dim).cuda().half()
            switchback = (
                SwitchBackLinear(dim, 4 * dim, vector_wise_quantization=vector_wise_quantization).cuda().half()
            )
            baseline = Linear8bitLt(dim, 4 * dim).cuda().half()
            switchback.weight.data.copy_(standard.weight)
            switchback.bias.data.copy_(standard.bias)
            baseline.weight.data.copy_(standard.weight)
            baseline.bias.data.copy_(standard.bias)

            x1 = torch.randn(batch, dim).cuda().half().requires_grad_(True)
            x2 = x1.clone().detach().requires_grad_(True)
            x3 = x1.clone().detach().requires_grad_(True)

            out_standard = standard(x1)
            (2**10 * out_standard.abs().mean()).backward()

            print(x2.dtype)
            out_sb = switchback(x2)
            (2**10 * out_sb.abs().mean()).backward()

            out_baseline = baseline(x3)
            (2**10 * out_baseline.abs().mean()).backward()

            err_sb = (out_standard - out_sb).abs().mean()
            err_baseline = (out_standard - out_baseline).abs().mean()
            print("OUT", err_sb, err_baseline)
            assert err_sb < 2 * err_baseline

            err_sb = (standard.bias.grad - switchback.bias.grad).abs().mean()
            err_baseline = (standard.bias.grad - baseline.bias.grad).abs().mean()

            print("GW2", err_sb, err_baseline)
            assert err_sb < 2 * err_baseline

            err_sb = (standard.weight.grad - switchback.weight.grad).abs().mean()
            err_baseline = (standard.weight.grad - baseline.weight.grad).abs().mean()

            print("GW1", err_sb, err_baseline)
            assert err_sb < 2 * err_baseline

            err_sb = (x1.grad - x2.grad).abs().mean()
            err_baseline = (x1.grad - x3.grad).abs().mean()

            print("GX1", err_sb, err_baseline)
            assert err_sb < 2 * err_baseline
