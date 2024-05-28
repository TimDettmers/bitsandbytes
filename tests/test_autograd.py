from typing import Tuple

import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.cextension import BNB_HIP_VERSION
from tests.helpers import (
    BOOLEAN_TRIPLES,
    BOOLEAN_TUPLES,
    TRUE_FALSE,
    describe_dtype,
    get_test_dims,
    id_formatter,
)

TRANSPOSE_VALS = [(False, True), (False, False)]


@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(32, 96, n=1), ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
@pytest.mark.parametrize(
    "funcs",
    [(torch.bmm, bnb.bmm_cublas), (torch.matmul, bnb.matmul_cublas)],
    ids=["func=bmm", "func=matmul"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=describe_dtype)
@pytest.mark.parametrize("req_grad", BOOLEAN_TUPLES, ids=id_formatter("req_grad"))
@pytest.mark.parametrize("transpose", BOOLEAN_TUPLES, ids=id_formatter("transpose"))
def test_matmul(dim1, dim2, dim3, dim4, funcs, dtype, req_grad: Tuple[bool, bool], transpose: Tuple[bool, bool]):
    if dim2 > 0:
        dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    dim4 = dim4 - (dim4 % 16)
    for i in range(25):
        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
            dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
            A = torch.randn(size=dimA, device="cuda", requires_grad=req_grad[0])
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1])
            target = torch.randn(size=(dim2, dim4), device="cuda", requires_grad=req_grad[1])
            torch.nn.init.xavier_uniform_(B)

            if not transpose[0] and not transpose[1]:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B)
            elif not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B.t())
            elif transpose[0] and not transpose[1]:
                out_torch = funcs[0](A.t(), B)
                out_bnb = funcs[1](A.t(), B)
            elif transpose[0] and transpose[1]:
                out_torch = funcs[0](A.t(), B.t())
                out_bnb = funcs[1](A.t(), B.t())

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() < n * 0.0175
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx == 0).sum().item() < n * 0.001

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.02
                torch.testing.assert_close(gradB1, gradB2, atol=0.18, rtol=0.3)

        # batched matrix multiply
        if funcs[0] in [torch.bmm, torch.matmul]:
            A = torch.randn(
                size=(dim1, dim2, dim3),
                device="cuda",
                requires_grad=req_grad[0],
            )
            B = torch.randn(
                size=(dim1, dim3, dim4),
                device="cuda",
                requires_grad=req_grad[1],
            )
            target = torch.randn(
                size=(dim1, dim2, dim4),
                device="cuda",
                requires_grad=req_grad[1],
            )
            torch.nn.init.xavier_uniform_(B)

            out_torch = funcs[0](A, B)
            out_bnb = funcs[1](A, B)

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() < n * 0.01
            torch.testing.assert_close(out_bnb, out_torch, atol=0.027, rtol=0.2)

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.02

        if funcs[0] in [torch.matmul]:
            dim1 = dim1 - (dim1 % 16)
            A = torch.randn(
                size=(dim1, dim2, dim3),
                device="cuda",
                requires_grad=req_grad[0],
            )
            dimB = (dim4, dim3) if transpose[1] else (dim3, dim4)
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1])
            target = torch.randn(
                size=(dim1, dim2, dim4),
                device="cuda",
                requires_grad=req_grad[1],
            )
            torch.nn.init.xavier_uniform_(B)

            if transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B.t())
            else:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B)

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() < n * 0.0175
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx == 0).sum().item() < n * 0.001

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.02


@pytest.mark.skipif(0 < BNB_HIP_VERSION < 601, reason="this test is supported on ROCm from 6.1")
@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [*get_test_dims(32, 96, n=1), 0], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
@pytest.mark.parametrize("decomp", [0.0, 6.0], ids=id_formatter("decomp"))
@pytest.mark.parametrize(
    "funcs",
    [(torch.matmul, bnb.matmul), (torch.matmul, bnb.research.switchback_bnb)],
    ids=["func=matmul", "func=switchback_bnb"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("req_grad", BOOLEAN_TRIPLES, ids=id_formatter("req_grad"))
@pytest.mark.parametrize("transpose", TRANSPOSE_VALS, ids=id_formatter("transpose"))
@pytest.mark.parametrize("has_fp16_weights", TRUE_FALSE, ids=id_formatter("has_fp16_weights"))
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
def test_matmullt(dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose, decomp, has_fp16_weights, has_bias):
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    outlier_dim = torch.randint(0, dimA[1], size=(dimA[1] // 8,), device="cuda")
    if has_bias == False:
        req_grad = list(req_grad)
        req_grad[2] = False

    for i in range(3):
        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            A = torch.randn(size=dimA, device="cuda", requires_grad=req_grad[0], dtype=dtype)
            if decomp == 6.0:
                with torch.no_grad():
                    A[:, outlier_dim] = 6.0
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1], dtype=dtype)
            target = torch.randn(
                size=(dim2, dim4),
                device="cuda",
                requires_grad=req_grad[1],
                dtype=dtype,
            )
            bias = None
            bias2 = None
            if has_bias:
                bias = torch.randn(dim4, device="cuda", dtype=dtype, requires_grad=req_grad[2])
                bias2 = bias.clone()
            torch.nn.init.xavier_uniform_(B)
            B2 = B.clone()

            state = bnb.MatmulLtState()
            state.threshold = decomp
            state.has_fp16_weights = has_fp16_weights
            if not has_fp16_weights:
                if not transpose[0] and not transpose[1]:
                    B2 = B2.t().contiguous()
                (
                    state.CB,
                    CBt,
                    state.SCB,
                    SCBt,
                    coo_tensorB,
                ) = bnb.functional.double_quant(B2.to(torch.float16))
                B2 = state.CB

            if not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B2, state=state, bias=bias2)
            elif not transpose[0] and not transpose[1]:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B2.t(), state=state, bias=bias2)

            if has_bias:
                out_torch += bias

            assert out_bnb.dtype == A.dtype, f"bnb matmullt received {A.dtype} but returned {out_bnb.dtype}"

            n = out_bnb.numel()
            err = torch.abs(out_bnb - out_torch).mean().item()
            # print(f'abs error {err:.4f}')

            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() <= n * (0.0175 if dtype == torch.float16 else 0.021)
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx == 0).sum().item() <= n * 0.001

            if has_fp16_weights:
                if any(req_grad):
                    out_bnb.data.copy_(out_torch)
                    torch.cuda.synchronize()
                    loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                    loss_bnb.backward()
                    gradA1 = A.grad
                    gradB1 = B.grad
                    A.grad = None
                    B.grad = None
                    if has_bias:
                        gradBias1 = bias.grad
                        bias.grad = None

                    loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                    loss_torch.backward()
                    gradA2 = A.grad
                    gradB2 = B.grad
                    A.grad = None
                    B.grad = None
                    if has_bias:
                        gradBias2 = bias.grad
                        bias.grad = None

                if req_grad[0]:
                    torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
                if req_grad[1]:
                    n = gradB1.numel()
                    if dim2 > 0:
                        assert torch.abs(gradB1).sum() > 0.0
                        assert torch.abs(gradB2).sum() > 0.0
                    else:
                        assert torch.abs(gradB1).sum() == 0.0
                        assert torch.abs(gradB2).sum() == 0.0
                    idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)

                    assert (idx == 0).sum().item() <= n * 0.1
                    idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                    assert (idx == 0).sum().item() <= n * 0.02
                    torch.testing.assert_close(gradB1, gradB2, atol=0.18, rtol=0.3)

                if req_grad[2]:
                    torch.testing.assert_close(gradBias1, gradBias2)


@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [*get_test_dims(32, 96, n=1), 0], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
@pytest.mark.parametrize("funcs", [(torch.matmul, bnb.matmul_4bit)], ids=["func=matmul"])
@pytest.mark.parametrize("req_grad", BOOLEAN_TRIPLES, ids=id_formatter("req_grad"))
@pytest.mark.parametrize("transpose", TRANSPOSE_VALS, ids=id_formatter("transpose"))
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("compress_statistics", TRUE_FALSE, ids=id_formatter("compress_statistics"))
@pytest.mark.parametrize("quant_type", ["fp4", "nf4"], ids=id_formatter("quant_type"))
def test_matmul_4bit(
    dim1,
    dim2,
    dim3,
    dim4,
    funcs,
    dtype,
    req_grad,
    transpose,
    has_bias,
    compress_statistics,
    quant_type,
):
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    if has_bias == False:
        req_grad = list(req_grad)
        req_grad[2] = False

    for i in range(3):
        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            A = torch.randn(size=dimA, device="cuda", requires_grad=req_grad[0], dtype=dtype)
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1], dtype=dtype)
            target = torch.randn(size=(dim2, dim4), device="cuda", requires_grad=req_grad[1], dtype=dtype)
            bias = None
            bias2 = None
            if has_bias:
                bias = torch.randn(dim4, device="cuda", dtype=dtype, requires_grad=req_grad[2])
                bias2 = bias.clone()
            torch.nn.init.xavier_uniform_(B)

            B2, quant_state = bnb.functional.quantize_4bit(
                B,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
            )

            if not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B2.t(), quant_state, bias=bias2)
            elif not transpose[0] and not transpose[1]:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B2, quant_state, bias=bias2)

            if has_bias:
                out_torch += bias

            assert out_bnb.dtype == A.dtype, f"bnb matmullt received {A.dtype} but returned {out_bnb.dtype}"

            n = out_bnb.numel()
            err = torch.abs(out_bnb - out_torch).float().mean().item()
            if n > 0:
                assert err < 0.115

                # assert err < 0.20
            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None
                if has_bias:
                    gradBias1 = bias.grad
                    bias.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None
                if has_bias:
                    gradBias2 = bias.grad
                    bias.grad = None

                if req_grad[0]:
                    torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)

                if req_grad[2]:
                    torch.testing.assert_close(gradBias1, gradBias2)


@pytest.mark.parametrize("dim1", get_test_dims(16, 64, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [*get_test_dims(32, 96, n=1), 0], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 96, n=1), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 96, n=1), ids=id_formatter("dim4"))
@pytest.mark.parametrize("req_grad", BOOLEAN_TRIPLES, ids=id_formatter("req_grad"))
@pytest.mark.parametrize("transpose", TRANSPOSE_VALS, ids=id_formatter("transpose"))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize(
    "funcs",
    [(torch.matmul, bnb.research.matmul_fp8_mixed), (torch.matmul, bnb.research.matmul_fp8_global)],
    ids=["matmul_fp8_mixed", "matmul_fp8_global"],
)
def test_matmul_fp8(dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose):
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    req_grad = list(req_grad)
    req_grad[2] = False

    for i in range(3):
        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            A = torch.randn(size=dimA, device="cuda", requires_grad=req_grad[0], dtype=dtype)
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1], dtype=dtype)
            target = torch.randn(size=(dim2, dim4), device="cuda", requires_grad=req_grad[1], dtype=dtype)

            torch.nn.init.xavier_uniform_(B)

            fw_code = bnb.functional.create_fp8_map(True, 4, 3, 8).to(A.device)
            bw_code = bnb.functional.create_fp8_map(True, 5, 2, 8).to(A.device)

            if not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B.t(), fw_code, bw_code)
            elif not transpose[0] and not transpose[1]:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B, fw_code, bw_code)

            assert out_bnb.dtype == A.dtype, f"bnb matmullt received {A.dtype} but returned {out_bnb.dtype}"

            n = out_bnb.numel()
            err = torch.abs(out_bnb - out_torch).float().mean().item()
            if n > 0:
                assert err < 0.115
                # assert err < 0.20
            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

                if req_grad[0]:
                    torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)

                if req_grad[1]:
                    n = gradB1.numel()
                    if dim2 > 0:
                        assert torch.abs(gradB1).sum() > 0.0
                        assert torch.abs(gradB2).sum() > 0.0
                    else:
                        assert torch.abs(gradB1).sum() == 0.0
                        assert torch.abs(gradB2).sum() == 0.0
                    idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)

                    assert (idx == 0).sum().item() <= n * 0.1
                    idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                    assert (idx == 0).sum().item() <= n * 0.02
                    grad_err = (gradB1 - gradB2).abs().mean()
                    assert grad_err.item() < 0.003
                    torch.testing.assert_close(gradB1, gradB2, atol=0.18, rtol=0.3)
