from itertools import product
import math
import random
import time

import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F
from bitsandbytes.cextension import BNB_HIP_VERSION, HIP_ENVIRONMENT, ROCM_GPU_ARCH
from tests.helpers import (
    BOOLEAN_TUPLES,
    TRUE_FALSE,
    describe_dtype,
    get_blocksizes,
    get_test_dims,
    id_formatter,
)

torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)
k = 20


def assert_all_approx_close(a, b, rtol=1e-3, atol=1e-3, count=0, throw=True):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    sumval = (idx == 0).sum().item()
    if sumval > count:
        if throw:
            print(f"Too many values not close: assert {sumval} < {count}")
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

    return sumval


class FFN(torch.nn.Module):
    def __init__(self, input_features, hidden_size, bias=True):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_features, hidden_size, bias=bias)
        self.fc2 = torch.nn.Linear(hidden_size, input_features, bias=bias)

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Timer:
    def __init__(self):
        self.starts = {}
        self.ends = {}
        self.agg = {}

    def tick(self, name="default"):
        if name not in self.starts:
            self.starts[name] = torch.cuda.Event(enable_timing=True)
            self.ends[name] = torch.cuda.Event(enable_timing=True)
            self.starts[name].record()
        else:
            ms = self.tock(name, evict=True, print_ms=False)

    def tock(self, name="default", evict=True, print_ms=True):
        if name in self.ends:
            self.ends[name].record()
            torch.cuda.synchronize()
            ms = self.starts[name].elapsed_time(self.ends[name])
            if name not in self.agg:
                self.agg[name] = 0.0
            self.agg[name] += ms
            if evict:
                self.starts.pop(name)
                self.ends.pop(name)

        if print_ms and name in self.agg:
            print(f"{name} took: {self.agg[name] / 1000.0:.5f}s")

        return self.agg[name]

    def reset(self):
        self.starts = {}
        self.ends = {}
        self.agg = {}
        print("Resetting benchmark data")


def setup():
    pass


def teardown():
    pass


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float", "half"])
def test_estimate_quantiles(dtype):
    A = torch.rand(1024, 1024, device="cuda")
    A = A.to(dtype)
    code = F.estimate_quantiles(A)

    percs = torch.linspace(1 / 512, 511 / 512, 256, device=A.device)
    torch.testing.assert_close(percs, code, atol=1e-3, rtol=1e-2)

    A = torch.randn(1024, 1024, device="cuda")
    A = A.to(dtype)
    code = F.estimate_quantiles(A)

    quantiles = torch.quantile(A.float(), percs)
    diff = torch.abs(code - quantiles)
    assert (diff > 5e-02).sum().item() == 0


def test_quantile_quantization():
    for i in range(100):
        A1 = torch.randn(1024, 1024, device="cuda")
        code = F.estimate_quantiles(A1)
        C = F.quantize_no_absmax(A1, code)
        A2 = F.dequantize_no_absmax(C, code)
        diff = torch.abs(A1 - A2).mean().item()
        assert diff < 0.0075

        A1 = torch.rand(1024, 1024, device="cuda")
        code = F.estimate_quantiles(A1)
        C = F.quantize_no_absmax(A1, code)
        A2 = F.dequantize_no_absmax(C, code)
        diff = torch.abs(A1 - A2).mean().item()
        torch.testing.assert_close(A1, A2, atol=5e-3, rtol=0)
        assert diff < 0.001


def test_dynamic_quantization():
    diffs = []
    reldiffs = []
    for i in range(100):
        A1 = torch.randn(1024, 1024, device="cuda")
        C, S = F.quantize(A1)
        A2 = F.dequantize(C, S)
        diff = torch.abs(A1 - A2)
        reldiff = diff / torch.abs(A1 + 1e-8)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
        assert diff.mean().item() < 0.0135
    print(sum(diffs) / len(diffs))
    print(sum(reldiffs) / len(reldiffs))

    for i in range(100):
        A1 = torch.rand(1024, 1024, device="cuda")
        C, S = F.quantize(A1)
        A2 = F.dequantize(C, S)
        diff = torch.abs(A1 - A2).mean().item()
        torch.testing.assert_close(A1, A2, atol=1e-2, rtol=0)
        assert diff < 0.004


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("nested", TRUE_FALSE, ids=id_formatter("nested"))
@pytest.mark.parametrize("blocksize", get_blocksizes(HIP_ENVIRONMENT))
@pytest.mark.parametrize("signed", TRUE_FALSE, ids=id_formatter("signed"))
def test_dynamic_blockwise_quantization(dtype, nested, blocksize, signed):
    # print('')
    diffs = []
    reldiffs = []
    for i in range(100):
        A1 = torch.randn(1024, 1024, device="cuda", dtype=dtype)
        C, S = F.quantize_blockwise(A1, blocksize=blocksize, nested=nested)
        A2 = F.dequantize_blockwise(C, S)
        diff = torch.abs(A1 - A2).float()
        reldiff = diff / torch.abs(A1.float() + 1e-8)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
    abserr = sum(diffs) / len(diffs)
    relerr = sum(reldiffs) / len(reldiffs)
    # print('nested=', nested, 'randn', blocksize, 'dtype', dtype, sum(diffs)/len(diffs))
    # print('nested=', nested, 'randn', blocksize, 'dtype', dtype, sum(reldiffs)/len(reldiffs))
    assert abserr < 0.011
    assert relerr < 0.018
    assert A2.dtype == dtype

    diffs = []
    code = F.create_dynamic_map(signed=signed)
    for i in range(100):
        A1 = torch.rand(1024, 1024, device="cuda", dtype=dtype)
        C, S = F.quantize_blockwise(A1, blocksize=blocksize, nested=nested, code=code)
        A2 = F.dequantize_blockwise(C, S)
        diff = torch.abs(A1 - A2).float()
        reldiff = diff / torch.abs(A1.float() + 1e-8)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
        # torch.testing.assert_close(A1, A2, atol=1e-2, rtol=0)
    abserr = sum(diffs) / len(diffs)
    relerr = sum(reldiffs) / len(reldiffs)
    if signed:
        assert abserr < 0.0035
        assert relerr < 0.015
    else:
        assert abserr < 0.00175
        assert relerr < 0.012
    assert A2.dtype == dtype
    # print('signed=', signed, 'nested=', nested, 'rand', blocksize, sum(diffs)/len(diffs))
    # print('signed=', signed, 'nested=', nested, 'rand', blocksize, sum(reldiffs)/len(reldiffs))


@pytest.mark.parametrize("gtype", [torch.float32, torch.float16], ids=["float", "half"])
def test_percentile_clipping(gtype):
    gnorm_vec1 = torch.zeros(100, device="cuda")
    gnorm_vec2 = torch.zeros(100, device="cuda")
    n = 4
    step = 0
    percentile = 5
    for i in range(k):
        step += 1
        g = torch.randn(n, n, dtype=gtype, device="cuda")
        gnorm1, clip2, gnorm_scale = F.percentile_clipping(g, gnorm_vec2, step, percentile=percentile)
        assert gnorm_scale == 1.0 if gnorm1 < clip2 else clip2 / gnorm1

        gnorm2 = torch.norm(g.float())
        if step == 1:
            gnorm_vec1[:] = gnorm2
        else:
            gnorm_vec1[step % 100] = gnorm2

        vals, idx = torch.sort(gnorm_vec1)
        clip1 = vals[percentile]

        torch.testing.assert_close(gnorm_vec1, torch.sqrt(gnorm_vec2))
        torch.testing.assert_close(clip1, clip2)
        torch.testing.assert_close(gnorm1, gnorm2)


def quant(x):
    max1 = torch.abs(x).max()
    x = torch.round(x / max1 * 127)
    return max1, x.to(torch.int8)


def dequant(c, maxC):
    return c.float() * (maxC / 127)


def mm_dequant(maxA, maxB, C):
    return C.float() * (maxA / 127) * (maxB / 127)


def quant_multi(x, dim):
    max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
    max1[max1 == 0] = 1.0
    x = torch.round(x / max1 * 127)
    return max1, x.to(torch.int8)


def quant_multi_chunk(x, dim, chunk_size=32):
    if dim == 1:
        x_chunked = einops.rearrange(x, "(c a) b -> c a b", c=chunk_size)
        max1 = torch.amax(torch.abs(x_chunked), dim=dim + 1, keepdim=True)
        max1 = torch.tile(max1, (1, 1, x.shape[1]))
        max1 = max1.view(x.shape)
    elif dim == 0:
        x_chunked = einops.rearrange(x, "a (b c) -> a b c", c=chunk_size)
        max1 = torch.amax(torch.abs(x_chunked), dim=dim, keepdim=True)
        max1 = torch.tile(max1, (x.shape[0], 1, 1))
        max1 = max1.view(x.shape)
    max1[max1 == 0] = 1.0
    x = torch.round(x / max1 * 127)
    return max1, x.to(torch.int8)


def quant_minmax(A):
    minA = A.min()
    maxA = A.max()


def mean(xx):
    return sum(xx) / float(len(xx))


methods = {
    "linear": (
        lambda x, dim: quant(x),
        lambda x, dim: quant(x),
        dequant,
        dequant,
        mm_dequant,
    ),
    "vectorwise": (quant_multi, quant_multi, dequant, dequant, mm_dequant),
}


@pytest.mark.parametrize("dim1", [1024 * 2], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [1024 * 16], ids=id_formatter("dim2"))
@pytest.mark.parametrize("quant_methods", methods.values(), ids=methods.keys())
@pytest.mark.parametrize("batched", TRUE_FALSE, ids=id_formatter("batched"))
def test_approx_igemm(dim1, dim2, quant_methods, batched):
    dim1 = dim1 - (dim1 % 32)
    dim2 = dim2 - (dim2 % 32)
    errors = []
    relerrors = []
    # print("")
    for i in range(5):
        if batched:
            A = torch.normal(0, 0.5, size=(32, dim1, dim2 // 32), device="cuda")
            B = torch.normal(0, 0.5, size=(32, dim2 // 32, dim1), device="cuda")
            maxA, Ac = quant_methods[0](A, 2)
            maxB, Bc = quant_methods[1](B, 1)
        else:
            A = torch.normal(0, 0.5, size=(dim1, dim2), device="cuda")
            B = torch.normal(0, 0.5, size=(dim2, dim1), device="cuda")
            maxA, Ac = quant_methods[0](A, 1)
            maxB, Bc = quant_methods[1](B, 0)
        torch.testing.assert_close(quant_methods[2](maxA, Ac), A, atol=0.025, rtol=0.05)
        if batched:
            out2 = torch.bmm(A, B)
            C = torch.bmm(Ac.float(), Bc.float())
        else:
            out2 = torch.mm(A, B)
            C = F.igemm(Ac, Bc)
        out = quant_methods[4](maxA, maxB, C)
        std = out2.std()
        out /= std
        out2 /= std
        err = torch.abs(out - out2)
        relerr = err / torch.abs(out2)
        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())
    # print(mean(errors))
    # print(mean(relerrors))


def test_stable_embedding():
    layer = bnb.nn.StableEmbedding(1024, 1024)
    layer.reset_parameters()


@pytest.mark.parametrize("hidden_dim", get_test_dims(32, 256, n=2), ids=id_formatter("hidden_dim"))
@pytest.mark.parametrize("batch_dim", get_test_dims(16, 256, n=2), ids=id_formatter("batch_dim"))
@pytest.mark.parametrize("seq_dim", get_test_dims(16, 256, n=2), ids=id_formatter("seq_dim"))
@pytest.mark.parametrize("transpose", BOOLEAN_TUPLES, ids=id_formatter("transpose"))
def test_igemm(hidden_dim, batch_dim, transpose, seq_dim):
    hidden_dim = hidden_dim - (hidden_dim % 32)
    batch_dim = batch_dim - (batch_dim % 16)
    seq_dim = seq_dim - (seq_dim % 16)
    for i in range(k):
        shapeA = (batch_dim, hidden_dim) if not transpose[0] else (hidden_dim, batch_dim)
        shapeB = (32 * random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32 * random.randint(1, 4))
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)
        if not transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.matmul(A.float(), B.t().float())
            out = F.igemm(A, B.t())
        elif transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.t().float(), B.float())
            out = F.igemm(A.t(), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.matmul(A.t().float(), B.t().float())
            out = F.igemm(A.t(), B.t())

        torch.testing.assert_close(out.float(), out2)

    for i in range(k):
        shapeA = (batch_dim, seq_dim, hidden_dim)
        shapeB = (32 * random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32 * random.randint(1, 4))
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)
        if not transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.matmul(A.float(), B.t().float())
            out = F.igemm(A, B.t())

        torch.testing.assert_close(out.float(), out2)


@pytest.mark.parametrize("seq_dim", get_test_dims(32, 512, n=3), ids=id_formatter("seq_dim"))
@pytest.mark.parametrize("hidden_dim", get_test_dims(32, 1024 * 4, n=3), ids=id_formatter("hidden_dim"))
@pytest.mark.parametrize("batch_dim", get_test_dims(2, 16, n=3), ids=id_formatter("batch_dim"))
def test_dim3_igemm(seq_dim, hidden_dim, batch_dim):
    seq_dim = seq_dim - (seq_dim % 32)
    hidden_dim = hidden_dim - (hidden_dim % 32)
    batch_dim = batch_dim - (batch_dim % 2)
    for i in range(25):
        A = torch.randint(-128, 127, size=(batch_dim, seq_dim, hidden_dim), device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=(batch_dim, seq_dim, 1024), device="cuda").to(torch.int8)
        out2 = torch.einsum("bsi, bso->io", A.float(), B.float())
        iout = torch.empty(A.shape[2], B.shape[2], dtype=torch.int32, device=A.device)
        out = F.igemm(A, B, out=iout)

        torch.testing.assert_close(out.float(), out2)


@pytest.mark.parametrize("seq_dim", get_test_dims(32, 512, n=2), ids=id_formatter("seq_dim"))
@pytest.mark.parametrize("hidden_dim", get_test_dims(32, 1024 * 4, n=2), ids=id_formatter("hidden_dim"))
@pytest.mark.parametrize("batch_dim", get_test_dims(2, 16, n=2), ids=id_formatter("batch_dim"))
@pytest.mark.parametrize("transpose", TRUE_FALSE, ids=id_formatter("transpose"))
def test_minmax_igemm(seq_dim, hidden_dim, batch_dim, transpose):
    def min_max(x):
        maxA = torch.amax(x, dim=2, keepdim=True)
        minA = torch.amin(x, dim=2, keepdim=True)
        scale = (maxA - minA) / 2.0
        return (127 * (x - minA - scale) / scale).to(torch.int8), minA, scale

    seq_dim = seq_dim - (seq_dim % 16)
    hidden_dim = hidden_dim - (hidden_dim % 16)
    batch_dim = batch_dim - (batch_dim % 2)
    errs = []
    relerrs = []
    errs2 = []
    relerrs2 = []
    for i in range(k):
        A = torch.normal(0.0, 0.5, size=(batch_dim, seq_dim, hidden_dim), device="cuda")
        if transpose:
            B = torch.normal(0, 0.5, size=(256, hidden_dim), device="cuda")
        else:
            B = torch.normal(0, 0.5, size=(hidden_dim, 256), device="cuda")
        Ac, minA, scale = min_max(A)
        if transpose:
            maxB, Bc = quant_multi(B, dim=(1 if transpose else 0))
            out = F.igemm(Ac, Bc.t())
            out2 = torch.matmul(A, B.t())
            offset = B.t().sum(0) * (minA + scale)
            out = out.float()
            out = (out * maxB.t() * scale / (127 * 127)) + offset

            maxA, Ac = quant_multi(A, dim=2)
            out3 = F.igemm(Ac, Bc.t())
            out3 = mm_dequant(maxA, maxB.t(), out3)
        else:
            maxB, Bc = quant_multi(B, dim=0)
            offset = B.sum(0) * (minA + scale)
            out = F.igemm(Ac, Bc)
            out2 = torch.matmul(A, B)
            out = out.float()
            out = (out * maxB * scale / (127 * 127)) + offset

            maxA, Ac = quant_multi(A, dim=2)
            out3 = F.igemm(Ac, Bc)
            out3 = mm_dequant(maxA, maxB, out3)

        std = out2.std()
        out2 /= std
        out /= std
        out3 /= std

        err = torch.abs(out - out2)
        relerr = err / (torch.abs(out2) + 1e-7)

        err2 = torch.abs(out3 - out2)
        relerr2 = err2 / (torch.abs(out2) + 1e-7)

        errs.append(err.mean().item())
        relerrs.append(relerr.mean().item())
        errs2.append(err2.mean().item())
        relerrs2.append(relerr2.mean().item())
    # print(mean(errs))
    # print(mean(relerrs))
    # print(mean(errs2))
    # print(mean(relerrs2))
    assert mean(errs) < 0.015
    assert mean(relerrs) < 0.3


@pytest.mark.parametrize("dim1", get_test_dims(1, 64, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(32, 128, n=2), ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 256, n=2), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 256, n=2), ids=id_formatter("dim4"))
@pytest.mark.parametrize("transpose", BOOLEAN_TUPLES, ids=id_formatter("transpose"))
def test_ibmm(dim1, dim2, dim3, dim4, transpose):
    dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    dim4 = dim4 - (dim4 % 16)
    for i in range(k):
        shapeA = (dim1, dim3, dim2) if transpose[0] else (dim1, dim2, dim3)
        shapeB = (dim1, dim4, dim3) if transpose[1] else (dim1, dim3, dim4)
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)

        if not transpose[0] and not transpose[1]:
            out2 = torch.bmm(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.bmm(A.float(), B.permute([0, 2, 1]).float())
            out = F.igemm(A, B.permute([0, 2, 1]))
        elif transpose[0] and not transpose[1]:
            out2 = torch.bmm(A.permute([0, 2, 1]).float(), B.float())
            out = F.igemm(A.permute([0, 2, 1]), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.bmm(A.permute([0, 2, 1]).float(), B.permute([0, 2, 1]).float())
            out = F.igemm(A.permute([0, 2, 1]), B.permute([0, 2, 1]))
        torch.testing.assert_close(out.float(), out2.float())


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.parametrize("dim1", get_test_dims(1, 64, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(32, 128, n=1), ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 256, n=1), ids=id_formatter("dim3"))
def test_vector_quant(dim1, dim2, dim3):
    dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    for i in range(k):
        A = torch.randn(size=(dim2, dim3), device="cuda")
        qA, SA = F.vectorwise_quant(A, dim=0)
        A1 = F.vectorwise_dequant(qA, SA)
        n = A1.numel()
        assert_all_approx_close(A1, A, atol=0.01, rtol=0.1, count=int(n * 0.002))


@pytest.mark.skipif(0 < BNB_HIP_VERSION < 601, reason="this test is supported on ROCm from 6.1")
@pytest.mark.parametrize("dim1", get_test_dims(2, 256, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(2, 256, n=2), ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(2, 256, n=2), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dtype", [torch.int8, torch.int32], ids=describe_dtype)
@pytest.mark.parametrize("orderA", ["row"], ids=id_formatter("orderA"))
@pytest.mark.parametrize(
    "orderOut", ["col", "row"] if HIP_ENVIRONMENT else ["col", "row", "col32"], ids=id_formatter("orderOut")
)
@pytest.mark.parametrize("transpose", [False], ids=id_formatter("transpose"))
@pytest.mark.parametrize("dims", [2, 3], ids=id_formatter("dims"))
def test_nvidia_transform(dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose):
    if dims == 3 and orderOut != "col32":
        return
    if dtype == torch.int32 and orderOut != "col32":
        return
    try:
        func = F.get_transform_func(dtype, orderA, orderOut, transpose)
    except ValueError as ve:
        pytest.skip(str(ve))  # skip if not supported

    if dims == 2:
        A = torch.randint(-128, 127, size=(dim1, dim2), device="cuda").to(dtype)
    elif dims == 3:
        A = torch.randint(-128, 127, size=(dim1, dim2, dim3), device="cuda").to(dtype)

    out, S = F.nvidia_transform(A, to_order=orderOut)

    if orderOut == "row":
        torch.testing.assert_close(A.flatten(), out.flatten())
    elif orderOut == "col":
        torch.testing.assert_close(A.t().flatten(), out.flatten())
    elif orderOut == "col32":
        if dims == 2:
            n = A.shape[0] * (A.shape[1] + (32 - (A.shape[1] % 32)))
        elif dims == 3:
            n = A.shape[0] * A.shape[1] * (A.shape[2] + (32 - (A.shape[2] % 32)))
        assert out.numel() == n
    elif orderOut == "col_turing":
        # 32 col 8 row tiles
        n = (A.shape[0] + (8 - A.shape[0] % 8)) * (A.shape[1] + (32 - (A.shape[1] % 32)))
        assert out.numel() == n
        total_coltile = (A.shape[1] // 32) + (1 if A.shape[1] % 32 != 0 else 0)
        for row in range(A.shape[0]):
            for col in range(A.shape[1]):
                i = row * A.shape[1]
                j = col

                coltile = (col // 32) + (1 if col % 32 != 0 else 0)
                rowtile = ((row // 8) + (1 if row % 8 != 0 else 0)) * total_coltile
                offset = 32 * 8 * (rowtile + coltile)
                col2 = col % 32
                row2 = (row % 8) * 32

                assert A.flatten()[i + j] == A[row, col]
                # assert A.flatten()[i+j] == out.flatten()[row2+col2]
                # torch.testing.assert_close(A.flatten()[i+j], A[row, col])
                # torch.testing.assert_close(A.flatten()[i+j], out.flatten()[row2+ col2+block_offset])

    if orderOut == "col32":
        out2, S = F.nvidia_transform(out, from_order=orderOut, to_order="row", state=S)
        torch.testing.assert_close(A, out2)


@pytest.mark.parametrize("dim1", get_test_dims(1, 256, n=1), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(32, 512, n=1), ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", get_test_dims(32, 1024, n=1), ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", get_test_dims(32, 1024, n=1), ids=id_formatter("dim4"))
@pytest.mark.parametrize("dims", (2, 3), ids=id_formatter("dims"))
@pytest.mark.parametrize("ldb", (0,), ids=id_formatter("ldb"))
@pytest.mark.parametrize("device", ("cuda", "cpu"), ids=id_formatter("device"))
def test_igemmlt_int(dim1, dim2, dim3, dim4, dims, ldb, device):
    for i in range(k):
        if dims == 2:
            A = torch.randint(-128, 127, size=(dim1, dim3), device=device).to(torch.int8)
        elif dims == 3:
            A = torch.randint(-128, 127, size=(dim1, dim2, dim3), device=device).to(torch.int8)
        B = torch.randint(-128, 127, size=(dim4, dim3), device=device).to(torch.int8)
        C1 = torch.matmul(A.float(), B.t().float())

        A2, SA = F.transform(A, "col32")
        B2, SB = F.transform(B, "col_turing")
        C2, SC = F.igemmlt(A2, B2, SA, SB)
        if device == "cpu":
            assert SC is None
        if device == "cuda":
            C3, S = F.nvidia_transform(C2, "row", state=SC)
        else:
            C3, S = C2, None
        torch.testing.assert_close(C1, C3.float())

        # transpose
        B = torch.randint(-128, 127, size=(dim3, dim4), device=device).to(torch.int8)
        C1 = torch.matmul(A.float(), B.float())

        B2t, SBt = F.transform(B, "col_turing", transpose=True)
        C2, SC = F.igemmlt(A2, B2t, SA, SBt)
        if device == "cuda":
            C3, S = F.nvidia_transform(C2, "row", state=SC)
        else:
            C3, S = C2, None
        torch.testing.assert_close(C1, C3.float())


@pytest.mark.parametrize("dim1", [32], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [32], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", [32], ids=id_formatter("dim3"))
@pytest.mark.parametrize("dim4", [32], ids=id_formatter("dim4"))
@pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
def test_igemmlt_half(dim1, dim2, dim3, dim4, dims):
    formatB = F.get_special_format_str()
    for i in range(k):
        if dims == 2:
            A = torch.normal(0, 0.5, size=(dim1, dim3), device="cuda").half()
        elif dims == 3:
            A = torch.normal(0, 0.5, size=(dim1, dim2, dim3), device="cuda").half()
        B = torch.randn((dim4, dim3), device="cuda").half()
        torch.nn.init.xavier_uniform_(B)
        C1 = torch.matmul(A, B.t())
        C2 = bnb.matmul(A, B.t())

        A = A.view(-1, A.shape[-1])

        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        CB, CBt, statsB, statsBt, coo_tensor = F.double_quant(B)
        C32A, SA = F.transform(CA, "col32")
        CxB, SB = F.transform(CB, to_order=formatB)
        out1_32, Sout1_32 = F.igemmlt(C32A, CxB, SA, SB)
        output = F.mm_dequant(out1_32, Sout1_32, statsAt, statsBt)

        # print('')
        # print(output.flatten()[:10])
        # print(C1.flatten()[:10])
        # print(C2.flatten()[:10])

        # torch.testing.assert_close(C1.view(-1, C1.shape[-1]), output, atol=0.025, rtol=0.05)

        # transpose
        # B = torch.randint(-128, 127, size=(dim3, dim4), device='cuda').to(torch.int8)
        # C1 = torch.matmul(A.float(), B.float())

        # B2t, SBt = F.transform2(B, 'col_turing', transpose=True)
        # C2, SC = F.igemmlt(A2, B2t, SA, SBt)
        # C3, S = F.transform(C2, 'row', state=SC)
        # torch.testing.assert_close(C1, C3.float())


@pytest.mark.parametrize(
    ("batch", "seq", "model", "hidden"),
    [
        pytest.param(2, 512, 4 * 1024, 3 * 4 * 1024, id="batch=2, seq=512, model=4k, hidden=12k"),
        pytest.param(2, 512, 5120, 3 * 5120, id="batch=2, seq=512, model=5k, hidden=15k"),
        pytest.param(2, 512, 12 * 1024, 4 * 12 * 1024, id="batch=2, seq=512, model=12k, hidden=48k"),
    ],
)
@pytest.mark.benchmark
def test_bench_8bit_training(batch, seq, model, hidden):
    formatB = F.get_special_format_str()
    A = torch.randn(batch, seq, model, device="cuda").half()
    grad = torch.randn(batch, seq, model, device="cuda").half()
    w1 = torch.randint(-128, 127, size=(hidden, model), device="cuda").half()
    w2 = torch.randint(-128, 127, size=(model, hidden), device="cuda").half()
    print("")

    # torch.cuda.synchronize()
    ## warmup
    # for i in range(100):
    #    torch.matmul(A, w1.t())
    # torch.cuda.synchronize()

    dtype = torch.int8
    A = A.view(-1, A.shape[-1]).contiguous()
    grad = grad.view(-1, grad.shape[-1]).contiguous()
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        out1 = torch.matmul(A, w1.t())  # fc1
        # out2 = torch.matmul(out1, w2.t())# fc2

        # d1 = torch.matmul(grad, w2) # delta1
        # d2 = torch.matmul(d1, w1) # delta2

        # grad1 = torch.einsum('bo,bh->oh', out1, grad) # grad w2
        # grad2 = torch.einsum('bh,bo->ho', A, d2) # grad w1

    torch.cuda.synchronize()
    t16 = time.time() - t0
    print(t16)

    # torch.cuda.empty_cache()

    # Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    # Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)

    # CTw1, Sw1 = F.transform2(Cw1, formatB)
    # CTw2, Sw2 = F.transform2(Cw2, formatB)
    # CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)
    # CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)

    # CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
    # C32A, SA = F.transform2(CA, 'col32')
    ## fc1
    # out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1, dtype=dtype)
    ##out1 = F.mm_dequant(out1_32, Sout1_32, statsAt, statsw1t)

    ## fc2
    # Cout1, Cout1t, statsout1, statsout1t, coo_tensor = F.double_quant(out1)
    # C32out1, Sout1 = F.transform2(Cout1, 'col32')
    # out2_32, Sout2_32 = F.igemmlt(C32out1, CTw2, Sout1, Sw2, dtype=dtype)
    ##out2 = F.mm_dequant(out2_32, Sout2_32, statsout1t, statsw2t)

    ## delta1
    # Cgrad, Cgradt, statsgrad, statsgradt, coo_tensor = F.double_quant(grad)
    # C32grad, Sgrad = F.transform2(Cgrad, 'col32')
    ##d1_32, Sd1_32 = F.igemmlt(C32grad, CTw2t, Sgrad, Sw2t, dtype=dtype)
    ##d1 = F.mm_dequant(d1_32, Sd1_32, statsgradt, statsw2)

    ## delta2
    # Cd1, Cd1t, statsd1, statsd1t, coo_tensor = F.double_quant(d1)
    # C32d1, Sd1 = F.transform2(Cd1, 'col32')
    ##d2_32, Sd2_32 = F.igemmlt(C32d1, CTw1t, Sd1, Sw1t, dtype=dtype)
    ##d2 = F.mm_dequant(d2_32, Sd2_32, statsd1t, statsw1)

    ## grad1
    # C32out1t, Sout1t = F.transform2(Cout1t, 'col32', transpose=True)
    # CTgradt, Sgradt = F.transform2(Cgradt, formatB, transpose=True)
    ##grad1_32, Sgrad1_32 = F.igemmlt(C32out1t, CTgradt, Sout1t, Sgradt, dtype=dtype)
    ##grad1 = F.mm_dequant(grad1_32, Sgrad1_32, statsout1, statsgrad)

    ## grad2
    # C32At, SAt = F.transform2(CAt, 'col32', transpose=True)
    # CTd1t, Sd1t = F.transform2(Cd1t, formatB, transpose=True)
    ##grad2_32, Sgrad2_32 = F.igemmlt(C32At, CTd1t, SAt, Sd1t, dtype=dtype)
    ##grad2 = F.mm_dequant(grad2_32, Sgrad2_32, statsA, statsd1)

    # Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)

    # Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    # Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)

    # CTw1, Sw1 = F.transform2(Cw1, formatB)
    # CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)
    # CTw2, Sw2 = F.transform2(Cw2, formatB)
    # CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(k):
    #    #Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    #    #CTw1, Sw1 = F.transform2(Cw1, formatB)
    #    #Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    #    #CTw1, Sw1 = F.transform2(Cw1, formatB)

    #    #CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A, threshold=3.5)
    #    CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
    #    #CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)
    #    #CTw2, Sw2 = F.transform2(Cw2, formatB)
    #    #CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)

    #    C32A, SA = F.transform2(CA, 'col32')

    #    # fc1
    #    out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1, dtype=dtype)
    #    #out1dn = F.mm_dequant(out1_32, Sout1_32, statsA, statsw1)

    #    #print(coo_tensor.nnz)
    #    #out1sp = F.spmm_coo(coo_tensor, w1.t())
    #    #print(w1.t().shape)
    #    #out1 = out1dn + out1sp

    #    # fc2
    #    Cout1, Cout1t, statsout1, statsout1t, coo_tensor = F.double_quant(out1)
    #    C32out1, Sout1 = F.transform2(Cout1, 'col32')
    #    out2_32, Sout2_32 = F.igemmlt(C32out1, CTw2, Sout1, Sw2, dtype=dtype)
    #    #out2 = F.mm_dequant(out2_32, Sout2_32, statsout1, statsw2)

    #    # delta1
    #    Cgrad, Cgradt, statsgrad, statsgradt, coo_tensor = F.double_quant(grad)
    #    C32grad, Sgrad = F.transform2(Cgrad, 'col32')
    #    d1_32, Sd1_32 = F.igemmlt(C32grad, CTw2t, Sgrad, Sw2t, dtype=dtype)
    #    #d1 = F.mm_dequant(d1_32, Sd1_32, statsgrad, statsw2t)

    #    # delta2
    #    Cd1, Cd1t, statsd1, statsd1t, coo_tensor = F.double_quant(d1)
    #    C32d1, Sd1 = F.transform2(Cd1, 'col32')
    #    d2_32, Sd2_32 = F.igemmlt(C32d1, CTw1t, Sd1, Sw1t, dtype=dtype)
    #    #d2 = F.mm_dequant(d2_32, Sd2_32, statsd1, statsw1t)

    #    # grad1
    #    #C32out1t, Sout1t = F.transform2(Cout1t, 'col32', transpose=True)
    #    #CTgradt, Sgradt = F.transform2(Cgradt, formatB, transpose=True)
    #    #grad1_32, Sgrad1_32 = F.igemmlt(C32out1t, CTgradt, Sout1t, Sgradt, dtype=dtype)
    #    #grad1 = F.mm_dequant(grad1_32, Sgrad1_32, statsout1t, statsgradt)

    #    ## grad2
    #    #C32At, SAt = F.transform2(CAt, 'col32', transpose=True)
    #    #CTd1t, Sd1t = F.transform2(Cd1t, formatB, transpose=True)
    #    #grad2_32, Sgrad2_32 = F.igemmlt(C32At, CTd1t, SAt, Sd1t, dtype=dtype)
    #    #grad2 = F.mm_dequant(grad2_32, Sgrad2_32, statsAt, statsd1t)

    # torch.cuda.synchronize()
    # t8 = time.time() - t0
    # print(t8)


@pytest.mark.parametrize("dim1", get_test_dims(64, 256, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim4", get_test_dims(64, 1024, n=2), ids=id_formatter("dim4"))
@pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
@pytest.mark.parametrize("formatB", ["col_turing", "col_ampere"], ids=id_formatter("formatB"))
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
def test_dequant_mm(dim1, dim4, dims, formatB, has_bias):
    inner = torch.randint(1, 128, size=(1,)).item()
    bias = None
    if has_bias:
        bias = torch.randn(dim4, device="cuda", dtype=torch.float16)
    formatB = F.get_special_format_str()
    for i in range(1):
        A = torch.randn(dim1, inner, device="cuda")
        B = torch.randn(dim4, inner, device="cuda")
        C1 = torch.matmul(A.half(), B.t().half())
        if has_bias:
            C1 += bias

        A1, maxA = F.vectorwise_quant(A, dim=1)
        B1, maxB = F.vectorwise_quant(B, dim=1)

        A2, SA = F.nvidia_transform(A1, "col32")
        B2, SB = F.nvidia_transform(B1, formatB)
        C2, SC = F.igemmlt(A2, B2, SA, SB)

        C3, S = F.nvidia_transform(C2, "row", state=SC)
        C4 = F.vectorwise_mm_dequant(C3.float(), maxA, maxB.t())
        if has_bias:
            C4 += bias

        # TODO: is something wrong here? If so, the problem goes deeper
        # n = C1.numel()
        # p = 0.06
        std = C1.std(0).view(1, -1)
        C1 /= std
        C4 /= std
        # assert_all_approx_close(C1, C4, atol=0.02, rtol=0.1, count=int(n*0.06))
        # assert (count / n < p), f"error in more than {p} of elements: {count}/{n}={count/n}"

        C5 = F.mm_dequant(C2, SC, maxA.flatten(), maxB.flatten(), bias=bias)
        # torch.testing.assert_close(C5, C4, atol=0.015, rtol=0.1)
        n = C5.numel()
        assert_all_approx_close(C1, C4, atol=0.015, rtol=0.1, count=int(0.01 * n))


@pytest.mark.parametrize("dim1", get_test_dims(64, 256, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim4", get_test_dims(64, 1024, n=2), ids=id_formatter("dim4"))
@pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
def test_dequant_mm_cpu(dim1, dim4, dims, has_bias):
    inner = torch.randint(1, 128, size=(1,)).item()
    bias = None
    if has_bias:
        bias = torch.randn(dim4, device="cpu", dtype=torch.bfloat16)
    for i in range(1):
        A = torch.randn(dim1, inner, device="cpu")
        B = torch.randn(dim4, inner, device="cpu")

        A1, maxA = F.vectorwise_quant(A, dim=1)
        B1, maxB = F.vectorwise_quant(B, dim=1)

        C2, SC = F.igemmlt(A1, B1, SA=None, SB=None)
        assert SC is None

        C3 = F.vectorwise_mm_dequant(C2.bfloat16(), maxA, maxB.t())
        if has_bias:
            C3 += bias

        C4 = F.mm_dequant(C2, SC, maxA.flatten(), maxB.flatten(), bias=bias)
        torch.testing.assert_close(C3.float(), C4.float(), atol=0.05, rtol=0.1)


@pytest.mark.parametrize("dim1", [1 * 1024], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [1 * 1024], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
def test_colrow_absmax(dim1, dim2, dims):
    for i in range(k):
        threshold = 3.0
        A = torch.randn(dim1, dim2, device="cuda").half()
        A_truncated = A.clone()
        A_truncated[torch.abs(A_truncated) >= 3.0] = 0.0
        if dims == 2:
            row_stats1, _ = torch.abs(A.float()).max(1)
            col_stats1, _ = torch.abs(A.float()).max(0)
            row_stats1_trunc, _ = torch.abs(A_truncated.float()).max(1)
            col_stats1_trunc, _ = torch.abs(A_truncated.float()).max(0)
        else:
            assert False

        row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(A, threshold=threshold)

        A_blocked = einops.rearrange(
            torch.abs(A),
            "(rows row_tiles) (cols block_size)-> rows cols row_tiles block_size",
            row_tiles=16,
            block_size=64 * 4,
        )
        nnz_rows1_counts = (torch.abs(A_blocked) >= threshold).sum(3).flatten()
        nnz_block_ptr1 = torch.zeros(
            nnz_rows1_counts.shape[0] + 1,
            dtype=nnz_rows1_counts.dtype,
            device=nnz_rows1_counts.device,
        )
        nnz_block_ptr1[1:] = nnz_rows1_counts.cumsum(0)

        torch.testing.assert_close(col_stats1_trunc, col_stats2)
        torch.testing.assert_close(row_stats1_trunc, row_stats2)
        torch.testing.assert_close(nnz_block_ptr1.int(), nnz_block_ptr2)

        row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(A, threshold=0.0)

        torch.testing.assert_close(col_stats1, col_stats2)
        torch.testing.assert_close(row_stats1, row_stats2)
        assert nnz_block_ptr2 is None


@pytest.mark.parametrize("dim1", get_test_dims(1, 4 * 1024, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(1, 4 * 1024, n=2), ids=id_formatter("dim2"))
@pytest.mark.parametrize("device", ["cuda", "cpu"], ids=id_formatter("device"))
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16], ids=id_formatter("dtype"))
def test_double_quant(dim1, dim2, device, dtype):
    if device == "cuda" and dtype == torch.bfloat16:
        pytest.skip("bfloat16 is not implemented for this operation on CUDA backend")
    for i in range(k):
        A = torch.randn(dim1, dim2, device=device).to(dtype)
        out_col1, Scol = F.vectorwise_quant(A, dim=0)
        out_row1, Srow = F.vectorwise_quant(A, dim=1)

        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)

        # max difference is 1 due to rounding differences
        torch.testing.assert_close(CA, out_row1, atol=1, rtol=0)
        torch.testing.assert_close(CAt, out_col1, atol=1, rtol=0)

        n = CAt.numel()
        num_not_close_rows = (torch.isclose(CA, out_row1, atol=1) == 0).sum().item()
        num_not_close_cols = (torch.isclose(CAt, out_col1, atol=1) == 0).sum().item()

        # allow for 1:500 error due to rounding differences
        min_error = 1 / 500
        if num_not_close_cols > (min_error * n):
            print(f"Min error exceeded {num_not_close_cols} elements are different. Error: {num_not_close_cols/n:.4f}")
            assert False
        if num_not_close_rows > (min_error * n):
            print(f"Min error exceeded {num_not_close_rows} elements are different. Error: {num_not_close_rows/n:.4f}")
            assert False

        torch.testing.assert_close(Srow.flatten().float(), statsA)
        torch.testing.assert_close(Scol.flatten().float(), statsAt)


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.parametrize(
    ("dim1", "dim4", "inner"),
    (
        pytest.param(dim1, dim4, inner, id=f"{dim1=},{dim4=},{inner=}")
        for (dim1, dim4, inner) in zip(
            get_test_dims(1, 4 * 1024, n=4),
            get_test_dims(1, 4 * 1024, n=4),
            get_test_dims(1, 4 * 1024, n=4),
        )
    ),
)
def test_integrated_igemmlt(dim1, dim4, inner):
    for i in range(k):
        A = torch.randn(dim1, inner, device="cuda").half()
        B = torch.randn(dim4, inner, device="cuda").half()

        out1 = torch.matmul(A.half(), B.t().half())

        C1a, C1b, stats1a, stats1b, coo_tensor = F.double_quant(A)
        C2a, C2b, stats2a, stats2b, coo_tensor = F.double_quant(B)
        A1, maxA = F.vectorwise_quant(A, dim=1)
        B1, maxB = F.vectorwise_quant(B, dim=1)

        torch.testing.assert_close(maxA.flatten().float(), stats1a)
        torch.testing.assert_close(maxB.flatten().float(), stats2a)
        torch.testing.assert_close(C1a, A1, rtol=0, atol=1)
        torch.testing.assert_close(C2a, B1, rtol=0, atol=1)

        A2, SA = F.nvidia_transform(C1a, "col32")
        B2, SB = F.nvidia_transform(C2a, "col_turing")
        outC32, SC = F.igemmlt(A2, B2, SA, SB)
        out2 = F.mm_dequant(outC32, SC, stats1a, stats2a)

        A2, SA = F.nvidia_transform(A1, "col32")
        B2, SB = F.nvidia_transform(B1, "col_turing")
        C2, SC = F.igemmlt(A2, B2, SA, SB)

        C3, S = F.nvidia_transform(C2, "row", state=SC)
        out3 = F.vectorwise_mm_dequant(C3.float(), maxA, maxB.t())

        err1 = torch.abs(out1 - out2).mean().item()
        err2 = torch.abs(out1 - out3).mean().item()
        assert err2 <= err1 * 1.025


@pytest.mark.parametrize(
    ("dim1", "dim4", "inner"),
    (
        pytest.param(dim1, dim4, inner, id=f"{dim1=},{dim4=},{inner=}")
        for (dim1, dim4, inner) in zip(
            get_test_dims(1, 4 * 1024, n=6),
            get_test_dims(1, 4 * 1024, n=6),
            get_test_dims(1, 4 * 1024, n=6),
        )
    ),
)
@pytest.mark.skip("Row scale has some bugs for ampere")
def test_igemmlt_row_scale(dim1, dim4, inner):
    formatB = F.get_special_format_str()
    err1, err2, err3 = [], [], []
    relerr1, relerr2 = [], []
    scale = 1
    for i in range(k):
        A = torch.randn(dim1, inner, device="cuda").half()
        B = torch.randn(dim4, inner, device="cuda").half()
        torch.nn.init.xavier_uniform_(B)
        C1 = torch.matmul(A, B.t())

        out1 = torch.matmul(A.half(), B.t().half())

        C1a, C1b, stats1a, stats1b, coo_tensor = F.double_quant(A)
        CB, absmaxB = F.vectorwise_quant(B, quant_type="linear")
        A2, SA = F.nvidia_transform(C1a, "col32")
        B2, SB = F.nvidia_transform(CB, formatB)
        A1, maxA = F.vectorwise_quant(A, dim=1)

        c = 10.0 * inner * scale
        row_scale = torch.ones_like(maxA) / c
        outC32, SC = F.igemmlt(A2, B2, SA, SB, dtype=torch.int8, row_scale=row_scale)
        C3, S = F.nvidia_transform(outC32, "row", state=SC)
        maxval = torch.abs(C3).max()
        if maxval == 127:
            scale = 1.5
        else:
            scale = maxval / 120
        out3 = C3 * maxA * absmaxB * c / (127 * 127)

        C4 = torch.matmul(C1a.float(), CB.float().t())

        C2a, C2b, stats2a, stats2b, coo_tensor = F.double_quant(B)
        B2, SB = F.nvidia_transform(C2a, formatB)
        outC32, SC = F.igemmlt(A2, B2, SA, SB)
        out2 = F.mm_dequant(outC32, SC, stats1a, stats2a)

        CA, SA = F.vectorwise_quant(A, dim=1, quant_type="vector")
        CB, SB = F.vectorwise_quant(B, dim=1, quant_type="linear")

        C = torch.matmul(CA.float(), CB.t().float())
        out4 = C * SA * SB / (127 * 127)
        # out4 = torch.clip(torch.round(C*SA/c), -127, 127)*c*SB/(127*127)

        # print('='*80)
        # print(out1)
        # print(out2)
        # print(out3)

        # print(out1)
        # print(out2)
        # print(out3)
        err1.append(torch.abs(out1 - out2).mean().item())
        err2.append(torch.abs(out1 - out3).mean().item())
        err3.append(torch.abs(out1 - out4).mean().item())

        # assert_all_approx_close(C3.float(), torch.round(C4*row_scale), rtol=0, atol=0, count=10)
    print("")
    print(sum(err1) / len(err1))
    print(sum(err2) / len(err2))
    print(sum(err3) / len(err3))


@pytest.mark.parametrize(
    ("dim1", "dim4", "inner"),
    [
        pytest.param(1024, 12288 * 4, 12288, id="1024, 12288*4, 12288"),
        pytest.param(2048, 4096 * 4, 4096, id="2048, 4096*4, 4096"),
    ],
)
@pytest.mark.skip("Row scale has some bugs for ampere")
@pytest.mark.benchmark
def test_row_scale_bench(dim1, dim4, inner):
    formatB = F.get_special_format_str()
    err1, err2, err3 = [], [], []
    relerr1, relerr2 = [], []
    scale = 1
    A = torch.randn(dim1, inner, device="cuda").half()
    B = torch.randn(dim4, inner, device="cuda").half()
    torch.nn.init.xavier_uniform_(B)
    # warmpup
    for i in range(k):
        C1 = torch.matmul(A, B.t())

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        C1 = torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print("16", time.time() - t0)

    C1a, C1b, stats1a, stats1b, coo_tensor = F.double_quant(A)
    CB, absmaxB = F.vectorwise_quant(B, quant_type="linear")
    A2, SA = F.nvidia_transform(C1a, "col32")
    B2, SB = F.nvidia_transform(CB, formatB)
    A1, maxA = F.vectorwise_quant(A, dim=1)

    c = 10.0 * inner * scale
    row_scale = maxA / c
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        outC32, SC = F.igemmlt(A2, B2, SA, SB, dtype=torch.int8, row_scale=row_scale)
    torch.cuda.synchronize()
    print("row-wise", time.time() - t0)

    C2a, C2b, stats2a, stats2b, coo_tensor = F.double_quant(B)
    B2, SB = F.nvidia_transform(C2a, formatB)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        outC32, SC = F.igemmlt(A2, B2, SA, SB)
    torch.cuda.synchronize()
    print("vector-wise", time.time() - t0)


@pytest.mark.parametrize("dim1", get_test_dims(2, 1024, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(2, 1024, n=2), ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", [0], ids=id_formatter("dim3"))
@pytest.mark.parametrize("dims", [2], ids=id_formatter("dims"))
@pytest.mark.parametrize("dtype", [torch.int8], ids=describe_dtype)
@pytest.mark.parametrize("orderA", ["row"], ids=id_formatter("orderA"))
@pytest.mark.parametrize("orderOut", ["col32", "col_turing", "col_ampere"], ids=id_formatter("orderOut"))
@pytest.mark.parametrize("transpose", TRUE_FALSE, ids=id_formatter("transpose"))
def test_transform(dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose):
    for i in range(k):
        if dims == 2:
            A = torch.randint(10, 99, size=(dim1, dim2), device="cuda").to(dtype)
        elif dims == 3:
            A = torch.randint(10, 99, size=(dim1, dim2, dim3), device="cuda").to(dtype)

        A.view(-1)[-1] = -1
        if transpose:
            At = A.t().contiguous()
            out1, S1 = F.nvidia_transform(At, to_order=orderOut)
        else:
            out1, S1 = F.nvidia_transform(A, to_order=orderOut)
        out2, S2 = F.transform(A, to_order=orderOut, transpose=transpose)

        assert S1[0][0] == S2[0][0]
        assert S1[0][1] == S2[0][1]
        # print(out1)
        # print(out2)

        torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("dim1", get_test_dims(2, 1024, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(2, 1024, n=2), ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim3", [0], ids=id_formatter("dim3"))
@pytest.mark.parametrize("dims", [2], ids=id_formatter("dims"))
@pytest.mark.parametrize("dtype", [torch.int8], ids=describe_dtype)
@pytest.mark.parametrize("orderA", ["row"], ids=id_formatter("orderA"))
@pytest.mark.parametrize("orderOut", ["col32", "col_turing", "col_ampere"], ids=id_formatter("orderOut"))
@pytest.mark.parametrize("transpose", TRUE_FALSE, ids=id_formatter("transpose"))
def test_transform_cpu(dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose):
    for i in range(k):
        if dims == 2:
            A = torch.randint(10, 99, size=(dim1, dim2), device="cpu").to(dtype)
        elif dims == 3:
            A = torch.randint(10, 99, size=(dim1, dim2, dim3), device="cpu").to(dtype)

        A.view(-1)[-1] = -1
        if transpose:
            out1 = A.t().contiguous()
        else:
            out1 = A
        out2, S2 = F.transform(A, to_order=orderOut, transpose=transpose)

        assert S2 is None

        torch.testing.assert_close(out1, out2)


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
def test_overflow():
    formatB = F.get_special_format_str()
    print(formatB)
    for i in range(2):
        a = torch.arange(5, 15).cuda().to(torch.int8).view(-1, 1)
        b = torch.arange(5, 15).cuda().to(torch.int8).view(-1, 1)

        Ca, Sa = F.nvidia_transform(a, "col32")
        Cb, Sb = F.nvidia_transform(b, formatB)

        c = F.igemmlt(Ca, Cb, Sa, Sb, dtype=torch.int8)
        c2 = torch.matmul(a.float(), b.float().t())


@pytest.mark.parametrize("dim1", get_test_dims(1, 4 * 1024, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(1, 4 * 1024, n=2), ids=id_formatter("dim2"))
@pytest.mark.parametrize("device", ["cuda", "cpu"], ids=id_formatter("device"))
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16], ids=id_formatter("dtype"))
def test_coo_double_quant(dim1, dim2, device, dtype):
    if device == "cuda" and dtype == torch.bfloat16:
        pytest.skip("bfloat16 is not implemented for this operation on CUDA backend")
    threshold = 3.00
    for i in range(k):
        A = torch.randn(dim1, dim2, device=device).to(dtype)

        idx = torch.abs(A) >= threshold
        CA2, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A, threshold=threshold)

        if idx.sum() > 0:
            assert coo_tensor is not None
        if coo_tensor is not None:
            A1 = A * idx
            A2 = torch.zeros_like(A)
            A2[coo_tensor.rowidx.long(), coo_tensor.colidx.long()] = coo_tensor.values
            torch.testing.assert_close(A1, A2)

            A1 = A * (idx == 0)
            A2 = (CA.float() * statsA.unsqueeze(1) / 127).to(dtype)
            torch.testing.assert_close(A1, A2, rtol=0.05, atol=1.5e-2)


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.parametrize("dim1", get_test_dims(1, 1 * 1024, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(1, 1 * 1024, n=2), ids=id_formatter("dim2"))
@pytest.mark.parametrize("transposed_B", TRUE_FALSE, ids=id_formatter("transposed_B"))
def test_spmm_coo(dim1, dim2, transposed_B):
    threshold = 1.5
    dim3 = torch.randint(32, 128, size=(1,)).item()
    # dim3 = 17
    for i in range(k):
        A = torch.randn(dim1, dim2).cuda().half()
        if transposed_B:
            B = torch.randn(dim3, dim2).cuda().half()
        else:
            B = torch.randn(dim2, dim3).cuda().half()

        idx = torch.abs(A) >= threshold
        nnz = (idx == 1).sum().item()
        rows, cols = torch.where(idx)
        values = A[idx]
        cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
        A2 = A * idx

        if transposed_B:
            out2 = F.spmm_coo(cooA, B.t())
            out1 = torch.matmul(A2, B.t())
        else:
            out2 = F.spmm_coo(cooA, B)
            out1 = torch.matmul(A2, B)

        assert_all_approx_close(out1, out2, rtol=0.01, atol=3.0e-2, count=30)


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.benchmark
def test_spmm_bench():
    batch = 2
    model = 1024 * 1
    hidden = model * 4
    seq = 1024
    dim1 = batch * seq
    dim2 = model
    dim3 = hidden
    threshold = 4
    A = torch.randn(dim1, dim2, device="cuda").half()
    B = torch.randn(dim2, dim3, device="cuda").half()
    for i in range(10):
        C1 = bnb.matmul(A, B.t())

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        C1 = bnb.matmul(A, B.t())
    torch.cuda.synchronize()
    t8 = time.time() - t0

    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    print(nnz / idx.numel())
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)

    for i in range(10):
        out2 = F.spmm_coo(cooA, B)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        out2 = F.spmm_coo(cooA, B)
    torch.cuda.synchronize()
    tsp = time.time() - t0
    print(tsp, t8)
    print(tsp / t8)


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.parametrize("dim1", get_test_dims(256, 1024, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(256, 1024, n=2), ids=id_formatter("dim2"))
def test_integrated_sparse_decomp(dim1, dim2):
    threshold = 3.0
    formatB = "col_turing"
    for i in range(k):
        A = torch.randn(dim1, dim2).cuda().half()
        w1 = torch.randn(dim1, dim2).cuda().half()
        out1 = torch.matmul(A, w1.t())

        Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
        CTw1, Sw1 = F.transform(Cw1, formatB)

        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        C32A, SA = F.transform(CA, "col32")

        out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1)
        out2 = F.mm_dequant(out1_32, Sout1_32, statsA, statsw1)

        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A, threshold=threshold)
        C32A, SA = F.transform(CA, "col32")

        out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1)
        out3 = F.mm_dequant(out1_32, Sout1_32, statsA, statsw1)

        assert coo_tensor is not None

        out4 = F.spmm_coo(coo_tensor, w1.t())
        out5 = out3 + out4

        err1 = torch.abs(out1 - out2).mean().item()
        err2 = torch.abs(out1 - out5).mean().item()
        assert err2 < err1


def test_matmuls():
    a = torch.randn(256, 512).half().cuda()
    b = torch.randn(256, 512).half().cuda()
    c1 = torch.matmul(a, b.t())
    c2 = bnb.matmul(a, b)
    c3 = bnb.matmul_cublas(a, b.t())

    err1 = torch.abs(c1 - c2).mean().item()
    err2 = torch.abs(c1 - c3).mean().item()
    assert err1 < 0.2
    assert err2 < 0.2
    print(err1, err2)


@pytest.mark.parametrize("dim1", [1 * 2048], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [12288], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dtype", [torch.float16], ids=describe_dtype)
@pytest.mark.parametrize("out_func", ["zeros", "ones"], ids=id_formatter("out_func"))
def test_spmm_coo_very_sparse(dim1, dim2, dtype, out_func):
    out_func = getattr(torch, out_func)

    threshold = 3.3
    # threshold = 2.8
    # threshold = 0.0
    A = torch.randn(dim1, dim2, device="cuda").half()
    if dtype == torch.float16:
        B = torch.randn(dim2, dim2 * 4, device="cuda").half()
        torch.nn.init.xavier_uniform_(B)
    else:
        B = torch.randn(dim2, dim2 * 4, device="cuda").half()
        torch.nn.init.xavier_uniform_(B)
        B, SB = F.vectorwise_quant(B, quant_type="linear")
        # B = torch.randint(-127, 127, size=(dim2, dim2*4), device='cuda').to(torch.int8)

    print("")
    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
    A2 = A * idx
    out1 = torch.matmul(A2.half(), B.half())
    out = out_func(out1.shape, dtype=torch.float16, device=out1.device)
    out1 += out.clone()
    out2 = F.spmm_coo_very_sparse(cooA, B, out=out)
    # print(B)
    # print(out1)
    # print(out2)
    p = 200 / (2048 * 12288 * 4)
    n = out1.numel()
    count = math.ceil(p * n)
    std = out1.std()
    out1 /= std
    out2 /= std
    assert_all_approx_close(out1, out2.half(), rtol=0.01, atol=3.0e-2, count=count)
    # assert_all_approx_close(out1, out2.half(), rtol=0.05, atol=0.01, count=count)

    idx_col = torch.randint(0, A2.shape[-1], size=(15,))

    # torch.testing.assert_close(out1, out2.half(), rtol=0.05, atol=0.001)

    # Bt = torch.randn(dim2*4, dim2, device='cuda').half()
    # torch.cuda.synchronize()
    # t0 = time.time()
    # print(A2.shape, B.shape)
    # for i in range(100):
    #   #out3 = F.spmm_coo(cooA, Bt.t())
    #   #out2 = F.spmm_coo(cooA, B)
    #   #out2 = F.spmm_coo_very_sparse(cooA, B)
    #   #out1 = torch.matmul(A, Bt.t())

    # torch.cuda.synchronize()
    # print(time.time() - t0)


def test_coo2csr():
    threshold = 1
    A = torch.randn(128, 128).half().cuda()
    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
    A2 = A * idx
    csrA = F.coo2csr(cooA)
    counts = csrA.rowptr[1:] - csrA.rowptr[:-1]
    assert counts.numel() == A.shape[0]

    torch.testing.assert_close(counts.long(), (A2 != 0).sum(1))
    idx = A2 != 0
    torch.testing.assert_close(A2[idx], csrA.values)


def test_coo2csc():
    threshold = 1
    A = torch.randn(128, 128).half().cuda()
    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
    A2 = A * idx
    cscA = F.coo2csc(cooA)
    counts = cscA.colptr[1:] - cscA.colptr[:-1]
    assert counts.numel() == A.shape[1]

    torch.testing.assert_close(counts.long(), (A2 != 0).sum(0))
    # torch uses row-major -> use transpose to transfer to col-major
    idx = A2.t() != 0
    torch.testing.assert_close(A2.t()[idx], cscA.values)


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.parametrize("dim1", [1 * 2048])
@pytest.mark.parametrize("dim2", [2048])
@pytest.mark.parametrize("dtype", [torch.int8])
def test_spmm_coo_dequant(dim1, dim2, dtype):
    threshold = 6.0
    # threshold = 2.8
    # threshold = 0.0
    A = torch.randn(dim1, dim2, device="cuda").half()
    B = torch.empty(dim2, dim2 * 4, device="cuda", dtype=torch.float16)
    torch.nn.init.xavier_uniform_(B)
    Bt = B.t().contiguous()

    CB, CBt, statsB, statsBt, coo_tensor = F.double_quant(B)

    rowidx = torch.randint(0, A.shape[-1], size=(15,))

    A[:, rowidx] = 8.0

    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
    A2 = A * idx
    out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
    out1 = torch.matmul(A2, B.half())
    out3 = F.spmm_coo_very_sparse(cooA, CBt.half())
    out3 = out3 * statsBt.half() / 127

    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    offset = counts.cumsum(0).int()
    max_count, max_idx = torch.sort(counts, descending=True)
    print(torch.median(max_count.float()))

    torch.testing.assert_close(out2, out3, rtol=0.05, atol=0.001)

    p = 200 / (2048 * 12288 * 4)
    n = out1.numel()
    count = math.ceil(p * n)
    assert_all_approx_close(out1, out2, rtol=0.01, atol=3.0e-2, count=count)

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(100):
    #   out2 = F.spmm_coo_very_sparse(cooA, B)
    # torch.cuda.synchronize()
    # print('fp16', time.time() - t0)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out2 = F.spmm_coo(cooA, B)
    torch.cuda.synchronize()
    print("cusparse fp16", time.time() - t0)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out2 = F.spmm_coo_very_sparse(cooA, CBt)
    torch.cuda.synchronize()
    print("int8", time.time() - t0)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
    torch.cuda.synchronize()
    print("int8+dequant", time.time() - t0)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out2 = torch.matmul(A, B)
    torch.cuda.synchronize()
    print("matmul", time.time() - t0)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out1 = bnb.matmul(A, Bt)
        out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
        out = out1 + out2
    torch.cuda.synchronize()
    print("sparse+ matmul", time.time() - t0)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out1 = bnb.matmul(A, Bt)
        torch.matmul(A[:, rowidx], Bt.t()[rowidx], out=out1)
    torch.cuda.synchronize()
    print("partial matmul", time.time() - t0)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        out1 = bnb.matmul(A, Bt)
    torch.cuda.synchronize()
    print("partial matmul", time.time() - t0)


@pytest.mark.parametrize(
    ("batch", "seq", "model", "hidden"),
    [pytest.param(1, 1, 6656, 4 * 6656, id="batch=1, seq=1, model=6656, hidden=26k")],
)
@pytest.mark.benchmark
def test_bench_matmul(batch, seq, model, hidden):
    iters = 1000
    formatB = F.get_special_format_str()

    A = torch.randn(batch, seq, model, device="cuda").half()
    B = torch.empty(hidden, model, dtype=torch.float16, device="cuda")
    torch.nn.init.xavier_uniform_(B)

    B_fp4, state = F.quantize_fp4(B)
    B_fp4_c, state_c = F.quantize_fp4(B, compress_statistics=True)

    B_nf4, state_nf4 = F.quantize_nf4(B)
    B_nf4_c, state_nf4_c = F.quantize_nf4(B, compress_statistics=True)

    linear8bit = bnb.nn.Linear8bitLt(model, hidden, False, False).cuda().half()
    linear8bit.eval()

    outliers = torch.randint(0, model, size=(5,)).cuda()
    A[:, :, outliers] = 8.0

    linearMixedBit = bnb.nn.Linear8bitLt(model, hidden, False, False, threshold=6.0).cuda().half()
    # linearMixedBit.eval()

    linear8bit_train = bnb.nn.Linear8bitLt(model, hidden, False).cuda().half()
    linear8bit_train_thresh = bnb.nn.Linear8bitLt(model, hidden, False, threshold=6.0).cuda().half()
    bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)

    # warmup
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print("")

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print(
        f"pytorch fp16: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s",
    )

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    bnb.matmul_4bit(A, B_fp4.t(), quant_state=state)
    # torch.cuda.synchronize()
    # print( f"bnb fp4: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s" )

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    bnb.matmul_4bit(A, B_fp4.t(), quant_state=state_c)
    # torch.cuda.synchronize()
    # print( f"bnb fp4 + compressed stats: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s" )

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)
    torch.cuda.synchronize()
    print(f"bnb nf4: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul_4bit(A, B_nf4_c.t(), quant_state=state_nf4_c)
    torch.cuda.synchronize()
    print(f"bnb nf4+DQ: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    bnb.matmul(A, B)
    # torch.cuda.synchronize()
    # print(f"CB -> CxB conversion (each iteration): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    bnb.matmul(A, B, threshold=6.0)
    # torch.cuda.synchronize()
    # print(f"CB -> CxB conversion + threshold: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(A, threshold=0.0)
    # C32A, SA = F.transform(CA, "col32")
    # CB, CBt, SCB, SCBt, coo_tensorB = F.double_quant(B)
    # CxB, SB = F.transform(CB, to_order=formatB)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    out32, Sout32 = F.igemmlt(C32A, CxB, SA, SB)
    # torch.cuda.synchronize()
    # print(f"no overhead matmul-lt: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # BA, statsB = F.vectorwise_quant(B, dim=1)
    # CxB, SB = F.nvidia_transform(CB, to_order=formatB)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    A2 = A.view(-1, A.shape[-1]).contiguous()
    #    CA, statsA = F.vectorwise_quant(A2, dim=1)
    #    C32A, SA = F.nvidia_transform(CA, "col32")
    #    out32, Sout32 = F.igemmlt(C32A, CxB, SA, SB)
    #    Cout, Sout = F.nvidia_transform(out32, "row", state=Sout32)
    #    F.vectorwise_mm_dequant(Cout, statsA, statsB.t())
    # torch.cuda.synchronize()
    # print(f"vector pytorch + nvidia: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # BA, statsB = F.vectorwise_quant(B, dim=1, quant_type="linear")
    # CxB, SB = F.nvidia_transform(CB, to_order=formatB)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    A2 = A.view(-1, A.shape[-1]).contiguous()
    #    CA, statsA = F.vectorwise_quant(A2, dim=1, quant_type="linear")
    #    C32A, SA = F.nvidia_transform(CA, "col32")
    #    out32, Sout32 = F.igemmlt(C32A, CxB, SA, SB)
    #    Cout, Sout = F.nvidia_transform(out32, "row", state=Sout32)
    #    out = Cout * statsB * statsA * (1.0 / (127 * 127))
    # torch.cuda.synchronize()
    # print(f"linear pytorch + nvidia: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # linear8bit(A)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    linear8bit(A)
    # torch.cuda.synchronize()
    # print( f"bnb linear8bitlt (eval): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # linearMixedBit(A)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    linearMixedBit(A)
    # torch.cuda.synchronize()
    # print( f"bnb linear8bitlt with threshold (eval): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # linear8bit_train(A)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    linear8bit_train(A)
    # torch.cuda.synchronize()
    # print( f"bnb linear8bitlt (training): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    # linear8bit_train_thresh(A)
    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    linear8bit_train(A)
    # torch.cuda.synchronize()
    # print( f"bnb linear8bitlt with threshold (training): [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")


def test_zeropoint():
    def quant_zp(x):
        dtype = x.dtype
        x = x.float()
        dyna = x.max() - x.min()
        if dyna == 0:
            dyna = 1
        qx = 254.0 / dyna
        minx = x.min()
        # zpx = torch.round(minx* qx)
        # zpx = 127 - torch.round(x.max()* qx)
        zpx = torch.round(x.min() * qx) - 127
        x = (qx * x) + zpx
        return x, qx, zpx

    batch = 2
    seq = 512
    model = 1024
    hidden = 4 * model
    A = torch.randn(batch * seq, model, device="cuda").half() * 0.1
    B = torch.randn(model, hidden, device="cuda").half() * 0.1

    C0 = torch.matmul(A, B)

    # A, SA = F.vectorwise_quant(A, quant_type='linear')
    # B, SB = F.vectorwise_quant(B, quant_type='linear')
    A = A.float()
    B = B.float()

    C1 = torch.matmul(A, B)
    C3 = bnb.matmul(A.half(), B.t().contiguous().half())

    zp = 1
    # C2 = torch.matmul(A-zp, B)
    # C2 += B.sum(0).view(1, -1)*zp
    C2 = torch.matmul(A, B - zp)
    C2 -= A.sum(1).view(-1, 1) * zp

    ca, cqa, cza = quant_zp(A)
    # print(ca.min(), ca.max())
    # print((ca - cza).min(), (ca - cza).max())

    zp = 1
    scale = 2.0
    C5 = torch.matmul((A * scale) - zp, B)
    C5 += B.sum(0) * zp
    C5 /= scale

    CA, qa, zpa = quant_zp(A)
    C4 = torch.matmul(CA, B)
    C4 -= B.sum(0) * zpa
    C4 /= qa

    zpb = 1
    zpa = 1
    qa = 2
    qb = 2
    C6 = torch.matmul((A * qa) + zpa, (B * qb) + zpb)
    C6 -= (qb * B.sum(0).view(1, -1) * zpa) + (qa * A.sum(1).view(-1, 1) * zpb)
    C6 -= zpa * zpb * A.shape[1]
    C6 /= qa * qb

    CA, qa, zpa = quant_zp(A)
    CB, qb, zpb = quant_zp(B)
    C7 = torch.matmul(CA, CB)
    C7 -= (qb * B.sum(0).view(1, -1) * zpa) + (qa * A.sum(1).view(-1, 1) * zpb)
    C7 -= zpa * zpb * A.shape[1]
    C7 /= qa * qb

    # print("")
    # print(C0.flatten()[:10])
    # print(C1.flatten()[:10])
    # print(C2.flatten()[:10])
    # print(C3.flatten()[:10])
    # print(C5.flatten()[:10])
    # print(C6.flatten()[:10])
    # print(C7.flatten()[:10])
    err1 = torch.abs(C1 - C2).mean().item()
    err2 = torch.abs(C1 - C3).mean().item()
    err3 = torch.abs(C1 - C4).mean().item()
    err4 = torch.abs(C1 - C5).mean().item()
    err5 = torch.abs(C1 - C6).mean().item()
    err6 = torch.abs(C1 - C7).mean().item()
    print(err1, err2, err3, err4, err5, err6)


@pytest.mark.skipif(0 < BNB_HIP_VERSION < 601, reason="this test is supported on ROCm from 6.1")
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_extract_outliers(device):
    for i in range(k):
        shapeA = (4096, 4096 * 4)
        idx = torch.unique(torch.randint(0, shapeA[1], size=(10,)).int()).to(device=device)
        A = torch.randint(-128, 127, size=shapeA, device=device).to(torch.int8)
        outliers1 = A[:, idx.long()]

        CA, SA = F.transform(A, "col_turing")

        outliers2 = F.extract_outliers(CA, SA, idx)

        assert outliers2.shape[0] == shapeA[0]
        assert outliers2.shape[1] == idx.numel()

        torch.testing.assert_close(outliers1, outliers2)

        CA, SA = F.transform(A, "col_ampere")

        outliers2 = F.extract_outliers(CA, SA, idx)

        assert outliers2.shape[0] == shapeA[0]
        assert outliers2.shape[1] == idx.numel()

        torch.testing.assert_close(outliers1, outliers2)


def test_blockwise_cpu_large():
    diffs = []
    reldiffs = []
    batch = 128
    seq = 128
    for hidden in [128]:  # , 14336]:
        for blocksize in [4096, 16384]:
            for i in range(2):
                A1 = torch.randn(batch, seq, hidden, device="cpu")
                t0 = time.time()
                C, S = F.quantize_blockwise(A1, blocksize=blocksize)
                A2 = F.dequantize_blockwise(C, S, blocksize=blocksize)
                print(time.time() - t0)
                diff = torch.abs(A1 - A2)
                reldiff = diff / torch.abs(A1 + 1e-8)
                diffs.append(diff.mean().item())
                reldiffs.append(reldiff.mean().item())
                assert diffs[-1] < 0.011
            # print(sum(diffs)/len(diffs))
            # print(sum(reldiffs)/len(reldiffs))


def test_fp8_quant():
    for e_bits in range(1, 7):
        p_bits = 7 - e_bits
        code = F.create_fp8_map(True, e_bits, p_bits).cuda()

        abserr = []
        relerr = []
        for i in range(100):
            A1 = torch.randn(1024, 1024, device="cuda")
            C, SC = F.quantize_blockwise(A1, code=code)
            A2 = F.dequantize_blockwise(C, SC)
            diff = torch.abs(A1 - A2)
            reldiff = diff / torch.abs(A1 + 1e-8)
            abserr.append(diff.mean().item())
            relerr.append(reldiff.mean().item())
            # assert diff < 0.0075
        # print(sum(abserr)/len(abserr))
        # print(sum(relerr)/len(relerr))

        abserr = []
        relerr = []
        for i in range(100):
            A1 = torch.rand(1024, 1024, device="cuda")
            C, SC = F.quantize_blockwise(A1, code=code)
            A2 = F.dequantize_blockwise(C, SC)
            diff = torch.abs(A1 - A2)
            reldiff = diff / torch.abs(A1 + 1e-8)
            abserr.append(diff.mean().item())
            relerr.append(reldiff.mean().item())
            # assert diff < 0.0075
        # print(sum(abserr)/len(abserr))
        # print(sum(relerr)/len(relerr))

        abserr = []
        relerr = []
        for i in range(100):
            A1 = torch.randn(1024, 1024, device="cuda")
            C, SC = F.quantize_blockwise(A1)
            A2 = F.dequantize_blockwise(C, SC)
            diff = torch.abs(A1 - A2)
            reldiff = diff / torch.abs(A1 + 1e-8)
            abserr.append(diff.mean().item())
            relerr.append(reldiff.mean().item())
            # assert diff < 0.0075
        # print(3, sum(abserr)/len(abserr))
        # print(3, sum(relerr)/len(relerr))


def test_few_bit_quant():
    # print('')
    for bits in range(2, 9):
        # print('='*30, bits, '='*30)
        for method in ["linear", "fp8", "dynamic", "quantile"]:
            abserrs = []
            relerrs = []
            code = None
            if method == "linear":
                code = F.create_linear_map(True, total_bits=bits).cuda()
            elif method == "fp8":
                ebits = math.ceil(bits / 2)
                pbits = bits - ebits - 1
                code = F.create_fp8_map(True, ebits, pbits, bits).cuda()
            elif method == "dynamic":
                code = F.create_dynamic_map(True, bits - 0, bits).cuda()
            elif method == "quantile":
                values = torch.randn(2048, 2048, device="cuda")
                code = F.create_quantile_map(values, bits).cuda()
            # for some data types we have no zero
            # for some data types we have one zero
            # for some data types we have two zeros
            assert torch.unique(code).numel() in [2**bits, 2**bits - 1], f"bits: {bits}, method: {method}"
            # print(method, (code==0).sum())
            assert code.numel() == 256
            for i in range(10):
                values = torch.randn(1, 32, device="cuda")
                values /= values.abs().max()
                # values[values.abs() < 1e-6] += 1e-5

                q1 = []
                v1 = []
                for v in values[0]:
                    idx = torch.abs(v - code).argmin()
                    q1.append(idx.item())
                    v1.append(code[idx].item())

                q1 = torch.Tensor(q1).cuda()
                v1 = torch.Tensor(v1).cuda()

                q2, S2 = F.quantize_blockwise(values, code=code)
                v2 = F.dequantize_blockwise(q2, S2)

                idx = torch.isclose(q1.int(), q2.int())
                err2 = torch.abs(v2 - values)
                abserrs.append(err2.mean().item())
                relerrs.append((err2 / (1e-10 + values).abs()).mean().item())
                if idx.sum():
                    # some weird cases
                    err1 = torch.abs(v1 - values).mean()
                    # assert err2.mean() <= err1

                else:
                    torch.testing.assert_close(q1, q2)
            # print(method, 'abserr:', sum(abserrs)/len(abserrs), 'relerr:', sum(relerrs)/len(relerrs))
    # assert False


def test_kbit_quantile_estimation():
    for i in range(100):
        data = torch.randn(1024, 1024, device="cuda")
        for bits in range(2, 9):
            p = np.linspace(1.3e-4, 1 - 1.3e-4, 2**bits)
            val1 = torch.Tensor(norm.ppf(p)).cuda()
            val2 = F.estimate_quantiles(data, offset=0, num_quantiles=2**bits)
            err = torch.abs(val1 - val2).mean()
            assert err < 0.038

    for i in range(100):
        data = torch.randn(1024, 1024, device="cuda")
        for bits in range(2, 4):
            total_values = 2**bits - 1
            p = np.linspace(0, 1, 2 * total_values + 1)
            idx = np.arange(1, 2 * total_values + 1, 2)
            p = p[idx]
            offset = 1 / (2 * total_values)
            p = np.linspace(offset, 1 - offset, total_values)
            val1 = torch.Tensor(norm.ppf(p)).cuda()
            val2 = F.estimate_quantiles(data, num_quantiles=2**bits - 1)
            err = torch.abs(val1 - val2).mean()
            assert err < 0.035


@pytest.mark.benchmark
def test_bench_dequantization():
    a = torch.rand(1024, 1024, device="cuda").half()
    code = F.create_fp8_map(True, 3, 0, 4).cuda()
    qa, SA = F.quantize_blockwise(a, code=code)
    print(qa.max())

    max_theoretical_mu = 1024 * 1024 * 2 / 1024**3 / 672 * 1000 * 1000
    # print(max_theoretical_mu)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        qa, SA = F.quantize_blockwise(a)
    torch.cuda.synchronize()
    # print((time.time()-t0)/1e6)


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
@pytest.mark.parametrize("blocksize", [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_4bit_quant(dtype, quant_type, blocksize, device):
    vals = list(product([0, 1], repeat=4))

    code = {}
    for bits in vals:
        result = 0
        bias = 3
        sign, e1, e2, p1 = bits
        idx = sign * 8 + e1 * 4 + e2 * 2 + p1 * 1
        sign = -1.0 if sign else 1.0
        exp = e1 * 2 + e2 * 1
        if exp == 0:
            # sub-normal
            if p1 == 0:
                result = 0
            else:
                result = sign * 0.0625
        else:
            # normal
            exp = 2 ** (-exp + bias + 1)
            frac = 1.5 if p1 else 1.0
            result = sign * exp * frac
        code[idx] = result

    A1 = torch.randn(1024, 1024, device=device, dtype=dtype)
    qa, SA = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
    A2 = F.dequantize_4bit(qa, SA, blocksize=blocksize, quant_type=quant_type)
    if device == "cpu":
        A2 = A2.t()

    err = (A1 - A2).abs().float()
    relerr = (err / (A1.abs().float() + 1e-8)).mean()
    idx = err > 1.0
    err = err.mean()

    assert A2.dtype == dtype

    # With larger block sizes, we can expect this to blow up.
    # At blocksize>=1024, don't even bother looking at relerr.
    if blocksize <= 64:
        assert err.item() < 0.1
        assert relerr.item() < 0.28
    elif blocksize <= 256:
        assert err.item() < 0.11
        assert relerr.item() < 0.30
    elif blocksize <= 512:
        assert err.item() < 0.12
        assert relerr.item() < 0.31
    elif quant_type == "fp4":
        # 1024 => 0.48, 2048 => 0.52, 4096 => 0.56
        assert err.item() < 0.08 + math.log2(blocksize) * 4e-2
    else:
        # 1024 => 0.8, 2048 => 0.88, 4096 => 0.96
        assert err.item() < math.log2(blocksize) * 8e-2


@pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
def test_4bit_compressed_stats(quant_type):
    blocksizes = [128, 64] if not HIP_ENVIRONMENT else [128]
    for blocksize in blocksizes:
        errs1 = []
        errs2 = []
        for i in range(10):
            A1 = torch.randn(1024, 1024, device="cuda").half()
            q2, SA2 = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
            q3, SA3 = F.quantize_4bit(A1, blocksize=blocksize, compress_statistics=True, quant_type=quant_type)
            A2 = F.dequantize_4bit(q2, SA2, quant_type=quant_type)
            A3 = F.dequantize_4bit(q3, SA3, quant_type=quant_type)

            err = (A1 - A2).abs().float()
            relerr = (err / (A1.abs().float() + 1e-15)).mean()
            err = err.mean()

            errs1.append(err.item())

            assert err.item() < 0.11
            assert relerr.item() < 0.28

            err = (A1 - A3).abs().float()
            relerr = (err / (A1.abs().float() + 1e-15)).mean()
            err = err.mean()

            errs2.append(err.item())

            assert err.item() < 0.11
            assert relerr.item() < 0.28

        # print(sum(errs1)/len(errs1), blocksize, quant_type)
        # print(sum(errs2)/len(errs2), blocksize, quant_type)


# @pytest.mark.parametrize("quant_type", ['fp4', 'nf4'])
@pytest.mark.parametrize("quant_type", ["nf4"])
@pytest.mark.benchmark
def test_bench_4bit_dequant(quant_type):
    blocksize = 256
    a = torch.rand(1024 * 12 * 4, 1024 * 12, device="cuda").half()
    qa, SA = F.quantize_4bit(a, blocksize=blocksize, quant_type=quant_type)

    input_size = a.numel() / 2
    output_size = a.numel() * 2
    num_bytes = input_size + output_size
    GB = num_bytes / 1e9
    max_theoretical_s = GB / 768
    # print(max_theoretical_s*1e6)
    b = torch.randn(128, 1024 * 12, device="cuda").half()

    iters = 100
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        F.dequantize_4bit(qa, SA, blocksize=blocksize, quant_type=quant_type)
        # b.copy_(a)
    torch.cuda.synchronize()
    # print((time.time()-t0)/iters*1e6)

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    torch.matmul(b, a.t())
    # torch.cuda.synchronize()
    # print((time.time()-t0)/iters*1e6)


def test_normal_map_tree():
    code = F.create_normal_map()
    values = code[:8].tolist() + code[-8:].tolist()
    num_pivots = 1
    # print(values)
    while num_pivots < 16:
        idx = list(range(16 // num_pivots // 2, 16, 16 // num_pivots))
        # print(idx)
        num_pivots *= 2
        pivots = []
        for i in idx:
            pivots.append((values[i - 1] + values[i]) / 2)
        # print(pivots)


@pytest.mark.skipif(
    HIP_ENVIRONMENT, reason="gemv 4bit tests are partially enabled on MI300, others being fixed for warpsize 64"
)
@pytest.mark.parametrize("double_quant", TRUE_FALSE, ids=lambda double_quant: f"DQ_{double_quant}")
@pytest.mark.parametrize("storage_type", ["nf4", "fp4"])
@pytest.mark.parametrize("kind", ["fc1", "fc2", "attn", "attn_packed"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize(
    "quant_storage",
    [torch.uint8, torch.float16, torch.bfloat16, torch.float32],
    ids=describe_dtype,
)
def test_gemv_4bit(dtype, storage_type, quant_storage, double_quant, kind):
    for dim in [128, 256, 512, 1024]:
        # for dim in [4*1024]:
        # for dim in [1*16]:
        errs1 = []
        errs2 = []
        errs3 = []
        relerrs1 = []
        relerrs2 = []
        relerrs3 = []
        max_errs1 = []
        max_errs2 = []
        max_errs3 = []

        for i in range(100):
            if kind == "fc1":
                A = torch.randn(1, dim, dtype=dtype, device="cuda")
                B = torch.randn(dim * 4, dim, dtype=dtype, device="cuda") / math.sqrt(dim)
            elif kind == "fc2":
                A = torch.randn(1, 4 * dim, dtype=dtype, device="cuda")
                B = torch.randn(dim, 4 * dim, dtype=dtype, device="cuda") / math.sqrt(dim)
            elif kind == "attn":
                A = torch.randn(1, dim, dtype=dtype, device="cuda")
                B = torch.randn(dim, dim, dtype=dtype, device="cuda") / math.sqrt(dim)
            elif kind == "attn_packed":
                A = torch.randn(1, dim, dtype=dtype, device="cuda")
                B = torch.randn(dim * 3, dim, dtype=dtype, device="cuda") / math.sqrt(dim)

            qB, state = F.quantize_4bit(
                B,
                quant_type=storage_type,
                compress_statistics=double_quant,
                quant_storage=quant_storage,
            )
            C3 = torch.matmul(A, B.t())
            C2 = F.gemv_4bit(A, qB.t(), state=state)
            A.requires_grad = True
            C1 = bnb.matmul_4bit(A, qB.t(), state)

            err1 = (C1 - C2).abs().float()
            err2 = (C3 - C2).abs().float()
            err3 = (C3 - C1).abs().float()

            mag1 = torch.abs(C1).float() + 1e-5
            mag2 = torch.abs(C3).float() + 1e-5
            mag3 = torch.abs(C3).float() + 1e-5

            relerr1 = err1 / mag1
            relerr2 = err2 / mag2
            relerr3 = err3 / mag3

            max_err1 = err1.max()
            max_err2 = err2.max()
            max_err3 = err3.max()

            errs1.append(err1.mean().item())
            errs2.append(err2.mean().item())
            errs3.append(err3.mean().item())

            relerrs1.append(relerr1.mean().item())
            relerrs2.append(relerr2.mean().item())
            relerrs3.append(relerr3.mean().item())

            max_errs1.append(max_err1.item())
            max_errs2.append(max_err2.item())
            max_errs3.append(max_err3.item())

            c = int(C1.numel() * 0.0014 * (dim / 256)) + 1

            c = assert_all_approx_close(C1, C2, 1e-5, 0.01, count=c, throw=False)
        err1 = sum(errs1) / len(errs1) / math.sqrt(dim)
        err2 = sum(errs2) / len(errs2) / math.sqrt(dim)
        err3 = sum(errs3) / len(errs3) / math.sqrt(dim)
        relerr1 = sum(relerrs1) / len(relerrs1) / math.sqrt(dim)
        relerr2 = sum(relerrs2) / len(relerrs2) / math.sqrt(dim)
        relerr3 = sum(relerrs3) / len(relerrs3) / math.sqrt(dim)
        maxerr1 = sum(max_errs1) / len(max_errs1) / math.sqrt(dim)
        maxerr2 = sum(max_errs2) / len(max_errs2) / math.sqrt(dim)
        maxerr3 = sum(max_errs3) / len(max_errs3) / math.sqrt(dim)
        absratio = err2 / err3
        relratio = relerr2 / relerr3
        maxratio = relerr2 / relerr3

        # for debugging if the tests fails
        #
        # print('='*80)
        # print(f'For matmul: {A.shape}, {B.shape}, {kind}, {dtype}, {storage_type}, double_quant={double_quant}:')
        # print(C1.flatten()[-20:])
        # print(C2.flatten()[-20:])
        # print(f'inference vs training abs: {err1}')
        # print(f'inference vs training rel: {relerr1}')
        # print(f'inference vs training max: {maxerr1}')
        # print(f'inference vs training vs torch err ratio abs: {absratio}')
        # print(f'inference vs training vs torch err ratio rel: {relratio}')
        # print(f'inference vs training vs torch err ratio max: {maxratio}')
        if dtype == torch.float16:
            if dim <= 512:
                assert err1 < 7e-5
                assert relerr1 < 0.0008
            else:
                assert err1 < 6e-5
                assert relerr1 < 2e-4
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.005 and relratio > 0.995
            assert maxratio < 1.005 and maxratio > 0.995
        elif dtype == torch.float32:
            if dim <= 512:
                assert err1 < 5e-8
                assert relerr1 < 1e-6
                assert maxerr1 < 1e-7
            else:
                assert err1 < 5e-8
                assert relerr1 < 8e-6
                assert maxerr1 < 1e-7
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.005 and relratio > 0.995
            assert maxratio < 1.005 and maxratio > 0.995
        elif dtype == torch.bfloat16:
            if dim <= 512:
                assert err1 < 6e-4
                assert relerr1 < 0.007
                assert maxerr1 < 0.015
            else:
                assert err1 < 2e-4
                assert relerr1 < 0.002
                assert maxerr1 < 0.0012
            assert absratio < 1.005 and absratio > 0.995
            assert relratio < 1.04 and relratio > 0.96
            assert maxratio < 1.02 and maxratio > 0.98


@pytest.mark.parametrize("kind", ["fc1", "fc2", "attn", "attn_packed"])
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
def test_gemv_4bit_cpu(dtype, quant_type, kind):
    """
    Test 4bit GEMV for CPU. It is simplified a lot from the cuda version, since
    the CPU backend does not support double_quant or quant_storage other than uint8.
    Also, the CPU backend has different numeric accuracy from that of CUDA
    """
    for dim in [128, 256, 512, 1024]:
        for i in range(10):
            if kind == "fc1":
                A = torch.randn(1, dim, dtype=dtype, device="cpu")
                B = torch.randn(dim * 4, dim, dtype=dtype, device="cpu") / math.sqrt(dim)
            elif kind == "fc2":
                A = torch.randn(1, 4 * dim, dtype=dtype, device="cpu")
                B = torch.randn(dim, 4 * dim, dtype=dtype, device="cpu") / math.sqrt(dim)
            elif kind == "attn":
                A = torch.randn(1, dim, dtype=dtype, device="cpu")
                B = torch.randn(dim, dim, dtype=dtype, device="cpu") / math.sqrt(dim)
            elif kind == "attn_packed":
                A = torch.randn(1, dim, dtype=dtype, device="cpu")
                B = torch.randn(dim * 3, dim, dtype=dtype, device="cpu") / math.sqrt(dim)

            qB, state = F.quantize_4bit(
                B,
                quant_type=quant_type,
                compress_statistics=False,
                quant_storage=torch.uint8,
            )
            dqB = F.dequantize_4bit(qB, state)
            C3 = torch.matmul(A, dqB)
            C2 = F.gemv_4bit(A, qB.t(), state=state)
            A.requires_grad = True
            C1 = bnb.matmul_4bit(A, qB.t(), state)

            c = int(C1.numel() * 0.0014 * (dim / 256)) + 1
            rtol = 1e-3 if dtype != torch.bfloat16 else 1e-2
            atol = 1e-2 if dtype != torch.bfloat16 else 5e-2
            assert_all_approx_close(C1, C2, rtol, atol, count=c)
            assert_all_approx_close(C3, C2, rtol, atol, count=c)


@pytest.mark.skip("Row scale has some bugs for ampere")
def test_managed():
    n = 32 * 10
    A = F.get_paged(n, n, dtype=torch.float32)
    B = F.get_paged(n, n, dtype=torch.uint8)
    B2 = F.get_paged(n, n, dtype=torch.float32)
    assert A.is_paged
    assert B.is_paged
    assert A.page_deviceid == 0
    assert B.page_deviceid == 0
    F.fill(A, 17.0)
    F.fill(B, 17)
    F.fill(B2, 2)
    assert (A == 17).sum().item() == n * n
    assert (B == 17).sum().item() == n * n
    C = A * B.float()
    assert (C == 289).sum().item() == n * n
    F._mul(A, B2)
    F._mul(A, B2)
    F._mul(A, B2)
    assert (A == 17 * (2**3)).sum().item() == n * n


# F.prefetch_tensor(A)
# F.prefetch_tensor(B)


# F.fill(B2, 17.0)
# F._mul(A, B2)

# F.prefetch_tensor(A, to_cpu=True)
# F.prefetch_tensor(B, to_cpu=True)
# F.prefetch_tensor(B2, to_cpu=True)
# torch.cuda.synchronize()

# assert (A==17).sum().item() == n*n

# torch.testing.assert_close(A, torch.ones(A.shape)*289)


@pytest.mark.parametrize("storage_type", ["nf4", "fp4"], ids=["nf4", "fp4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("double_quant", [False], ids=["DQ_True"])
@pytest.mark.skipif(
    HIP_ENVIRONMENT and ROCM_GPU_ARCH == "gfx90a",
    reason="this test is not supported on ROCm with gfx90a architecture yet",
)
def test_gemv_eye_4bit(storage_type, dtype, double_quant):
    dims = 10
    torch.random.manual_seed(np.random.randint(0, 412424242))
    dims = get_test_dims(0, 8192, n=dims)
    dims = [dim + (64 - (dim % 64)) for dim in dims]
    # for dim in [576, 5120, 3520, 5184, 1280, 4992, 5312, 2048]:
    for dim in dims:
        A = torch.normal(0, 0.1, size=(1, 1, dim), dtype=dtype, device="cuda")
        B = torch.eye(dim, dtype=dtype, device="cuda")

        qB, state = F.quantize_4bit(B, quant_type=storage_type, compress_statistics=double_quant)
        C3 = torch.matmul(A, B.t())
        C2 = bnb.matmul_4bit(A, qB.t(), state)
        A.requires_grad = True
        C1 = bnb.matmul_4bit(A, qB.t(), state)

        torch.testing.assert_close(A, C3)
        torch.testing.assert_close(A, C1)
        torch.testing.assert_close(A, C2)
        # torch.testing.assert_close(A, C1, rtol=1e-5, atol=0.00001)
        # torch.testing.assert_close(A, C2, rtol=1e-5, atol=0.080)
