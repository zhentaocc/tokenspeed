# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The torch MXFP4 KV reference codec (CPU)."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.mxfp4_reference import (
    e2m1_decode,
    hadamard_matrix,
    mxfp4_dequantize_reference,
    mxfp4_fake_quantize,
    mxfp4_quantize_reference,
)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def test_e2m1_table_and_sign():
    codes = torch.arange(16, dtype=torch.uint8)
    values = e2m1_decode(codes)
    torch.testing.assert_close(values[:8], _E2M1)
    torch.testing.assert_close(values[8:], -_E2M1)


def test_packing_order_and_scale():
    # One group scaled by 2: amax 12 -> scale 2**ceil(log2(2)) = 2, UE8M0 128.
    x = torch.zeros(64)
    x[0], x[1], x[31] = 1.0, -12.0, 3.0
    packed, scales = mxfp4_quantize_reference(x)
    assert packed.shape == (32,) and scales.shape == (2,)
    assert scales.tolist() == [128, 1]  # the zero group clamps to the floor
    # element 0 (0.5 * 2) is code 1 in the low nibble, element 1 (-6 * 2) is
    # code 15 in the high nibble.
    assert packed[0].item() == (1 | (15 << 4))
    torch.testing.assert_close(mxfp4_dequantize_reference(packed, scales), x)


def test_round_half_to_even_code():
    # Scale 1 (amax 6); midpoints between neighbouring E2M1 values.
    x = torch.tensor([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] * 8)
    out = mxfp4_fake_quantize(x)
    expected = torch.tensor([6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0] * 8)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_representable_rows_round_trip(dtype):
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (5, 4, 256), dtype=torch.uint8)
    exponents = torch.randint(110, 140, (5, 4, 8), dtype=torch.int32)
    values = e2m1_decode(codes).reshape(5, 4, 8, 32)
    values = values * torch.pow(2.0, (exponents - 127).float())[..., None]
    x = values.reshape(5, 4, 256).to(dtype)
    torch.testing.assert_close(mxfp4_fake_quantize(x), x, rtol=0, atol=0)


def test_ceil_never_clips_and_mse_is_no_worse():
    torch.manual_seed(1)
    x = torch.randn(512, 512) * torch.rand(512, 1) * 10
    x[:, ::37] *= 20  # outliers
    ceil = mxfp4_fake_quantize(x, exponent_mode="ceil")
    mse = mxfp4_fake_quantize(x, exponent_mode="mse")
    _, scales = mxfp4_quantize_reference(x)
    scale = torch.pow(2.0, scales.float() - 127)
    # ceil keeps every scaled value inside E2M1's range: nothing saturates.
    assert torch.all(x.reshape(512, 16, 32).abs().amax(-1) / scale <= 6.0)
    ceil_err = (ceil - x).reshape(512, 16, 32).square().sum(-1)
    mse_err = (mse - x).reshape(512, 16, 32).square().sum(-1)
    assert torch.all(mse_err <= ceil_err)
    assert mse_err.sum() < ceil_err.sum()


def test_hadamard_is_orthonormal_and_helps_outliers():
    h = hadamard_matrix(512, "cpu")
    torch.testing.assert_close(h @ h, torch.eye(512), atol=1e-5, rtol=0)
    torch.manual_seed(2)
    x = torch.randn(256, 512)
    x[:, 7] *= 50  # one outlier channel shared by every row
    plain = (mxfp4_fake_quantize(x) - x).square().sum()
    rotated = (mxfp4_fake_quantize(x, hadamard=True) - x).square().sum()
    assert rotated < plain


def test_rejects_rows_not_multiple_of_64():
    with pytest.raises(ValueError):
        mxfp4_quantize_reference(torch.zeros(32))
