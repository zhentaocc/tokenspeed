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

"""The simulated MXFP4 KV-cache debug hook (CPU)."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.mxfp4_reference import mxfp4_fake_quantize

from tokenspeed.runtime.layers.attention.kv_cache.simulated_quant import (
    SimulatedKVQuant,
    simulate_latent_write_,
)

_RANK, _ROPE = 512, 64


def test_parse_recipe():
    assert SimulatedKVQuant.parse("mxfp4") == SimulatedKVQuant()
    recipe = SimulatedKVQuant.parse("mxfp4, mse,hadamard,rope")
    assert recipe.exponent_mode == "mse"
    assert recipe.hadamard and recipe.quantize_rope
    with pytest.raises(ValueError):
        SimulatedKVQuant.parse("nvfp4")
    with pytest.raises(ValueError):
        SimulatedKVQuant.parse("mxfp4,typo")
    with pytest.raises(ValueError):
        SimulatedKVQuant.parse("mxfp4,k_only,v_only")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("rope", [False, True])
def test_latent_write_rounds_only_written_rows(dtype, paged, rope):
    torch.manual_seed(0)
    slots_total, page = 64, 16
    rows = (torch.randn(slots_total, 1, _RANK + _ROPE) * 2).to(dtype)
    cache = rows.clone()
    view = cache.view(slots_total // page, page, 1, -1) if paged else cache
    slots = torch.tensor([3, 17, 40, 41])
    mask = torch.tensor([True, True, False, True])
    recipe = SimulatedKVQuant.parse("mxfp4,rope" if rope else "mxfp4")
    simulate_latent_write_(recipe, view, slots, mask, _RANK)

    written = slots[mask]
    untouched = torch.ones(slots_total, dtype=torch.bool)
    untouched[written] = False
    assert torch.equal(
        cache[untouched].view(torch.uint8), rows[untouched].view(torch.uint8)
    )
    got = cache[written, 0].float()
    src = rows[written, 0].float()
    torch.testing.assert_close(
        got[:, :_RANK], mxfp4_fake_quantize(src[:, :_RANK]).to(dtype).float()
    )
    expected_rope = mxfp4_fake_quantize(src[:, _RANK:]) if rope else src[:, _RANK:]
    torch.testing.assert_close(got[:, _RANK:], expected_rope.to(dtype).float())
    # Idempotent: rounding the rounded rows again changes nothing.
    again = cache.clone()
    again_view = again.view(slots_total // page, page, 1, -1) if paged else again
    simulate_latent_write_(recipe, again_view, slots, mask, _RANK)
    assert torch.equal(again.view(torch.uint8), cache.view(torch.uint8))


def test_hadamard_latent_error_is_smaller_on_outlier_channels():
    torch.manual_seed(1)
    x = torch.randn(128, 1, _RANK + _ROPE)
    x[..., 11] *= 40
    plain = SimulatedKVQuant.parse("mxfp4").round_latent(x, _RANK)
    rotated = SimulatedKVQuant.parse("mxfp4,hadamard").round_latent(x, _RANK)
    plain_err = (plain - x)[..., :_RANK].square().sum()
    rotated_err = (rotated - x)[..., :_RANK].square().sum()
    assert rotated_err < plain_err
    # The RoPE part is untouched without ``rope``.
    torch.testing.assert_close(rotated[..., _RANK:], x[..., _RANK:])
