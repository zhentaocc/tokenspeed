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

"""Simulated MXFP4 KV cache: a debug hook for accuracy studies.

``TOKENSPEED_SIMULATE_KV_QUANT`` names a recipe; every KV row a layer writes is
then rounded through MXFP4 in place, right after the write, so every later
reader of the cache sees the rounded values while the cache keeps its storage
dtype. It measures the accuracy of an MXFP4 KV cache before the packed pools
and kernels exist. It is slow and is not a serving mode.

Recipe: comma-separated tokens, starting with ``mxfp4``:

* ``mse``: pick each group's exponent from ``{ceil, ceil - 1}`` by squared
  error instead of always ``ceil``.
* ``hadamard``: rotate the MLA latent (the non-RoPE part) by the orthonormal
  Hadamard matrix before rounding and back after. MLA keys and values are
  the same latent, so a real cache folds this rotation into the absorbed key
  and value projections at load time.
* ``rope``: also round the MLA key's RoPE part (otherwise it keeps the cache
  dtype, as FlashMLA's FP8 layout keeps it in BF16).
* ``k_only`` / ``v_only``: for MHA caches, round only the keys or the values.

Example: ``TOKENSPEED_SIMULATE_KV_QUANT=mxfp4,mse,hadamard``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.kvcache.mxfp4_reference import mxfp4_fake_quantize

from tokenspeed.runtime.utils.env import envs

_RECIPE_FLAGS = ("mse", "hadamard", "rope", "k_only", "v_only")


@dataclass(frozen=True)
class SimulatedKVQuant:
    exponent_mode: str = "ceil"
    hadamard: bool = False
    quantize_rope: bool = False
    keys: bool = True
    values: bool = True

    @classmethod
    def parse(cls, spec: str) -> SimulatedKVQuant:
        tokens = [token.strip() for token in spec.split(",") if token.strip()]
        if not tokens or tokens[0] != "mxfp4":
            raise ValueError(
                f"TOKENSPEED_SIMULATE_KV_QUANT must start with 'mxfp4', got {spec!r}"
            )
        flags = set(tokens[1:])
        unknown = flags.difference(_RECIPE_FLAGS)
        if unknown:
            raise ValueError(
                f"unknown simulated KV quant flags {sorted(unknown)}; "
                f"known: {list(_RECIPE_FLAGS)}"
            )
        if {"k_only", "v_only"} <= flags:
            raise ValueError("k_only and v_only are exclusive")
        return cls(
            exponent_mode="mse" if "mse" in flags else "ceil",
            hadamard="hadamard" in flags,
            quantize_rope="rope" in flags,
            keys="v_only" not in flags,
            values="k_only" not in flags,
        )

    def round_latent(self, rows: torch.Tensor, kv_lora_rank: int) -> torch.Tensor:
        """MLA latent rows ``[..., kv_lora_rank + rope_dim]`` rounded through MXFP4."""
        nope = mxfp4_fake_quantize(
            rows[..., :kv_lora_rank],
            exponent_mode=self.exponent_mode,
            hadamard=self.hadamard,
        )
        rope = rows[..., kv_lora_rank:]
        if self.quantize_rope:
            rope = mxfp4_fake_quantize(rope, exponent_mode=self.exponent_mode)
        return torch.cat((nope, rope), dim=-1)

    def round_heads(self, rows: torch.Tensor) -> torch.Tensor:
        """MHA K or V rows ``[..., head_dim]`` rounded through MXFP4."""
        return mxfp4_fake_quantize(
            rows, exponent_mode=self.exponent_mode, hadamard=self.hadamard
        )


@functools.cache
def simulated_kv_quant() -> SimulatedKVQuant | None:
    """The process's simulated KV quantization recipe, or None when off."""
    spec = envs.TOKENSPEED_SIMULATE_KV_QUANT.get()
    if not spec:
        return None
    return SimulatedKVQuant.parse(spec)


def _row_index(cache: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Index tuple of ``slots`` into a ``[slots, ...]`` or a
    ``[pages, page_size, ...]`` row view."""
    slots = slots.long()
    if cache.dim() == 3:
        return (slots,)
    if cache.dim() == 4:
        page_size = cache.shape[1]
        return (slots // page_size, slots % page_size)
    raise ValueError(f"unsupported KV row view of shape {tuple(cache.shape)}")


def round_written_rows_(
    cache: torch.Tensor,
    slots: torch.Tensor,
    write_mask: torch.Tensor | None,
    round_rows,
) -> None:
    """Replace the rows at ``slots`` of ``cache`` with ``round_rows(rows)``.

    Rows that ``write_mask`` excludes are rewritten unchanged; every op keeps
    a fixed shape, so this also runs inside a captured graph.
    """
    if slots.numel() == 0:
        return
    index = _row_index(cache, slots)
    # Gather and scatter bytes: FP8 tensors do not support index_put.
    storage = cache.view(torch.uint8) if cache.element_size() == 1 else cache
    rows = storage[index].view(cache.dtype).float()
    rounded = round_rows(rows)
    if write_mask is not None:
        keep = write_mask.reshape(-1, *([1] * (rows.dim() - 1)))
        rounded = torch.where(keep, rounded, rows)
    storage[index] = rounded.to(cache.dtype).view(storage.dtype)


def simulate_latent_write_(
    recipe: SimulatedKVQuant,
    cache: torch.Tensor,
    slots: torch.Tensor,
    write_mask: torch.Tensor | None,
    kv_lora_rank: int,
) -> None:
    """Round the MLA latent rows just written at ``slots`` through MXFP4."""
    if not isinstance(cache, torch.Tensor):
        raise NotImplementedError(
            "simulated MXFP4 KV needs a plain latent cache (BF16 or FP8 rows), "
            "not per-token-head planes"
        )
    round_written_rows_(
        cache,
        slots,
        write_mask,
        lambda rows: recipe.round_latent(rows, kv_lora_rank),
    )
