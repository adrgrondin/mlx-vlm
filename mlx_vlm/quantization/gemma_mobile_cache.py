"""Static hybrid KV cache for Gemma 4 QAT mobile (wNa8o8).

The mobile checkpoints ship precomputed per-layer scalar ``k_cache_scale`` /
``v_cache_scale`` (``float32``, shape ``[]``) for a **static**, per-tensor
symmetric KV-cache quantization:

* **Global (full-attention) layers → 4-bit** KV cache.
* **Local (sliding) layers → 8-bit** KV cache.

This is *not* the runtime codebook scheme (TurboQuant) — it is uniform
symmetric quantization with a fixed scale baked in at QAT time, so there is no
dynamic calibration overhead.

The cache classes subclass MLX's ``KVCache`` / ``RotatingKVCache`` and override
``update_and_fetch`` to quantize incoming K/V (packed ``uint8`` for 4-bit,
``int8`` for 8-bit) before storing, and dequantize on fetch. The rotation /
append logic of the parent classes operates on the packed tensors directly
(slicing / concatenate / assignment are dtype-agnostic), so only the fetch
needs dequantization.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx

from ..models.cache import KVCache, RotatingKVCache
from .gemma_mobile import unpack_int4
from .gemma_mobile_quantize import pack_int4_row


# ---------------------------------------------------------------------------
# Per-tensor symmetric quantization helpers (static scale)
# ---------------------------------------------------------------------------

def quantize_kv_static(kv: mx.array, scale: mx.array, bits: int) -> mx.array:
    """Quantize K/V with a static scalar scale → int8 (8-bit) or packed uint8 (4-bit)."""
    lo = -(2 ** (bits - 1))
    hi = 2 ** (bits - 1) - 1
    q = mx.clip(mx.round(kv / scale), lo, hi).astype(mx.int8)
    return pack_int4_row(q) if bits == 4 else q


def dequantize_kv_static(
    qkv: mx.array, scale: mx.array, bits: int, head_dim: int
) -> mx.array:
    """Dequantize packed/int K/V back to float using the static scalar scale."""
    if bits == 4:
        ints = unpack_int4(qkv, head_dim)
    else:
        ints = qkv.astype(mx.int8)
    return ints.astype(scale.dtype) * scale


# ---------------------------------------------------------------------------
# Quantized cache classes
# ---------------------------------------------------------------------------

class GemmaStaticQuantizedKVCache(KVCache):
    """Full-attention KV cache with static 4-bit (default) KV quantization."""

    def __init__(
        self,
        key_scale: mx.array,
        value_scale: mx.array,
        bits: int = 4,
        step: int = 256,
    ):
        super().__init__()
        self.key_scale = key_scale
        self.value_scale = value_scale
        # Use ``kv_bits`` (not ``bits``) so ``scaled_dot_product_attention`` does
        # not mistake us for MLX's group-quantized ``QuantizedKVCache`` (which
        # has ``bits`` + ``group_size`` and returns *packed* K/V). We dequantize
        # on fetch and return fp K/V, so the standard SDPA path is correct.
        self.kv_bits = bits
        self.k_head_dim: Optional[int] = None
        self.v_head_dim: Optional[int] = None

    def _quant(self, kv: mx.array, scale: mx.array) -> mx.array:
        return quantize_kv_static(kv, scale, self.kv_bits)

    def _dequant(self, qkv: mx.array, scale: mx.array, head_dim: int) -> mx.array:
        return dequantize_kv_static(qkv, scale, self.kv_bits, head_dim)

    def update_and_fetch(self, keys, values):
        if self.k_head_dim is None:
            self.k_head_dim = keys.shape[-1]
            self.v_head_dim = values.shape[-1]
        qk = self._quant(keys, self.key_scale)
        qv = self._quant(values, self.value_scale)
        rqk, rqv = super().update_and_fetch(qk, qv)
        return (
            self._dequant(rqk, self.key_scale, self.k_head_dim),
            self._dequant(rqv, self.value_scale, self.v_head_dim),
        )

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.kv_bits, self.k_head_dim, self.v_head_dim)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.kv_bits, self.k_head_dim, self.v_head_dim = (
            int(v[0]), int(v[1]), int(v[2]), int(v[3])
        )


class GemmaStaticQuantizedRotatingKVCache(RotatingKVCache):
    """Sliding-window KV cache with static 8-bit (default) KV quantization."""

    def __init__(
        self,
        key_scale: mx.array,
        value_scale: mx.array,
        max_size: int,
        bits: int = 8,
        keep: int = 0,
        step: int = 256,
    ):
        super().__init__(max_size=max_size, keep=keep)
        self.key_scale = key_scale
        self.value_scale = value_scale
        self.kv_bits = bits
        self.k_head_dim: Optional[int] = None
        self.v_head_dim: Optional[int] = None

    def _quant(self, kv: mx.array, scale: mx.array) -> mx.array:
        return quantize_kv_static(kv, scale, self.kv_bits)

    def _dequant(self, qkv: mx.array, scale: mx.array, head_dim: int) -> mx.array:
        return dequantize_kv_static(qkv, scale, self.kv_bits, head_dim)

    def update_and_fetch(self, keys, values):
        if self.k_head_dim is None:
            self.k_head_dim = keys.shape[-1]
            self.v_head_dim = values.shape[-1]
        qk = self._quant(keys, self.key_scale)
        qv = self._quant(values, self.value_scale)
        rqk, rqv = super().update_and_fetch(qk, qv)
        return (
            self._dequant(rqk, self.key_scale, self.k_head_dim),
            self._dequant(rqv, self.value_scale, self.v_head_dim),
        )

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (self.keep, self.max_size, self.offset, self._idx, self.kv_bits),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        self.keep, self.max_size, self.offset, self._idx, self.kv_bits = map(
            int, v
        )


# ---------------------------------------------------------------------------
# KV-scale extraction (from raw HF weights, before sanitize drops them)
# ---------------------------------------------------------------------------

_KV_SCALE_RE = re.compile(r"layers\.(\d+)\.self_attn\.(k|v)_cache_scale")


def extract_gemma_kv_scales(weights: Dict[str, mx.array]) -> Dict[int, Tuple[mx.array, mx.array]]:
    """Extract ``{layer_idx: (key_scale, value_scale)}`` from raw HF weights.

    Matches both the HF multimodal layout (``model.language_model.layers.i…``)
    and the MLX text layout (``language_model.model.layers.i…``).
    """
    scales: Dict[int, Dict[str, mx.array]] = {}
    for k, v in weights.items():
        m = _KV_SCALE_RE.search(k)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        which = m.group(2)
        scales.setdefault(layer_idx, {})[which] = v
    return {
        i: (s["k"], s["v"])
        for i, s in sorted(scales.items())
        if "k" in s and "v" in s
    }


def build_gemma_static_caches(
    layer_types,
    kv_scales: Dict[int, Tuple[mx.array, mx.array]],
    sliding_window: int,
    num_kv_shared_layers: int = 0,
    num_hidden_layers: int = 0,
) -> list:
    """Build the hybrid static-quantized cache list for a Gemma 4 mobile model.

    Global (full-attention) layers get a 4-bit ``GemmaStaticQuantizedKVCache``;
    local (sliding) layers get an 8-bit ``GemmaStaticQuantizedRotatingKVCache``.
    Layers without a precomputed scale fall back to the fp16 cache.
    """
    first_kv_shared = num_hidden_layers - num_kv_shared_layers
    caches = []
    for i in range(first_kv_shared):
        is_full = layer_types[i] == "full_attention"
        if i in kv_scales:
            k_scale, v_scale = kv_scales[i]
            if is_full:
                caches.append(
                    GemmaStaticQuantizedKVCache(k_scale, v_scale, bits=4)
                )
            else:
                caches.append(
                    GemmaStaticQuantizedRotatingKVCache(
                        k_scale, v_scale, max_size=sliding_window, keep=0, bits=8
                    )
                )
        elif is_full:
            caches.append(KVCache())
        else:
            caches.append(RotatingKVCache(max_size=sliding_window, keep=0))
    return caches
