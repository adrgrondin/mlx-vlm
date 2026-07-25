"""Tests for the Gemma 4 QAT Mobile static hybrid KV cache (Phase 4)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.quantization.gemma_mobile_cache import (
    GemmaStaticQuantizedKVCache,
    GemmaStaticQuantizedRotatingKVCache,
    build_gemma_static_caches,
    dequantize_kv_static,
    extract_gemma_kv_scales,
    quantize_kv_static,
)


def _rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Quantize / dequantize helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_dequantize_kv_roundtrip(bits):
    rng = _rng()
    kv = mx.array(rng.standard_normal((1, 1, 4, 8)).astype(np.float32))
    scale = mx.array(0.5)
    q = quantize_kv_static(kv, scale, bits)
    dq = dequantize_kv_static(q, scale, bits, 8)
    err = float(mx.max(mx.abs(kv - dq)))
    # Symmetric uniform quantization error is bounded by scale/2.
    assert err < 0.5, f"bits={bits} err={err}"


def test_quantize_kv_4bit_packs_to_uint8():
    kv = mx.zeros((1, 1, 1, 8))
    q = quantize_kv_static(kv, mx.array(1.0), 4)
    assert q.dtype == mx.uint8
    assert q.shape == (1, 1, 1, 4)  # 8 / 2 = 4 packed bytes


def test_quantize_kv_8bit_is_int8():
    kv = mx.zeros((1, 1, 1, 8))
    q = quantize_kv_static(kv, mx.array(1.0), 8)
    assert q.dtype == mx.int8
    assert q.shape == (1, 1, 1, 8)


# ---------------------------------------------------------------------------
# Cache classes
# ---------------------------------------------------------------------------

def test_static_kv_cache_prefill_and_decode():
    rng = _rng()
    cache = GemmaStaticQuantizedKVCache(mx.array(0.5), mx.array(0.5), bits=4)
    k = mx.array(rng.standard_normal((1, 1, 3, 8)).astype(np.float32))
    v = mx.array(rng.standard_normal((1, 1, 3, 8)).astype(np.float32))
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == (1, 1, 3, 8)
    assert np.all(np.isfinite(np.array(k_out)))
    # decode
    k1 = mx.array(rng.standard_normal((1, 1, 1, 8)).astype(np.float32))
    v1 = mx.array(rng.standard_normal((1, 1, 1, 8)).astype(np.float32))
    k_out, v_out = cache.update_and_fetch(k1, v1)
    assert k_out.shape == (1, 1, 4, 8)
    assert cache.offset == 4


def test_static_rotating_kv_cache_rotation():
    rng = _rng()
    cache = GemmaStaticQuantizedRotatingKVCache(
        mx.array(0.5), mx.array(0.5), max_size=4, bits=8
    )
    k = mx.array(rng.standard_normal((1, 1, 3, 8)).astype(np.float32))
    v = mx.array(rng.standard_normal((1, 1, 3, 8)).astype(np.float32))
    cache.update_and_fetch(k, v)
    # decode past max_size -> rotation
    for _ in range(5):
        k1 = mx.array(rng.standard_normal((1, 1, 1, 8)).astype(np.float32))
        v1 = mx.array(rng.standard_normal((1, 1, 1, 8)).astype(np.float32))
        k_out, v_out = cache.update_and_fetch(k1, v1)
        assert k_out.shape[2] <= 4, "rotating cache bounded by max_size"
        assert np.all(np.isfinite(np.array(k_out)))


def test_static_kv_cache_no_bits_attr():
    """The cache must NOT expose ``bits`` (would trigger MLX's quantized SDPA)."""
    cache = GemmaStaticQuantizedKVCache(mx.array(0.5), mx.array(0.5), bits=4)
    assert not hasattr(cache, "bits")
    assert cache.kv_bits == 4


# ---------------------------------------------------------------------------
# Scale extraction + cache builder
# ---------------------------------------------------------------------------

def test_extract_gemma_kv_scales():
    weights = {
        "model.language_model.layers.0.self_attn.k_cache_scale": mx.array(0.1),
        "model.language_model.layers.0.self_attn.v_cache_scale": mx.array(0.2),
        "model.language_model.layers.3.self_attn.k_cache_scale": mx.array(0.3),
        "model.language_model.layers.3.self_attn.v_cache_scale": mx.array(0.4),
        "language_model.model.layers.5.self_attn.k_cache_scale": mx.array(0.5),
        "language_model.model.layers.5.self_attn.v_cache_scale": mx.array(0.6),
        "other.weight": mx.zeros((1,)),
    }
    scales = extract_gemma_kv_scales(weights)
    assert sorted(scales.keys()) == [0, 3, 5]
    assert np.isclose(float(scales[0][0]), 0.1)
    assert np.isclose(float(scales[0][1]), 0.2)
    assert np.isclose(float(scales[5][1]), 0.6)


def test_build_gemma_static_caches():
    layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    kv_scales = {i: (mx.array(0.5), mx.array(0.5)) for i in range(4)}
    caches = build_gemma_static_caches(
        layer_types, kv_scales, sliding_window=8,
        num_kv_shared_layers=0, num_hidden_layers=4,
    )
    assert len(caches) == 4
    assert isinstance(caches[0], GemmaStaticQuantizedRotatingKVCache)
    assert caches[0].kv_bits == 8  # sliding -> 8-bit
    assert isinstance(caches[1], GemmaStaticQuantizedKVCache)
    assert caches[1].kv_bits == 4  # full -> 4-bit
    assert isinstance(caches[3], GemmaStaticQuantizedKVCache)
    assert caches[3].kv_bits == 4


def test_build_gemma_static_caches_fallback_fp16():
    """Layers without a scale fall back to the fp16 cache."""
    from mlx_vlm.models.cache import KVCache, RotatingKVCache

    layer_types = ["sliding_attention", "full_attention"]
    caches = build_gemma_static_caches(
        layer_types, {}, sliding_window=8,
        num_kv_shared_layers=0, num_hidden_layers=2,
    )
    assert isinstance(caches[0], RotatingKVCache)
    assert isinstance(caches[1], KVCache)
