"""Tests for the Gemma 4 QAT Mobile Metal ``qmv`` kernel (Phase 5).

Validates the fused ``gemma_mobile_matmul`` Metal fast path against a numpy
ground-truth reference for int2/int4/int8, across batch sizes and realistic
dimensions, and verifies the fallback heuristics.

Note: MLX's own ``matmul`` on Metal uses a different accumulation order (and
lower intermediate precision) than a naive fp32 dot-product, so the kernel is
compared against a **numpy** reference (the mathematical ground truth), not
against ``dequantize_weight`` + ``x @ w.T``.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.quantization.gemma_mobile import (
    GemmaQuantizedLinear,
    _gemma_qmv_kernel,
    apply_srq,
    dequantize_weight,
    gemma_mobile_matmul,
)


# ---------------------------------------------------------------------------
# Packing helpers (mirror the format spec / test_gemma_mobile_quant.py)
# ---------------------------------------------------------------------------

def _pack_int2_row(vals: np.ndarray) -> np.ndarray:
    u = (vals.astype(np.int32) + 2).astype(np.uint8)
    packed = np.zeros(u.shape[0] // 4, dtype=np.uint8)
    for i in range(4):
        packed |= u[i::4] << (2 * i)
    return packed


def _pack_int4_row(vals: np.ndarray) -> np.ndarray:
    u = (vals.astype(np.int32) + 8).astype(np.uint8)
    packed = np.zeros(u.shape[0] // 2, dtype=np.uint8)
    packed |= u[0::2] & 0x0F
    packed |= (u[1::2] & 0x0F) << 4
    return packed


def _quant_per_channel(W: np.ndarray, bits: int):
    if bits == 2:
        denom, lo, hi = 1.0, -2, 1
    elif bits == 4:
        denom, lo, hi = 7.0, -8, 7
    else:
        denom, lo, hi = 127.0, -128, 127
    scale = (np.max(np.abs(W), axis=1, keepdims=True) / denom).astype(np.float32)
    q = np.clip(np.round(W / scale), lo, hi).astype(np.int8)
    return q, scale


def _pack_weight(q: np.ndarray, bits: int) -> np.ndarray:
    if bits == 2:
        return np.stack([_pack_int2_row(r) for r in q])
    if bits == 4:
        return np.stack([_pack_int4_row(r) for r in q])
    return q.astype(np.int8)


def _make_quant_weight(out_d, in_d, bits, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((out_d, in_d)).astype(np.float32)
    q, scale = _quant_per_channel(W, bits)
    packed = _pack_weight(q, bits)
    return mx.array(packed), mx.array(scale), q, scale


def _numpy_ref(x_np, q, scale):
    """Ground-truth fp32 matmul: ``x @ (q * scale).T`` computed in numpy."""
    w = q.astype(np.float32) * scale
    return x_np @ w.T


# ---------------------------------------------------------------------------
# Kernel availability
# ---------------------------------------------------------------------------

def test_qmv_kernel_available_on_metal():
    """The qmv kernel should compile on a Metal-capable system."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    for bits in (2, 4, 8):
        assert _gemma_qmv_kernel(bits) is not None


# ---------------------------------------------------------------------------
# Correctness vs numpy ground-truth (realistic dims = multiples of 512)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
def test_gemma_mobile_matmul_matches_reference(bits, batch):
    """The Metal qmv path must match the numpy fp32 ground-truth reference."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    out_d, in_d = 512, 512
    packed, scale, q, s = _make_quant_weight(out_d, in_d, bits, seed=42)
    rng = np.random.default_rng(99)
    x_np = rng.standard_normal((batch, in_d)).astype(np.float32)
    x = mx.array(x_np)

    got = np.array(gemma_mobile_matmul(x, packed, scale, bits, in_d))
    ref = _numpy_ref(x_np, q, s)

    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_gemma_mobile_matmul_int8_uses_fallback():
    """int8 weights should use the pure-MLX path (Metal kernel is int2/int4 only)."""
    out_d, in_d = 512, 512
    packed, scale, q, s = _make_quant_weight(out_d, in_d, 8, seed=7)
    x_np = np.random.default_rng(1).standard_normal((1, in_d)).astype(np.float32)
    x = mx.array(x_np)
    got = np.array(gemma_mobile_matmul(x, packed, scale, 8, in_d))
    ref = _numpy_ref(x_np, q, s)
    # Pure-MLX path uses MLX's matmul which has a different accumulation order
    # than numpy; allow a slightly looser tolerance for the fallback.
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=0.1)


def test_gemma_mobile_matmul_large_batch_fallback():
    """Batch > 16 should fall back to pure-MLX (prefill path)."""
    out_d, in_d = 512, 512
    packed, scale, q, s = _make_quant_weight(out_d, in_d, 2, seed=5)
    x_np = np.random.default_rng(2).standard_normal((32, in_d)).astype(np.float32)
    x = mx.array(x_np)
    got = np.array(gemma_mobile_matmul(x, packed, scale, 2, in_d))
    ref = _numpy_ref(x_np, q, s)
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=0.1)


def test_gemma_mobile_matmul_non_aligned_dims_fallback():
    """Non-multiple-of-512 input dims should fall back to pure-MLX."""
    out_d, in_d = 64, 100  # not a multiple of 512
    packed, scale, q, s = _make_quant_weight(out_d, in_d, 2, seed=3)
    x_np = np.random.default_rng(4).standard_normal((1, in_d)).astype(np.float32)
    x = mx.array(x_np)
    got = np.array(gemma_mobile_matmul(x, packed, scale, 2, in_d))
    ref = _numpy_ref(x_np, q, s)
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=0.1)


# ---------------------------------------------------------------------------
# GemmaQuantizedLinear end-to-end with the Metal path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 4])
def test_quantized_linear_metal_path(bits):
    """GemmaQuantizedLinear should produce correct output via the Metal path."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    out_d, in_d = 512, 512
    packed, scale, q, s = _make_quant_weight(out_d, in_d, bits, seed=10)
    lin = GemmaQuantizedLinear(in_d, out_d, bits, bias=True, dtype=mx.float32)
    lin.weight = packed
    lin.weight_scale = scale
    lin.bias = mx.zeros((out_d,), dtype=mx.float32)
    x_np = np.random.default_rng(20).standard_normal((4, in_d)).astype(np.float32)
    x = mx.array(x_np)
    got = np.array(lin(x))
    ref = _numpy_ref(x_np, q, s)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_quantized_linear_metal_with_srq():
    """SRQ scales should be applied correctly around the Metal matmul."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    out_d, in_d = 512, 512
    packed, scale, q, s = _make_quant_weight(out_d, in_d, 4, seed=30)
    lin = GemmaQuantizedLinear(
        in_d, out_d, 4, bias=False, dtype=mx.float32,
        input_scale=True, output_scale=True,
    )
    lin.weight = packed
    lin.weight_scale = scale
    lin.input_activation_scale = mx.array(2.0)
    lin.output_activation_scale = mx.array(0.0)  # no-op
    x_np = np.random.default_rng(40).standard_normal((2, in_d)).astype(np.float32)
    x = mx.array(x_np)
    got = np.array(lin(x))
    xq = np.array(apply_srq(x, mx.array(2.0)))
    ref = _numpy_ref(xq, q, s)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_quantized_linear_metal_with_both_srq():
    """Both input and output SRQ should be fused correctly into the Metal kernel."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    out_d, in_d = 512, 512
    packed, scale, q, s = _make_quant_weight(out_d, in_d, 4, seed=31)
    lin = GemmaQuantizedLinear(
        in_d, out_d, 4, bias=False, dtype=mx.float32,
        input_scale=True, output_scale=True,
    )
    lin.weight = packed
    lin.weight_scale = scale
    lin.input_activation_scale = mx.array(2.0)
    lin.output_activation_scale = mx.array(3.0)
    x_np = np.random.default_rng(41).standard_normal((2, in_d)).astype(np.float32)
    x = mx.array(x_np)
    got = np.array(lin(x))
    # Reference: input SRQ → matmul → output SRQ (computed in MLX for parity)
    xq = apply_srq(x, mx.array(2.0))
    out = xq @ mx.array((q.astype(np.float32) * s).T.astype(np.float32))
    ref = np.array(apply_srq(out, mx.array(3.0)))
    # The Metal kernel and the MLX matmul use different accumulation orders
    # (~1e-3 relative).  The output SRQ ``round`` amplifies this: an element
    # near a rounding boundary can flip by 1 integer × output_scale (3.0).
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=3.0)


@pytest.mark.parametrize("in_d", [512, 256])
def test_quantized_linear_metal_int8(in_d):
    """int8 weights should use the Metal kernel (with fused SRQ) for decode."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    out_d = 256 if in_d == 512 else 1536
    packed, scale, q, s = _make_quant_weight(out_d, in_d, 8, seed=32)
    lin = GemmaQuantizedLinear(in_d, out_d, 8, bias=False, dtype=mx.float32)
    lin.weight = packed
    lin.weight_scale = scale
    x_np = np.random.default_rng(42).standard_normal((1, in_d)).astype(np.float32)
    x = mx.array(x_np)
    got = np.array(lin(x))
    ref = _numpy_ref(x_np, q, s)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# qmm kernel (P3: batched prefill, batch > 16)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("batch", [17, 32, 64])
def test_gemma_mobile_qmm_matches_reference(bits, batch):
    """The batched qmm kernel (batch > 1, out >= 2048) must match numpy."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    # out_d >= 2048 to trigger the qmm path (below that, qmv/fallback is used).
    if bits == 8:
        out_d, in_d = 2048, 256
    else:
        out_d, in_d = 2048, 2048
    packed, scale, q, s = _make_quant_weight(out_d, in_d, bits, seed=50)
    rng = np.random.default_rng(99)
    x_np = rng.standard_normal((batch, in_d)).astype(np.float32)
    x = mx.array(x_np)
    got = np.array(gemma_mobile_matmul(x, packed, scale, bits, in_d))
    ref = _numpy_ref(x_np, q, s)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-2)


def test_gemma_mobile_qmm_large_dims():
    """qmm kernel with realistic dimensions (out=6144, in=1536, batch=64)."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    out_d, in_d = 6144, 1536
    packed, scale, q, s = _make_quant_weight(out_d, in_d, 4, seed=60)
    rng = np.random.default_rng(60)
    x_np = rng.standard_normal((64, in_d)).astype(np.float32)
    x = mx.array(x_np)
    got = np.array(gemma_mobile_matmul(x, packed, scale, 4, in_d))
    ref = _numpy_ref(x_np, q, s)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-1)


def test_gemma_mobile_qmm_with_srq():
    """qmm kernel with fused input + output SRQ."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    out_d, in_d = 2048, 2048
    packed, scale, q, s = _make_quant_weight(out_d, in_d, 4, seed=70)
    rng = np.random.default_rng(70)
    x_np = rng.standard_normal((32, in_d)).astype(np.float32)
    x = mx.array(x_np)
    in_s, out_s = mx.array(2.0), mx.array(3.0)
    got = np.array(gemma_mobile_matmul(x, packed, scale, 4, in_d, input_scale=in_s, output_scale=out_s))
    xq = np.array(apply_srq(x, in_s))
    w = q.astype(np.float32) * s
    ref = np.array(apply_srq(mx.array(xq @ w.T), out_s))
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=3.0)


def test_gemma_mobile_qmm_per_row_output_scale():
    """qmm kernel with per-row output SRQ (fused q/k/v case)."""
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    out_d, in_d = 2048, 2048
    packed, scale, q, s = _make_quant_weight(out_d, in_d, 4, seed=71)
    rng = np.random.default_rng(71)
    x_np = rng.standard_normal((32, in_d)).astype(np.float32)
    x = mx.array(x_np)
    in_s = mx.array(2.0)
    # Per-row output scale: first half 3.0, second half 1.5 (like q vs k/v).
    out_s_row = mx.array(np.concatenate([np.full(1024, 3.0), np.full(1024, 1.5)]).astype(np.float32))
    got = np.array(gemma_mobile_matmul(x, packed, scale, 4, in_d, input_scale=in_s, output_scale=out_s_row))
    xq = np.array(apply_srq(x, in_s))
    w = q.astype(np.float32) * s
    out_ref = xq @ w.T
    out_s_np = np.array(out_s_row)
    ref = np.clip(np.round(out_ref / out_s_np), -128, 127) * out_s_np
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=3.0)
