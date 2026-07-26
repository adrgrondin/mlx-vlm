"""Gemma 4 QAT Mobile (wNa8o8) quantization backend for mlx-vlm.

Implements the ``quant_method: "gemma"`` format used by Google's Gemma 4 QAT
mobile checkpoints (e.g. ``google/gemma-4-E2B-it-qat-mobile-transformers``):

* **Weights** — per-output-channel *symmetric* uniform quantization at 2/4/8
  bits. ``int2``/``int4`` are packed in ``uint8`` (4 / 2 values per byte,
  LSB-first); ``int8`` is stored directly. A per-channel ``float`` scale
  (``[out, 1]``) dequantizes as ``unpack(weight) * weight_scale``.
* **Static activations (SRQ)** — optional scalar ``input_activation_scale`` /
  ``output_activation_scale`` per layer fake-quantize activations to int8 and
  back (``scale == 0`` ⇒ no-op / uncalibrated).
* **Embeddings** — packed table + per-row (or block-wise) scale.

This mirrors the HuggingFace transformers reference implementation
(``src/transformers/integrations/gemma_quant.py``) bit-for-bit.

Pure-MLX dequantize-on-forward: weights stay packed in resident memory
(delivering the ~1 GB / 0.84 GB footprint that is the point of the format) and
are unpacked transiently per layer inside the matmul. Fused Metal kernels are a
follow-on optimization (see plan §5 Phase 5).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map_with_path

# ---------------------------------------------------------------------------
# Bit-exact unpacking (matches gemma_quant._unpack_int2 / _unpack_int4)
# ---------------------------------------------------------------------------

_U2_MASK = mx.array(0x03, mx.uint8)
_U4_MASK = mx.array(0x0F, mx.uint8)
_SHIFT_2 = mx.array(2, mx.uint8)
_SHIFT_4 = mx.array(4, mx.uint8)
_SHIFT_6 = mx.array(6, mx.uint8)


def unpack_int2(packed: mx.array, in_features: int) -> mx.array:
    """Unpack int2 from uint8 (4 values/byte, LSB-first) → int8 in [-2, 1]."""
    if packed.dtype != mx.uint8:
        raise ValueError(f"int2 weights must be uint8, got {packed.dtype}.")
    v0 = (packed & _U2_MASK).astype(mx.int8) - 2
    v1 = ((packed >> _SHIFT_2) & _U2_MASK).astype(mx.int8) - 2
    v2 = ((packed >> _SHIFT_4) & _U2_MASK).astype(mx.int8) - 2
    v3 = (packed >> _SHIFT_6).astype(mx.int8) - 2
    out = mx.stack([v0, v1, v2, v3], axis=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :in_features]


def unpack_int4(packed: mx.array, in_features: int) -> mx.array:
    """Unpack int4 from uint8 (2 values/byte, low-nibble-first) → int8 in [-8, 7]."""
    if packed.dtype != mx.uint8:
        raise ValueError(f"int4 weights must be uint8, got {packed.dtype}.")
    low = (packed & _U4_MASK).astype(mx.int8) - 8
    high = (packed >> _SHIFT_4).astype(mx.int8) - 8
    out = mx.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :in_features]


def unpack_int(packed: mx.array, num_bits: int, in_features: int) -> mx.array:
    """Dispatch unpacking on the bit width; returns signed int8 values."""
    if num_bits == 2:
        return unpack_int2(packed, in_features)
    if num_bits == 4:
        return unpack_int4(packed, in_features)
    if num_bits == 8:
        if packed.dtype != mx.int8:
            raise ValueError(f"int8 weights must be int8, got {packed.dtype}.")
        return packed
    raise ValueError(f"Unsupported num_bits {num_bits}; expected 2, 4, or 8.")


# ---------------------------------------------------------------------------
# Static Range Quantization (SRQ) for activations
# ---------------------------------------------------------------------------

def apply_srq(x: mx.array, scale: mx.array, bits: int = 8) -> mx.array:
    """Static Range Quantization rounding/clipping (fake-quant to int8 and back).

    ``scale == 0`` means the layer is uncalibrated ⇒ no-op. The guard uses
    ``mx.where`` (not ``scale.item()``) so it stays on-device and compile-friendly.
    """
    max_value = 2 ** (bits - 1) - 1
    min_value = -max_value - 1
    scale = scale.astype(x.dtype)
    calibrated = scale != 0
    safe_scale = mx.where(calibrated, scale, mx.ones_like(scale))
    q = mx.clip(mx.round(x / safe_scale), min_value, max_value) * safe_scale
    return mx.where(calibrated, q, x)


# ---------------------------------------------------------------------------
# Weight dequantization
# ---------------------------------------------------------------------------

def _channel_scale(weight_scale: mx.array) -> mx.array:
    """Coerce a per-channel scale to a broadcastable [..., 1] shape."""
    if weight_scale.ndim >= 1 and weight_scale.shape[-1] == 1:
        return weight_scale
    return weight_scale.reshape(*weight_scale.shape, 1) if weight_scale.ndim else weight_scale


def dequantize_weight(
    weight: mx.array,
    weight_scale: mx.array,
    num_bits: int,
    in_features: int,
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    """Dequantize a packed per-channel weight → real matrix ``unpack(w) * scale``.

    ``weight`` is ``[out, packed_in]`` (uint8/int8); result is ``[out, in]``.
    """
    ints = unpack_int(weight, num_bits, in_features)
    out = ints.astype(weight_scale.dtype) * _channel_scale(weight_scale)
    return out if dtype is None else out.astype(dtype)


# ---------------------------------------------------------------------------
# Metal fused qmv kernel (Phase 5: single-token / small-batch decode)
# ---------------------------------------------------------------------------

# Two simdgroups (64 lanes) per threadgroup. Each simdgroup computes
# OUTPUTS_PER_SIMDGROUP output rows, sharing the activation reads across them
# (each lane reads its VALUES_PER_THREAD activations once and reuses them for
# every output row in its group — ~4x fewer activation reads per row). The grid
# is (64, ceil(out/8), batch) with threadgroup (64, 1, 1): grid-x < 256 so a
# single threadgroup is dispatched in x (the two simdgroups), grid-y carries
# groups of OUTPUTS_PER_THREADGROUP output rows, and grid-z carries the
# input/batch rows. Each lane owns a contiguous slice of VALUES_PER_THREAD
# activations and the matching packed weight bytes per output row; it unpacks,
# multiply-accumulates, and ``simd_sum`` reduces each row's accumulator across
# the 32 lanes of its simdgroup. SRQ (static activation quantization) is fused:
# input SRQ when reading activations, output SRQ when writing the result.
_GEMMA_QMV_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint input_row = threadgroup_position_in_grid.z;
    uint input_dims = x_shape[1];
    uint output_dims = weight_shape[0];
    uint packed_in = weight_shape[1];
    uint output_start = threadgroup_position_in_grid.y * OUTPUTS_PER_THREADGROUP
        + simd_group * OUTPUTS_PER_SIMDGROUP;

    // Input SRQ scale is scalar (shared across all output rows); 0.0 means
    // uncalibrated → no-op.  Output SRQ scale is per-row (a scalar broadcast to
    // [output_dims] for standalone layers, or a true per-row array for fused
    // q/k/v where q and k/v have different output scales).
    float in_s = static_cast<float>(input_scale[0]);

    float accumulators[OUTPUTS_PER_SIMDGROUP] = {0.0f};
    constexpr uint VALUES_PER_THREAD = 16;
    constexpr uint BYTES_PER_THREAD = VALUES_PER_THREAD / VALUES_PER_BYTE;
    constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;

    for (uint block_start = lane * VALUES_PER_THREAD;
         block_start < input_dims;
         block_start += BLOCK_SIZE) {
        // Read + input-SRQ x values ONCE, shared across OUTPUTS_PER_SIMDGROUP rows.
        float x_thread[VALUES_PER_THREAD];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
            float x_val = static_cast<float>(
                x[input_row * input_dims + block_start + i]);
            // Fused input SRQ: fake-quant to int8 and back.
            if (in_s != 0.0f) {
                x_val = clamp(round(x_val / in_s), -128.0f, 127.0f) * in_s;
            }
            x_thread[i] = x_val;
        }
        uint packed_start = block_start / VALUES_PER_BYTE;
        for (uint row = 0; row < OUTPUTS_PER_SIMDGROUP; ++row) {
            uint output_row = output_start + row;
            if (output_row >= output_dims) break;
            float row_sum = 0.0f;
            #pragma clang loop unroll(full)
            for (uint b = 0; b < BYTES_PER_THREAD; ++b) {
                UNPACK_BLOCK
            }
            accumulators[row] += row_sum;
        }
    }

    for (uint row = 0; row < OUTPUTS_PER_SIMDGROUP; ++row) {
        accumulators[row] = simd_sum(accumulators[row]);
        uint output_row = output_start + row;
        if (lane == 0 && output_row < output_dims) {
            float result = accumulators[row] * static_cast<float>(weight_scale[output_row]);
            // Fused output SRQ: fake-quant the matmul output to int8 and back.
            // OUTPUT_SCALE_PER_ROW is a compile-time bool (templated): true for
            // fused q/k/v (per-row scales), false for standalone layers (scalar).
            float out_s;
            if (OUTPUT_SCALE_PER_ROW) {
                out_s = static_cast<float>(output_scale[output_row]);
            } else {
                out_s = static_cast<float>(output_scale[0]);
            }
            if (out_s != 0.0f) {
                result = clamp(round(result / out_s), -128.0f, 127.0f) * out_s;
            }
            out[input_row * output_dims + output_row] = static_cast<T>(result);
        }
    }
"""


def _gemma_unpack_block(num_bits: int) -> str:
    """Metal snippet that unpacks BYTES_PER_THREAD packed bytes and MACs.

    Values are cast to ``float`` *before* subtracting the signed offset so the
    arithmetic stays in floating-point (avoiding unsigned-integer underflow
    when the masked value is smaller than the offset, e.g. ``(uint)0 - 2``).
    """
    if num_bits == 2:
        return r"""
                    uint byte = uint(weight[output_row * packed_in + packed_start + b]);
                    row_sum += (float(byte & 0x3) - 2.0f) * x_thread[b * 4 + 0]
                             + (float((byte >> 2) & 0x3) - 2.0f) * x_thread[b * 4 + 1]
                             + (float((byte >> 4) & 0x3) - 2.0f) * x_thread[b * 4 + 2]
                             + (float(byte >> 6) - 2.0f) * x_thread[b * 4 + 3];"""
    if num_bits == 4:
        return r"""
                    uint byte = uint(weight[output_row * packed_in + packed_start + b]);
                    row_sum += (float(byte & 0xF) - 8.0f) * x_thread[b * 2 + 0]
                             + (float(byte >> 4) - 8.0f) * x_thread[b * 2 + 1];"""
    if num_bits == 8:
        return r"""
                    int v = int(weight[output_row * packed_in + packed_start + b]);
                    row_sum += float(v) * x_thread[b];"""
    raise ValueError(f"Unsupported num_bits {num_bits}")


# Two simdgroups (64 lanes) per threadgroup; each simdgroup computes
# OUTPUTS_PER_SIMDGROUP output rows, sharing the activation reads across them.
_OUTPUTS_PER_SIMDGROUP = 4
_OUTPUTS_PER_THREADGROUP = _OUTPUTS_PER_SIMDGROUP * 2  # 8

# Minimum output dims for the qmm (batched) kernel to beat the qmv (GEMV) kernel
# in prefill.  Below this, the qmm grid overhead dominates; above it, the weight
# sharing across batch rows wins (benchmarked at batch=16: qmm is 2-3x faster
# for out >= 2048, slightly slower for out <= 1536).
_QMM_MIN_OUTPUT_DIMS = 2048


@lru_cache(maxsize=None)
def _gemma_qmv_kernel(num_bits: int, per_row_output_scale: bool = False):
    if not hasattr(mx, "metal") or not mx.metal.is_available():
        return None
    values_per_byte = {2: 4, 4: 2, 8: 1}[num_bits]
    source = (
        _GEMMA_QMV_SOURCE.replace("VALUES_PER_BYTE", str(values_per_byte))
        .replace("OUTPUTS_PER_SIMDGROUP", str(_OUTPUTS_PER_SIMDGROUP))
        .replace("OUTPUTS_PER_THREADGROUP", str(_OUTPUTS_PER_THREADGROUP))
        .replace(
            "OUTPUT_SCALE_PER_ROW",
            "true" if per_row_output_scale else "false",
        )
        .replace("UNPACK_BLOCK", _gemma_unpack_block(num_bits))
    )
    return mx.fast.metal_kernel(
        name=f"mlx_vlm_gemma_mobile_qmv_b{num_bits}_pr{int(per_row_output_scale)}",
        input_names=["x", "weight", "weight_scale", "input_scale", "output_scale"],
        output_names=["out"],
        source=source,
    )


# ---------------------------------------------------------------------------
# Metal fused qmm kernel (P3: batched prefill, batch > 16)
# ---------------------------------------------------------------------------

_GEMMA_QMM_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

# Tiled GEMM using simdgroup matrix operations (adapted from one_bit.py's qmm).
# Each threadgroup (128 threads = 4 simdgroups) computes a [BM=32, BN=32] output
# tile.  Weights are dequantized on-the-fly when loading the weight tile (no full
# weight materialization).  SRQ is fused: input SRQ on the x tile load, per-row
# weight_scale + output SRQ on the output write.
_GEMMA_QMM_SOURCE = r"""
    constexpr short BM = 32;
    constexpr short BK = 32;
    constexpr short BN = 32;
    constexpr short TILE_STRIDE = BK + 16 / sizeof(T);

    const int M = x_shape[0];
    const int K = x_shape[1];
    const int N = weight_shape[0];
    const int packed_in = weight_shape[1];
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint thread_index = thread_index_in_threadgroup;
    // Grid mapping: grid.x carries batch tiles (×128 dispatch multiplier),
    // grid.y carries output tiles.  This puts the ×128 overhead on the small
    // batch dimension instead of the large output dimension, reducing grid-x
    // from output_tiles×128 (up to ~1M for lm_head) to batch_tiles×128 (~128).
    const int tile_row = threadgroup_position_in_grid.x * BM;
    const int tile_column = threadgroup_position_in_grid.y * BN;
    const int load_row = thread_index / 4;
    const int load_column = (thread_index % 4) * 8;
    const int simd_row = simd_group / 2;
    const int simd_column = simd_group % 2;

    const short quad = lane / 4;
    const short fragment_row = (quad & 4) + ((lane / 2) % 4);
    const short fragment_column = (quad & 2) * 2 + (lane % 2) * 2;

    float in_s = static_cast<float>(input_scale[0]);

    threadgroup T x_tile[BM * TILE_STRIDE];
    threadgroup T weight_tile[BN * TILE_STRIDE];

    metal::simdgroup_matrix<float, 8, 8> accumulators[2][2];
    #pragma clang loop unroll(full)
    for (short row = 0; row < 2; ++row) {
        #pragma clang loop unroll(full)
        for (short column = 0; column < 2; ++column) {
            accumulators[row][column].thread_elements()[0] = 0.0f;
            accumulators[row][column].thread_elements()[1] = 0.0f;
        }
    }

    for (int k_start = 0; k_start < K; k_start += BK) {
        int input_row = tile_row + load_row;
        int input_column = k_start + load_column;

        // Load x tile with fused input SRQ.
        #pragma clang loop unroll(full)
        for (short element = 0; element < 8; ++element) {
            float x_val = (input_row < M)
                ? static_cast<float>(x[input_row * K + input_column + element])
                : 0.0f;
            if (in_s != 0.0f) {
                x_val = clamp(round(x_val / in_s), -128.0f, 127.0f) * in_s;
            }
            x_tile[load_row * TILE_STRIDE + load_column + element] =
                static_cast<T>(x_val);
        }

        // Load + dequantize weight tile (per-row scale applied after the matmul).
        UNPACK_TILE_BLOCK

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (short k_fragment = 0; k_fragment < BK; k_fragment += 8) {
            metal::simdgroup_matrix<float, 8, 8> a[2];
            metal::simdgroup_matrix<float, 8, 8> b[2];
            #pragma clang loop unroll(full)
            for (short tile = 0; tile < 2; ++tile) {
                int a_row = simd_row * 16 + tile * 8 + fragment_row;
                int b_row = simd_column * 16 + tile * 8 + fragment_column;
                #pragma clang loop unroll(full)
                for (short element = 0; element < 2; ++element) {
                    a[tile].thread_elements()[element] = static_cast<float>(
                        x_tile[a_row * TILE_STRIDE + k_fragment +
                               fragment_column + element]);
                    b[tile].thread_elements()[element] = static_cast<float>(
                        weight_tile[(b_row + element) * TILE_STRIDE +
                                    k_fragment + fragment_row]);
                }
            }

            #pragma clang loop unroll(full)
            for (short row = 0; row < 2; ++row) {
                #pragma clang loop unroll(full)
                for (short column = 0; column < 2; ++column) {
                    metal::simdgroup_matrix<float, 8, 8> result;
                    simdgroup_multiply_accumulate(
                        result, a[row], b[column],
                        accumulators[row][column]);
                    accumulators[row][column] = result;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (short row = 0; row < 2; ++row) {
        #pragma clang loop unroll(full)
        for (short column = 0; column < 2; ++column) {
            int output_row = tile_row + simd_row * 16 + row * 8 + fragment_row;
            int output_column =
                tile_column + simd_column * 16 + column * 8 + fragment_column;
            #pragma clang loop unroll(full)
            for (short element = 0; element < 2; ++element) {
                if (output_row < M && output_column + element < N) {
                    float result =
                        accumulators[row][column].thread_elements()[element];
                    result *= static_cast<float>(
                        weight_scale[output_column + element]);
                    float out_s;
                    if (OUTPUT_SCALE_PER_ROW) {
                        out_s = static_cast<float>(
                            output_scale[output_column + element]);
                    } else {
                        out_s = static_cast<float>(output_scale[0]);
                    }
                    if (out_s != 0.0f) {
                        result =
                            clamp(round(result / out_s), -128.0f, 127.0f) * out_s;
                    }
                    out[output_row * N + output_column + element] =
                        static_cast<T>(result);
                }
            }
        }
    }
"""


def _gemma_qmm_unpack_tile(num_bits: int) -> str:
    """Metal snippet that loads + dequantizes 8 packed weight elements into the
    weight tile (per-row scale is applied after the matmul, not here)."""
    if num_bits == 2:
        return r"""
        int w_row = tile_column + load_row;
        int k_col = k_start + load_column;
        int packed_col = k_col / 4;
        uchar bytes[2];
        #pragma clang loop unroll(full)
        for (short j = 0; j < 2; ++j)
            bytes[j] = (w_row < N) ? weight[w_row * packed_in + packed_col + j] : 0;
        #pragma clang loop unroll(full)
        for (short element = 0; element < 8; ++element) {
            uchar byte_val = bytes[element / 4];
            float w_val = float((byte_val >> (2 * (element % 4))) & 0x3) - 2.0f;
            weight_tile[load_row * TILE_STRIDE + load_column + element] =
                static_cast<T>(w_val);
        }"""
    if num_bits == 4:
        return r"""
        int w_row = tile_column + load_row;
        int k_col = k_start + load_column;
        int packed_col = k_col / 2;
        uchar bytes[4];
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; ++j)
            bytes[j] = (w_row < N) ? weight[w_row * packed_in + packed_col + j] : 0;
        #pragma clang loop unroll(full)
        for (short element = 0; element < 8; ++element) {
            uchar byte_val = bytes[element / 2];
            float w_val = float((byte_val >> (4 * (element % 2))) & 0xF) - 8.0f;
            weight_tile[load_row * TILE_STRIDE + load_column + element] =
                static_cast<T>(w_val);
        }"""
    if num_bits == 8:
        return r"""
        int w_row = tile_column + load_row;
        int k_col = k_start + load_column;
        #pragma clang loop unroll(full)
        for (short element = 0; element < 8; ++element) {
            int v = (w_row < N)
                ? int(weight[w_row * packed_in + k_col + element]) : 0;
            weight_tile[load_row * TILE_STRIDE + load_column + element] =
                static_cast<T>(float(v));
        }"""
    raise ValueError(f"Unsupported num_bits {num_bits}")


@lru_cache(maxsize=None)
def _gemma_qmm_kernel(num_bits: int, per_row_output_scale: bool = False):
    if not hasattr(mx, "metal") or not mx.metal.is_available():
        return None
    source = (
        _GEMMA_QMM_SOURCE.replace("UNPACK_TILE_BLOCK", _gemma_qmm_unpack_tile(num_bits))
        .replace(
            "OUTPUT_SCALE_PER_ROW",
            "true" if per_row_output_scale else "false",
        )
    )
    return mx.fast.metal_kernel(
        name=f"mlx_vlm_gemma_mobile_qmm_b{num_bits}_pr{int(per_row_output_scale)}",
        input_names=["x", "weight", "weight_scale", "input_scale", "output_scale"],
        output_names=["out"],
        header=_GEMMA_QMM_HEADER,
        source=source,
    )


# ---------------------------------------------------------------------------
# Mobile → MLX native format conversion (utility, not used in the hot path)
# ---------------------------------------------------------------------------
#
# ``mobile_to_mlx`` converts the mobile per-channel packed format to MLX's
# group_size=128 uint32 format bit-exactly.  It is kept as a utility for
# experimentation; the hot path uses the custom qmv/qmm Metal kernels, which
# fuse SRQ and avoid the memory overhead of the uint32 format.  Benchmarked:
# MLX's native ``quantized_matmul`` is 1.2-3x faster for the matmul itself but
# slower end-to-end due to SRQ overhead + memory pressure (the uint32 format
# + per-group scales/biases add ~328 MB for lm_head, causing cache pressure).

def mobile_to_mlx(
    weight: mx.array,
    weight_scale: mx.array,
    num_bits: int,
    in_features: int,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Convert mobile packed weights to MLX's uint32 quantized format (group_size=128).

    Returns ``(w_mlx, scales, biases)`` for ``mx.quantized_matmul``.  The
    conversion is bit-exact: ``mx.dequantize(w_mlx, scales, biases, ...)`` equals
    ``dequantize_weight(weight, weight_scale, ...)``.  Per-channel (one scale per
    output row) symmetric quantization maps to group_size=128 with the per-channel
    scale broadcast across all groups in a row and a constant bias of
    ``-shift * scale`` (where ``shift`` re-centers the signed int range to
    unsigned).

    Requires ``in_features % 128 == 0`` (MLX's smallest supported group_size).
    Memory-efficient: for int2/int4 the packed bytes already contain the shifted
    unsigned values, so we pack 4 consecutive bytes into one uint32 (little-endian)
    without unpacking — one byte position at a time to keep peak memory low.
    """
    shift = {2: 2, 4: 8, 8: 128}[num_bits]
    n_groups = in_features // 128
    out_dims = weight.shape[0]

    if num_bits in (2, 4):
        # Mobile uint8 bytes already hold the shifted unsigned values (LSB-first
        # within each byte).  MLX packs the same values LSB-first into uint32, so
        # 4 consecutive bytes map directly to 1 uint32 (little-endian).  Process
        # one byte position at a time to avoid materialising a large intermediate.
        w = weight  # [out, packed_in] uint8
        packed = w[..., 0::4].astype(mx.uint32)
        packed = packed + w[..., 1::4].astype(mx.uint32) * mx.array(
            256, dtype=mx.uint32
        )
        packed = packed + w[..., 2::4].astype(mx.uint32) * mx.array(
            65536, dtype=mx.uint32
        )
        packed = packed + w[..., 3::4].astype(mx.uint32) * mx.array(
            16777216, dtype=mx.uint32
        )
    else:  # int8 — shift signed to unsigned, then pack 4 values per uint32.
        q = (weight.astype(mx.int32) + shift).astype(mx.uint32)  # [out, in]
        packed = q[..., 0::4]
        packed = packed + q[..., 1::4] * mx.array(256, dtype=mx.uint32)
        packed = packed + q[..., 2::4] * mx.array(65536, dtype=mx.uint32)
        packed = packed + q[..., 3::4] * mx.array(16777216, dtype=mx.uint32)

    # Per-group scales/biases: broadcast the per-channel scale across all groups.
    ws = weight_scale.astype(mx.float32)
    scales = mx.broadcast_to(ws, (out_dims, n_groups))
    biases = mx.broadcast_to((-shift * ws).astype(mx.float32), (out_dims, n_groups))
    return packed, scales, biases


def gemma_mobile_matmul(
    x: mx.array,
    weight: mx.array,
    weight_scale: mx.array,
    num_bits: int,
    in_features: int,
    input_scale: Optional[mx.array] = None,
    output_scale: Optional[mx.array] = None,
) -> mx.array:
    """Quantized matmul ``x @ dequant(weight).T`` with fused SRQ activations.

    Single-token / small-batch decode of packed int2/int4 weights uses the
    fused ``qmv`` Metal kernel (reads packed uint8 + per-channel scale, never
    materializing the full fp weight). SRQ (static activation quantization) is
    fused into the kernel — input SRQ when reading activations, output SRQ when
    writing the result — eliminating ~2× the kernel launches per layer.

    Batched prefill, int8 weights, and any case where Metal is unavailable or
    the input dimension is not a multiple of 512 (which would cause out-of-bounds
    lane reads) fall back to the pure-MLX dequant + matmul with inline SRQ.
    """
    output_dims = weight.shape[0]
    output_shape = (*x.shape[:-1], output_dims)
    x_2d = x.reshape(-1, in_features)
    batch = x_2d.shape[0]

    # Input SRQ scale is scalar (read as input_scale[0] in the kernel).
    _zero = mx.array([0.0], dtype=x.dtype)
    in_s = input_scale.reshape(1).astype(x.dtype) if input_scale is not None else _zero
    # Output SRQ scale: a 0-d scalar for standalone layers (passed as a 1-element
    # array, read as output_scale[0]) or a 1-d [output_dims] per-row array for
    # fused q/k/v (where q and k/v have different output scales).  The kernel has
    # a compile-time template (OUTPUT_SCALE_PER_ROW) for each case so standalone
    # layers never pay for a per-row read or a broadcast materialization.
    if output_scale is not None:
        out_s_raw = output_scale.astype(x.dtype)
        if out_s_raw.ndim == 0:
            _per_row_out = False
            out_s = out_s_raw.reshape(1)
        else:
            _per_row_out = True
            out_s = out_s_raw
    else:
        _per_row_out = False
        out_s = _zero

    # Prefill (batch > 1) with large output dims: tiled ``qmm`` Metal kernel.
    # The qmm kernel tiles both batch and output, sharing weight reads across
    # batch rows — 2-3x faster than the qmv (GEMV) kernel for large output dims
    # (q_proj, MLP, lm_head).  For small output dims the qmv kernel is faster
    # (less grid overhead), so qmm is only used above _QMM_MIN_OUTPUT_DIMS.
    # Requires in_features % 32 == 0 (BK tile alignment).
    _can_qmm = (
        batch > 1
        and in_features % 32 == 0
        and num_bits in (2, 4, 8)
        and output_dims >= _QMM_MIN_OUTPUT_DIMS
    )
    if _can_qmm:
        kernel = _gemma_qmm_kernel(num_bits, _per_row_out)
        if kernel is not None:
            # grid-x carries batch tiles (×128 dispatch multiplier), grid-y
            # carries output tiles.  MLX's metal_kernel dispatches only a fraction
            # of grid-x threadgroups when grid-x is small (a dispatch quirk also
            # present in one_bit.py's qmm), so we multiply by 128 to ensure full
            # dispatch.  By putting the ×128 on the small batch dimension instead
            # of the large output dimension, grid-x stays ~128 (batch=16) instead
            # of up to ~1M (lm_head), reducing dispatch overhead.  Out-of-bounds
            # threadgroups are bounds-checked and do no work.
            out = kernel(
                inputs=[x_2d, weight, weight_scale, in_s, out_s],
                template=[("T", x.dtype)],
                grid=(
                    (batch + 31) // 32 * 128,
                    (output_dims + 31) // 32,
                    1,
                ),
                threadgroup=(128, 1, 1),
                output_shapes=[(batch * output_dims,)],
                output_dtypes=[x.dtype],
            )[0]
            return out.reshape(output_shape)

    # Decode (batch == 1) and small-batch prefill (batch <= 16) with small
    # output dims: fused ``qmv`` Metal kernel (GEMV, two simdgroups per output
    # row group, sharing activation reads).  int2/int4 need in_features % 512
    # == 0; int8 (per-layer PLE gates/projections) only needs in_features % 16.
    _can_qmv = batch <= 16 and (
        (num_bits in (2, 4) and in_features % 512 == 0)
        or (num_bits == 8 and in_features % 16 == 0)
    )
    if _can_qmv:
        kernel = _gemma_qmv_kernel(num_bits, _per_row_out)
        if kernel is not None:
            out = kernel(
                inputs=[x_2d, weight, weight_scale, in_s, out_s],
                template=[("T", x.dtype)],
                grid=(
                    64,
                    (output_dims + _OUTPUTS_PER_THREADGROUP - 1)
                    // _OUTPUTS_PER_THREADGROUP,
                    batch,
                ),
                threadgroup=(64, 1, 1),
                output_shapes=[(batch * output_dims,)],
                output_dtypes=[x.dtype],
            )[0]
            return out.reshape(output_shape)

    # Pure-MLX fallback (no Metal / unaligned dims): dequant + matmul.
    if input_scale is not None:
        x_2d = apply_srq(x_2d, input_scale)
    w = dequantize_weight(weight, weight_scale, num_bits, in_features, x.dtype)
    out = (x_2d @ w.T).reshape(output_shape)
    if output_scale is not None:
        out = apply_srq(out, output_scale)
    return out


# ---------------------------------------------------------------------------
# Fused q/k/v projection (P4)
# ---------------------------------------------------------------------------

def _qkv_input_scales_match(q, k, v) -> bool:
    """True when the three projections' input SRQ scales are identical.

    Gemma 4 QAT calibrates q/k/v on the same input (the layer-norm output), so
    they share one ``input_activation_scale``.  Fusion is only valid when they
    match; otherwise each projection must quantize x differently and cannot
    share a single fused matmul.
    """
    q_has, k_has, v_has = q._has_input_scale, k._has_input_scale, v._has_input_scale
    if q_has != k_has or q_has != v_has:
        return False
    if not q_has:
        return True
    return bool(
        mx.array_equal(q.input_activation_scale, k.input_activation_scale)
        and mx.array_equal(q.input_activation_scale, v.input_activation_scale)
    )


def _build_per_row_output_scale(q, k, v, dtype: mx.Dtype) -> mx.array:
    """Per-row [q_out + k_out + v_out] output SRQ scale for the fused matmul."""
    qd, kd, vd = q.output_dims, k.output_dims, v.output_dims
    parts = []
    for proj, dim in ((q, qd), (k, kd), (v, vd)):
        if proj._has_output_scale:
            parts.append(mx.broadcast_to(proj.output_activation_scale.astype(dtype), (dim,)))
        else:
            parts.append(mx.zeros((dim,), dtype=dtype))
    return mx.concatenate(parts)


def gemma_fused_qkv_matmul(attn: nn.Module, x: mx.array) -> Optional[mx.array]:
    """Fused q/k/v quantized matmul for Gemma mobile attention layers.

    Concatenates the packed q/k/v weights (lazily, cached on the attention
    module) and runs a **single** ``gemma_mobile_matmul`` with per-row output
    SRQ, replacing three separate Metal kernel launches with one.  The input
    SRQ is shared (q/k/v share one ``input_activation_scale``); the output SRQ is
    per-row so q rows and k/v rows keep their distinct output scales.

    Returns the concatenated ``[..., q_out + k_out + v_out]`` output (the caller
    splits it), or ``None`` when fusion is not possible (projections are not all
    gemma-quantized, or input SRQ scales differ) so the caller can fall back to
    three separate projections.
    """
    cache = getattr(attn, "_fused_qkv_cache", None)
    if cache is None:
        q = getattr(attn, "q_proj", None)
        k = getattr(attn, "k_proj", None)
        v = getattr(attn, "v_proj", None)
        if (
            q is None or k is None or v is None
            or not all(getattr(p, "mode", None) == "gemma" for p in (q, k, v))
        ):
            attn._fused_qkv_cache = False
            return None
        if not _qkv_input_scales_match(q, k, v):
            attn._fused_qkv_cache = False
            return None
        weight = mx.concatenate([q.weight, k.weight, v.weight], axis=0)
        weight_scale = mx.concatenate(
            [q.weight_scale, k.weight_scale, v.weight_scale], axis=0
        )
        in_s = q.input_activation_scale if q._has_input_scale else None
        out_s = _build_per_row_output_scale(q, k, v, x.dtype)
        cache = (weight, weight_scale, q.num_bits, q.input_dims, in_s, out_s)
        attn._fused_qkv_cache = cache

    if cache is False:
        return None

    weight, weight_scale, num_bits, in_features, in_s, out_s = cache
    return gemma_mobile_matmul(
        x, weight, weight_scale, num_bits, in_features,
        input_scale=in_s, output_scale=out_s,
    )


def precompile_gemma_mobile_kernels(dtype: mx.Dtype = mx.bfloat16) -> None:
    """JIT-compile all custom Metal kernel variants up front (at load time).

    Gemma mobile uses 12 custom Metal kernels (qmv × 6 + qmm × 6).  Each is
    JIT-compiled by MLX on first use (~35–80 ms per variant, ~0.5 s total).
    Without precompilation the *first* prompt pays this cost, showing up as low
    prompt tok/s.  Calling this once during model load moves the compilation into
    the (already slow) load phase so the first prompt runs at steady-state speed.

    Only the dtype template and input dtypes matter for compilation; grid/shapes
    are runtime parameters, so tiny dummy tensors suffice.
    """
    if not hasattr(mx, "metal") or not mx.metal.is_available():
        return

    _zero = mx.array([0.0], dtype=dtype)

    # qmv (decode / small-batch GEMV) — batch=1, one output threadgroup.
    for num_bits in (2, 4, 8):
        in_features = 512 if num_bits in (2, 4) else 16
        packed_in = in_features // {2: 4, 4: 2, 8: 1}[num_bits]
        w_dtype = mx.uint8 if num_bits in (2, 4) else mx.int8
        out_dims = _OUTPUTS_PER_THREADGROUP  # 8
        x = mx.zeros((1, in_features), dtype=dtype)
        w = mx.zeros((out_dims, packed_in), dtype=w_dtype)
        ws = mx.ones((out_dims, 1), dtype=dtype)
        for per_row in (False, True):
            kernel = _gemma_qmv_kernel(num_bits, per_row)
            if kernel is None:
                continue
            out_s = mx.zeros((out_dims,), dtype=dtype) if per_row else _zero
            _ = kernel(
                inputs=[x, w, ws, _zero, out_s],
                template=[("T", dtype)],
                grid=(64, 1, 1),
                threadgroup=(64, 1, 1),
                output_shapes=[(out_dims,)],
                output_dtypes=[dtype],
            )[0]
            mx.eval(_)

    # qmm (batched prefill GEMM) — one 32×32 output tile, one batch tile.
    for num_bits in (2, 4, 8):
        in_features = 32  # BK tile alignment
        packed_in = in_features // {2: 4, 4: 2, 8: 1}[num_bits]
        w_dtype = mx.uint8 if num_bits in (2, 4) else mx.int8
        out_dims = 32  # one output tile
        batch = 32  # one batch tile
        x = mx.zeros((batch, in_features), dtype=dtype)
        w = mx.zeros((out_dims, packed_in), dtype=w_dtype)
        ws = mx.ones((out_dims, 1), dtype=dtype)
        for per_row in (False, True):
            kernel = _gemma_qmm_kernel(num_bits, per_row)
            if kernel is None:
                continue
            out_s = mx.zeros((out_dims,), dtype=dtype) if per_row else _zero
            _ = kernel(
                inputs=[x, w, ws, _zero, out_s],
                template=[("T", dtype)],
                grid=((batch + 31) // 32 * 128, (out_dims + 31) // 32, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(batch * out_dims,)],
                output_dtypes=[dtype],
            )[0]
            mx.eval(_)


# ---------------------------------------------------------------------------
# Quantized layers
# ---------------------------------------------------------------------------

class GemmaQuantizedLinear(nn.Module):
    """Linear with packed int2/4/8 per-channel weights and SRQ activations."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_bits: int,
        bias: bool = False,
        dtype: mx.Dtype = mx.float32,
        input_scale: bool = False,
        output_scale: bool = False,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bits = num_bits
        self.mode = "gemma"

        if num_bits == 2:
            packed_in = (input_dims + 3) // 4
            w_dtype = mx.uint8
        elif num_bits == 4:
            packed_in = (input_dims + 1) // 2
            w_dtype = mx.uint8
        elif num_bits == 8:
            packed_in = input_dims
            w_dtype = mx.int8
        else:
            raise ValueError(f"Unsupported num_bits {num_bits}; expected 2, 4, or 8.")

        self.weight = mx.zeros((output_dims, packed_in), dtype=w_dtype)
        self.weight_scale = mx.ones((output_dims, 1), dtype=dtype)

        self._has_input_scale = input_scale
        self._has_output_scale = output_scale
        if input_scale:
            self.input_activation_scale = mx.zeros((), dtype=dtype)
        if output_scale:
            self.output_activation_scale = mx.zeros((), dtype=dtype)

        if bias:
            self.bias = mx.zeros((output_dims,), dtype=dtype)
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        # SRQ scales are fused into the matmul (Metal kernel or inline pure-MLX)
        # to avoid ~8 separate element-wise kernel launches per layer.
        in_s = self.input_activation_scale if self._has_input_scale else None
        out_s = self.output_activation_scale if self._has_output_scale else None

        out = gemma_mobile_matmul(
            x,
            self.weight,
            self.weight_scale,
            self.num_bits,
            self.input_dims,
            input_scale=in_s,
            output_scale=out_s,
        )
        if "bias" in self:
            out = out + self.bias
        return out

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"bias={'bias' in self}, num_bits={self.num_bits}"
        )


class MmapEmbeddingLookup:
    """On-disk (CPU-mmap) lookup for a large packed embedding table.

    The packed ``embedding_quantized`` table is held as a read-only ``mmap`` view
    of the safetensors file (demand-paged by the OS).  Per-token row gathers page
    in only the needed rows, so the full table never enters MLX unified memory.
    This mirrors LiteRT-LM's ``EmbeddingLookupText``, which places large
    embedding tables on the CPU because they "may use too much memory on the
    accelerator".

    Only the small gathered row block is transferred to MLX; the dequantization
    (unpack + scale) is done on the MLX side via
    ``GemmaQuantizedEmbedding._dequant_rows``.
    """

    def __init__(self, safetensors_path, quant_key, num_bits, embedding_dim):
        import json as _json
        import mmap as _mmap
        import struct as _struct

        self.safetensors_path = safetensors_path
        self.quant_key = quant_key
        self.num_bits = num_bits
        self.embedding_dim = embedding_dim

        self._f = open(safetensors_path, "rb")
        hlen = _struct.unpack("<Q", self._f.read(8))[0]
        header = _json.loads(self._f.read(hlen))
        data_start = 8 + hlen

        if quant_key not in header:
            self._f.close()
            raise KeyError(
                f"Tensor {quant_key!r} not found in {safetensors_path}"
            )
        info = header[quant_key]
        off0, _ = info["data_offsets"]
        shape = info["shape"]
        dtype_str = info["dtype"]

        # U8 -> numpy uint8 (int2/int4 packed), I8 -> numpy int8 (int8).
        np_dtype = np.uint8 if dtype_str == "U8" else np.int8
        self._mm = _mmap.mmap(self._f.fileno(), 0, access=_mmap.ACCESS_READ)
        count = int(shape[0]) * int(shape[1])
        self._quant = np.frombuffer(
            self._mm, dtype=np_dtype, count=count, offset=data_start + off0
        ).reshape(shape)

    def gather(self, ids_np):
        """Return the packed rows (uint8/int8 numpy) for the given token ids."""
        return self._quant[ids_np]

    def close(self):
        # Release the numpy view before closing the mmap (otherwise the view's
        # exported pointer prevents ``mmap.close``).
        self._quant = None
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass


class GemmaQuantizedEmbedding(nn.Module):
    """Embedding with packed int2/4/8 table and per-row (or block-wise) scale.

    The architectural ``embed_scale`` is applied by the surrounding model (as in
    mlx-vlm's Gemma 4), so this layer returns the *unscaled* dequantized rows.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int,
        embed_scale: float = 1.0,
        dtype: mx.Dtype = mx.float32,
        num_blocks: int = 1,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.scalar_embed_scale = embed_scale
        self.output_dtype = dtype
        self.bits = num_bits
        self.mode = "gemma"

        if num_bits == 2:
            packed_dim = (embedding_dim + 3) // 4
            w_dtype = mx.uint8
        elif num_bits == 4:
            packed_dim = (embedding_dim + 1) // 2
            w_dtype = mx.uint8
        elif num_bits == 8:
            packed_dim = embedding_dim
            w_dtype = mx.int8
        else:
            raise ValueError(f"Unsupported num_bits {num_bits}; expected 2, 4, or 8.")

        self.embedding_quantized = mx.zeros(
            (num_embeddings, packed_dim), dtype=w_dtype
        )
        # Per-row scale uses shape (num_emb, 1); block-wise uses (num_emb, n_blocks)
        # where block_size = embedding_dim // n_blocks.  The placeholder must match
        # the checkpoint's scale shape for strict weight loading.
        self.embedding_scale = mx.ones((num_embeddings, num_blocks), dtype=dtype)
        self.freeze()

    def _dequant_rows(self, quant_rows: mx.array, scale_rows: mx.array) -> mx.array:
        ints = unpack_int(quant_rows, self.num_bits, self.embedding_dim)
        out_dtype = self.output_dtype
        if scale_rows.shape[-1] == 1:
            return ints.astype(out_dtype) * scale_rows.astype(out_dtype)
        # Block-wise scale: reshape ints to (..., num_blocks, block_size) and
        # broadcast-multiply by the per-block scale (..., num_blocks, 1), then
        # flatten the last two axes back to (..., embedding_dim).
        num_blocks = scale_rows.shape[-1]
        block_size = self.embedding_dim // num_blocks
        ints = ints.reshape(*ints.shape[:-1], num_blocks, block_size)
        scaled = ints.astype(out_dtype) * scale_rows[..., None].astype(out_dtype)
        return scaled.reshape(scaled.shape[:-2] + (-1,))

    def offload_to_mmap(self, safetensors_path: str, quant_key: str) -> None:
        """Offload the packed embedding table to an on-disk mmap lookup.

        Replaces the (possibly lazy) ``embedding_quantized`` parameter with a
        tiny dummy and stores an :class:`MmapEmbeddingLookup` so subsequent
        ``__call__`` invocations gather rows from disk instead of from unified
        memory.  The per-row ``embedding_scale`` (small) stays resident for the
        dequant multiply.

        Must be called AFTER ``load_weights`` (so the table shape/dtype is known)
        and BEFORE ``mx.eval(model.parameters())`` (so the full table is never
        materialized into unified memory).
        """
        lookup = MmapEmbeddingLookup(
            safetensors_path, quant_key, self.num_bits, self.embedding_dim
        )
        object.__setattr__(self, "_mmap_lookup", lookup)
        # Use dict setitem (not object.__setattr__) so the old array is dropped
        # from the module's parameter dict and actually freed; ``parameters()``
        # then sees the 1-element dummy and never materializes the full table.
        w_dtype = self["embedding_quantized"].dtype
        self["embedding_quantized"] = mx.zeros((1,), dtype=w_dtype)

    @property
    def weight(self) -> mx.array:
        """Full dequantized table (without architectural embed_scale)."""
        if getattr(self, "_mmap_lookup", None) is not None:
            # Rare path (tied lm_head / input_embeddings): materialize the full
            # table on demand from the mmap.  This pages the whole table into
            # memory and defeats the offload, but standard generation (untied
            # lm_head, as in Gemma 4 QAT mobile) never calls this.
            all_ids = mx.arange(self.num_embeddings, dtype=mx.int32)
            return self.__call__(all_ids)
        return self._dequant_rows(self.embedding_quantized, self.embedding_scale)

    def __call__(self, input_ids: mx.array) -> mx.array:
        lookup = getattr(self, "_mmap_lookup", None)
        if lookup is not None:
            # Gather packed rows from the on-disk mmap (CPU), transfer only the
            # small row block to MLX, then dequantize on-device.  The full table
            # stays paged out on disk.
            ids_np = np.asarray(input_ids)
            quant_rows = mx.array(lookup.gather(ids_np))
            scales = self.embedding_scale[input_ids]
            return self._dequant_rows(quant_rows, scales)
        rows = self.embedding_quantized[input_ids]
        scales = self.embedding_scale[input_ids]
        return self._dequant_rows(rows, scales)

    def as_linear(self, x: mx.array) -> mx.array:
        """Matmul against the dequantized table (tied-lm-head / logits path)."""
        return x @ self.weight.T

    def _extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, num_bits={self.num_bits}"
        )


# ---------------------------------------------------------------------------
# Per-module bit resolution (mirrors GemmaQuantizationConfig + replace_with_quant_layers)
# ---------------------------------------------------------------------------

def is_gemma_quant_config(quantization_config: Optional[Dict[str, Any]]) -> bool:
    return (
        quantization_config is not None
        and quantization_config.get("quant_method") == "gemma"
    )


def _normalize_module_path(path: str) -> str:
    """Map an mlx-vlm leaf-module path to the HF Gemma4 namespace used by
    ``module_quant_configs`` regexes.

    mlx-vlm wraps the transformer under ``LanguageModel.model`` (HF keeps these
    directly under ``language_model``), and ``lm_head`` lives under
    ``language_model`` in mlx-vlm text models but at top level in HF.
    """
    if path.startswith("language_model.model."):
        path = "language_model." + path[len("language_model.model.") :]
    if path == "language_model.lm_head":
        path = "lm_head"
    return path


def _path_contains_segment(path: str, segment: str) -> bool:
    """True if ``segment`` is a whole path segment of ``path``."""
    return (
        path == segment
        or path.startswith(segment + ".")
        or path.endswith("." + segment)
        or ("." + segment + ".") in path
    )


def _compile_module_quant_configs(
    module_quant_configs: Optional[Dict[str, Any]]
) -> List[Tuple[re.Pattern, int]]:
    items: List[Tuple[re.Pattern, int]] = []
    for pattern, opts in (module_quant_configs or {}).items():
        bits = opts.get("num_bits") if isinstance(opts, dict) else opts
        if bits is None:
            continue
        items.append((re.compile(pattern), int(bits)))
    return items


def resolve_module_bits(
    path: str, quantization_config: Dict[str, Any]
) -> Optional[int]:
    """Resolve the bit width for a leaf module path, or ``None`` to skip it.

    ``None`` means the module is in ``modules_to_not_convert``. Unmatched modules
    fall back to the config's default ``num_bits`` (4 for the mobile schema).
    """
    normalized = _normalize_module_path(path)

    for entry in quantization_config.get("modules_to_not_convert") or []:
        entry = entry[len("model.") :] if entry.startswith("model.") else entry
        if entry and _path_contains_segment(normalized, entry):
            return None

    for regex, bits in _compile_module_quant_configs(
        quantization_config.get("module_quant_configs")
    ):
        if regex.search(normalized):
            return bits

    return int(quantization_config.get("num_bits", 4))


# ---------------------------------------------------------------------------
# Module replacement (mirrors one_bit.replace_one_bit_modules)
# ---------------------------------------------------------------------------

def _strip_clippable_linear_clips(model: nn.Module) -> None:
    """Remove clipping parameters from ``ClippableLinear`` wrappers whose
    inner linear has been replaced with a ``GemmaQuantizedLinear``.

    Mobile (wNa8o8) checkpoints use SRQ (static activation quantization), not
    the input/output clamping of ``ClippableLinear``.  The clip params
    (``input_min`` / ``input_max`` / ``output_min`` / ``output_max``) are
    created unconditionally by ``ClippableLinear.__init__`` (default
    ``use_clipping=True``) but are absent from mobile checkpoints, so they must
    be removed for ``strict=True`` weight loading to succeed.

    Uses duck typing (``use_clipping`` attribute + ``linear`` child) to avoid
    importing the model module and creating a circular dependency.
    """
    for _path, module in model.named_modules():
        if not hasattr(module, "use_clipping") or not hasattr(module, "linear"):
            continue
        if not isinstance(getattr(module, "linear"), GemmaQuantizedLinear):
            continue
        if getattr(module, "use_clipping", False):
            module.use_clipping = False
            for attr in ("input_min", "input_max", "output_min", "output_max"):
                if hasattr(module, attr):
                    delattr(module, attr)


def replace_with_gemma_quant_layers(
    model: nn.Module,
    quantization_config: Dict[str, Any],
    weights: Optional[Dict[str, mx.array]] = None,
    dtype: mx.Dtype = mx.float16,
) -> nn.Module:
    """Replace ``nn.Linear``/``nn.Embedding`` leaves with their Gemma quantized
    counterparts, driven by ``quantization_config.module_quant_configs``.

    A layer is replaced only when (a) it is not in ``modules_to_not_convert``,
    and (b) the checkpoint actually carries a packed ``<path>.weight_scale`` for
    it (so unquantized / skipped modules stay as their original fp layer).
    """
    quantize_embeddings = quantization_config.get("quantize_embeddings", False)

    def replace(path: str, module: nn.Module) -> nn.Module:
        bits = resolve_module_bits(path, quantization_config)
        if bits is None:
            return module

        if isinstance(module, nn.Linear):
            if weights is not None and f"{path}.weight_scale" not in weights:
                return module  # not quantized in this checkpoint
            has_in_scale = weights is not None and f"{path}.input_activation_scale" in weights
            has_out_scale = weights is not None and f"{path}.output_activation_scale" in weights
            out_dims, in_dims = module.weight.shape
            new = GemmaQuantizedLinear(
                in_dims,
                out_dims,
                bits,
                bias="bias" in module,
                dtype=dtype,
                input_scale=has_in_scale,
                output_scale=has_out_scale,
            )
            if "bias" in module:
                new.bias = module.bias
            return new

        if isinstance(module, nn.Embedding):
            if not quantize_embeddings:
                return module
            scale_key = f"{path}.embedding_scale"
            if weights is not None and scale_key not in weights:
                return module  # not quantized in this checkpoint
            num_emb, dim = module.weight.shape
            # Determine block-wise vs per-row scale from the checkpoint shape.
            num_blocks = 1
            if weights is not None:
                scale_shape = weights[scale_key].shape
                if len(scale_shape) >= 2 and scale_shape[-1] != 1:
                    num_blocks = scale_shape[-1]
            return GemmaQuantizedEmbedding(
                num_emb, dim, bits, embed_scale=1.0, dtype=dtype, num_blocks=num_blocks
            )

        return module

    leaves = tree_map_with_path(
        replace, model.leaf_modules(), is_leaf=nn.Module.is_module
    )
    model.update_modules(leaves)

    # Mobile checkpoints use SRQ, not ClippableLinear clipping.  Remove the
    # clip params from any ClippableLinear whose inner linear was just
    # replaced with a GemmaQuantizedLinear so strict weight loading succeeds.
    _strip_clippable_linear_clips(model)

    return model


def offload_gemma_embeddings(
    model: nn.Module,
    weight_files,
    min_size_bytes: int = 8 * 1024 * 1024,
) -> int:
    """Offload large ``GemmaQuantizedEmbedding`` tables to on-disk mmap lookups.

    For every ``GemmaQuantizedEmbedding`` whose packed table exceeds
    ``min_size_bytes``, replace the in-memory ``embedding_quantized`` with an
    :class:`MmapEmbeddingLookup` backed by the safetensors file.  The full table
    then never enters MLX unified memory; per-token row gathers page in only the
    needed rows (mirrors LiteRT-LM's ``EmbeddingLookupText``).

    Must be called AFTER ``model.load_weights`` (so table shapes are known) and
    BEFORE ``mx.eval(model.parameters())`` (so the full table is never
    materialized).

    Returns the number of embedding tables offloaded.
    """
    import json as _json
    import os as _os
    import struct as _struct

    if _os.environ.get("MLX_VLM_NO_EMBED_OFFLOAD") or not weight_files:
        return 0

    # Map every "*.embedding_quantized" safetensors key to its file + shape.
    emb_keys = {}  # key -> (file_path, shape)
    for wf in weight_files:
        try:
            with open(wf, "rb") as f:
                hlen = _struct.unpack("<Q", f.read(8))[0]
                header = _json.loads(f.read(hlen))
        except (OSError, ValueError):
            continue
        for k, info in header.items():
            if k.endswith(".embedding_quantized"):
                emb_keys[k] = (str(wf), tuple(info["shape"]))

    if not emb_keys:
        return 0

    quantized_model = (
        model.language_model._model
        if getattr(model, "_is_text_model", False)
        else model
    )

    n_offloaded = 0
    for path, module in quantized_model.named_modules():
        if not isinstance(module, GemmaQuantizedEmbedding):
            continue
        quant = module["embedding_quantized"]
        shape = tuple(quant.shape)
        size = int(shape[0]) * int(shape[1]) if len(shape) == 2 else 0
        if size < min_size_bytes:
            continue

        quant_key = f"{path}.embedding_quantized"
        info = emb_keys.get(quant_key)
        if info is None:
            # Fallback: match by shape (unique for Gemma 4's two big tables).
            matches = [k for k, v in emb_keys.items() if v[1] == shape]
            if len(matches) != 1:
                continue
            quant_key = matches[0]

        file_path = emb_keys[quant_key][0]
        module.offload_to_mmap(file_path, quant_key)
        n_offloaded += 1

    return n_offloaded


# ---------------------------------------------------------------------------
# Helpers for conversion / introspection
# ---------------------------------------------------------------------------

def gemma_quant_layer_paths(
    model: nn.Module, quantization_config: Dict[str, Any], weights: Dict[str, mx.array]
) -> Dict[str, int]:
    """Return ``{path: num_bits}`` for every leaf that would be gemma-quantized."""
    out: Dict[str, int] = {}
    quantize_embeddings = quantization_config.get("quantize_embeddings", False)
    for path, module in model.leaf_modules().items():
        bits = resolve_module_bits(path, quantization_config)
        if bits is None:
            continue
        if isinstance(module, nn.Linear):
            if f"{path}.weight_scale" not in weights:
                continue
        elif isinstance(module, nn.Embedding):
            if not quantize_embeddings or f"{path}.embedding_scale" not in weights:
                continue
        else:
            continue
        out[path] = bits
    return out
