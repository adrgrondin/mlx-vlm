"""Conversion / quantization for the Gemma 4 QAT mobile (wNa8o8) format.

Two paths:

* **Transcode** (lossless): remap a HuggingFace ``-mobile-transformers``
  checkpoint into the mlx-vlm key layout, preserving the packed int2/4/8
  weights bit-exact. This is the recommended path — the QAT checkpoints are
  already trained for this format, so there is no quantization error.
* **PTQ** (post-training quantization): quantize an *unquantized* Gemma 4
  checkpoint into the mobile format with per-channel symmetric uniform weights
  and the E2B/E4B module bit assignment. Lower quality than QAT; useful for
  custom fine-tunes.

The packed storage layout matches
:mod:`mlx_vlm.quantization.gemma_mobile` (and the HF reference
``gemma_quant.py``): int2/int4 in uint8 (4 / 2 per byte, LSB-first), int8
directly, with a per-output-channel float scale.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .gemma_mobile import resolve_module_bits

# ---------------------------------------------------------------------------
# Packing (inverse of gemma_mobile.unpack_int2 / unpack_int4)
# ---------------------------------------------------------------------------

def pack_int2_row(ints: mx.array) -> mx.array:
    """Pack signed int2 values ([-2, 1]) into uint8, 4/byte, LSB-first.

    ``ints`` is [..., in] int8; returns [..., ceil(in/4)] uint8.
    """
    if ints.dtype != mx.int8:
        ints = ints.astype(mx.int8)
    u = (ints + 2).astype(mx.uint8)
    # pad the last byte to a multiple of 4
    pad = (-u.shape[-1]) % 4
    if pad:
        u = mx.pad(u, [(0, 0)] * (u.ndim - 1) + [(0, pad)])
    u = u.reshape(*u.shape[:-1], -1, 4)
    packed = (u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)).astype(
        mx.uint8
    )
    return packed


def pack_int4_row(ints: mx.array) -> mx.array:
    """Pack signed int4 values ([-8, 7]) into uint8, 2/byte, low-nibble-first."""
    if ints.dtype != mx.int8:
        ints = ints.astype(mx.int8)
    u = (ints + 8).astype(mx.uint8)
    pad = (-u.shape[-1]) % 2
    if pad:
        u = mx.pad(u, [(0, 0)] * (u.ndim - 1) + [(0, pad)])
    u = u.reshape(*u.shape[:-1], -1, 2)
    packed = ((u[..., 0] & 0x0F) | ((u[..., 1] & 0x0F) << 4)).astype(mx.uint8)
    return packed


def pack_int_row(ints: mx.array, bits: int) -> mx.array:
    if bits == 2:
        return pack_int2_row(ints)
    if bits == 4:
        return pack_int4_row(ints)
    if bits == 8:
        return ints.astype(mx.int8)
    raise ValueError(f"Unsupported num_bits {bits}; expected 2, 4, or 8.")


# ---------------------------------------------------------------------------
# PTQ: per-channel symmetric uniform quantization
# ---------------------------------------------------------------------------

def _symmetric_scale(weight: mx.array, bits: int) -> mx.array:
    """Per-output-channel symmetric scale: max_abs / (2**(b-1) - 1).

    For int2 the divisor is 1 so the levels {-2,-1,0,1} are all representable
    (the codebook is asymmetric around 0 by design).
    """
    denom = 1.0 if bits == 2 else float(2 ** (bits - 1) - 1)
    max_abs = mx.max(mx.abs(weight), axis=-1, keepdims=True)
    return max_abs / denom


def quantize_per_channel_sym(
    weight: mx.array, bits: int
) -> Tuple[mx.array, mx.array]:
    """Quantize a 2D weight [out, in] to packed int + per-channel scale.

    Returns ``(packed_weight, weight_scale)`` where ``weight_scale`` is
    ``[out, 1]`` float32 and ``packed_weight`` is uint8 (int2/int4) or int8.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got shape {weight.shape}.")
    scale = _symmetric_scale(weight, bits).astype(mx.float32)
    lo = -(2 ** (bits - 1))
    hi = 2 ** (bits - 1) - 1
    q = mx.clip(mx.round(weight / scale), lo, hi).astype(mx.int8)
    return pack_int_row(q, bits), scale


def quantize_embedding_per_row(
    weight: mx.array, bits: int
) -> Tuple[mx.array, mx.array]:
    """Quantize an embedding table [num_emb, dim] to packed int + per-row scale."""
    return quantize_per_channel_sym(weight, bits)


# ---------------------------------------------------------------------------
# Module bit assignment builders (E2B / E4B)
# ---------------------------------------------------------------------------

def build_e2b_mobile_quantization_config() -> Dict[str, Any]:
    """The ``quantization_config`` for Gemma 4 E2B mobile (35 layers).

    Mirrors ``google/gemma-4-E2B-it-qat-mobile-transformers``.
    """
    return {
        "quant_method": "gemma",
        "num_bits": 4,
        "quantize_embeddings": True,
        "module_quant_configs": {
            r"^lm_head$": {"num_bits": 2},
            r"language_model\.embed_tokens$": {"num_bits": 2},
            r"language_model\.embed_tokens_per_layer$": {"num_bits": 4},
            r"language_model\.layers\.(\d|1[0-4])\.mlp\.": {"num_bits": 4},
            r"language_model\.layers\.\d+\.mlp\.": {"num_bits": 2},
            r"language_model\.layers\.\d+\.per_layer_input_gate$": {"num_bits": 8},
            r"language_model\.layers\.\d+\.per_layer_projection$": {"num_bits": 8},
            r"language_model\.layers\.\d+\.self_attn\.": {"num_bits": 4},
            r"vision_tower": {"num_bits": 8},
            r"audio_tower(?!.*lconv1d\.linear_start)": {"num_bits": 2},
            r"audio_tower\.layers\.\d+\.lconv1d\.linear_start\.": {"num_bits": 4},
        },
        "modules_to_not_convert": [
            "model.vision_tower.patch_embedder",
            "model.audio_tower.subsample_conv_projection",
            "model.audio_tower.output_proj",
            "relative_k_proj",
            "model.embed_audio",
            "model.embed_vision",
            "per_layer_model_projection",
        ],
    }


def build_e4b_mobile_quantization_config() -> Dict[str, Any]:
    """The ``quantization_config`` for Gemma 4 E4B mobile (42 layers).

    Mirrors ``google/gemma-4-E4B-it-qat-mobile-transformers``. Unlike E2B, E4B
    keeps **all** MLP layers at 4-bit (no targeted 2-bit decode-only MLPs) and
    quantizes ``embed_tokens_per_layer`` to 2-bit.
    """
    cfg = build_e2b_mobile_quantization_config()
    # Start from E2B but override the two differing rules.
    mqc = {
        k: v
        for k, v in cfg["module_quant_configs"].items()
        if "mlp" not in k and "embed_tokens_per_layer" not in k
    }
    mqc[r"language_model\.embed_tokens_per_layer$"] = {"num_bits": 2}
    mqc[r"language_model\.layers\.\d+\.mlp\."] = {"num_bits": 4}
    cfg["module_quant_configs"] = mqc
    return cfg


# ---------------------------------------------------------------------------
# Transcode: HF mobile-transformers -> mlx-vlm layout (lossless)
# ---------------------------------------------------------------------------

# Keys whose presence means "this is a Gemma mobile checkpoint".
_GEMMA_MOBILE_SUFFIXES = (".weight_scale", ".embedding_scale", ".input_activation_scale")

# Tensors stored in the checkpoint but not consumed by the model (KV-cache
# quantization scales, used by the quantized cache in Phase 4).
_DROP_SUFFIXES = ("k_cache_scale", "v_cache_scale")

# Multimodal prefixes to drop when extracting a text-only checkpoint.
_MULTIMODAL_PREFIXES = (
    "model.vision_tower.",
    "model.audio_tower.",
    "model.embed_vision.",
    "model.embed_audio.",
)


def transcode_gemma_mobile_weights(
    weights: Dict[str, mx.array],
    text_only: bool = True,
) -> Dict[str, mx.array]:
    """Remap HF ``-mobile-transformers`` keys to the mlx-vlm layout, losslessly.

    HF layout (``Gemma4ForCausalLM``): ``model.language_model.<...>``,
    ``lm_head.<...>``, ``model.vision_tower.<...>`` ...

    mlx-vlm text layout (``gemma4_text``): ``language_model.model.<...>``,
    ``language_model.lm_head.<...>``.

    KV-cache quantization scales (``k_cache_scale`` / ``v_cache_scale``) are
    dropped — they are consumed by the quantized cache, not the model. When
    ``text_only`` is set, vision/audio/embed_* tensors are dropped too.
    """
    out: Dict[str, mx.array] = {}
    for k, v in weights.items():
        if any(s in k for s in _DROP_SUFFIXES):
            continue
        if text_only and k.startswith(_MULTIMODAL_PREFIXES):
            continue

        if k.startswith("model.language_model."):
            rest = k[len("model.language_model."):]
            new_key = f"language_model.model.{rest}"
        elif k.startswith("lm_head."):
            new_key = f"language_model.{k}"
        elif k.startswith("model."):
            # vision_tower / audio_tower / embed_* (multimodal, non-text-only)
            new_key = k[len("model."):]
        else:
            new_key = k
        out[new_key] = v
    return out


def is_gemma_mobile_checkpoint(weights: Dict[str, mx.array]) -> bool:
    """Heuristic: does this weight dict look like a Gemma mobile checkpoint?"""
    return any(k.endswith(_GEMMA_MOBILE_SUFFIXES) for k in weights)


# ---------------------------------------------------------------------------
# PTQ: walk a model and quantize its leaves into the mobile format
# ---------------------------------------------------------------------------

def quantize_model_gemma_mobile(
    model: nn.Module,
    quantization_config: Dict[str, Any],
    weights: Optional[Dict[str, mx.array]] = None,
    dtype: mx.Dtype = mx.float32,
) -> Tuple[Dict[str, mx.array], Dict[str, int]]:
    """Quantize an unquantized model's leaves into the Gemma mobile format.

    Returns ``(new_weights, path_to_bits)``. ``new_weights`` contains packed
    int weights + scales for quantized leaves and the original (fp) weights for
    skipped / unquantized leaves. ``path_to_bits`` maps each quantized leaf path
    to its bit width (for building the saved ``quantization`` config).
    """
    leaves = tree_flatten(model.leaf_modules(), is_leaf=lambda x: isinstance(x, nn.Module))
    quantize_embeddings = quantization_config.get("quantize_embeddings", False)
    new_weights: Dict[str, mx.array] = {}
    path_to_bits: Dict[str, int] = {}

    for path, mod in leaves:
        bits = resolve_module_bits(path, quantization_config)
        if isinstance(mod, nn.Linear):
            if bits is None:
                new_weights[f"{path}.weight"] = mod.weight.astype(dtype)
                continue
            packed, scale = quantize_per_channel_sym(mod.weight.astype(mx.float32), bits)
            new_weights[f"{path}.weight"] = packed
            new_weights[f"{path}.weight_scale"] = scale
            path_to_bits[path] = bits
        elif isinstance(mod, nn.Embedding):
            if bits is None or not quantize_embeddings:
                new_weights[f"{path}.weight"] = mod.weight.astype(dtype)
                continue
            packed, scale = quantize_embedding_per_row(mod.weight.astype(mx.float32), bits)
            new_weights[f"{path}.embedding_quantized"] = packed
            new_weights[f"{path}.embedding_scale"] = scale
            path_to_bits[path] = bits
        # other leaves (norms, etc.) are handled by the caller via model.parameters()

    # Carry over non-leaf parameters (norms, layer_scalar, per_layer_model_projection, ...).
    for path, val in tree_flatten(model.parameters()):
        if path in new_weights or any(path.startswith(qp + ".") for qp in path_to_bits):
            continue
        if path.endswith(".weight"):
            mp = path[: -len(".weight")]
            if mp in path_to_bits:
                continue  # already replaced by packed storage
        new_weights[path] = val.astype(dtype) if val.dtype in (mx.float32, mx.bfloat16) else val

    return new_weights, path_to_bits


# ---------------------------------------------------------------------------
# Conversion helpers (transcode config promotion + weight casting)
# ---------------------------------------------------------------------------

# Multimodal-only config keys dropped when extracting a text-only checkpoint.
_MULTIMODAL_CONFIG_KEYS = (
    "vision_config",
    "audio_config",
    "image_token_id",
    "audio_token_id",
    "video_token_id",
    "boi_token_id",
    "boa_token_id",
    "eoi_token_id",
    "eoa_token_id",
    "eoa_token_index",
    "vision_soft_tokens_per_image",
    "vision_soft_tokens_per_video_frame",
    "audio_soft_tokens_per_image",
    "audio_ms_per_token",
)


def promote_text_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a ``gemma4`` (multimodal) config into a ``gemma4_text`` one.

    Promotes ``text_config`` to the top level, carries over ``quantization_config``
    and generation-relevant fields, and drops vision/audio config + multimodal
    token ids. Used by the text-only transcode path.
    """
    text_config = dict(config.get("text_config", {}))
    # Mobile checkpoints store quantization_config at the top level.
    if "quantization_config" in config:
        text_config["quantization_config"] = config["quantization_config"]
    for k in ("tie_word_embeddings", "eos_token_id", "bos_token_id", "pad_token_id"):
        if k in config and k not in text_config:
            text_config[k] = config[k]
    for k in _MULTIMODAL_CONFIG_KEYS:
        text_config.pop(k, None)
    return text_config


def cast_gemma_mobile_weights(
    weights: Dict[str, mx.array], dtype: mx.Dtype = mx.float16
) -> Dict[str, mx.array]:
    """Cast floating-point tensors to ``dtype``; keep packed int weights bit-exact.

    The packed int2/int4/int8 weights (``uint8``/``int8``) are preserved
    losslessly. Scales and unquantized fp parameters are narrowed to ``dtype``
    (default ``float16``) to hit the resident-memory target.
    """
    out: Dict[str, mx.array] = {}
    for k, v in weights.items():
        if v.dtype in (mx.uint8, mx.int8):
            out[k] = v
        elif mx.issubdtype(v.dtype, mx.floating):
            out[k] = v.astype(dtype)
        else:
            out[k] = v
    return out


def select_mobile_quant_config(num_layers: int) -> Dict[str, Any]:
    """Pick the E2B or E4B mobile ``quantization_config`` for PTQ by layer count."""
    if num_layers <= 35:
        return build_e2b_mobile_quantization_config()
    return build_e4b_mobile_quantization_config()
