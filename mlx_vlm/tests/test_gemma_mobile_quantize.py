"""Tests for the Gemma 4 QAT Mobile conversion / quantization module.

Covers:
* pack ↔ unpack round-trips (int2 / int4 / int8, including odd lengths).
* PTQ per-channel symmetric quantization error bounds.
* E2B / E4B ``quantization_config`` builders.
* Transcode key remapping (lossless, HF ``-mobile-transformers`` → mlx-vlm).
* Config promotion (gemma4 → gemma4_text) + weight casting + config selection.
* End-to-end PTQ → ``replace_with_gemma_quant_layers`` → load → forward.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.quantization.gemma_mobile import (
    GemmaQuantizedEmbedding,
    GemmaQuantizedLinear,
    dequantize_weight,
    replace_with_gemma_quant_layers,
    unpack_int2,
    unpack_int4,
)
from mlx_vlm.quantization.gemma_mobile_quantize import (
    build_e2b_mobile_quantization_config,
    build_e4b_mobile_quantization_config,
    cast_gemma_mobile_weights,
    is_gemma_mobile_checkpoint,
    pack_int2_row,
    pack_int4_row,
    pack_int_row,
    promote_text_config,
    quantize_embedding_per_row,
    quantize_model_gemma_mobile,
    quantize_per_channel_sym,
    select_mobile_quant_config,
    transcode_gemma_mobile_weights,
)


# ---------------------------------------------------------------------------
# Packing round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 4])
def test_pack_unpack_roundtrip(bits):
    rng = np.random.default_rng(0)
    lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    vals = rng.integers(lo, hi + 1, size=(5, 24), dtype=np.int8)
    packed = pack_int_row(mx.array(vals), bits)
    unpack = unpack_int2 if bits == 2 else unpack_int4
    out = unpack(packed, 24)
    np.testing.assert_array_equal(np.array(out), vals)


@pytest.mark.parametrize("bits", [2, 4])
def test_pack_unpack_odd_length(bits):
    """Packing pads the last byte; unpacking trims to ``in_features``."""
    rng = np.random.default_rng(1)
    lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    for n in (1, 3, 5, 7, 9):
        vals = rng.integers(lo, hi + 1, size=(3, n), dtype=np.int8)
        packed = pack_int_row(mx.array(vals), bits)
        unpack = unpack_int2 if bits == 2 else unpack_int4
        out = unpack(packed, n)
        np.testing.assert_array_equal(np.array(out), vals)


def test_pack_int8_dispatch():
    vals = mx.array([[-128, 0, 127, -1]], dtype=mx.int8)
    packed = pack_int_row(vals, 8)
    assert packed.dtype == mx.int8
    np.testing.assert_array_equal(np.array(packed), np.array(vals))


def test_pack_int_invalid_bits():
    with pytest.raises(ValueError):
        pack_int_row(mx.zeros((1, 4), dtype=mx.int8), 3)


# ---------------------------------------------------------------------------
# PTQ per-channel symmetric quantization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 4, 8])
def test_quantize_per_channel_sym_error_bounds(bits):
    rng = np.random.default_rng(42)
    W = rng.standard_normal((6, 32)).astype(np.float32)
    packed, scale = quantize_per_channel_sym(mx.array(W), bits)
    assert scale.shape == (6, 1)
    dq = dequantize_weight(packed, scale, bits, 32, mx.float32)
    err = float(mx.max(mx.abs(mx.array(W) - dq)))
    # Error grows as bits shrink; int2 is the coarsest.
    bounds = {2: 1.5, 4: 0.25, 8: 0.02}
    assert err < bounds[bits], f"bits={bits} err={err} > {bounds[bits]}"


def test_quantize_per_channel_sym_int2_levels():
    """int2 codebook is {-2,-1,0,1}; scale = max_abs / 1."""
    W = np.array([[0.0, 0.4, -0.4, 1.0, -1.0]], dtype=np.float32)
    packed, scale = quantize_per_channel_sym(mx.array(W), 2)
    out = unpack_int2(packed, 5)
    # max_abs = 1.0, scale = 1.0; round(0.4)=0, round(-0.4)=0, round(1)=1, round(-1)=-1
    np.testing.assert_array_equal(np.array(out[0]), [0, 0, 0, 1, -1])


def test_quantize_per_channel_sym_requires_2d():
    with pytest.raises(ValueError):
        quantize_per_channel_sym(mx.zeros((4,), dtype=mx.float32), 4)


def test_quantize_embedding_per_row_matches_per_channel():
    rng = np.random.default_rng(7)
    W = rng.standard_normal((5, 16)).astype(np.float32)
    ep, es = quantize_embedding_per_row(mx.array(W), 4)
    lp, ls = quantize_per_channel_sym(mx.array(W), 4)
    np.testing.assert_array_equal(np.array(ep), np.array(lp))
    np.testing.assert_allclose(np.array(es), np.array(ls))


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def test_build_e2b_config_structure():
    cfg = build_e2b_mobile_quantization_config()
    assert cfg["quant_method"] == "gemma"
    assert cfg["num_bits"] == 4
    assert cfg["quantize_embeddings"] is True
    mqc = cfg["module_quant_configs"]
    assert mqc[r"^lm_head$"]["num_bits"] == 2
    assert mqc[r"language_model\.embed_tokens$"]["num_bits"] == 2
    assert mqc[r"language_model\.layers\.(\d|1[0-4])\.mlp\."]["num_bits"] == 4
    assert mqc[r"language_model\.layers\.\d+\.mlp\."]["num_bits"] == 2
    assert mqc[r"language_model\.layers\.\d+\.self_attn\."]["num_bits"] == 4
    assert "per_layer_model_projection" in cfg["modules_to_not_convert"]


def test_build_e4b_config_structure():
    cfg = build_e4b_mobile_quantization_config()
    assert cfg["quant_method"] == "gemma"
    mqc = cfg["module_quant_configs"]
    # E4B: ALL MLP layers at 4-bit (no targeted 2-bit decode-only MLPs).
    assert mqc[r"language_model\.layers\.\d+\.mlp\."]["num_bits"] == 4
    assert not any(
        "1[0-4]" in pat for pat in mqc if "mlp" in pat
    ), "E4B should not have the E2B 0-14 split rule"
    # E4B: embed_tokens_per_layer is 2-bit (E2B is 4-bit).
    assert mqc[r"language_model\.embed_tokens_per_layer$"]["num_bits"] == 2


# ---------------------------------------------------------------------------
# Transcode key remapping
# ---------------------------------------------------------------------------

def _real_hf_mobile_weights():
    """Representative keys from ``google/gemma-4-E2B-it-qat-mobile-transformers``."""
    return {
        "model.language_model.embed_tokens.weight": mx.zeros((4, 4)),
        "model.language_model.embed_tokens.embedding_quantized": mx.zeros(
            (4, 1), dtype=mx.uint8
        ),
        "model.language_model.embed_tokens.embedding_scale": mx.zeros((4, 1)),
        "model.language_model.embed_tokens_per_layer.embedding_scale": mx.zeros(
            (4, 35)
        ),
        "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros((4, 4)),
        "model.language_model.layers.0.self_attn.q_proj.weight_scale": mx.zeros(
            (4, 1)
        ),
        "model.language_model.layers.0.self_attn.q_proj.input_activation_scale": mx.zeros(
            ()
        ),
        "model.language_model.layers.0.self_attn.q_proj.output_activation_scale": mx.zeros(
            ()
        ),
        "model.language_model.layers.0.self_attn.k_cache_scale": mx.zeros(()),
        "model.language_model.layers.0.self_attn.v_cache_scale": mx.zeros(()),
        "model.language_model.per_layer_model_projection.weight": mx.zeros((4, 4)),
        "lm_head.weight": mx.zeros((4, 4)),
        "lm_head.weight_scale": mx.zeros((4, 1)),
        "lm_head.input_activation_scale": mx.zeros(()),
        "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight": mx.zeros(
            (4, 4)
        ),
        "model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight": mx.zeros(
            (4, 4)
        ),
        "model.embed_vision.embedding_projection.weight": mx.zeros((4, 4)),
        "model.embed_audio.embedding_projection.weight": mx.zeros((4, 4)),
    }


def test_transcode_text_only():
    out = transcode_gemma_mobile_weights(_real_hf_mobile_weights(), text_only=True)
    assert "language_model.model.embed_tokens.embedding_scale" in out
    assert "language_model.model.embed_tokens_per_layer.embedding_scale" in out
    assert "language_model.model.layers.0.self_attn.q_proj.weight_scale" in out
    assert (
        "language_model.model.layers.0.self_attn.q_proj.input_activation_scale" in out
    )
    assert "language_model.lm_head.weight" in out
    assert "language_model.lm_head.weight_scale" in out
    assert "language_model.model.per_layer_model_projection.weight" in out
    assert not any("k_cache_scale" in k for k in out)
    assert not any("v_cache_scale" in k for k in out)
    assert not any("vision_tower" in k for k in out)
    assert not any("audio_tower" in k for k in out)
    assert not any("embed_vision" in k for k in out)
    assert not any("embed_audio" in k for k in out)


def test_transcode_multimodal_keeps_encoders():
    out = transcode_gemma_mobile_weights(_real_hf_mobile_weights(), text_only=False)
    assert "vision_tower.encoder.layers.0.mlp.down_proj.linear.weight" in out
    assert "audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight" in out
    assert "embed_vision.embedding_projection.weight" in out
    assert "embed_audio.embedding_projection.weight" in out
    assert not any("k_cache_scale" in k for k in out)


def test_transcode_preserves_tensor_values():
    src = _real_hf_mobile_weights()
    src["lm_head.weight"] = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    out = transcode_gemma_mobile_weights(src, text_only=True)
    np.testing.assert_array_equal(
        np.array(out["language_model.lm_head.weight"]),
        np.array(src["lm_head.weight"]),
    )


def test_is_gemma_mobile_checkpoint():
    assert is_gemma_mobile_checkpoint(_real_hf_mobile_weights())
    assert not is_gemma_mobile_checkpoint({"a.weight": mx.zeros((1,))})


# ---------------------------------------------------------------------------
# End-to-end PTQ -> replace -> load -> forward
# ---------------------------------------------------------------------------

def _make_tiny_gemma4(h=16, inter=32, vocab=64, n_layers=18):
    class MLP(nn.Module):
        def __init__(self, h, inter):
            super().__init__()
            self.gate_proj = nn.Linear(h, inter, bias=False)
            self.down_proj = nn.Linear(inter, h, bias=False)
            self.up_proj = nn.Linear(h, inter, bias=False)

    class Attn(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.q_proj = nn.Linear(h, h, bias=False)
            self.o_proj = nn.Linear(h, h, bias=False)

    class Layer(nn.Module):
        def __init__(self, h, inter):
            super().__init__()
            self.mlp = MLP(h, inter)
            self.self_attn = Attn(h)

    class TextModel(nn.Module):
        def __init__(self, h, inter, vocab, n):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, h)
            self.layers = [Layer(h, inter) for _ in range(n)]
            self.per_layer_model_projection = nn.Linear(h, h, bias=False)

    class LM(nn.Module):
        def __init__(self, h, inter, vocab):
            super().__init__()
            self.model = TextModel(h, inter, vocab, n_layers)
            self.lm_head = nn.Linear(h, vocab, bias=False)

    class M(nn.Module):
        def __init__(self, h, inter, vocab):
            super().__init__()
            self.language_model = LM(h, inter, vocab)

    return M(h, inter, vocab)


def test_quantize_model_gemma_mobile_structure():
    model = _make_tiny_gemma4()
    qc = build_e2b_mobile_quantization_config()
    new_weights, path_to_bits = quantize_model_gemma_mobile(model, qc, dtype=mx.float16)

    assert path_to_bits["language_model.lm_head"] == 2
    assert path_to_bits["language_model.model.embed_tokens"] == 2
    assert path_to_bits["language_model.model.layers.0.mlp.gate_proj"] == 4
    assert path_to_bits["language_model.model.layers.14.mlp.gate_proj"] == 4
    assert path_to_bits["language_model.model.layers.15.mlp.gate_proj"] == 2
    assert path_to_bits["language_model.model.layers.0.self_attn.q_proj"] == 4
    # per_layer_model_projection is in modules_to_not_convert -> stays fp.
    assert (
        "language_model.model.per_layer_model_projection.weight" in new_weights
    )
    assert (
        "language_model.model.per_layer_model_projection" not in path_to_bits
    )
    # Packed weights present for quantized leaves.
    assert (
        new_weights["language_model.lm_head.weight"].dtype == mx.uint8
    )  # 2-bit -> uint8
    assert new_weights["language_model.lm_head.weight_scale"].shape == (64, 1)


def test_quantize_model_gemma_mobile_load_and_forward():
    model = _make_tiny_gemma4()
    qc = build_e2b_mobile_quantization_config()
    new_weights, _ = quantize_model_gemma_mobile(model, qc, dtype=mx.float16)

    fresh = _make_tiny_gemma4()
    fresh = replace_with_gemma_quant_layers(fresh, qc, new_weights, dtype=mx.float16)
    assert isinstance(fresh.language_model.lm_head, GemmaQuantizedLinear)
    assert isinstance(
        fresh.language_model.model.embed_tokens, GemmaQuantizedEmbedding
    )
    assert isinstance(
        fresh.language_model.model.layers[0].mlp.gate_proj, GemmaQuantizedLinear
    )
    # per_layer_model_projection stays a plain Linear.
    assert isinstance(
        fresh.language_model.model.per_layer_model_projection, nn.Linear
    )

    fresh.load_weights(list(new_weights.items()), strict=False)
    mx.eval(fresh.parameters())

    # Forward through the quantized lm_head + embed_tokens directly.
    ids = mx.array([1, 2, 3])
    emb = fresh.language_model.model.embed_tokens(ids)
    logits = fresh.language_model.lm_head(emb)
    assert logits.shape == (3, 64)
    assert np.all(np.isfinite(np.array(logits)))


# ---------------------------------------------------------------------------
# Conversion helpers (config promotion, weight casting, config selection)
# ---------------------------------------------------------------------------

def test_promote_text_config():
    mm_cfg = {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "tie_word_embeddings": False,
            "dtype": "float32",
            "eos_token_id": [1, 2],
        },
        "quantization_config": {"quant_method": "gemma", "num_bits": 4},
        "tie_word_embeddings": False,
        "vision_config": {"hidden_size": 16},
        "audio_config": {"hidden_size": 16},
        "image_token_id": 100,
        "audio_token_id": 101,
        "eos_token_id": [1, 2],
    }
    out = promote_text_config(mm_cfg)
    assert out["model_type"] == "gemma4_text"
    assert out["hidden_size"] == 1536
    assert out["quantization_config"]["quant_method"] == "gemma"
    assert out["tie_word_embeddings"] is False
    assert "vision_config" not in out
    assert "audio_config" not in out
    assert "image_token_id" not in out
    assert "audio_token_id" not in out


def test_cast_gemma_mobile_weights_keeps_int():
    weights = {
        "a.weight": mx.zeros((4, 4), dtype=mx.uint8),
        "b.weight": mx.zeros((4, 4), dtype=mx.int8),
        "c.weight_scale": mx.ones((4, 1), dtype=mx.float32),
        "d.weight": mx.ones((4, 4), dtype=mx.bfloat16),
    }
    out = cast_gemma_mobile_weights(weights, mx.float16)
    assert out["a.weight"].dtype == mx.uint8  # packed int2/int4 preserved
    assert out["b.weight"].dtype == mx.int8  # packed int8 preserved
    assert out["c.weight_scale"].dtype == mx.float16  # fp cast
    assert out["d.weight"].dtype == mx.float16  # fp cast


def test_select_mobile_quant_config():
    e2b = select_mobile_quant_config(35)
    assert e2b["module_quant_configs"][r"language_model\.layers\.(\d|1[0-4])\.mlp\."][
        "num_bits"
    ] == 4
    e4b = select_mobile_quant_config(42)
    assert e4b["module_quant_configs"][r"language_model\.layers\.\d+\.mlp\."][
        "num_bits"
    ] == 4
    assert (
        r"language_model\.layers\.(\d|1[0-4])\.mlp\."
        not in e4b["module_quant_configs"]
    )
