"""Tests for the Gemma 4 QAT Mobile (wNa8o8 / ``quant_method: "gemma"``) backend.

Covers bit-exact unpacking, SRQ activations, per-channel dequantization, the
quantized Linear/Embedding layers, per-module bit resolution against the E2B
``module_quant_configs`` table, and module replacement.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.quantization.gemma_mobile import (
    GemmaQuantizedEmbedding,
    GemmaQuantizedLinear,
    MmapEmbeddingLookup,
    _strip_clippable_linear_clips,
    apply_srq,
    dequantize_weight,
    offload_gemma_embeddings,
    replace_with_gemma_quant_layers,
    resolve_module_bits,
    unpack_int,
    unpack_int2,
    unpack_int4,
)


# ---------------------------------------------------------------------------
# Packing helpers (mirror gemma_quant.py layout)
# ---------------------------------------------------------------------------

def _pack_int2_row(vals: np.ndarray) -> np.ndarray:
    """Pack a row of signed int2 values ([-2, 1]) into uint8, 4/byte, LSB-first."""
    u = (vals.astype(np.int32) + 2).astype(np.uint8)
    packed = np.zeros(u.shape[0] // 4, dtype=np.uint8)
    for i in range(4):
        packed |= u[i::4] << (2 * i)
    return packed


def _pack_int4_row(vals: np.ndarray) -> np.ndarray:
    """Pack a row of signed int4 values ([-8, 7]) into uint8, 2/byte, low-nibble-first."""
    u = (vals.astype(np.int32) + 8).astype(np.uint8)
    packed = np.zeros(u.shape[0] // 2, dtype=np.uint8)
    packed |= u[0::2] & 0x0F
    packed |= (u[1::2] & 0x0F) << 4
    return packed


# ---------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------

def test_unpack_int2_bit_exact():
    vals = np.array([-2, -1, 0, 1, -2, 1, 0, -1, 1, -2, -1, 0], dtype=np.int8)
    packed = mx.array(_pack_int2_row(vals)[None, :])
    out = unpack_int2(packed, vals.size)
    np.testing.assert_array_equal(np.array(out[0]), vals)


def test_unpack_int4_bit_exact():
    vals = np.array([-8, 7, 0, -1, 3, -8, 7, 5, -4, 4, 1, -2], dtype=np.int8)
    packed = mx.array(_pack_int4_row(vals)[None, :])
    out = unpack_int4(packed, vals.size)
    np.testing.assert_array_equal(np.array(out[0]), vals)


def test_unpack_int_dispatch():
    p2 = mx.array(_pack_int2_row(np.array([-2, 1, 0, -1], dtype=np.int8))[None, :])
    np.testing.assert_array_equal(np.array(unpack_int(p2, 2, 4)[0]), [-2, 1, 0, -1])
    p4 = mx.array(_pack_int4_row(np.array([-8, 7], dtype=np.int8))[None, :])
    np.testing.assert_array_equal(np.array(unpack_int(p4, 4, 2)[0]), [-8, 7])
    p8 = mx.array(np.array([[-128, 0, 127, -1]], dtype=np.int8))
    np.testing.assert_array_equal(np.array(unpack_int(p8, 8, 4)[0]), [-128, 0, 127, -1])


def test_unpack_int_invalid_bits():
    with pytest.raises(ValueError):
        unpack_int(mx.zeros((1, 1), dtype=mx.uint8), 3, 4)


# ---------------------------------------------------------------------------
# SRQ activations
# ---------------------------------------------------------------------------

def test_apply_srq_noop_when_uncalibrated():
    x = mx.array([0.0, 0.5, 1.0, 100.0, -100.0])
    out = apply_srq(x, mx.array(0.0))
    np.testing.assert_array_equal(np.array(out), np.array(x))


def test_apply_srq_rounds_and_clips():
    x = mx.array([0.0, 0.4, 0.5, 0.6, 1000.0, -1000.0])
    out = apply_srq(x, mx.array(1.0))
    # round to nearest int (banker's rounding), clip to [-128, 127]
    expected = np.array([0.0, 0.0, 0.0, 1.0, 127.0, -128.0])
    np.testing.assert_allclose(np.array(out), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Dequantization + quantized layers
# ---------------------------------------------------------------------------

def _quant_per_channel(W: np.ndarray, bits: int):
    if bits == 2:
        denom = 1.0
        lo, hi = -2, 1
    elif bits == 4:
        denom = 7.0
        lo, hi = -8, 7
    else:
        denom = 127.0
        lo, hi = -128, 127
    scale = (np.max(np.abs(W), axis=1, keepdims=True) / denom).astype(np.float32)
    q = np.clip(np.round(W / scale), lo, hi).astype(np.int8)
    return q, scale


def _pack_weight(q: np.ndarray, bits: int) -> np.ndarray:
    out = np.zeros((q.shape[0], q.shape[1] // (4 // bits)), dtype=np.uint8) if bits < 8 else q
    if bits == 2:
        return np.stack([_pack_int2_row(r) for r in q])
    if bits == 4:
        return np.stack([_pack_int4_row(r) for r in q])
    return q


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_dequantize_weight_matches_reference(bits):
    rng = np.random.default_rng(0)
    out_d, in_d = 5, 8
    W = rng.standard_normal((out_d, in_d)).astype(np.float32)
    q, scale = _quant_per_channel(W, bits)
    packed = mx.array(_pack_weight(q, bits))
    got = dequantize_weight(packed, mx.array(scale), bits, in_d, mx.float32)
    ref = mx.array((q.astype(np.float32) * scale))
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-6)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_gemma_quantized_linear_matches_reference(bits):
    rng = np.random.default_rng(1)
    out_d, in_d = 6, 8
    W = rng.standard_normal((out_d, in_d)).astype(np.float32)
    q, scale = _quant_per_channel(W, bits)
    packed = mx.array(_pack_weight(q, bits))
    lin = GemmaQuantizedLinear(in_d, out_d, bits, bias=False, dtype=mx.float32)
    lin.weight = packed
    lin.weight_scale = mx.array(scale)
    x = mx.array(rng.standard_normal((3, in_d)).astype(np.float32))
    got = lin(x)
    ref = x @ mx.array((q.astype(np.float32) * scale)).T
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-6)


def test_gemma_quantized_linear_with_srq_scales():
    rng = np.random.default_rng(2)
    out_d, in_d = 4, 8
    W = rng.standard_normal((out_d, in_d)).astype(np.float32)
    q, scale = _quant_per_channel(W, 4)
    lin = GemmaQuantizedLinear(
        in_d, out_d, 4, bias=False, dtype=mx.float32, input_scale=True, output_scale=True
    )
    lin.weight = mx.array(_pack_weight(q, 4))
    lin.weight_scale = mx.array(scale)
    lin.input_activation_scale = mx.array(2.0)
    lin.output_activation_scale = mx.array(0.0)  # uncalibrated -> no-op on output
    x = mx.array(rng.standard_normal((2, in_d)).astype(np.float32))
    out = lin(x)
    assert out.shape == (2, out_d)
    # output SRQ is a no-op (scale 0); input SRQ quantizes x to int8 with scale 2
    xq = apply_srq(x, mx.array(2.0))
    ref = xq @ mx.array((q.astype(np.float32) * scale)).T
    np.testing.assert_allclose(np.array(out), np.array(ref), atol=1e-6)


def test_gemma_quantized_embedding_matches_reference():
    rng = np.random.default_rng(3)
    num_emb, dim = 5, 8
    W = rng.standard_normal((num_emb, dim)).astype(np.float32)
    q, scale = _quant_per_channel(W, 2)
    emb = GemmaQuantizedEmbedding(num_emb, dim, 2, embed_scale=1.0, dtype=mx.float32)
    emb.embedding_quantized = mx.array(np.stack([_pack_int2_row(r) for r in q]))
    emb.embedding_scale = mx.array(scale)
    ids = mx.array([0, 2, 4])
    got = emb(ids)
    ref = mx.array((q[ids].astype(np.float32) * scale[ids]))
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-6)
    # as_linear uses the full dequantized table
    x = mx.array(rng.standard_normal((2, dim)).astype(np.float32))
    ref = x @ mx.array((q.astype(np.float32) * scale)).T
    np.testing.assert_allclose(np.array(emb.as_linear(x)), np.array(ref), atol=1e-6)


# ---------------------------------------------------------------------------
# Per-module bit resolution (E2B mobile-transformers config)
# ---------------------------------------------------------------------------

E2B_QC = {
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


@pytest.mark.parametrize(
    "path,expected",
    [
        ("language_model.lm_head", 2),
        ("language_model.model.embed_tokens", 2),
        ("language_model.model.embed_tokens_per_layer", 4),
        ("language_model.model.layers.0.mlp.gate_proj", 4),
        ("language_model.model.layers.14.mlp.down_proj", 4),
        ("language_model.model.layers.15.mlp.gate_proj", 2),
        ("language_model.model.layers.34.mlp.up_proj", 2),
        ("language_model.model.layers.0.self_attn.q_proj", 4),
        ("language_model.model.layers.20.self_attn.o_proj", 4),
        ("language_model.model.layers.5.per_layer_input_gate", 8),
        ("language_model.model.layers.5.per_layer_projection", 8),
        ("vision_tower.layers.0.self_attn.q_proj", 8),
        ("audio_tower.layers.0.lconv1d.linear_start.linear", 4),
        ("audio_tower.layers.0.lconv1d.linear_end.linear", 2),
        # skipped (modules_to_not_convert)
        ("language_model.model.per_layer_model_projection", None),
        ("language_model.model.layers.0.self_attn.relative_k_proj", None),
        ("vision_tower.patch_embedder", None),
        ("audio_tower.subsample_conv_projection", None),
        ("embed_vision.embedding_projection", None),
        ("embed_audio.embedding_projection", None),
    ],
)
def test_resolve_module_bits_e2b(path, expected):
    assert resolve_module_bits(path, E2B_QC) == expected


# ---------------------------------------------------------------------------
# Module replacement
# ---------------------------------------------------------------------------

def _make_gemma4_like_model(h=16, inter=32, vocab=64, n_layers=18):
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


def _build_packed_weights(model, bits_for_path):
    weights = {}
    for path, mod in tree_flatten(model.leaf_modules(), is_leaf=lambda x: isinstance(x, nn.Module)):
        bits = bits_for_path(path, mod)
        if bits is None:
            continue
        if isinstance(mod, nn.Linear):
            od, ind = mod.weight.shape
            packed_in = (ind + 3) // 4 if bits == 2 else (ind + 1) // 2 if bits == 4 else ind
            wtype = mx.uint8 if bits < 8 else mx.int8
            weights[f"{path}.weight"] = mx.zeros((od, packed_in), dtype=wtype)
            weights[f"{path}.weight_scale"] = mx.ones((od, 1), dtype=mx.float32)
        elif isinstance(mod, nn.Embedding):
            nemb, dim = mod.weight.shape
            packed_dim = (dim + 3) // 4 if bits == 2 else (dim + 1) // 2 if bits == 4 else dim
            wtype = mx.uint8 if bits < 8 else mx.int8
            weights[f"{path}.embedding_quantized"] = mx.zeros((nemb, packed_dim), dtype=wtype)
            weights[f"{path}.embedding_scale"] = mx.ones((nemb, 1), dtype=mx.float32)
    return weights


def test_replace_with_gemma_quant_layers():
    model = _make_gemma4_like_model()
    qc = {
        "quant_method": "gemma",
        "num_bits": 4,
        "quantize_embeddings": True,
        "module_quant_configs": {
            r"^lm_head$": {"num_bits": 2},
            r"language_model\.embed_tokens$": {"num_bits": 2},
            r"language_model\.layers\.(\d|1[0-4])\.mlp\.": {"num_bits": 4},
            r"language_model\.layers\.\d+\.mlp\.": {"num_bits": 2},
            r"language_model\.layers\.\d+\.self_attn\.": {"num_bits": 4},
        },
        "modules_to_not_convert": ["per_layer_model_projection"],
    }
    weights = _build_packed_weights(model, lambda p, m: resolve_module_bits(p, qc))
    # per_layer_model_projection must stay fp (no weight_scale)
    weights.pop("language_model.model.per_layer_model_projection.weight_scale", None)

    model = replace_with_gemma_quant_layers(model, qc, weights, dtype=mx.float16)

    lm = model.language_model
    assert isinstance(lm.lm_head, GemmaQuantizedLinear) and lm.lm_head.num_bits == 2
    assert isinstance(lm.model.embed_tokens, GemmaQuantizedEmbedding) and lm.model.embed_tokens.num_bits == 2
    assert isinstance(lm.model.layers[0].mlp.gate_proj, GemmaQuantizedLinear) and lm.model.layers[0].mlp.gate_proj.num_bits == 4
    assert isinstance(lm.model.layers[14].mlp.gate_proj, GemmaQuantizedLinear) and lm.model.layers[14].mlp.gate_proj.num_bits == 4
    assert isinstance(lm.model.layers[15].mlp.gate_proj, GemmaQuantizedLinear) and lm.model.layers[15].mlp.gate_proj.num_bits == 2
    assert isinstance(lm.model.layers[0].self_attn.q_proj, GemmaQuantizedLinear) and lm.model.layers[0].self_attn.q_proj.num_bits == 4
    # skipped module stays a plain Linear
    assert isinstance(lm.model.per_layer_model_projection, nn.Linear) and not isinstance(
        lm.model.per_layer_model_projection, GemmaQuantizedLinear
    )


def test_replace_skips_layers_without_packed_weights():
    model = _make_gemma4_like_model()
    qc = {"quant_method": "gemma", "num_bits": 4, "quantize_embeddings": True, "module_quant_configs": {}, "modules_to_not_convert": []}
    # No weights at all -> nothing should be replaced
    model = replace_with_gemma_quant_layers(model, qc, weights={}, dtype=mx.float16)
    assert isinstance(model.language_model.lm_head, nn.Linear) and not isinstance(
        model.language_model.lm_head, GemmaQuantizedLinear
    )


# ---------------------------------------------------------------------------
# Block-wise embedding scale (PLE embeddings use (num_emb, n_blocks) scales)
# ---------------------------------------------------------------------------

def test_gemma_quantized_embedding_num_blocks_init_shape():
    """num_blocks controls the placeholder scale shape for strict loading."""
    emb = GemmaQuantizedEmbedding(10, 8, 4, num_blocks=4)
    assert emb.embedding_scale.shape == (10, 4)
    emb1 = GemmaQuantizedEmbedding(10, 8, 4, num_blocks=1)
    assert emb1.embedding_scale.shape == (10, 1)


def test_gemma_quantized_embedding_blockwise_dequant():
    """Block-wise scale (num_blocks > 1) dequantizes each block by its own scale."""
    rng = np.random.default_rng(42)
    num_emb, dim, num_blocks = 4, 8, 4
    block_size = dim // num_blocks
    W = rng.standard_normal((num_emb, dim)).astype(np.float32)
    # Per-block symmetric int4 quantization
    blocks = W.reshape(num_emb, num_blocks, block_size)
    max_abs = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), 1e-8)
    scales = (max_abs / 8.0).squeeze(-1)  # (num_emb, num_blocks)
    q = np.clip(np.round(blocks / scales[..., None]), -8, 7).astype(np.int8)
    q_flat = q.reshape(num_emb, dim)

    emb = GemmaQuantizedEmbedding(num_emb, dim, 4, dtype=mx.float32, num_blocks=num_blocks)
    emb.embedding_quantized = mx.array(np.stack([_pack_int4_row(r) for r in q_flat]))
    emb.embedding_scale = mx.array(scales)

    ids = mx.array([0, 1, 3])
    got = emb(ids)
    ref = (q[ids].astype(np.float32) * scales[ids][..., None]).reshape(len(ids), dim)
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-6)


# ---------------------------------------------------------------------------
# ClippableLinear clip stripping (mobile checkpoints use SRQ, not clipping)
# ---------------------------------------------------------------------------

class _FakeClippableLinear(nn.Module):
    """Minimal stand-in for ClippableLinear (duck-typed via use_clipping + linear)."""

    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        self.use_clipping = True
        self.input_min = mx.array(float("-inf"))
        self.input_max = mx.array(float("inf"))
        self.output_min = mx.array(float("-inf"))
        self.output_max = mx.array(float("inf"))

    def __call__(self, x):
        return self.linear(x)


def test_strip_clippable_clips_removes_params():
    """_strip_clippable_linear_clips removes clip params when inner is quantized."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _FakeClippableLinear(8, 8)

        def __call__(self, x):
            return self.layer(x)

    model = M()
    assert model.layer.use_clipping is True
    assert hasattr(model.layer, "input_min")
    model.layer.linear = GemmaQuantizedLinear(8, 8, 2, bias=False)
    _strip_clippable_linear_clips(model)
    assert model.layer.use_clipping is False
    for attr in ("input_min", "input_max", "output_min", "output_max"):
        assert not hasattr(model.layer, attr)
    assert isinstance(model.layer.linear, GemmaQuantizedLinear)


def test_strip_clippable_clips_preserves_unquantized():
    """Unquantized ClippableLinear (plain nn.Linear inner) keeps its clip params."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _FakeClippableLinear(8, 8)

        def __call__(self, x):
            return self.layer(x)

    model = M()
    _strip_clippable_linear_clips(model)
    assert model.layer.use_clipping is True
    assert hasattr(model.layer, "input_min")


def test_replace_strips_clippable_clips_end_to_end():
    """replace_with_gemma_quant_layers strips clips from ClippableLinear wrappers."""

    class AudioLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffw = _FakeClippableLinear(8, 8)

        def __call__(self, x):
            return self.ffw(x)

    class AudioTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [AudioLayer()]

        def __call__(self, x):
            return self.layers[0](x)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_tower = AudioTower()

        def __call__(self, x):
            return self.audio_tower(x)

    model = M()
    qc = {
        "quant_method": "gemma",
        "num_bits": 4,
        "quantize_embeddings": False,
        "module_quant_configs": {r"audio_tower": {"num_bits": 2}},
        "modules_to_not_convert": [],
    }
    weights = {
        "audio_tower.layers.0.ffw.linear.weight": mx.zeros((8, 2), dtype=mx.uint8),
        "audio_tower.layers.0.ffw.linear.weight_scale": mx.ones((8, 1), dtype=mx.float32),
    }
    model = replace_with_gemma_quant_layers(model, qc, weights, dtype=mx.float16)
    ffw = model.audio_tower.layers[0].ffw
    assert isinstance(ffw.linear, GemmaQuantizedLinear)
    assert ffw.use_clipping is False
    for attr in ("input_min", "input_max", "output_min", "output_max"):
        assert not hasattr(ffw, attr)


# ---------------------------------------------------------------------------
# On-disk mmap embedding offload (Phase 1 memory optimization)
# ---------------------------------------------------------------------------

def _write_safetensors(path, tensors):
    """Write a minimal safetensors file.  ``tensors``: {name: (bytes, dtype_str, shape)}."""
    import json
    import struct

    header = {}
    offset = 0
    data = b""
    for name, (db, dtype_str, shape) in tensors.items():
        header[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(db)],
        }
        offset += len(db)
        data += db
    hjson = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(data)


def _make_offload_embedding(num_emb, dim, num_bits, seed=0):
    """Build a per-row GemmaQuantizedEmbedding with random packed weights + scale."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((num_emb, dim)).astype(np.float32)
    q, scale = _quant_per_channel(W, num_bits)
    pack_fn = _pack_int2_row if num_bits == 2 else _pack_int4_row
    packed = np.stack([pack_fn(r) for r in q])
    emb = GemmaQuantizedEmbedding(num_emb, dim, num_bits, dtype=mx.float32, num_blocks=1)
    emb.embedding_quantized = mx.array(packed)
    emb.embedding_scale = mx.array(scale)
    return emb


@pytest.mark.parametrize("num_bits", [2, 4])
def test_offload_embedding_matches_in_memory(tmp_path, num_bits):
    """Offloaded __call__ (mmap gather + MLX dequant) matches the in-memory path."""
    num_emb, dim = 32, 16
    emb = _make_offload_embedding(num_emb, dim, num_bits, seed=1)
    packed_np = np.array(emb.embedding_quantized)
    scale_np = np.array(emb.embedding_scale)
    path = tmp_path / "emb.safetensors"
    _write_safetensors(str(path), {
        "emb.embedding_quantized": (packed_np.tobytes(), "U8", packed_np.shape),
        "emb.embedding_scale": (scale_np.tobytes(), "F32", scale_np.shape),
    })
    ids = mx.array([0, 3, 7, 15, 31])
    ref = emb(ids)
    mx.eval(ref)
    emb.offload_to_mmap(str(path), "emb.embedding_quantized")
    assert getattr(emb, "_mmap_lookup", None) is not None
    got = emb(ids)
    mx.eval(got)
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-6)


def test_offload_frees_dict_item(tmp_path):
    """offload_to_mmap replaces the dict item so parameters() sees the dummy."""
    emb = _make_offload_embedding(32, 16, 2, seed=2)
    packed_np = np.array(emb.embedding_quantized)
    scale_np = np.array(emb.embedding_scale)
    path = tmp_path / "emb.safetensors"
    _write_safetensors(str(path), {
        "emb.embedding_quantized": (packed_np.tobytes(), "U8", packed_np.shape),
        "emb.embedding_scale": (scale_np.tobytes(), "F32", scale_np.shape),
    })
    assert emb["embedding_quantized"].shape == packed_np.shape
    emb.offload_to_mmap(str(path), "emb.embedding_quantized")
    # Both the attribute and the dict item must be the 1-element dummy so the
    # full table is actually freed (object.__setattr__ would leave the dict item).
    assert emb.embedding_quantized.shape == (1,)  # attribute -> dict
    assert emb["embedding_quantized"].shape == (1,)  # dict item
    assert emb.parameters()["embedding_quantized"].shape == (1,)  # parameters()


def test_offload_weight_materializes_on_demand(tmp_path):
    """The weight property materializes the full table from the mmap (rare path)."""
    emb = _make_offload_embedding(16, 8, 4, seed=3)
    packed_np = np.array(emb.embedding_quantized)
    scale_np = np.array(emb.embedding_scale)
    path = tmp_path / "emb.safetensors"
    _write_safetensors(str(path), {
        "emb.embedding_quantized": (packed_np.tobytes(), "U8", packed_np.shape),
        "emb.embedding_scale": (scale_np.tobytes(), "F32", scale_np.shape),
    })
    ref_weight = emb.weight
    mx.eval(ref_weight)
    emb.offload_to_mmap(str(path), "emb.embedding_quantized")
    got_weight = emb.weight
    mx.eval(got_weight)
    np.testing.assert_allclose(np.array(got_weight), np.array(ref_weight), atol=1e-6)


def test_offload_gemma_embeddings_selects_by_size(tmp_path):
    """offload_gemma_embeddings offloads tables above the size threshold only."""
    big = _make_offload_embedding(64, 128, 4, seed=4)  # 64 * 64 = 4096 bytes
    small = _make_offload_embedding(8, 16, 2, seed=5)  # 8 * 4 = 32 bytes

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = big
            self.embed_tokens_per_layer = small

        def __call__(self, x):
            return self.embed_tokens(x)

    model = M()
    path = tmp_path / "model.safetensors"
    _write_safetensors(str(path), {
        "embed_tokens.embedding_quantized": (
            np.array(big.embedding_quantized).tobytes(),
            "U8",
            np.array(big.embedding_quantized).shape,
        ),
        "embed_tokens.embedding_scale": (
            np.array(big.embedding_scale).tobytes(),
            "F32",
            np.array(big.embedding_scale).shape,
        ),
        "embed_tokens_per_layer.embedding_quantized": (
            np.array(small.embedding_quantized).tobytes(),
            "U8",
            np.array(small.embedding_quantized).shape,
        ),
        "embed_tokens_per_layer.embedding_scale": (
            np.array(small.embedding_scale).tobytes(),
            "F32",
            np.array(small.embedding_scale).shape,
        ),
    })
    n = offload_gemma_embeddings(model, [str(path)], min_size_bytes=100)
    assert n == 1
    assert getattr(model.embed_tokens, "_mmap_lookup", None) is not None
    assert getattr(model.embed_tokens_per_layer, "_mmap_lookup", None) is None
    # The offloaded (large) table still produces correct output.
    ids = mx.array([0, 10, 40, 63])
    ref = big(ids)
    mx.eval(ref)
    got = model.embed_tokens(ids)
    mx.eval(got)
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-6)


def test_offload_gemma_embeddings_env_disable(tmp_path):
    """MLX_VLM_NO_EMBED_OFFLOAD=1 disables offloading."""
    import os

    big = _make_offload_embedding(64, 128, 4, seed=6)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = big

        def __call__(self, x):
            return self.embed_tokens(x)

    model = M()
    path = tmp_path / "model.safetensors"
    _write_safetensors(str(path), {
        "embed_tokens.embedding_quantized": (
            np.array(big.embedding_quantized).tobytes(),
            "U8",
            np.array(big.embedding_quantized).shape,
        ),
        "embed_tokens.embedding_scale": (
            np.array(big.embedding_scale).tobytes(),
            "F32",
            np.array(big.embedding_scale).shape,
        ),
    })
    old = os.environ.get("MLX_VLM_NO_EMBED_OFFLOAD")
    os.environ["MLX_VLM_NO_EMBED_OFFLOAD"] = "1"
    try:
        n = offload_gemma_embeddings(model, [str(path)], min_size_bytes=100)
    finally:
        if old is None:
            os.environ.pop("MLX_VLM_NO_EMBED_OFFLOAD", None)
        else:
            os.environ["MLX_VLM_NO_EMBED_OFFLOAD"] = old
    assert n == 0
    assert getattr(model.embed_tokens, "_mmap_lookup", None) is None
