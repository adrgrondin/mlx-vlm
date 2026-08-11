"""Compiled decode fast-path for Gemma 4 QAT mobile models (LiteRT-LM inspired).

The direct port of LiteRT-LM's "single compiled program" approach: compile the
whole single-token decode forward with ``mx.compile``, passing the KV cache as
plain arrays (external buffers) instead of the stateful ``KVCache`` /
``RotatingKVCache`` objects that ``mx.compile`` rejects.

See ``GEMMA4_QAT_MOBILE_LITERT_PLAN.md`` for the full investigation. Summary:

* **Bit-exact** vs the eager path (max |eager − compiled| = 0.0).
* **Neutral** on speed (the custom Metal matmul kernels are opaque to
  ``mx.compile``, so they cannot fuse with the surrounding RMSNorm/residual;
  the decode is GPU-compute-bound on weight-read bandwidth, not Python/launch
  bound).  Kept as an opt-in foundation (``MLX_VLM_FAST_DECODE=1``) for future
  kernel-level fusion work.

Design:
* The KV cache is flattened to per-source-layer ``keys`` / ``values`` arrays.
  Layers 0..14 are source layers (``kv_shared_only=False``); layers 15..34 are
  KV-shared and reuse the KV of layer 13 (sliding) / 14 (full) via
  ``previous_kvs``.
* The KV update is a growing ``mx.concatenate`` (full attention) or
  ``concatenate(...)[..., -window:, :]`` (sliding, ``keep=0`` ⇒ equivalent to
  the rotating cache).  The ``-window:`` slice is a constant, so it is
  compile-friendly.  ``mx.compile`` handles the growing shapes without
  recompiling.
* For decode (single new token) the attention mask is ``None`` for both full
  and sliding layers: the fetched keys are already pre-filtered (all keys for
  full; last ``window`` for sliding), so the new token attends to all of them.
* Fused q/k/v (P4) caches are pre-built eagerly so the compiled graph uses the
  pre-concatenated weights as constants.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ...quantization.gemma_mobile import gemma_fused_qkv_matmul, gemma_mobile_matmul
from ..base import LanguageModelOutput


def fast_decode_enabled() -> bool:
    """Whether the compiled decode fast-path is opt-in (env-gated)."""
    return os.environ.get("MLX_VLM_FAST_DECODE", "0") == "1"


class FastDecodeState:
    """Holds the flattened KV cache arrays + current position for the fast path."""

    __slots__ = ("keys_list", "values_list", "offset", "src_layers")

    def __init__(self, keys_list, values_list, offset, src_layers):
        self.keys_list = keys_list
        self.values_list = values_list
        self.offset = offset
        self.src_layers = src_layers


def _source_layers(lm) -> List[int]:
    m = lm.model
    return [i for i in range(len(m.layers)) if not m.layers[i].self_attn.kv_shared_only]


def _prebuild_fused_qkv(lm, dtype) -> dict:
    """Eagerly populate the fused q/k/v weight caches so the compiled graph
    captures the pre-concatenated weights as constants (no per-step rebuild)."""
    m = lm.model
    fused = {}
    hidden = m.config.hidden_size
    for i in _source_layers(lm):
        attn = m.layers[i].self_attn
        if (not attn.use_k_eq_v) and getattr(attn.q_proj, "mode", None) == "gemma":
            gemma_fused_qkv_matmul(attn, mx.zeros((1, 1, hidden), dtype=dtype))
            fused[i] = attn._fused_qkv_cache
    return fused


def build_compiled_decode(lm):
    """Build (once per model) the ``mx.compile``-d single-token decode forward.

    Returns ``(compiled_fn, src_layers)``.  The compiled function signature is
    ``compiled_fn(input_ids, offset, keys_list, values_list) -> (logits,
    new_keys, new_values)`` where ``keys_list`` / ``values_list`` are indexed by
    source-layer index (None for shared layers).
    """
    m = lm.model
    NL = m.config.num_hidden_layers
    WIN = m.config.sliding_window
    prev_kvs = m.previous_kvs
    src_layers = _source_layers(lm)
    # Use the model's parameter dtype for the dummy used to prebuild fused qkv.
    from mlx.utils import tree_flatten

    leaves = tree_flatten(m.parameters())
    dtype = mx.bfloat16
    for item in leaves:
        v = item[1] if isinstance(item, tuple) else item
        if isinstance(v, mx.array) and mx.issubdtype(v.dtype, mx.floating):
            dtype = v.dtype
            break
    fused_cache = _prebuild_fused_qkv(lm, dtype)
    tie = lm.tie_word_embeddings
    softcap = lm.final_logit_softcapping

    @mx.compile
    def compiled_decode(input_ids, offset, keys_list, values_list):
        h = m.embed_tokens(input_ids) * m.embed_scale
        per_layer_inputs = m.get_per_layer_inputs(input_ids)
        per_layer_inputs = m.project_per_layer_inputs(h, per_layer_inputs)
        per_layer_list = [per_layer_inputs[:, :, i, :] for i in range(NL)]

        intermediates = [None] * NL
        new_keys: List[Any] = []
        new_values: List[Any] = []
        for idx in range(NL):
            layer = m.layers[idx]
            attn = layer.self_attn
            prev_idx = prev_kvs[idx]
            shared_kv = intermediates[prev_idx]
            B, L, _ = h.shape
            residual = h
            hh = layer.input_layernorm(h)
            if not attn.kv_shared_only:  # source layer: compute q/k/v, update KV
                fc = fused_cache.get(idx)
                if fc is not None and fc is not False:
                    weight, weight_scale, num_bits, in_features, in_s, out_s = fc
                    fused = gemma_mobile_matmul(
                        hh, weight, weight_scale, num_bits, in_features,
                        input_scale=in_s, output_scale=out_s,
                    )
                    qd = attn.n_heads * attn.head_dim
                    kvd = attn.n_kv_heads * attn.head_dim
                    queries = fused[..., :qd].reshape(B, L, attn.n_heads, attn.head_dim)
                    keys = fused[..., qd:qd + kvd].reshape(B, L, attn.n_kv_heads, attn.head_dim)
                    values = fused[..., qd + kvd:].reshape(B, L, attn.n_kv_heads, attn.head_dim)
                else:
                    queries = attn.q_proj(hh).reshape(B, L, attn.n_heads, attn.head_dim)
                    keys = attn.k_proj(hh).reshape(B, L, attn.n_kv_heads, attn.head_dim)
                    values = attn.v_proj(hh).reshape(B, L, attn.n_kv_heads, attn.head_dim)
                queries = attn.q_norm(queries)
                keys = attn.k_norm(keys)
                keys = keys.transpose(0, 2, 1, 3)
                keys = attn.rope(keys, offset=offset)
                values = attn.v_norm(values)
                values = values.transpose(0, 2, 1, 3)
                queries = queries.transpose(0, 2, 1, 3)
                queries = attn.rope(queries, offset=offset)
                keys = mx.concatenate([keys_list[idx], keys], axis=2)
                values = mx.concatenate([values_list[idx], values], axis=2)
                if attn.is_sliding:
                    keys = keys[..., -WIN:, :]
                    values = values[..., -WIN:, :]
                fetched_k, fetched_v = keys, values
            else:  # KV-shared layer: only q, reuse source KV
                queries = attn.q_proj(hh).reshape(B, L, attn.n_heads, attn.head_dim)
                queries = attn.q_norm(queries)
                queries = queries.transpose(0, 2, 1, 3)
                queries = attn.rope(queries, offset=offset)
                fetched_k, fetched_v = shared_kv
            out = mx.fast.scaled_dot_product_attention(
                queries, fetched_k, fetched_v, scale=attn.scale, mask=None
            )
            out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            out = attn.o_proj(out)
            kvs = (fetched_k, fetched_v)
            hh = layer.post_attention_layernorm(out)
            hh = residual + hh
            residual = hh
            hh = layer.pre_feedforward_layernorm(hh)
            hh = layer.mlp(hh)
            hh = layer.post_feedforward_layernorm(hh)
            hh = residual + hh
            if layer.per_layer_input_gate is not None and per_layer_list[idx] is not None:
                residual = hh
                gate = layer.per_layer_input_gate(hh)
                gate = nn.gelu_approx(gate)
                gate = mx.multiply(gate, per_layer_list[idx])
                gate = layer.per_layer_projection(gate)
                gate = layer.post_per_layer_input_norm(gate)
                hh = residual + gate
            if layer.layer_scalar is not None:
                hh = hh * layer.layer_scalar
            h = hh
            intermediates[idx] = kvs
            if not attn.kv_shared_only:
                new_keys.append(fetched_k)
                new_values.append(fetched_v)
        h = m.norm(h)
        if tie:
            logits = m.embed_tokens.as_linear(h)
        else:
            logits = lm.lm_head(h)
        if softcap is not None:
            logits = mx.tanh(logits / softcap) * softcap
        return logits, new_keys, new_values

    return compiled_decode, src_layers


def init_fast_decode(lm, prompt_cache) -> Optional[FastDecodeState]:
    """Extract the KV state from the eager prompt cache after prefill.

    Returns ``None`` if the fast path is not usable (e.g. no source-layer caches).
    """
    # The fast path uses the mobile-format custom kernels (gemma_mobile_matmul),
    # which need the packed weights.  The native compiled path frees them at load
    # time (precompile_native_functions), so the two are incompatible — bail out
    # and let the eager native path handle decode instead of reading zeroed weights.
    if getattr(lm.model, "_mobile_weights_freed", False):
        return None
    src_layers = _source_layers(lm)
    if not src_layers or len(prompt_cache) < len(src_layers):
        return None
    keys_list: List[Optional[mx.array]] = [None] * len(lm.model.layers)
    values_list: List[Optional[mx.array]] = [None] * len(lm.model.layers)
    for i in src_layers:
        c = prompt_cache[i]
        if c is None or c.keys is None:
            return None
        k, v = c.state
        keys_list[i] = k
        values_list[i] = v
    offset = int(prompt_cache[src_layers[0]].offset)
    return FastDecodeState(keys_list, values_list, offset, src_layers)


def fast_decode_step(lm, state: FastDecodeState, token: mx.array):
    """Run one compiled decode step; update ``state`` in place; return logits."""
    fn = getattr(lm, "_fast_decode_fn", None)
    if fn is None:
        compiled_fn, src_layers = build_compiled_decode(lm)
        lm._fast_decode_fn = compiled_fn
        lm._fast_decode_src = src_layers
        fn = compiled_fn
    logits, new_keys, new_values = fn(
        token, mx.array(state.offset), state.keys_list, state.values_list
    )
    # Write the updated KV back, indexed by source-layer id.
    for j, i in enumerate(state.src_layers):
        state.keys_list[i] = new_keys[j]
        state.values_list[i] = new_values[j]
    state.offset += 1
    return LanguageModelOutput(logits=logits)
