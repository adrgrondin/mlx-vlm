from functools import lru_cache, partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import RMSNorm

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_causal_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..rope_utils import initialize_rope
from .config import TextConfig


@partial(mx.compile, shapeless=True)
def geglu(gate, x):
    return nn.gelu_approx(gate) * x


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale (with_scale=False, scale_shift=0.0)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class RMSNormZeroShift(nn.Module):
    """Gemma4 RMSNorm with scale_shift=0.0 (weight used directly, no +1 offset)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    return mx.tanh(x / softcap) * softcap


# ---------------------------------------------------------------------------
# Native quantized_matmul compiled decode path (LiteRT-LM inspired).
#
# MLX's native ``mx.quantized_matmul`` is 1.03–1.61× faster than the custom
# qmv Metal kernel for the raw matmul (benchmarked on M5 Max across all
# bit-widths and realistic sizes).  Crucially, unlike the custom ``metal_kernel``
# (opaque to ``mx.compile``), the native matmul IS compile-friendly, so
# ``mx.compile`` fuses the surrounding element-wise ops (RMSNorms, residual
# adds, gelu, SRQ activations, layer_scalar) with the matmul into a single
# compiled graph — the true LiteRT-style fusion.
#
# The SRQ (static activation quantization) that the custom kernel fuses inline
# is applied as a compile-friendly element-wise op (``_srq``) that fuses with
# the adjacent norm/residual.  The KV cache update + SDPA stay EAGER (outside
# the compiled function) to avoid the O(n²) growing-concat regression.
#
# Weight conversion (``mobile_to_mlx``) is bit-exact: the per-channel mobile
# format maps to group_size=128 with the per-channel scale broadcast and a
# constant bias of ``-shift * scale``.
# ---------------------------------------------------------------------------

_NATIVE_GROUP_SIZE = 128


def _srq(x, s):
    """Compile-friendly static fake-quantization (SRQ).

    Matches the custom qmv kernel's internal float32 SRQ:
    ``clamp(round(x / s), -128, 127) * s``.  When ``s == 0`` the SRQ is a
    no-op (returns ``x`` unchanged).  ``s`` may be a scalar or per-row array.
    """
    s = s.astype(mx.float32)
    is_zero = s == 0
    s_safe = mx.where(is_zero, mx.array(1.0, dtype=mx.float32), s)
    q = mx.clip(mx.round(x.astype(mx.float32) / s_safe), -128.0, 127.0)
    return mx.where(is_zero, x, q * s_safe)


def _qlinear_native_args(m):
    """Extract native ``mx.quantized_matmul`` arguments from a
    ``GemmaQuantizedLinear``.

    Returns ``(wq, scales, biases, in_s, out_s)`` where ``wq``, ``scales``,
    ``biases`` are for ``mx.quantized_matmul(group_size=128)`` and ``in_s``,
    ``out_s`` are SRQ scales (zero array when the layer has no SRQ).  The
    weight conversion is bit-exact and cached on the module.
    """
    from ...quantization.gemma_mobile import mobile_to_mlx

    cached = getattr(m, "_native_w", None)
    if cached is None:
        wq, scales, biases = mobile_to_mlx(
            m.weight, m.weight_scale, m.num_bits, m.input_dims
        )
        object.__setattr__(m, "_native_w", (wq, scales, biases))
    else:
        wq, scales, biases = cached

    dtype = m.weight_scale.dtype
    in_s = m.input_activation_scale if m._has_input_scale else mx.array(
        [0.0], dtype=dtype
    )
    out_s = m.output_activation_scale if m._has_output_scale else mx.array(
        [0.0], dtype=dtype
    )
    return (wq, scales, biases, in_s, out_s)


def _free_mobile_weights(model):
    """Free mobile-format weights after native conversion.

    After ``_qlinear_native_args`` has been called for all decoder layers
    (triggering lazy conversion to native ``quantized_matmul`` format), this
    function evaluates all native weights in a single ``mx.eval`` (breaking
    the lazy references to the mobile-format source arrays) and then replaces
    the mobile-format ``weight`` / ``weight_scale`` with tiny dummy arrays.

    This recovers ~2 GB of memory since the native compiled path (used for both
    prefill and decode) no longer needs the mobile-format weights.  The SRQ
    scales (``input_activation_scale``, ``output_activation_scale``) are
    preserved because the native path still reads them.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    text_model = lm.model if hasattr(lm, "model") else lm
    layers = text_model.layers

    # Collect all native weights (lazy) from all layers that use the native path.
    all_native = []
    mobile_modules = []
    for layer in layers:
        native = layer._native_args
        if not native:
            continue
        # pre_attn_args and post_attn_args contain the native weights
        # interleaved with norm weights and SRQ scales.  The native weights
        # are the ones cached as _native_w on each GemmaQuantizedLinear.
        for module in _iter_gemma_quant_linear(layer):
            cached = getattr(module, "_native_w", None)
            if cached is not None:
                all_native.extend(cached)
                mobile_modules.append(module)
            # Also handle fused qkv
        fused = getattr(layer.self_attn, "_fused_qkv_native", None)
        if fused is not None:
            all_native.extend(fused[:3])  # wq, scales, biases

    if not mobile_modules:
        return

    # Single eval to materialize all native weights (breaks lazy references
    # to the mobile-format source arrays).
    mx.eval(*all_native)

    # Now safe to free the mobile-format weights.  Use dict setitem (not
    # object.__setattr__) so the old array is dropped from the module's parameter
    # dict and actually freed from unified memory.
    for module in mobile_modules:
        w_dtype = module["weight"].dtype
        s_dtype = module["weight_scale"].dtype
        module["weight"] = mx.zeros((1,), dtype=w_dtype)
        module["weight_scale"] = mx.zeros((1,), dtype=s_dtype)


def _iter_gemma_quant_linear(layer):
    """Yield all GemmaQuantizedLinear modules in a decoder layer."""
    attn = layer.self_attn
    mlp = layer.mlp
    yield attn.q_proj
    if not attn.is_kv_shared_layer:
        yield attn.k_proj
        yield attn.v_proj
    yield attn.o_proj
    yield mlp.gate_proj
    yield mlp.up_proj
    yield mlp.down_proj
    if layer.per_layer_input_gate is not None:
        yield layer.per_layer_input_gate
        yield layer.per_layer_projection


def precompile_native_functions(model, shapes=(1, 16, 32, 64, 128, 256)):
    """Precompile native compiled functions for common prompt lengths.

    Called at load time (alongside ``precompile_gemma_mobile_kernels``) to
    eliminate the per-shape ``@mx.compile`` JIT cost that would otherwise be
    paid on the first forward pass of each unique prompt length.  Also
    triggers weight conversion and frees mobile-format weights.

    Three critical details that make this work:

    1. **Run AFTER load_weights** — ``_qlinear_native_args`` converts and
       caches native weights from ``m.weight``/``m.weight_scale``.  If called
       before ``load_weights``, the cached native weights are garbage from
       uninitialized tensors.  The caller (``utils.load_model``) ensures this.

    2. **Eval the output** — ``@mx.compile`` traces the function at call time
       but only generates and caches the Metal shaders when the output graph
       is *evaluated*.  Calling ``text_model(dummy_ids)`` without ``mx.eval``
       builds the graph lazily but never compiles.  We must capture and eval the
       output of each dummy pass.

    3. **Convert + free layer-by-layer** — Instead of materializing all native
       weights at once (which would peak at mobile + native ≈ 4.5 GB), we
       convert, eval, and free each layer's mobile weights one at a time.  This
       keeps the conversion peak at roughly half-mobile + half-native ≈ 3.2 GB.

    4. **Compile via direct function calls, not full forward passes** — Running
       ``text_model(dummy_ids)`` for each shape accumulates activations across
       all 35 layers (~0.8 GB for seq_len=256), pushing the peak to 3.9 GB.
       Instead, we call each unique compiled function directly with a tiny dummy
       input (~4 MB), triggering compilation with negligible activation memory.
       Only the first source layer and first KV-shared layer are needed (their
       compiled functions are shared across all layers via ``lru_cache``).
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    text_model = lm.model if hasattr(lm, "model") else lm
    layers = getattr(text_model, "layers", None)
    if layers is None:
        return

    # Convert, materialize, and free mobile weights layer by layer to keep
    # peak memory low (mobile + native never coexist in full).
    any_native = False
    for layer in layers:
        native = layer._get_native_args()
        if not native:
            continue
        any_native = True

        # Collect this layer's native weights (lazy) for eval.
        layer_native = []
        for module in _iter_gemma_quant_linear(layer):
            cached = getattr(module, "_native_w", None)
            if cached is not None:
                layer_native.extend(cached)
        fused = getattr(layer.self_attn, "_fused_qkv_native", None)
        if fused is not None:
            layer_native.extend(fused[:3])

        # Materialize native weights (breaks lazy refs to mobile weights).
        if layer_native:
            mx.eval(*layer_native)

        # Free this layer's mobile weights.  Use dict setitem (not
        # object.__setattr__) so the old array is dropped from the module's
        # parameter dict and actually freed from unified memory (object.__setattr__
        # only shadows the attribute via __dict__, leaving the dict item — and the
        # materialized array — alive).
        for module in _iter_gemma_quant_linear(layer):
            w_dtype = module["weight"].dtype
            s_dtype = module["weight_scale"].dtype
            module["weight"] = mx.zeros((1,), dtype=w_dtype)
            module["weight_scale"] = mx.zeros((1,), dtype=s_dtype)

    if not any_native:
        return

    # Pre-set flags so Gemma4TextModel.__call__ doesn't try to free weights
    # during the dummy passes (it frees on the 2nd forward pass).
    object.__setattr__(text_model, "_weights_converted", True)
    object.__setattr__(text_model, "_mobile_weights_freed", True)

    # Compile the @mx.compile functions for common prompt lengths.
    #
    # For small shapes (≤ 32 tokens), run the full forward pass — this warms
    # up both the compiled functions AND the MLX built-in ops (RMSNorm, SDPA,
    # RoPE, embeddings, per-layer inputs) with negligible activation memory
    # (~0.1 GB for seq_len=32).  For larger shapes, call the compiled functions
    # directly with tiny dummy inputs (~4 MB) to avoid accumulating ~0.8 GB
    # of activations across 35 layers (which pushed the peak to 3.9 GB).
    #
    # The compiled functions are shared across layers via lru_cache, so we
    # only need to call the first source layer and first KV-shared layer for
    # the direct-compile path.
    hidden_size = text_model.config.hidden_size
    per_layer_dim = getattr(text_model, "hidden_size_per_layer_input", 0)
    dtype = text_model.layers[0].input_layernorm.weight.dtype

    # Find the first source and first KV-shared layer (different compiled fns).
    first_source = None
    first_kvshared = None
    for layer in layers:
        native = layer._native_args
        if not native:
            continue
        if native[0] and first_source is None:
            first_source = layer
        elif not native[0] and first_kvshared is None:
            first_kvshared = layer
        if first_source is not None and first_kvshared is not None:
            break

    compile_layers = [l for l in (first_source, first_kvshared) if l is not None]
    dummy_offset = mx.array(0)
    # Shapes small enough that a full forward pass is cheap (< ~0.1 GB activations).
    full_pass_shapes = {s for s in shapes if s <= 32}

    for seq_len in shapes:
        if seq_len in full_pass_shapes:
            # Full forward pass: warms up compiled fns + MLX built-in ops.
            dummy_ids = mx.array([[2] * seq_len], dtype=mx.int32)
            try:
                out = text_model(dummy_ids)
                mx.eval(out)
            except Exception:
                pass
        else:
            # Direct compile: call compiled fns with tiny dummy inputs.
            dummy_x = mx.zeros((1, seq_len, hidden_size), dtype=dtype)
            dummy_residual = mx.zeros((1, seq_len, hidden_size), dtype=dtype)
            dummy_attn_out = mx.zeros((1, seq_len, hidden_size), dtype=dtype)
            dummy_pli = mx.zeros((1, seq_len, per_layer_dim), dtype=dtype)

            for layer in compile_layers:
                native = layer._native_args
                if not native:
                    continue
                _is_src, pre_fn, pre_args, post_fn, post_args = native
                try:
                    pre_out = pre_fn(dummy_x, *pre_args, dummy_offset)
                    mx.eval(pre_out)
                except Exception:
                    pass
                try:
                    post_out = post_fn(
                        dummy_residual, dummy_attn_out, dummy_pli, *post_args
                    )
                    mx.eval(post_out)
                except Exception:
                    pass


def _qlinear_args(m):
    """Extract mobile-format arguments for ``gemma_mobile_matmul`` (fallback)."""
    in_s = m.input_activation_scale if m._has_input_scale else mx.array(
        [0.0], dtype=m.weight_scale.dtype
    )
    out_s = m.output_activation_scale if m._has_output_scale else mx.array(
        [0.0], dtype=m.weight_scale.dtype
    )
    return (m.weight, m.weight_scale, m.num_bits, m.input_dims, in_s, out_s)


def _get_rope_freqs(attn):
    """Extract rope frequencies for compiled functions.

    For ``ProportionalRoPE`` (full attention), returns ``rope._freqs`` (includes
    ``inf`` for non-rotated dims via ``partial_rotary_factor``).  For ``nn.RoPE``
    (sliding attention), precomputes ``base^(arange(0, dims, 2) / dims)``.
    """
    rope = attn.rope
    freqs = getattr(rope, "_freqs", None)
    if freqs is not None:
        return freqs
    # nn.RoPE: precompute freqs (MLX uses base^(arange/dims), verified bit-exact).
    return mx.power(
        rope.base,
        mx.arange(0, rope.dims, 2, dtype=mx.float32) / rope.dims,
    )


def _build_fused_qkv_native(attn):
    """Build concatenated native-format q/k/v weights for a fused matmul.

    Returns ``(wq, scales, biases, in_s, out_s)`` where the weights are
    concatenated along the output dimension and ``out_s`` is a per-row array
    (q rows and k/v rows may have different output SRQ scales).
    """
    cached = getattr(attn, "_fused_qkv_native", None)
    if cached is not None:
        return cached

    q = _qlinear_native_args(attn.q_proj)
    k = _qlinear_native_args(attn.k_proj)
    v = _qlinear_native_args(attn.v_proj)
    q_wq, q_sc, q_bi, q_in, q_out = q
    k_wq, k_sc, k_bi, k_in, k_out = k
    v_wq, v_sc, v_bi, v_in, v_out = v

    wq = mx.concatenate([q_wq, k_wq, v_wq], axis=0)
    scales = mx.concatenate([q_sc, k_sc, v_sc], axis=0)
    biases = mx.concatenate([q_bi, k_bi, v_bi], axis=0)
    in_s = q_in  # shared input SRQ scale

    # Per-row output SRQ scale (broadcast scalars to per-row, zeros for no-SRQ).
    dtype = q_out.dtype
    parts = []
    for proj, dim, s in (
        (attn.q_proj, attn.q_proj.output_dims, q_out),
        (attn.k_proj, attn.k_proj.output_dims, k_out),
        (attn.v_proj, attn.v_proj.output_dims, v_out),
    ):
        if proj._has_output_scale:
            parts.append(mx.broadcast_to(s.astype(dtype), (dim,)))
        else:
            parts.append(mx.zeros((dim,), dtype=dtype))
    out_s = mx.concatenate(parts)

    result = (wq, scales, biases, in_s, out_s)
    object.__setattr__(attn, "_fused_qkv_native", result)
    return result


@lru_cache(maxsize=None)
def _get_compiled_pre_attn_source(n_heads, head_dim, n_kv_heads):
    """Factory: compiled pre-attention for source layers.

    ``n_heads``, ``head_dim``, ``n_kv_heads`` are captured as Python constants
    (needed for tensor slicing/reshaping inside the compiled graph).
    """
    qd = n_heads * head_dim
    kvd = n_kv_heads * head_dim

    @mx.compile
    def _fn(
        x,
        input_norm_w, input_norm_eps,
        qkv_wq, qkv_scales, qkv_biases, qkv_in_s, qkv_out_s,
        q_norm_w, q_norm_eps,
        k_norm_w, k_norm_eps,
        v_norm_w, v_norm_eps,
        rope_freqs, offset,
    ):
        h = mx.fast.rms_norm(x, input_norm_w, input_norm_eps).astype(mx.float32)
        h = _srq(h, qkv_in_s)
        qkv = mx.quantized_matmul(
            h, qkv_wq, qkv_scales, qkv_biases,
            group_size=_NATIVE_GROUP_SIZE, bits=4,
        )
        qkv = _srq(qkv, qkv_out_s).astype(x.dtype)
        B, L, _ = qkv.shape
        queries = qkv[..., :qd].reshape(B, L, n_heads, head_dim)
        keys = qkv[..., qd : qd + kvd].reshape(B, L, n_kv_heads, head_dim)
        values = qkv[..., qd + kvd :].reshape(B, L, n_kv_heads, head_dim)

        queries = mx.fast.rms_norm(queries, q_norm_w, q_norm_eps)
        keys = mx.fast.rms_norm(keys, k_norm_w, k_norm_eps)
        values = mx.fast.rms_norm(values, v_norm_w, v_norm_eps)

        keys = keys.transpose(0, 2, 1, 3)
        keys = mx.fast.rope(
            keys, head_dim, traditional=False, base=None, scale=1.0,
            offset=offset, freqs=rope_freqs,
        )
        values = values.transpose(0, 2, 1, 3)
        queries = queries.transpose(0, 2, 1, 3)
        queries = mx.fast.rope(
            queries, head_dim, traditional=False, base=None, scale=1.0,
            offset=offset, freqs=rope_freqs,
        )
        return queries, keys, values

    return _fn


@lru_cache(maxsize=None)
def _get_compiled_pre_attn_kvshared(n_heads, head_dim):
    """Factory: compiled pre-attention for KV-shared layers."""

    @mx.compile
    def _fn(
        x,
        input_norm_w, input_norm_eps,
        q_wq, q_scales, q_biases, q_in_s, q_out_s,
        q_norm_w, q_norm_eps,
        rope_freqs, offset,
    ):
        h = mx.fast.rms_norm(x, input_norm_w, input_norm_eps).astype(mx.float32)
        h = _srq(h, q_in_s)
        q = mx.quantized_matmul(
            h, q_wq, q_scales, q_biases,
            group_size=_NATIVE_GROUP_SIZE, bits=4,
        )
        q = _srq(q, q_out_s).astype(x.dtype)
        B, L, _ = q.shape
        queries = q.reshape(B, L, n_heads, head_dim)
        queries = mx.fast.rms_norm(queries, q_norm_w, q_norm_eps)
        queries = queries.transpose(0, 2, 1, 3)
        queries = mx.fast.rope(
            queries, head_dim, traditional=False, base=None, scale=1.0,
            offset=offset, freqs=rope_freqs,
        )
        return queries

    return _fn


@lru_cache(maxsize=None)
def _get_compiled_post_attn_mlp_ple(mlp_bits, ple_bits):
    """Factory: compiled post-attention + MLP + PLE.

    ``mlp_bits`` and ``ple_bits`` are captured as Python constants for
    ``mx.quantized_matmul`` specialization.
    """

    @mx.compile
    def _fn(
        residual,
        attn_output,
        per_layer_input,
        post_attn_w, post_attn_eps,
        pre_ff_w, pre_ff_eps,
        post_ff_w, post_ff_eps,
        ple_norm_w, ple_norm_eps,
        layer_scalar,
        o_wq, o_scales, o_biases, o_in_s, o_out_s,
        gate_wq, gate_scales, gate_biases, gate_in_s, gate_out_s,
        up_wq, up_scales, up_biases, up_in_s, up_out_s,
        down_wq, down_scales, down_biases, down_in_s, down_out_s,
        ple_gwq, ple_gscales, ple_gbiases, ple_gin_s, ple_gout_s,
        ple_pwq, ple_pscales, ple_pbiases, ple_pin_s, ple_pout_s,
    ):
        dt = attn_output.dtype

        # Post-attention: o_proj → norm → residual
        h = _srq(attn_output.astype(mx.float32), o_in_s)
        h = mx.quantized_matmul(
            h, o_wq, o_scales, o_biases,
            group_size=_NATIVE_GROUP_SIZE, bits=4,
        )
        h = _srq(h, o_out_s).astype(dt)
        h = mx.fast.rms_norm(h, post_attn_w, post_attn_eps)
        h = residual + h

        # MLP: pre_ff_norm → gate/up → gelu → down → post_ff_norm → residual
        residual = h
        h = mx.fast.rms_norm(h, pre_ff_w, pre_ff_eps).astype(mx.float32)
        gate = mx.quantized_matmul(
            _srq(h, gate_in_s), gate_wq, gate_scales, gate_biases,
            group_size=_NATIVE_GROUP_SIZE, bits=mlp_bits,
        )
        gate = _srq(gate, gate_out_s).astype(dt)
        up = mx.quantized_matmul(
            _srq(h, up_in_s), up_wq, up_scales, up_biases,
            group_size=_NATIVE_GROUP_SIZE, bits=mlp_bits,
        )
        up = _srq(up, up_out_s).astype(dt)
        h = nn.gelu_approx(gate) * up
        h = h.astype(mx.float32)
        down = mx.quantized_matmul(
            _srq(h, down_in_s), down_wq, down_scales, down_biases,
            group_size=_NATIVE_GROUP_SIZE, bits=mlp_bits,
        )
        down = _srq(down, down_out_s).astype(dt)
        h = mx.fast.rms_norm(down, post_ff_w, post_ff_eps)
        h = residual + h

        # Per-layer input (PLE): gate → gelu → multiply → proj → norm → residual
        residual = h
        h = h.astype(mx.float32)
        gate = mx.quantized_matmul(
            _srq(h, ple_gin_s), ple_gwq, ple_gscales, ple_gbiases,
            group_size=_NATIVE_GROUP_SIZE, bits=ple_bits,
        )
        gate = _srq(gate, ple_gout_s).astype(dt)
        gate = nn.gelu_approx(gate)
        gate = mx.multiply(gate, per_layer_input)
        gate = gate.astype(mx.float32)
        gate = mx.quantized_matmul(
            _srq(gate, ple_pin_s), ple_pwq, ple_pscales, ple_pbiases,
            group_size=_NATIVE_GROUP_SIZE, bits=ple_bits,
        )
        gate = _srq(gate, ple_pout_s).astype(dt)
        gate = mx.fast.rms_norm(gate, ple_norm_w, ple_norm_eps)
        h = residual + gate

        h = h * layer_scalar
        return h

    return _fn


class MLP(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int = 0):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    """Expert router: norm -> scale -> project -> top-k -> renormalize."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.eps = config.rms_norm_eps
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x: mx.array):
        x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)

        expert_scores = self.proj(x)

        top_k_indices = mx.argpartition(
            expert_scores, kth=-self.config.top_k_experts, axis=-1
        )
        top_k_indices = top_k_indices[..., -self.config.top_k_experts :]

        top_k_weights = mx.take_along_axis(expert_scores, top_k_indices, axis=-1)
        top_k_weights = mx.softmax(top_k_weights, axis=-1)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]

        return top_k_indices, top_k_weights


class GeGLU(nn.Module):
    """GELU-gated linear unit activation for SwitchGLU."""

    def __call__(self, x, gate):
        return geglu(gate, x)


class Experts(nn.Module):
    """Sparse MoE using SwitchGLU with gather_mm."""

    def __init__(self, config: TextConfig):
        super().__init__()
        from ..switch_layers import SwitchGLU

        self.switch_glu = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.moe_intermediate_size,
            num_experts=config.num_experts,
            activation=GeGLU(),
            bias=False,
        )

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        w = mx.expand_dims(top_k_weights, -1)
        y = self.switch_glu(x, top_k_indices)
        return (w * y).sum(-2)


class Attention(nn.Module):
    def __init__(
        self,
        config: TextConfig,
        layer_idx: int,
        kv_shared_only: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.kv_shared_only = kv_shared_only

        self.head_dim = (
            config.global_head_dim
            if self.layer_type == "full_attention"
            and hasattr(config, "global_head_dim")
            and config.global_head_dim
            else config.head_dim
        )

        dim = config.hidden_size
        self.n_heads = config.num_attention_heads

        # K-eq-V for full attention layers (26B/31B models)
        self.use_k_eq_v = (
            getattr(config, "attention_k_eq_v", False) and not self.is_sliding
        )
        if self.use_k_eq_v and config.num_global_key_value_heads is not None:
            self.n_kv_heads = config.num_global_key_value_heads
        else:
            self.n_kv_heads = config.num_key_value_heads

        self.scale = 1.0

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        if not kv_shared_only:
            self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            if not self.use_k_eq_v:
                self.v_proj = nn.Linear(
                    dim, self.n_kv_heads * self.head_dim, bias=False
                )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if not kv_shared_only:
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        # RoPE (with partial rotation support)
        layer_key = "sliding_attention" if self.is_sliding else "full_attention"
        rope_params = config.rope_parameters.get(layer_key, {})
        rope_theta = rope_params.get("rope_theta", 10000.0)

        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=config.rope_traditional,
            base=rope_theta,
            scaling_config=rope_params,
            max_position_embeddings=config.max_position_embeddings,
        )

        # KV sharing (2B/4B models)
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
        return_attn_output: bool = False,
    ) -> mx.array:
        B, L, _ = x.shape

        # P4: fused q/k/v projection for Gemma mobile quantized layers.  When
        # the three projections are gemma-quantized and share an input SRQ scale
        # (the common case), a single Metal matmul replaces three launches.
        fused_qkv = None
        if (
            shared_kv is None
            and not self.use_k_eq_v
            and getattr(self.q_proj, "mode", None) == "gemma"
        ):
            from ...quantization.gemma_mobile import gemma_fused_qkv_matmul

            fused_qkv = gemma_fused_qkv_matmul(self, x)

        if fused_qkv is not None:
            qd = self.n_heads * self.head_dim
            kvd = self.n_kv_heads * self.head_dim
            queries = fused_qkv[..., :qd].reshape(B, L, self.n_heads, self.head_dim)
            keys = fused_qkv[..., qd : qd + kvd].reshape(
                B, L, self.n_kv_heads, self.head_dim
            )
            values = fused_qkv[..., qd + kvd :].reshape(
                B, L, self.n_kv_heads, self.head_dim
            )
        else:
            queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
            if shared_kv is None:
                keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
                # k_eq_v: values from raw k_proj (before k_norm)
                if self.use_k_eq_v:
                    values = keys
                else:
                    values = self.v_proj(x).reshape(
                        B, L, self.n_kv_heads, self.head_dim
                    )

        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
        else:
            offset = mx.array(cache.offset) if cache is not None else 0

            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        if return_attn_output:
            return output, (keys, values), offset
        return self.o_proj(output), (keys, values), offset

    def attn_sdpa(self, queries, keys, values, cache, mask):
        """Eager SDPA between the compiled pre/post-attention segments.

        Takes already-transposed [B, heads, L, head_dim] queries/keys/values
        and returns the reshaped attention output [B, L, hidden].
        """
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        B = output.shape[0]
        L = output.shape[2]
        return output.transpose(0, 2, 1, 3).reshape(B, L, -1)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: TextConfig,
        layer_idx: int,
        kv_shared_only: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx, kv_shared_only=kv_shared_only)
        self.mlp = MLP(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MoE (26B model)
        self.enable_moe = getattr(config, "enable_moe_block", False)
        if self.enable_moe:
            self.router = Router(config)
            self.experts = Experts(config)
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        # Per-layer input gating (2B/4B models)
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # Layer scalar (all text layers)
        self.layer_scalar = mx.ones((1,))

        # Native quantized_matmul compiled decode path (LiteRT-LM inspired).
        # Usable when all relevant linear layers are gemma-quantized, there is
        # no MoE, and all input dims are divisible by 128.  Weights are extracted
        # lazily on first decode call.
        self._native_args = None  # cached (is_source, pre_attn_args, post_attn_args) or False

    def _get_native_args(self):
        """Lazily extract and cache native compiled-path arguments.

        Returns ``(is_source, pre_attn_args, post_attn_args)`` or ``False`` if
        the native path is not usable (non-gemma layers, MoE, or dims not
        divisible by 128).
        """
        if self._native_args is not None:
            return self._native_args

        attn = self.self_attn
        mlp = self.mlp
        ple_g = self.per_layer_input_gate
        ple_p = self.per_layer_projection

        # Check all relevant linear layers are gemma-quantized.
        post_layers = [attn.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj, ple_g, ple_p]
        if attn.is_kv_shared_layer:
            pre_layers = [attn.q_proj]
        else:
            pre_layers = [attn.q_proj, attn.k_proj, attn.v_proj]
        all_layers = post_layers + pre_layers
        if not all(getattr(m, "mode", None) == "gemma" for m in all_layers):
            self._native_args = False
            return False

        # Check all input dims divisible by 128 (MLX group_size constraint).
        all_dims = [m.input_dims for m in all_layers]
        if any(d % _NATIVE_GROUP_SIZE != 0 for d in all_dims):
            self._native_args = False
            return False

        # Post-attention: get compiled fn (specialized by bit-widths) + args.
        post_attn_fn = _get_compiled_post_attn_mlp_ple(
            mlp.gate_proj.num_bits, ple_g.num_bits
        )
        post_attn_args = (
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
            self.pre_feedforward_layernorm.weight,
            self.pre_feedforward_layernorm.eps,
            self.post_feedforward_layernorm.weight,
            self.post_feedforward_layernorm.eps,
            self.post_per_layer_input_norm.weight,
            self.post_per_layer_input_norm.eps,
            self.layer_scalar,
            *_qlinear_native_args(attn.o_proj),
            *_qlinear_native_args(mlp.gate_proj),
            *_qlinear_native_args(mlp.up_proj),
            *_qlinear_native_args(mlp.down_proj),
            *_qlinear_native_args(ple_g),
            *_qlinear_native_args(ple_p),
        )

        # Pre-attention: get compiled fn (specialized by shapes) + args.
        rope_freqs = _get_rope_freqs(attn)
        if attn.is_kv_shared_layer:
            pre_attn_fn = _get_compiled_pre_attn_kvshared(
                attn.n_heads, attn.head_dim
            )
            pre_attn_args = (
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                *_qlinear_native_args(attn.q_proj),
                attn.q_norm.weight,
                attn.q_norm.eps,
                rope_freqs,
            )
            is_source = False
        else:
            # v_norm is RMSNormNoScale (no weight) — use ones.
            v_norm = attn.v_norm
            v_norm_w = getattr(v_norm, "weight", None)
            if v_norm_w is None:
                v_norm_w = mx.ones((attn.head_dim,), dtype=attn.q_norm.weight.dtype)
            pre_attn_fn = _get_compiled_pre_attn_source(
                attn.n_heads, attn.head_dim, attn.n_kv_heads
            )
            pre_attn_args = (
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                *_build_fused_qkv_native(attn),
                attn.q_norm.weight, attn.q_norm.eps,
                attn.k_norm.weight, attn.k_norm.eps,
                v_norm_w, v_norm.eps,
                rope_freqs,
            )
            is_source = True

        self._native_args = (
            is_source, pre_attn_fn, pre_attn_args, post_attn_fn, post_attn_args
        )
        return self._native_args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ) -> mx.array:
        residual = x

        # Native quantized_matmul compiled path for both prefill (batch > 1) and
        # decode (batch = 1).  Compiles pre-attention (norm + qkv + norms + rope)
        # and post-attention (o_proj + norms + MLP + PLE + layer_scalar) using
        # MLX's native quantized_matmul (compile-friendly, faster than the custom
        # qmv/qmm kernel).  The KV cache update + SDPA stay eager between the two
        # compiled segments.  The compiled functions use @mx.compile (per-shape);
        # shapeless=True was attempted but fails because the pre-attention functions
        # read .shape for tensor slicing (MLX cannot infer slice output shapes
        # with unknown dimensions).
        if (
            not self.enable_moe
            and per_layer_input is not None
            and self.per_layer_input_gate is not None
        ):
            native = self._get_native_args()
            if native:
                (
                    is_source, pre_attn_fn, pre_attn_args,
                    post_attn_fn, post_attn_args,
                ) = native
                attn = self.self_attn

                if is_source:
                    # offset must be an mx.array (not a Python int) so that
                    # @mx.compile treats it as a runtime input, not a constant.
                    # Otherwise precompilation (cache=None → int 0) and real
                    # prefill (cache → mx.array) compile different versions.
                    offset = mx.array(cache.offset) if cache is not None else mx.array(0)
                    queries, keys, values = pre_attn_fn(
                        x, *pre_attn_args, offset
                    )
                    if cache is not None:
                        keys, values = cache.update_and_fetch(keys, values)
                    attn_output = attn.attn_sdpa(
                        queries, keys, values, cache, mask
                    )
                    shared_kv = (keys, values)
                else:
                    if offset is None:
                        offset = mx.array(0)
                    queries = pre_attn_fn(x, *pre_attn_args, offset)
                    kv = shared_kv if shared_kv is not None else (None, None)
                    attn_output = attn.attn_sdpa(
                        queries, kv[0], kv[1], None, mask
                    )

                h = post_attn_fn(
                    residual, attn_output, per_layer_input, *post_attn_args
                )
                return h, shared_kv, offset

        # Eager path (MoE, non-gemma-quantized, or no PLE)
        h = self.input_layernorm(x)
        h, shared_kv, offset = self.self_attn(
            h, mask, cache, shared_kv=shared_kv, offset=offset
        )
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h

        if self.enable_moe:
            h1 = self.pre_feedforward_layernorm(h)
            h1 = self.mlp(h1)
            h1 = self.post_feedforward_layernorm_1(h1)

            top_k_indices, top_k_weights = self.router(h)
            h2 = self.pre_feedforward_layernorm_2(h)
            h2 = self.experts(h2, top_k_indices, top_k_weights)
            h2 = self.post_feedforward_layernorm_2(h2)

            h = h1 + h2
        else:
            h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)

        h = self.post_feedforward_layernorm(h)
        h = residual + h

        # Per-layer input gating
        if (
            self.per_layer_input_gate is not None
            and self.per_layer_projection is not None
            and self.post_per_layer_input_norm is not None
            and per_layer_input is not None
        ):
            residual = h
            gate = self.per_layer_input_gate(h)
            gate = nn.gelu_approx(gate)
            gate = mx.multiply(gate, per_layer_input)
            gate = self.per_layer_projection(gate)
            gate = self.post_per_layer_input_norm(gate)
            h = residual + gate

        if self.layer_scalar is not None:
            h = h * self.layer_scalar

        return h, shared_kv, offset


class Gemma4TextModel(nn.Module):
    def __init__(self, config: TextConfig, kv_shared_only: bool = False):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        first_kv_shared = config.num_hidden_layers - num_kv_shared
        self.layers = [
            DecoderLayer(
                config,
                layer_idx=i,
                kv_shared_only=kv_shared_only
                or (num_kv_shared > 0 and i >= first_kv_shared),
            )
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.first_kv_shared_layer_idx = first_kv_shared
        self.previous_kvs = list(range(len(self.layers)))
        if num_kv_shared > 0:
            N = len(self.layers)
            M = N - num_kv_shared
            kvs_by_type = {}
            for i in range(M):
                kvs_by_type[self.layers[i].layer_type] = i
            for j in range(M, N):
                self.previous_kvs[j] = kvs_by_type[self.layers[j].layer_type]

        # Per-layer input embeddings (2B/4B models)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            self.embed_tokens_per_layer_scale = config.hidden_size_per_layer_input**0.5
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_projection_scale = config.hidden_size**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = RMSNormZeroShift(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        result = self.embed_tokens_per_layer(input_ids)
        result = result * self.embed_tokens_per_layer_scale
        return result.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: mx.array,
        per_layer_inputs: Optional[mx.array] = None,
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def _block_sequence_ids_for_mask(self, mm_token_type_ids: mx.array) -> mx.array:
        is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
        prev = mx.concatenate(
            [
                mx.zeros_like(is_vision[:, :1]),
                is_vision[:, :-1],
            ],
            axis=1,
        )
        starts = is_vision & ~prev
        group_ids = mx.cumsum(starts.astype(mx.int32), axis=1) - 1
        return mx.where(is_vision, group_ids, mx.zeros_like(group_ids) - 1)

    def _apply_blockwise_bidirectional_overlay(
        self,
        base_mask: mx.array,
        mm_token_type_ids: mx.array,
    ) -> mx.array:
        if mm_token_type_ids is None:
            return base_mask
        if mm_token_type_ids.shape[1] != base_mask.shape[-1]:
            return base_mask
        if base_mask.shape[-2] != base_mask.shape[-1]:
            return base_mask

        block_sequence_ids = self._block_sequence_ids_for_mask(mm_token_type_ids)
        q_blocks = mx.expand_dims(block_sequence_ids, -1)
        k_blocks = mx.expand_dims(block_sequence_ids, -2)
        same_block = (q_blocks != -1) & (q_blocks == k_blocks)
        return base_mask | mx.expand_dims(same_block, 1)

    def _make_masks(self, h, cache, mm_token_type_ids: Optional[mx.array] = None):
        """Create attention masks, deduplicated by layer type."""
        mask = {}
        masks = []
        has_visual_tokens = (
            mm_token_type_ids is not None
            and int(mx.sum((mm_token_type_ids == 1) | (mm_token_type_ids == 2)).item())
            > 0
        )
        # Audio spans are sequential; keep mixed image+audio prompts causal to
        # avoid the vision block overlay dominating quantized unified models.
        use_bidirectional_vision = (
            getattr(self.config, "use_bidirectional_attention", None) == "vision"
            and mm_token_type_ids is not None
            and has_visual_tokens
            and h.shape[1] > 1
        )
        for l, c in zip(self.layers, cache):
            if l.layer_type not in mask:
                if l.layer_type == "full_attention":
                    # Full attention can use MLX's causal mask even when
                    # prefilling against an existing KV prefix. Only materialize
                    # a mask for batch left-padding or the Gemma 4 vision
                    # bidirectional overlay.
                    return_array = (
                        use_bidirectional_vision
                        or getattr(c, "left_padding", None) is not None
                    )
                    mask["full_attention"] = create_attention_mask(
                        h, c, return_array=return_array
                    )
                elif l.layer_type == "sliding_attention":
                    return_array = (
                        h.shape[1] > 1
                        and c is not None
                        and int(mx.max(mx.array(c.offset)).item()) > 0
                    ) or use_bidirectional_vision
                    mask["sliding_attention"] = create_attention_mask(
                        h, c, window_size=self.window_size, return_array=return_array
                    )
                if (
                    use_bidirectional_vision
                    and isinstance(mask[l.layer_type], str)
                    and mask[l.layer_type] == "causal"
                ):
                    window = (
                        self.window_size
                        if l.layer_type == "sliding_attention"
                        else None
                    )
                    mask[l.layer_type] = create_causal_mask(
                        h.shape[1], window_size=window
                    )
                if use_bidirectional_vision and isinstance(
                    mask[l.layer_type], mx.array
                ):
                    mask[l.layer_type] = self._apply_blockwise_bidirectional_overlay(
                        mask[l.layer_type],
                        mm_token_type_ids,
                    )
            masks.append(mask[l.layer_type])
        return masks

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        per_layer_inputs: Optional[mx.array] = None,
        mm_token_type_ids: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_sink: Optional[list] = None,
        shared_kv_sink: Optional[dict] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
            h = h * self.embed_scale
        else:
            h = inputs_embeds

        # Per-layer inputs (2B/4B models)
        if self.hidden_size_per_layer_input:
            if inputs is not None and per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(inputs)
            elif per_layer_inputs is not None:
                target_len = h.shape[1]
                if per_layer_inputs.shape[1] != target_len:
                    cache_offset = next(
                        (
                            int(c.offset)
                            for c in (cache or [])
                            if c is not None and hasattr(c, "offset")
                        ),
                        0,
                    )
                    max_start = max(per_layer_inputs.shape[1] - target_len, 0)
                    start = min(cache_offset, max_start)
                    per_layer_inputs = per_layer_inputs[:, start : start + target_len]
            if per_layer_inputs is not None or inputs is not None:
                per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        # Build cache + masks
        if cache is None:
            cache = [None] * len(self.layers)
        else:
            cache = cache + [None] * (len(self.layers) - len(cache))

        if mask is None:
            if mm_token_type_ids is None:
                mm_token_type_ids = token_type_ids
            masks = self._make_masks(h, cache, mm_token_type_ids)
        else:
            masks = [mask] * len(self.layers)

        # Forward through layers
        if per_layer_inputs is not None:
            per_layer_inputs = [
                per_layer_inputs[:, :, i, :] for i, _ in enumerate(self.layers)
            ]
        else:
            per_layer_inputs = [None] * len(self.layers)

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        intermediates = [(None, None)] * len(self.layers)
        for idx, (layer, c, m, prev_idx, pli) in enumerate(
            zip(
                self.layers,
                cache,
                masks,
                self.previous_kvs,
                per_layer_inputs,
            )
        ):
            kvs, offset = intermediates[prev_idx]
            h, kvs, offset = layer(
                h, m, c, per_layer_input=pli, shared_kv=kvs, offset=offset
            )
            intermediates[idx] = (kvs, offset)
            if hidden_sink is not None and idx in capture_set:
                hidden_sink.append(h)

        # Weight conversion is lazy (triggered by _get_native_args inside
        # each layer).  We free the mobile-format weights on the SECOND forward
        # pass (first decode step), not the first (prefill), so the prefill
        # has zero mx.eval overhead.  By the second pass, the prefill's matmuls
        # have already evaluated the native weights, so the mx.eval inside
        # _free_mobile_weights is a no-op.
        if getattr(self, "_weights_converted", False):
            if not getattr(self, "_mobile_weights_freed", False):
                _free_mobile_weights(self)
                object.__setattr__(self, "_mobile_weights_freed", True)
        else:
            object.__setattr__(self, "_weights_converted", True)

        if shared_kv_sink is not None:
            for idx, layer in enumerate(self.layers):
                kvs, _ = intermediates[idx]
                if kvs is not None:
                    shared_kv_sink[layer.layer_type] = kvs

        # Match HF's `_can_record_outputs={"hidden_states": Gemma4TextDecoderLayer}`
        # — the recorded value is the LAST decoder layer's output, captured
        # BEFORE the final RMSNorm. Speculative verification can reuse this
        # hidden for deferred logits; MTP drafters normalize it via
        # LanguageModel.speculative_draft_hidden before consuming it.
        if hidden_sink is not None and not capture_set:
            hidden_sink.append(h)

        if kwargs.pop("skip_final_norm", False):
            return h

        h = self.norm(h)

        return h


class LanguageModel(nn.Module):
    supports_logits_to_keep = True

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma4TextModel(config)
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)
        # Untied output head (Gemma 4 QAT mobile checkpoints set
        # ``tie_word_embeddings: false`` and ship a separate 2-bit ``lm_head``).
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Lazily built compiled decode fast-path (LiteRT-LM inspired; opt-in via
        # MLX_VLM_FAST_DECODE=1).  See models/gemma4/fast_decode.py.
        self._fast_decode_fn = None
        self._fast_decode_src = None

    def init_fast_decode(self, prompt_cache):
        """Extract the flattened KV state from the eager cache after prefill.

        Returns a ``FastDecodeState`` (or ``None`` if the fast path is not
        usable).  Used by ``generate_step`` when ``MLX_VLM_FAST_DECODE=1``.
        """
        from .fast_decode import init_fast_decode

        return init_fast_decode(self, prompt_cache)

    def fast_decode_step(self, state, token: mx.array):
        """Run one compiled single-token decode step; update ``state`` in place."""
        from .fast_decode import fast_decode_step

        return fast_decode_step(self, state, token)

    def logits_from_hidden(self, hidden: mx.array) -> mx.array:
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        if self.final_logit_softcapping is not None:
            logits = logit_softcap(self.final_logit_softcapping, logits)
        return logits

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.logits_from_hidden(self.model.norm(hidden))

    def speculative_draft_hidden(self, hidden: mx.array) -> mx.array:
        return self.model.norm(hidden)

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache
        prefill_kwargs = prefill_kwargs or {}
        if getattr(self, "no_chunked_prefill", False):
            return False

        token_types = prefill_kwargs.get("mm_token_type_ids", None)
        if token_types is None:
            token_types = prefill_kwargs.get("token_type_ids", None)
        if (
            getattr(self.config, "use_bidirectional_attention", None) == "vision"
            and token_types is not None
        ):
            has_visual = int(mx.sum((token_types == 1) | (token_types == 2)).item()) > 0
            has_audio = int(mx.sum(token_types == 3).item()) > 0
            if has_visual and not has_audio:
                return False

        if draft_model is not None:
            return (
                draft_kind == "mtp"
                and bool(prefill_kwargs.get("return_hidden", False))
                and bool(prefill_kwargs.get("return_shared_kv", False))
            )

        return True

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        per_layer_inputs: Optional[mx.array] = None,
        capture_layer_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        hidden_sink: Optional[list] = (
            []
            if capture_layer_ids is not None or kwargs.pop("return_hidden", False)
            else None
        )
        shared_kv_sink: Optional[dict] = (
            {} if kwargs.pop("return_shared_kv", False) else None
        )
        # Allow callers to pass pre-allocated sinks directly.
        hidden_sink = kwargs.pop("hidden_sink", hidden_sink)
        shared_kv_sink = kwargs.pop("shared_kv_sink", shared_kv_sink)
        logits_to_keep = kwargs.pop("logits_to_keep", None)

        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            per_layer_inputs=per_layer_inputs,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
            shared_kv_sink=shared_kv_sink,
            **kwargs,
        )
        if logits_to_keep:
            out = out[:, -int(logits_to_keep) :, :]
        out = self.logits_from_hidden(out)
        return LanguageModelOutput(
            logits=out,
            hidden_states=hidden_sink,
            shared_kv_states=shared_kv_sink,
        )

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        """Rewind target KV caches after a speculative-decoding round.

        Gemma 4 has only KV/RotatingKV caches (no SSM/GDN), so this is a
        simple trim + per-row tail-zero. ``gdn_states`` is accepted (and
        ignored) for API parity with qwen3_5's hook.
        """
        del gdn_states  # API-parity placeholder; Gemma 4 has no SSM/GDN state.
        if isinstance(accepted, int):
            accepted = mx.array([accepted])
        if isinstance(accepted, (list, tuple)):
            accepted = mx.array(accepted, dtype=mx.int32)

        max_a = int(accepted.max().item())
        n = max_a + 1
        trim = block_size - n
        is_batch = accepted.size > 1
        valid_ends = accepted + 1

        for c in caches:
            if c is None:
                continue

            if trim > 0 and hasattr(c, "trim"):
                c.trim(trim)
            if is_batch and hasattr(c, "_idx") and c.keys is not None and max_a > 0:
                kv_len = c._idx
                ve = valid_ends.tolist()
                verify_start = kv_len - n
                for bi in range(accepted.shape[0]):
                    start = verify_start + int(ve[bi])
                    if start < kv_len:
                        zero_row_tail = getattr(c, "zero_row_tail", None)
                        if callable(zero_row_tail):
                            zero_row_tail(bi, start, kv_len)
                        else:
                            c.keys[bi, :, start:kv_len, :] = 0
                            c.values[bi, :, start:kv_len, :] = 0
        return max_a

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "self_attn.rotary_emb" in k:
                continue
            if self._is_unused_shared_kv_weight(k):
                continue
            if any(
                s in k for s in ["input_max", "input_min", "output_max", "output_min"]
            ):
                if "vision_tower" not in k and "audio_tower" not in k:
                    continue
            sanitized[k] = v
        return sanitized

    def _is_unused_shared_kv_weight(self, key: str) -> bool:
        prefix = "language_model.model.layers."
        if not key.startswith(prefix):
            return False

        parts = key[len(prefix) :].split(".")
        if len(parts) < 4 or parts[1] != "self_attn":
            return False

        try:
            layer_idx = int(parts[0])
        except ValueError:
            return False
        if layer_idx >= len(self.model.layers):
            return False

        attn = self.model.layers[layer_idx].self_attn
        if not getattr(attn, "is_kv_shared_layer", False):
            return False

        return parts[2] in {"k_proj", "v_proj", "k_norm", "v_norm"}

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    @property
    def quant_predicate(self):
        def predicate(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if "router" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        # Gemma 4 QAT mobile: use the precomputed static KV-cache scales for a
        # hybrid 4-bit (global) / 8-bit (local) quantized cache when present.
        kv_scales = getattr(self.model, "_gemma_kv_scales", None)
        if kv_scales:
            from ...quantization.gemma_mobile_cache import (
                build_gemma_static_caches,
            )

            return build_gemma_static_caches(
                self.config.layer_types,
                kv_scales,
                self.config.sliding_window,
                num_kv_shared_layers=getattr(
                    self.config, "num_kv_shared_layers", 0
                ),
                num_hidden_layers=self.config.num_hidden_layers,
            )
        caches = []
        for layer_type in self.config.layer_types[
            : self.model.first_kv_shared_layer_idx
        ]:
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(
                        max_size=self.config.sliding_window, keep=0))
        return caches
