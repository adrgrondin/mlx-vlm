import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .utils import (
    MODEL_CONVERSION_DTYPES,
    create_model_card,
    fetch_from_hub,
    get_model_path,
    load_config,
    load_processor,
    make_shards,
    save_config,
    save_weights,
    skip_multimodal_module,
    upload_to_hub,
)

QUANT_RECIPES = [
    "mixed_2_6",
    "mixed_3_4",
    "mixed_3_5",
    "mixed_3_6",
    "mixed_3_8",
    "mixed_4_6",
    "mixed_4_8",
]

# Gemma 4 QAT mobile (wNa8o8) conversion modes, selected via --quant-predicate.
GEMMA_MOBILE_MODES = ["gemma_mobile", "gemma_mobile_ptq"]


def _resolve_compute_dtype(dtype, config):
    """Resolve the MLX compute/save dtype for the gemma mobile conversion."""
    if dtype is not None:
        return getattr(mx, dtype)
    src = (
        config.get("text_config", {}).get("dtype")
        or config.get("dtype")
        or "float16"
    )
    return getattr(mx, src, mx.float16)


def _load_raw_weights(model_path):
    """Load every ``*.safetensors`` file in ``model_path`` into one dict."""
    weight_files = sorted(
        wf for wf in glob.glob(str(model_path / "*.safetensors"))
    )
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    return weights


def _save_weights_dict(save_path, weights):
    """Save a flat ``{key: mx.array}`` dict as sharded safetensors + index."""
    save_path.mkdir(parents=True, exist_ok=True)
    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )
    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}
    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        mx.save_safetensors(
            str(save_path / shard_name), shard, metadata={"format": "mlx"}
        )
        for weight_name in shard:
            index_data["weight_map"][weight_name] = shard_name
    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }
    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)


def _gemma_mobile_num_layers(target) -> int:
    """Best-effort layer count for selecting the E2B/E4B PTQ config."""
    layers = getattr(target, "layers", None)
    if layers is None:
        lm = getattr(target, "language_model", None)
        layers = getattr(lm, "layers", None) if lm is not None else None
    return len(layers) if layers is not None else 0


def _convert_gemma_mobile(
    model_path: Path,
    mlx_path: Path,
    mode: str,
    text_only: bool,
    dtype: Optional[str],
    upload_repo: Optional[str],
    trust_remote_code: bool,
    revision: Optional[str],
):
    """Convert a Gemma 4 checkpoint to the MLX QAT mobile (wNa8o8) format.

    ``mode="transcode"``: lossless remap of an HF ``-mobile-transformers``
    checkpoint (packed weights preserved bit-exact). Recommended — the QAT
    checkpoints are already trained for this format.
    ``mode="ptq"``: post-training quantization of an *unquantized* Gemma 4
    checkpoint into the mobile format. Lower quality than QAT; useful for
    custom fine-tunes.
    """
    from .quantization.gemma_mobile import replace_with_gemma_quant_layers
    from .quantization.gemma_mobile_quantize import (
        cast_gemma_mobile_weights,
        is_gemma_mobile_checkpoint,
        promote_text_config,
        quantize_model_gemma_mobile,
        select_mobile_quant_config,
        transcode_gemma_mobile_weights,
    )

    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)
    config = load_config(
        model_path, trust_remote_code=trust_remote_code, revision=revision
    )
    compute_dtype = _resolve_compute_dtype(dtype, config)

    if mode == "transcode":
        qc = config.get("quantization_config") or config.get("text_config", {}).get(
            "quantization_config"
        )
        if qc is None or qc.get("quant_method") != "gemma":
            raise ValueError(
                "Transcode requires a source with quant_method='gemma' "
                "(an HF -mobile-transformers checkpoint). Use gemma_mobile_ptq "
                "to quantize an unquantized Gemma 4."
            )
        print("[INFO] Transcoding Gemma 4 QAT mobile checkpoint (lossless)")
        weights = _load_raw_weights(model_path)
        if not is_gemma_mobile_checkpoint(weights):
            raise ValueError("Source does not look like a Gemma mobile checkpoint.")
        weights = transcode_gemma_mobile_weights(weights, text_only=text_only)
        weights = cast_gemma_mobile_weights(weights, compute_dtype)
        if text_only:
            config = promote_text_config(config)
        _save_weights_dict(mlx_path, weights)
        processor = None  # loaded from the target after files + config are saved
    else:  # ptq
        print("[INFO] PTQ-quantizing Gemma 4 into the mobile format")
        model, config, _ = fetch_from_hub(
            model_path, lazy=True, trust_remote_code=trust_remote_code
        )
        processor = None
        target = (
            model.language_model._model
            if getattr(model, "_is_text_model", False)
            else model
        )
        num_layers = _gemma_mobile_num_layers(target)
        qc = select_mobile_quant_config(num_layers)
        new_weights, _ = quantize_model_gemma_mobile(
            target, qc, dtype=compute_dtype
        )
        target = replace_with_gemma_quant_layers(
            target, qc, new_weights, dtype=compute_dtype
        )
        target.load_weights(list(new_weights.items()), strict=False)
        mx.eval(target.parameters())
        config["quantization_config"] = qc
        config["quantization"] = qc
        save_weights(mlx_path, target, donate_weights=True)

    # Copy Python and JSON files (except the index, already regenerated).
    # For text-only transcode, skip the multimodal image/audio processor configs
    # so the target loads as a clean text-only (tokenizer-only) checkpoint.
    _text_only_skip = {"preprocessor_config.json", "processor_config.json"}
    for pattern in ["*.py", "*.json", "*.jinja"]:
        for file in glob.glob(str(model_path / pattern)):
            fname = Path(file).name
            if fname == "model.safetensors.index.json":
                continue
            if text_only and mode == "transcode" and fname in _text_only_skip:
                continue
            shutil.copy(file, mlx_path)
    # Copy folders (tokenizer assets, etc.).
    for item in model_path.iterdir():
        if item.is_dir():
            dest = mlx_path / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    save_config(config, mlx_path / "config.json")

    # For text-only transcode, strip the multimodal ``processor_class`` from
    # ``tokenizer_config.json`` so ``AutoProcessor`` resolves to a
    # tokenizer-only processor instead of the full ``Gemma4Processor`` (which
    # requires a vision/audio feature extractor that we deliberately dropped).
    if text_only and mode == "transcode":
        tok_cfg_path = mlx_path / "tokenizer_config.json"
        if tok_cfg_path.exists():
            with open(tok_cfg_path) as f:
                tok_cfg = json.load(f)
            if "processor_class" in tok_cfg:
                tok_cfg.pop("processor_class")
                with open(tok_cfg_path, "w") as f:
                    json.dump(tok_cfg, f, indent=2)

    # Load the processor from the (now complete) target so a text-only transcode
    # resolves to a tokenizer-only processor, then re-save it in canonical form.
    processor = load_processor(
        mlx_path,
        add_detokenizer=False,
        eos_token_ids=config.get("eos_token_id", None),
        trust_remote_code=trust_remote_code,
    )
    processor.save_pretrained(mlx_path)

    hf_repo = None if Path(str(model_path)).exists() else str(model_path)
    create_model_card(mlx_path, hf_repo)
    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo)


def _quantization_params(
    q_group_size: Optional[int], q_bits: Optional[int], q_mode: str
):
    mode_defaults = {
        "affine": (64, 4),
        "mxfp4": (32, 4),
        "nvfp4": (16, 4),
        "mxfp8": (32, 8),
    }
    group_size, bits = mode_defaults[q_mode]
    return {
        "group_size": q_group_size or group_size,
        "bits": q_bits or bits,
        "mode": q_mode,
    }


def _preserve_existing_deepseek_v4_quantization(
    config: dict,
    model: nn.Module,
    q_group_size: Optional[int],
    q_bits: Optional[int],
    q_mode: str,
):
    quantization_config = config.get("quantization_config", {})
    if (
        config.get("model_type") != "deepseek_v4"
        or "quantization" in config
        or not isinstance(quantization_config, dict)
        or quantization_config.get("quant_method") != "fp8"
    ):
        return

    from .models.deepseek_v4.language import make_quantization_config

    quantization = make_quantization_config(model)
    quantization.update(_quantization_params(q_group_size, q_bits, q_mode))
    config["quantization"] = quantization
    config["quantization_config"] = quantization


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    group_size = 64

    recipe_config = {
        "mixed_2_6": (2, 6),
        "mixed_3_4": (3, 4),
        "mixed_3_5": (3, 5),
        "mixed_3_6": (3, 6),
        "mixed_3_8": (3, 8),
        "mixed_4_6": (4, 6),
        "mixed_4_8": (4, 8),
    }

    if recipe not in recipe_config:
        raise ValueError(f"Invalid quant recipe {recipe}")

    low_bits, high_bits = recipe_config[recipe]

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """

        if skip_multimodal_module(path):
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[1] % group_size != 0:
            return False

        path_parts = path.split(".")
        index = 0

        if len(path_parts) > layer_location:
            element = path_parts[layer_location]
            if element.isdigit():
                index = int(element)

        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )

        if use_more_bits and ("v_proj" in path or "down_proj" in path):
            return {"group_size": group_size, "bits": high_bits}

        if "lm_head" in path or "embed_tokens" in path:
            return {"group_size": group_size, "bits": high_bits}

        return {"group_size": group_size, "bits": low_bits}

    return mixed_quant_predicate


def _has_decoder(module):
    if module is None:
        return False
    for _, sub in module.named_modules():
        if getattr(sub, "self_attn", None) is not None and getattr(sub, "mlp", None):
            return True
    return False


def _build_multimodal_awq_run(model, processor, config, calibration_data):
    """Build a calibration forward that routes media+text through the full model.

    Returns ``(run, n_samples)``, or ``(None, 0)`` when the model has no
    vision/audio modality so the caller falls back to text calibration.
    """
    from .prompt_utils import apply_chat_template
    from .quant import (
        load_calibration_media,
        synthetic_calibration_audio,
        synthetic_calibration_images,
    )
    from .utils import prepare_inputs

    has_vision = bool(config.get("vision_config")) or (
        getattr(model, "vision_tower", None) is not None
    )
    has_audio = bool(config.get("audio_config")) or (
        getattr(model, "audio_tower", None) is not None
    )
    if not has_vision and not has_audio:
        return None, 0

    if calibration_data:
        images, audios = load_calibration_media(calibration_data)
    else:
        images = synthetic_calibration_images(8) if has_vision else []
        audios = synthetic_calibration_audio(8) if has_audio else []
        print(
            "[INFO] AWQ: using synthetic calibration media; pass "
            "--calibration-data for real image/audio samples."
        )

    samples = [(im, None, 1, 0) for im in images]
    samples += [(None, au, 0, 1) for au in audios]
    if not samples:
        return None, 0

    cfg = model.config

    def run():
        for image, audio, n_img, n_aud in samples:
            prompt = (
                "Describe this image in detail."
                if n_img
                else "Describe what you hear in this audio."
            )
            formatted = apply_chat_template(
                processor, cfg, prompt, num_images=n_img, num_audios=n_aud
            )
            inputs = prepare_inputs(
                processor,
                images=[image] if image is not None else None,
                audio=[audio] if audio is not None else None,
                prompts=formatted,
                image_token_index=getattr(cfg, "image_token_index", None),
                add_special_tokens=False,
                pad_to_uniform_size=False,
            )
            extra = {
                k: v
                for k, v in inputs.items()
                if k not in ("input_ids", "pixel_values", "attention_mask")
            }
            mx.eval(
                model(
                    inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    mask=inputs.get("attention_mask"),
                    **extra,
                )
            )

    return run, len(samples)


def _apply_awq_calibration(
    model,
    processor,
    config,
    target,
    q_bits,
    q_group_size,
    calibration="text",
    calibration_data=None,
):
    """Calibrate (text or multimodal) and apply AWQ scaling to the decoder."""
    from .quant import DEFAULT_CALIBRATION_TEXT, apply_awq, collect_activation_stats

    tokenizer = getattr(processor, "tokenizer", processor)
    language_model = getattr(model, "language_model", None)
    root = language_model if _has_decoder(language_model) else target

    if calibration == "multimodal":
        run, n = _build_multimodal_awq_run(model, processor, config, calibration_data)
        if run is not None:
            print(f"[INFO] AWQ: multimodal calibration on {n} media samples.")
            stats = collect_activation_stats(root, run)
            summary = apply_awq(
                root, stats, bits=q_bits or 4, group_size=q_group_size or 64
            )
            print(f"[INFO] AWQ scaling applied: {summary}")
            return
        print("[INFO] AWQ: no vision/audio modality found; using text calibration.")

    probe = mx.array([tokenizer.encode(DEFAULT_CALIBRATION_TEXT[0])])
    forward = None
    for candidate in (getattr(root, "model", None), root, language_model):
        if candidate is None:
            continue
        try:
            mx.eval(candidate(probe))
            forward = candidate
            break
        except Exception:
            continue
    if forward is None:
        raise RuntimeError("Could not run a calibration forward pass for AWQ.")

    def run():
        for text in DEFAULT_CALIBRATION_TEXT:
            mx.eval(forward(mx.array([tokenizer.encode(text)])))

    stats = collect_activation_stats(root, run)
    summary = apply_awq(root, stats, bits=q_bits or 4, group_size=q_group_size or 64)
    print(f"[INFO] AWQ scaling applied: {summary}")


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    q_mode: str = "affine",
    quant_method: str = "rtn",
    calibration: str = "text",
    calibration_data: Optional[str] = None,
    dtype: Optional[str] = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    trust_remote_code: bool = True,
    quant_predicate: Optional[str] = None,
    gemma_mobile_text_only: bool = True,
):
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)

    # Gemma 4 QAT mobile (wNa8o8) conversion: transcode (lossless) or PTQ.
    # These are distinct from the affine nn.quantize path below, so handle them
    # before instantiating the full model.
    if quant_predicate in GEMMA_MOBILE_MODES:
        _convert_gemma_mobile(
            model_path,
            mlx_path,
            mode="ptq" if quant_predicate == "gemma_mobile_ptq" else "transcode",
            text_only=gemma_mobile_text_only,
            dtype=dtype,
            upload_repo=upload_repo,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        return

    model, config, processor = fetch_from_hub(
        model_path, lazy=True, trust_remote_code=trust_remote_code
    )

    model_quant_predicate = getattr(model, "quant_predicate", None)

    def base_quant_predicate(path, module):
        if skip_multimodal_module(path):
            return False
        if model_quant_predicate is not None:
            return model_quant_predicate(path, module)
        return True

    # TODO: Remove once all LM models are migrated
    # Text-only models wrap the real mlx-lm model under `language_model._model`.
    # nn.Module.parameters() can't reach that underscore child, so dtype-cast,
    # quantization, and save_weights must operate on the inner model -- the same
    # one load_model quantizes (see utils.load_model). Otherwise convert writes
    # an empty safetensors and mixed-bit per-layer keys don't match on reload.
    target = (
        model.language_model._model
        if getattr(model, "_is_text_model", False)
        else model
    )

    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(quant_predicate, target)

    quant_predicate = quant_predicate or base_quant_predicate

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype is None and (text_config := config.get("text_config", None)):
        dtype = text_config.get("dtype", None)
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def set_dtype(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(dtype)
            else:
                return v

        target.update(tree_map_with_path(set_dtype, target.parameters()))

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        from .quant_utils import quantize_model

        _preserve_existing_deepseek_v4_quantization(
            config, target, q_group_size, q_bits, q_mode
        )

        if quant_method == "awq":
            print("[INFO] Calibrating (AWQ)")
            _apply_awq_calibration(
                model,
                processor,
                config,
                target,
                q_bits,
                q_group_size,
                calibration=calibration,
                calibration_data=calibration_data,
            )

        print("[INFO] Quantizing")
        config.setdefault("vision_config", {})
        target, config = quantize_model(
            target,
            config,
            q_group_size,
            q_bits,
            mode=q_mode,
            quant_predicate=quant_predicate,
        )

    if dequantize:
        from .quant_utils import dequantize_model

        print("[INFO] Dequantizing")
        target = dequantize_model(target)

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    save_weights(mlx_path, target, donate_weights=True)

    # Copy Python and JSON files from the model path to the MLX path
    for pattern in ["*.py", "*.json"]:
        files = glob.glob(str(model_path / pattern))
        for file in files:
            # Skip the index file - save_weights() already generated the correct one
            if Path(file).name == "model.safetensors.index.json":
                continue
            shutil.copy(file, mlx_path)

    # Copy folders from the model path to the MLX path
    for item in model_path.iterdir():
        if item.is_dir():
            dest = mlx_path / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    # Not every remote-code processor inherits ProcessorMixin — Mage-VL's `MageVLProcessor`
    # deliberately does not ("We deliberately do NOT inherit transformers.ProcessorMixin"), so it
    # has no save_pretrained. The weights are already written by this point; losing the whole
    # conversion to the sidecar-copy step would be absurd. Fall back to copying the processor
    # files verbatim, which is what save_pretrained would have produced anyway.
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(mlx_path)
    else:
        # NOTE: no local `import shutil` here — convert.py already imports it at module scope,
        # and a function-local import would make the name local for the WHOLE function, unbinding
        # the earlier uses. That failure reads as "cannot access local variable 'shutil'".
        src = Path(hf_path)
        if src.is_dir():
            for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
                for f in src.glob(pattern):
                    if f.name not in ("config.json", "model.safetensors.index.json"):
                        shutil.copy2(f, Path(mlx_path) / f.name)
        print("[INFO] processor lacks save_pretrained; copied processor files verbatim")

    save_config(config, config_path=mlx_path / "config.json")

    hf_repo = None if Path(hf_path).exists() else hf_path
    create_model_card(mlx_path, hf_repo)

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )
    parser.add_argument(
        "--hf-path",
        "--model",
        type=str,
        help="Path to the model. This can be a local path or a Hugging Face Hub model identifier.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="Hugging Face revision (branch), when converting a model from the Hub.",
        default=None,
    )
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-mode",
        help="The quantization mode.",
        type=str,
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        default="affine",
    )
    parser.add_argument(
        "--quant-method",
        help="Weight quantization method.",
        type=str,
        choices=["rtn", "awq"],
        default="rtn",
    )
    parser.add_argument(
        "--calibration",
        help="AWQ calibration inputs: text (default) or multimodal (image/audio+text).",
        type=str,
        choices=["text", "multimodal"],
        default="text",
    )
    parser.add_argument(
        "--calibration-data",
        help="Optional directory of real images/audio for --calibration multimodal.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameter. Defaults to config.json's `torch_dtype` or the current model weights dtype",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
    )
    parser.add_argument(
        "--quant-predicate",
        help=(
            "Mixed-bit quantization recipe, or a Gemma 4 QAT mobile mode: "
            "'gemma_mobile' (lossless transcode of an HF -mobile-transformers "
            "checkpoint) or 'gemma_mobile_ptq' (post-training quantization of "
            "an unquantized Gemma 4)."
        ),
        choices=QUANT_RECIPES + GEMMA_MOBILE_MODES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--gemma-mobile-text-only",
        help=(
            "For gemma_mobile transcode: extract a text-only checkpoint (drop "
            "vision/audio encoders). Default: True."
        ),
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        help="Trust remote code.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_vlm.convert ...` directly is deprecated."
        " Use `mlx_vlm.convert ...` or `python -m mlx_vlm convert ...` instead."
    )
    main()
