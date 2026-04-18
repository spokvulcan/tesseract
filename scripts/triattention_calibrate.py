#!/usr/bin/env python3
"""Generate a TriAttention calibration artifact for a Qwen3.5-PARO checkpoint.

The artifact is a PyTorch `.pt` file in the format consumed by
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/TriAttentionCalibrationArtifact.swift`.
It is named by the model's *content* fingerprint — SHA-256 over `config.json` +
`tokenizer.json` + sorted `(filename, size, file-bytes)` of every
`.safetensors` — so the same binary is reusable across any machine that
downloads the same HuggingFace checkpoint. Algorithm matches
`ModelFingerprint.computeContentFingerprint` in Swift.

Four modes are supported:

  (a) Re-key an existing upstream stats file:
        scripts/triattention_calibrate.py \\
            --model-dir <local model dir> \\
            --stats-pt <upstream/calibration.pt> \\
            --output TriAttention/v1

      Use this when upstream's `scripts/calibrate.py` from
      https://github.com/WeianMao/triattention has already been run on a
      GPU box or on an fp16 checkpoint. We filter non-full-attention layers
      out and save the result under the right fingerprint-keyed name.

  (b) MLX + ParoQuant calibration (Apple Silicon, supports paroquant):
        scripts/triattention_calibrate.py \\
            --model-dir <local PARO checkpoint> \\
            --input <plain text file> \\
            --output TriAttention/v1

      Auto-selected when `config.json` declares `quantization_config.quant_method
      == "paroquant"`. Requires `paroquant[mlx]` + `mlx-lm`.

  (c) HuggingFace transformers calibration (fp16 or AWQ checkpoints):
        scripts/triattention_calibrate.py --backend hf \\
            --model-dir <fp16 model dir> --input <plain text file> \\
            --output TriAttention/v1

      Fallback for checkpoints that HF transformers can load.
      `paroquant`-quantized weights are not supported by stock transformers
      — use mode (b) for those, or mode (d) for MLX-native quant.

  (d) MLX-native calibration (standard MLX affine quantization, incl. MoE):
        scripts/triattention_calibrate.py --backend mlx-native \\
            --model-dir <local MLX-quant checkpoint> \\
            --input <plain text file> \\
            --output TriAttention/v1

      Auto-selected when `config.json` has a top-level `quantization` field
      but no `quantization_config.quant_method == "paroquant"`. Loads via
      `mlx_lm.utils.load` (LLM-only; silently ignores `vision_config`).
      Covers Qwen3.5 and `qwen3_5_moe` checkpoints shipped by mlx-community
      / unsloth. Requires `mlx-lm`.

This script is developer-only. The shipped app never runs calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Hash-input chunk size when streaming `.safetensors` bodies. Matches the
# 8 MiB used by Swift `ModelFingerprint.hashFileContents`.
CHUNK_SIZE = 8 * 1024 * 1024


def compute_content_fingerprint(model_dir: Path) -> str:
    """Mirror of `ModelFingerprint.computeContentFingerprint` in Swift.

    SHA-256 over: config.json bytes + 0x00 + tokenizer.json bytes + 0x00 +
    for each `.safetensors` in filename-sorted order:
      name UTF-8 + 0x00 + size (Int64 LE) + 0x00 + file bytes + 0x0a.
    """
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    hasher = hashlib.sha256()
    for optional in ("config.json", "tokenizer.json"):
        path = model_dir / optional
        if path.exists():
            hasher.update(path.read_bytes())
        hasher.update(b"\x00")

    safetensors = sorted(
        (p for p in model_dir.iterdir() if p.suffix.lower() == ".safetensors"),
        key=lambda p: p.name,
    )
    for path in safetensors:
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(path.stat().st_size.to_bytes(8, "little", signed=True))
        hasher.update(b"\x00")
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
        hasher.update(b"\x0a")

    return hasher.hexdigest()


def read_model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def full_attention_layer_indices(config: dict[str, Any]) -> list[int]:
    """Full-attention layers for Qwen3.5-PARO.

    Prefers explicit `layer_types` (nested under `text_config` for
    multimodal checkpoints). Falls back to `(i+1) % full_attention_interval == 0`
    — matching the rule used by `Qwen35DecoderLayer` in
    `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`.
    """
    text_config = config.get("text_config", config)

    layer_types = text_config.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        return [i for i, t in enumerate(layer_types) if t == "full_attention"]

    num_layers = text_config.get("num_hidden_layers")
    interval = text_config.get("full_attention_interval", 4)
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("config.json lacks both layer_types and num_hidden_layers")
    return [i for i in range(num_layers) if (i + 1) % interval == 0]


def load_stats_pt(path: Path) -> dict[str, Any]:
    """Load an upstream-produced `.pt` and minimally validate its shape."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level payload must be a dict")
    if "metadata" not in payload or "stats" not in payload:
        raise ValueError(f"{path}: expected keys 'metadata' and 'stats'")
    return payload


def filter_to_full_attention(
    payload: dict[str, Any],
    full_attention_indices: list[int],
) -> dict[str, Any]:
    """Drop stats for layers that are not full-attention and rewrite metadata.

    The TriAttention runtime rejects artifacts whose sampled-head set does not
    exactly match the full-attention layer set (see `TriAttentionQwen35Runtime
    .makeState` — it requires `observedLayerIndices == validLayerIndices`).
    """
    allowed = set(full_attention_indices)
    stats_in = payload["stats"]

    filtered_stats: dict[str, Any] = {}
    kept_pairs: list[tuple[int, int]] = []

    for key, entry in stats_in.items():
        layer_idx, head_idx = parse_stats_key(key)
        if layer_idx in allowed:
            filtered_stats[key] = entry
            kept_pairs.append((layer_idx, head_idx))

    if not filtered_stats:
        raise ValueError(
            "Upstream stats contained no full-attention layers. "
            f"Expected some of {sorted(allowed)}."
        )

    observed_layers = {layer for layer, _ in kept_pairs}
    missing = allowed - observed_layers
    if missing:
        raise ValueError(
            f"Upstream stats missing full-attention layers: {sorted(missing)}. "
            "Runtime requires every full-attention layer to be represented."
        )

    metadata = dict(payload["metadata"])
    metadata["sampled_heads"] = [[int(l), int(h)] for l, h in sorted(kept_pairs)]
    return {"metadata": metadata, "stats": filtered_stats}


def parse_stats_key(key: str) -> tuple[int, int]:
    if not key.startswith("layer") or "_head" not in key:
        raise ValueError(f"Malformed stats key: {key!r}")
    layer_part, _, head_part = key[len("layer") :].partition("_head")
    return int(layer_part), int(head_part)


def _install_gated_delta_net_eager_patch() -> None:
    """Monkey-patch `mlx_lm.models.qwen3_5.GatedDeltaNet.__call__` to force eager
    evaluation after each intermediate op.

    With mlx 0.31.1 + mlx-lm 0.31.2, the lazy-evaluated GatedDeltaNet forward
    produces all-NaN outputs on Qwen3.5 PARO and MLX-native checkpoints
    (4B/9B/27B PARO and qwen3_5_moe). Forcing eager evaluation breaks the bad
    kernel fusion and produces bit-identical results to the original
    (verified against the shipped 4B artifact: max abs diff 0.0). Both
    `qwen3_5_moe` and `qwen3_5` share this class, so one patch covers both.
    Remove once upstream MLX fixes the regression.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as _qwen3_5

    def _eager_gdn_call(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs); mx.eval(qkv)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim); mx.eval(z)
        b = self.in_proj_b(inputs); mx.eval(b)
        a = self.in_proj_a(inputs); mx.eval(a)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0); mx.eval(qkv)
        conv_input = mx.concatenate([conv_state, qkv], axis=1); mx.eval(conv_input)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input)); mx.eval(conv_out)
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        mx.eval(q, k, v)
        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6); mx.eval(q)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6); mx.eval(k)
        out, state = _qwen3_5.gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training,
        )
        mx.eval(out)
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        out = self.norm(out, z); mx.eval(out)
        out = self.out_proj(out.reshape(B, S, -1)); mx.eval(out)
        return out

    _qwen3_5.GatedDeltaNet.__call__ = _eager_gdn_call


def _collect_q_norm_stats(
    *,
    model: Any,
    encode: Any,
    full_indices: list[int],
    input_path: Path,
    max_length: int,
    attn_implementation: str,
    log_tag: str,
) -> dict[str, Any]:
    """Probe Q at each full-attention layer's `q_norm`, run one forward pass,
    and compute per-head real/imag/abs-mean stats.

    Returns the payload in the format consumed by
    `TriAttentionCalibrationArtifact.swift`. `encode` is expected to return
    a list of token ids (matches `processor.encode` / `tokenizer.encode`).
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import torch

    class _QNormProbe(nn.Module):
        def __init__(self, inner, sink, key):
            super().__init__()
            self.inner = inner
            self._sink = sink
            self._key = key

        def __call__(self, x):
            y = self.inner(x)
            self._sink[self._key] = y
            return y

    text_model = model.language_model.model
    captured: dict[int, mx.array] = {}
    for layer_idx in full_indices:
        attn = text_model.layers[layer_idx].self_attn
        if not hasattr(attn, "q_norm"):
            raise RuntimeError(
                f"layer {layer_idx} self_attn has no q_norm — "
                "not a supported Qwen3.5-family attention layer"
            )
        attn.q_norm = _QNormProbe(attn.q_norm, captured, layer_idx)

    text = input_path.read_text(encoding="utf-8")
    token_ids = encode(text)
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    print(f"[{log_tag}] tokenized length: {len(token_ids)}", file=sys.stderr)

    inputs = mx.array([list(token_ids)])
    print(f"[{log_tag}] running forward pass ...", file=sys.stderr)
    _ = model(inputs)
    mx.eval(list(captured.values()))

    stats: dict[str, dict[str, torch.Tensor]] = {}
    sampled_heads: list[tuple[int, int]] = []
    head_dim = 0

    for layer_idx in sorted(captured.keys()):
        q = captured[layer_idx]
        q_np = np.asarray(q.astype(mx.float32))
        assert q_np.ndim == 4 and q_np.shape[0] == 1, (
            f"unexpected captured shape {q_np.shape} for layer {layer_idx}"
        )
        num_heads = q_np.shape[2]
        head_dim = q_np.shape[3]
        freq_count = head_dim // 2

        for head_idx in range(num_heads):
            head = q_np[0, :, head_idx, :]
            real = head[:, :freq_count]
            imag = head[:, freq_count:]
            stats[f"layer{layer_idx:02d}_head{head_idx:02d}"] = {
                "q_mean_real": torch.from_numpy(real.mean(axis=0).copy()).float(),
                "q_mean_imag": torch.from_numpy(imag.mean(axis=0).copy()).float(),
                "q_abs_mean": torch.from_numpy(
                    np.sqrt(real * real + imag * imag).mean(axis=0).copy()
                ).float(),
            }
            sampled_heads.append((layer_idx, head_idx))
        del q_np, q
        captured[layer_idx] = None

    metadata: dict[str, Any] = {
        "num_traces": 1,
        "head_dim": int(head_dim),
        "dtype": "float16",
        "use_chat_template": False,
        "system_prompt": "",
        "attn_implementation": attn_implementation,
        "rope_style": "half",
        "rope_type": "default",
        "sampled_heads": [[int(l), int(h)] for l, h in sampled_heads],
    }

    return {"metadata": metadata, "stats": stats}


def run_mlx_calibration(
    model_dir: Path,
    input_path: Path,
    max_length: int,
) -> dict[str, Any]:
    """MLX + ParoQuant calibration path.

    Uses `paroquant.inference.backends.mlx.load` to load the quantized
    checkpoint, then substitutes each full-attention layer's `q_norm` with
    a probe that records its output. The probed tensor is the pre-RoPE Q
    (since Qwen3.5's attention applies `q_norm → transpose → rope`), which
    matches upstream's post-invert-RoPE representation when attention
    scaling is 1.0 — Qwen3.5 config has no rope_scaling, so this holds.
    """
    from paroquant.inference.backends.mlx.load import load as paro_load

    _install_gated_delta_net_eager_patch()

    print(f"[calibrate-mlx] loading {model_dir} ...", file=sys.stderr)
    model, processor, is_vlm = paro_load(str(model_dir), force_text=True)
    assert not is_vlm, "TriAttention calibration runs against the text path only"

    return _collect_q_norm_stats(
        model=model,
        encode=processor.encode,
        full_indices=full_attention_layer_indices(read_model_config(model_dir)),
        input_path=input_path,
        max_length=max_length,
        attn_implementation="paroquant-mlx",
        log_tag="calibrate-mlx",
    )


def run_mlx_native_calibration(
    model_dir: Path,
    input_path: Path,
    max_length: int,
) -> dict[str, Any]:
    """MLX-native calibration path for standard MLX-quantized Qwen3.5 family.

    Uses `mlx_lm.utils.load` (LLM-only — silently ignores `vision_config`).
    Handles both plain `qwen3_5` and `qwen3_5_moe` checkpoints with standard
    MLX affine quantization. Passes `lazy=True` to delay weight
    materialization (peak memory still approaches the full weight size once
    the forward pass runs — close any process holding a loaded model before
    calibrating large checkpoints).
    """
    from mlx_lm.utils import load as mlx_load

    _install_gated_delta_net_eager_patch()

    print(f"[calibrate-mlx-native] loading {model_dir} (lazy) ...", file=sys.stderr)
    model, tokenizer = mlx_load(str(model_dir), lazy=True)

    return _collect_q_norm_stats(
        model=model,
        encode=tokenizer.encode,
        full_indices=full_attention_layer_indices(read_model_config(model_dir)),
        input_path=input_path,
        max_length=max_length,
        attn_implementation="mlx-native",
        log_tag="calibrate-mlx-native",
    )


def run_hf_calibration(
    model_dir: Path,
    input_path: Path,
    max_length: int,
    device: str,
    attn_implementation: str,
) -> dict[str, Any]:
    """Port of upstream `scripts/calibrate.py` scoped to our use case.

    Runs one forward pass on `input_path`, hooks every decoder layer's
    `self_attn` to capture pre-RoPE Q, inverts RoPE, and computes per-head
    frequency statistics. Assumes the checkpoint is loadable by
    `AutoModelForCausalLM` — i.e. not paroquant-quantized.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device_obj = torch.device(device)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.eval()

    text_cfg = getattr(config, "text_config", config)
    num_heads = int(getattr(text_cfg, "num_attention_heads"))
    head_dim = int(
        getattr(text_cfg, "head_dim", getattr(text_cfg, "hidden_size") // num_heads)
    )

    backbone = getattr(model, "model", model)
    layer_list = getattr(backbone, "layers", None)
    if layer_list is None:
        raise RuntimeError("Model layout unexpected: `model.model.layers` missing.")

    if hasattr(backbone, "rotary_emb"):
        rotary = backbone.rotary_emb
    elif hasattr(layer_list[0], "self_attn") and hasattr(layer_list[0].self_attn, "rotary_emb"):
        rotary = layer_list[0].self_attn.rotary_emb
    else:
        raise RuntimeError("Could not locate rotary_emb module.")
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0))

    text = input_path.read_text(encoding="utf-8")
    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device_obj)
    seq_len = input_ids.shape[1]
    print(f"[calibrate] tokenized length: {seq_len}", file=sys.stderr)

    position_ids = torch.arange(seq_len, device=device_obj).unsqueeze(0)
    probe = torch.zeros(1, seq_len, head_dim, device=device_obj, dtype=dtype)
    cos_table, sin_table = rotary(probe, position_ids)

    captured: dict[int, torch.Tensor] = {}
    full_attention_indices = set(full_attention_layer_indices(json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8"))))

    def make_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return
            q = module.q_proj(hidden_states)
            bsz, q_len, _ = hidden_states.shape
            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            pos = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
            p = torch.zeros(1, q_len, head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            cos, sin = rotary(p, pos)
            q_rot = (q * cos.unsqueeze(1)) + (_rotate_half(q) * sin.unsqueeze(1))
            captured[layer_idx] = (q_rot * attn_scale).detach()
        return hook_fn

    handles = []
    for layer_idx, layer_module in enumerate(layer_list):
        if layer_idx not in full_attention_indices:
            continue
        attn = getattr(layer_module, "self_attn", None)
        if attn is None:
            continue
        handles.append(
            attn.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
        )

    print("[calibrate] running forward pass ...", file=sys.stderr)
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()

    stats: dict[str, dict[str, torch.Tensor]] = {}
    sampled_heads: list[tuple[int, int]] = []

    for layer_idx, q_rot in captured.items():
        cos = cos_table[:, :seq_len, :].unsqueeze(1)
        sin = sin_table[:, :seq_len, :].unsqueeze(1)
        q_base = _invert_half_rope(q_rot, cos, sin, attn_scale)

        for head_idx in range(num_heads):
            q_head = q_base[0, head_idx]
            freq = q_head.shape[-1] // 2
            real = q_head[..., :freq].to(torch.float32).contiguous()
            imag = q_head[..., freq:].to(torch.float32).contiguous()

            stats[f"layer{layer_idx:02d}_head{head_idx:02d}"] = {
                "q_mean_real": real.mean(dim=0).cpu(),
                "q_mean_imag": imag.mean(dim=0).cpu(),
                "q_abs_mean": torch.sqrt(real * real + imag * imag).mean(dim=0).cpu(),
            }
            sampled_heads.append((layer_idx, head_idx))

    rope_scaling = getattr(text_cfg, "rope_scaling", None) or {}
    rope_type = (
        rope_scaling.get("rope_type")
        or rope_scaling.get("type")
        or getattr(text_cfg, "rope_type", "default")
        or "default"
    )

    metadata: dict[str, Any] = {
        "num_traces": 1,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "use_chat_template": False,
        "system_prompt": "",
        "attn_implementation": attn_implementation,
        "rope_style": "half",
        "rope_type": rope_type,
        "sampled_heads": [[int(l), int(h)] for l, h in sampled_heads],
    }

    return {"metadata": metadata, "stats": stats}


def _rotate_half(x):
    import torch

    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _invert_half_rope(rotated, cos, sin, scale: float):
    import torch

    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    return base * cos_unit - _rotate_half(base) * sin_unit


def save_artifact(payload: dict[str, Any], output_path: Path) -> None:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path), pickle_protocol=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a TriAttention calibration artifact named by the model's "
            "content fingerprint."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Local path to the HuggingFace checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory — typically TriAttention/v1",
    )
    parser.add_argument(
        "--stats-pt",
        type=Path,
        help="Pre-generated upstream calibration .pt to re-key by fingerprint.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Plain-text calibration input (modes b, c, d).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "mlx", "mlx-native", "hf"),
        default="auto",
        help=(
            "Calibration backend when --input is given. "
            "`auto` uses mlx for paroquant checkpoints, mlx-native for MLX-quant "
            "checkpoints (incl. qwen3_5_moe), hf otherwise."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum calibration token length. Default 8192.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device for the hf backend. Default mps (Apple Silicon).",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Torch attention impl for the hf backend. Default sdpa.",
    )
    args = parser.parse_args()

    fingerprint = compute_content_fingerprint(args.model_dir)
    print(f"[calibrate] content fingerprint: {fingerprint}", file=sys.stderr)

    config = read_model_config(args.model_dir)
    full_indices = full_attention_layer_indices(config)
    print(
        f"[calibrate] full-attention layers ({len(full_indices)}): {full_indices}",
        file=sys.stderr,
    )

    if args.stats_pt is not None:
        payload = load_stats_pt(args.stats_pt)
        payload = filter_to_full_attention(payload, full_indices)
        print(
            f"[calibrate] re-keyed {len(payload['stats'])} heads from {args.stats_pt}",
            file=sys.stderr,
        )
    else:
        if args.input is None:
            parser.error("--input is required when --stats-pt is not provided")
        backend = args.backend
        if backend == "auto":
            quant_method = (config.get("quantization_config") or {}).get("quant_method")
            has_mlx_quant = "quantization" in config  # MLX-native affine quant
            if quant_method == "paroquant":
                backend = "mlx"
            elif has_mlx_quant:
                backend = "mlx-native"
            else:
                backend = "hf"
        print(f"[calibrate] backend: {backend}", file=sys.stderr)
        if backend == "mlx":
            payload = run_mlx_calibration(
                model_dir=args.model_dir,
                input_path=args.input,
                max_length=args.max_length,
            )
        elif backend == "mlx-native":
            payload = run_mlx_native_calibration(
                model_dir=args.model_dir,
                input_path=args.input,
                max_length=args.max_length,
            )
        else:
            payload = run_hf_calibration(
                model_dir=args.model_dir,
                input_path=args.input,
                max_length=args.max_length,
                device=args.device,
                attn_implementation=args.attn_implementation,
            )

    output_path = args.output / f"{fingerprint}.pt"
    save_artifact(payload, output_path)
    print(f"[calibrate] wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
