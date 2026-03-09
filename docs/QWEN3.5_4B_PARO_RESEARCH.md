# Qwen3.5-4B-PARO — Compatibility Research

**Date**: 2026-03-09
**Model**: [z-lab/Qwen3.5-4B-PARO](https://huggingface.co/z-lab/Qwen3.5-4B-PARO)
**Verdict**: **Not compatible** with current Tesseract setup without significant work.

---

## Model Overview

| Property | Value |
|---|---|
| Base model | Qwen/Qwen3.5-4B |
| Architecture | `Qwen3_5ForConditionalGeneration` (multimodal) |
| Text model type | `qwen3_5_text` |
| Parameters | ~4B (full), ~1B effective (INT4) |
| Quantization | ParoQuant (Pairwise Rotation Quantization), 4-bit |
| Weight format | safetensors (single file, 3.82 GB) |
| License | Apache 2.0 |
| Context window | 262,144 tokens |
| Hidden size | 2560 |
| Layers | 32 |
| Attention heads | 16 (4 KV heads) |
| Intermediate size | 9216 |
| Vocab size | 248,320 |

### What is ParoQuant?

ParoQuant is a state-of-the-art INT4 quantization method (ICLR 2026) that uses **learned pairwise rotations** to suppress weight outliers. It claims to close the accuracy gap with FP16 while running at near-AWQ speed. The key parameter is `krot=8` (8 rotation pairs per group).

Paper: [arXiv:2511.10645](https://arxiv.org/abs/2511.10645)

---

## Compatibility Analysis

### 1. Quantization Format — **INCOMPATIBLE**

This is the primary blocker.

**Current Tesseract stack** (mlx-swift-lm) supports these quantization modes:
- `affine` — standard MLX per-group linear quantization (what all our models use)
- `mxfp4` / `mxfp8` — Microscaling FP4/FP8
- `nvfp4` — NVIDIA FP4

**ParoQuant requires**:
- Custom `paroquant` quant method with `krot=8` rotation matrices
- Special dequantization: applies learned pairwise rotation matrices before/after standard INT4 dequant
- Dynamic quantization exclusions for specific layers (`linear_attn.in_proj_a/b`, `mlp.gate`, `mlp.shared_expert_gate`)

The `paroquant` quant method is **not recognized** by mlx-swift-lm's `QuantizationMode` enum. Loading this model would fail at weight deserialization — the framework wouldn't know how to dequantize the INT4 weights with rotation matrices.

### 2. Model Architecture — **LIKELY INCOMPATIBLE**

Qwen3.5 introduces a **hybrid attention** architecture that alternates between two layer types:
- `linear_attention` (24 of 32 layers) — linear complexity attention with conv kernels
- `full_attention` (8 of 32 layers) — standard transformer attention

This is fundamentally different from standard Qwen3 (all full attention). Key novel components:
- `linear_conv_kernel_dim: 4` — 1D convolution in linear attention layers
- `linear_key_head_dim: 128`, `linear_num_key_heads: 16` — separate KV geometry for linear layers
- `linear_num_value_heads: 32`, `linear_value_head_dim: 128`
- `partial_rotary_factor: 0.25` — only 25% of head dim gets RoPE
- `attn_output_gate: true` — gated attention output

The mlx-swift-lm `LLMTypeRegistry` may have a `qwen3_5` entry (50+ model types registered), but the linear attention layers and hybrid architecture are Qwen3.5-specific and likely require dedicated Swift implementation that may not exist yet in the Swift library.

### 3. Multimodal Architecture — **Mismatch but Workable**

The model is `Qwen3_5ForConditionalGeneration` (vision+text), not a text-only model. It includes:
- Vision encoder config (ViT with 24 layers, patch_size=16)
- Image/video token IDs

However, Tesseract only needs the text component. The `text_config` portion could theoretically be used standalone if the architecture were otherwise compatible.

### 4. Python MLX Support — **Exists, but Not Useful for Us**

The `paroquant` Python package (`pip install "paroquant[mlx]"`) supports Apple Silicon via Python MLX. This means:
- It works with `mlx-lm` (Python) — ✅
- It does NOT work with `mlx-swift-lm` (Swift) — ❌

Tesseract uses Swift MLX exclusively. The Python support doesn't help us.

---

## What We Currently Use

| Model ID | Repo | Quant | Size |
|---|---|---|---|
| **qwen3.5-4b** (default) | `mlx-community/Qwen3.5-4B-MLX-8bit` | MLX affine 8-bit | 5 GB |
| qwen3-4b-instruct-2507 | `mlx-community/Qwen3-4B-Instruct-2507-8bit` | MLX affine 8-bit | 4.5 GB |
| qwen3-4b-thinking-2507 | `lmstudio-community/Qwen3-4B-Thinking-2507-MLX-8bit` | MLX affine 8-bit | 4.5 GB |
| qwen3-4b-thinking-opus-distill | `nightmedia/...qx86-hi-mlx` | MLX affine 8-bit | 3.8 GB |
| nanbeige4.1-3b | `mlx-community/Nanbeige4.1-3B-8bit` | MLX affine 8-bit | 4.2 GB |

All models use standard MLX affine quantization in safetensors format.

---

## What Would Be Needed to Support ParoQuant

To run this model in Tesseract, we would need:

1. **Add ParoQuant dequantization to mlx-swift-lm** — Implement the pairwise rotation dequant kernel in Metal/MLX Swift. This is non-trivial: each group of weights needs rotation matrix application during dequant.

2. **Implement Qwen3.5 hybrid attention in Swift** — Write the `linear_attention` layer type with 1D convolution, separate KV head geometry, partial RoPE, and gated output. This is a significant architecture addition (~500-1000 lines of Swift).

3. **Handle dynamic quantization exclusions** — Some layers (gates, projections) are kept in higher precision. The loading code needs to handle mixed precision per-layer.

4. **Test and validate** — Ensure the pairwise rotation dequant produces correct outputs matching the Python implementation.

**Estimated effort**: 2-4 weeks of focused work, assuming familiarity with both MLX internals and the ParoQuant paper.

---

## Alternatives

If the goal is a smaller/faster Qwen3.5-4B variant:

| Option | Format | Size | Compatible? |
|---|---|---|---|
| `mlx-community/Qwen3.5-4B-MLX-8bit` | MLX 8-bit | 5 GB | ✅ Already in use |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | MLX 4-bit | ~2.5 GB | ✅ If it exists |
| Standard MLX `quantize()` on Qwen3.5-4B | MLX affine 4-bit | ~2.5 GB | ✅ Can self-quantize |
| `z-lab/Qwen3.5-4B-PARO` | ParoQuant INT4 | 3.82 GB | ❌ Needs custom work |

The simplest path to a smaller Qwen3.5-4B is to use MLX's built-in `quantize()` with 4-bit affine quantization, which is natively supported by mlx-swift-lm. The quality will be slightly lower than ParoQuant's rotation-based approach, but it works out of the box.

---

## Conclusion

**ParoQuant is an interesting quantization technique but is not compatible with Tesseract's Swift MLX stack.** The two blockers are:

1. Custom quantization format requiring new Metal kernels
2. Novel hybrid attention architecture (linear + full) requiring new Swift model code

Both the `mlx-community/Qwen3.5-4B-MLX-8bit` (already in use) and a potential standard MLX 4-bit quant are better options for our setup. If ParoQuant support lands in upstream mlx-swift-lm in the future, it would become viable without custom work.
