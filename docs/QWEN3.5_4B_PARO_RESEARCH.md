# Qwen3.5-4B-PARO — Compatibility Research

**Date**: 2026-03-09 (revised 2026-03-12)
**Model**: [z-lab/Qwen3.5-4B-PARO](https://huggingface.co/z-lab/Qwen3.5-4B-PARO)
**Verdict**: **Compatible** — now the default ParoQuant agent model.

---

## Model Overview

| Property | Value |
|---|---|
| Base model | Qwen/Qwen3.5-4B |
| Architecture | `Qwen3_5ForConditionalGeneration` (multimodal wrapper) |
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

## Compatibility Status

### 1. Quantization Format — **SUPPORTED**

ParoQuant Metal kernel and `RotateQuantizedLinear` are implemented in `Features/Agent/ParoQuant/`. The rotation kernel, AutoAWQ weight conversion, and weight format are identical between Qwen3 and Qwen3.5 PARO models (same krot=8, group_size=128, bits=4).

### 2. Model Architecture — **SUPPORTED**

Qwen3.5's hybrid attention architecture (linear + full attention layers) is registered as `qwen3_5` in mlx-swift-lm's `LLMTypeRegistry`. The VLM config wrapper is handled by flattening `text_config` to top-level in the loader, overriding `model_type` to `"qwen3_5"`.

### 3. VLM Wrapper — **Handled**

The model is `Qwen3_5ForConditionalGeneration` (vision+text), but Tesseract only needs the text component. The loader flattens `text_config` to top-level and sets `model_type` to `"qwen3_5"`, allowing `typeRegistry.createModel()` to instantiate the text-only architecture. Vision-related weight keys are ignored via relaxed verification (`.shapeMismatch` only, no `.noUnusedKeys`).

### 4. Dynamic Quantization Exclusions — **Naturally Handled**

Some layers (`linear_attn.in_proj_a/b`, `mlp.gate`, `mlp.shared_expert_gate`) are intentionally kept in FP16. These have regular `.weight` keys without `.qweight`/`.scales`/`.theta`, so the existing loader logic skips them automatically:
- `convertAutoAWQ()` requires `.qweight` + `.theta` → skips
- `patchRotationLayers()` requires `.theta` → skips
- `isCheckpointQuantizedLayer()` requires `.scales` → skips

### 5. AWQ/Sanitize Ordering — **Fixed**

Python does AWQ conversion BEFORE sanitize; the Swift loader matches this order (step 6: AWQ, step 7: sanitize). This matters because sanitize may remap key prefixes.

---

## Current Usage

| Model ID | Repo | Quant | Size |
|---|---|---|---|
| **qwen3.5-4b** (default) | `mlx-community/Qwen3.5-4B-MLX-8bit` | MLX affine 8-bit | 5 GB |
| **qwen3.5-4b-paro** | `z-lab/Qwen3.5-4B-PARO` | ParoQuant INT4 | 3.8 GB |

---

## Size Comparison

| Model | Quant | Size | Quality (approx) |
|---|---|---|---|
| `Qwen3.5-4B-MLX-8bit` | MLX affine 8-bit | 5 GB | ~98% of FP16 |
| `Qwen3.5-4B-PARO` | ParoQuant INT4 | ~3.8 GB | ~99% of FP16 |
| Standard MLX 4-bit | MLX affine 4-bit | ~2.5 GB | ~95% of FP16 |

ParoQuant INT4 gives better quality than standard INT4 at similar size, and uses ~24% less memory than the 8-bit default.

---

## Implementation Details

See `spec/PAROQUANT_SWIFT_PORT.md` for the full implementation spec including:
- Metal rotation kernel
- `RotateQuantizedLinear` module
- AutoAWQ→MLX weight conversion
- VLM config flattening
- Load flow and verification
