# ParoQuant Swift MLX Port — Implementation Spec

**Date**: 2026-03-09 (revised 2026-03-12)
**Goal**: Support ParoQuant INT4 models in Tesseract's Swift MLX stack.
**Target model**: `z-lab/Qwen3.5-4B-PARO` (`Qwen3_5ForConditionalGeneration` — text model extracted via config flattening).

---

## Executive Summary

ParoQuant adds exactly **one operation** before the standard `quantizedMatmul`: a pairwise Givens rotation of the input activations. The Python MLX implementation is ~150 lines of Python + 75 lines of Metal. All building blocks exist in Swift MLX (`MLXFast.metalKernel`, `quantizedMM`). This is a straightforward port.

The Qwen3.5-4B-PARO model uses a VLM wrapper config (`Qwen3_5ForConditionalGeneration`) with text model parameters nested under `text_config`. The loader flattens this to top-level before passing to `BaseConfiguration` and `typeRegistry.createModel()`, overriding `model_type` to `"qwen3_5"` (the registered text-only type).

### Scope

| What | Status |
|---|---|
| `RotateQuantizedLinear` module (Swift) | Implemented |
| Metal rotation kernel (port from Python) | Implemented |
| AutoAWQ→MLX weight conversion | Implemented |
| Weight loading & layer patching | Implemented |
| VLM config flattening (`text_config` → top-level) | Implemented |
| `Qwen3_5` model architecture (hybrid attention) | Already supported by mlx-swift-lm |
| `quantizedMM` (standard INT4 dequant) | Already exists |
| `MLXFast.metalKernel` (custom Metal dispatch) | Already exists |

---

## Target Model Details

| Property | Value |
|---|---|
| Base model | Qwen/Qwen3.5-4B |
| Architecture | `Qwen3_5ForConditionalGeneration` (multimodal wrapper) |
| Text model type | `qwen3_5_text` → flattened as `qwen3_5` |
| Layers | 32 |
| Hidden size | 2560 |
| Attention heads | 16 (4 KV heads) |
| Intermediate size | 9216 |
| Vocab size | 248,320 |
| Quantization | ParoQuant INT4 (krot=8, group_size=128, bits=4) |
| Size | ~3.8 GB |
| Context window | 262,144 tokens |

---

## How ParoQuant Works

Standard INT4 quantized linear:
```
y = quantizedMatmul(x, W_quantized, scales, biases)
```

ParoQuant INT4 quantized linear:
```
x_rotated = paroRotate(x, theta, pairs, channelScales)  // ← the only new thing
y = quantizedMatmul(x_rotated, W_quantized, scales, biases)
```

The rotation suppresses weight outliers by applying learned Givens rotations to the input before the quantized matmul. The rotations are pre-trained and stored alongside the quantized weights.

### Extra weights per layer (rotation parameters)

| Parameter | Shape | Type | Purpose |
|---|---|---|---|
| `theta` | `[krot, inputDims/2]` | float16 | Rotation angles |
| `pairs` | `[krot, inputDims]` | int16 | Channel pair indices |
| `channel_scales` | `[1, inputDims]` | float16 | Per-channel rescaling |

For Qwen3.5-4B (`inputDims=2560`, `krot=8`): ~120KB per layer, ~3.8MB total for 32 layers. Negligible overhead.

### The rotation algorithm

For each of `krot=8` rotation rounds:
1. Load input activations into shared memory, fused with `channelScales`
2. For each pair `(i, j)` in the group: apply 2D Givens rotation with angle `theta[k]`
   ```
   a' = a * cos(θ) + b * sin(θ)
   b' = b * cos(θ) - a * sin(θ)
   ```
3. Write rotated activations back

Key insight: pairs are **independent within each group** (each channel appears in at most one pair per round), so the entire group can be rotated in parallel across threads.

---

## VLM Config Handling

Qwen3.5-4B-PARO's `config.json` has a VLM wrapper structure:

```json
{
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        ...
    },
    "quantization_config": {
        "quant_method": "paroquant",
        "bits": 4,
        "group_size": 128,
        "krot": 8,
        ...
    }
}
```

The loader flattens `text_config` to top-level (text values win on conflicts) and overrides `model_type` to `"qwen3_5"` — the type registered in mlx-swift-lm for text-only Qwen3.5 models. This lets `typeRegistry.createModel()` instantiate the correct architecture without any vision encoder.

---

## Dynamic Quantization Exclusions

Qwen3.5-4B-PARO config declares these layers excluded from quantization:

```json
"dynamic": {
    "-:.*linear_attn.in_proj_a": {},
    "-:.*linear_attn.in_proj_b": {},
    "-:.*mlp.gate": {},
    "-:.*mlp.shared_expert_gate": {}
}
```

These layers have regular `.weight` keys (FP16), NOT `.qweight`/`.scales`. The loader handles this correctly with no special code:
- `convertAutoAWQ()` only converts layers with `.qweight` + `.theta` → skips excluded layers
- `patchRotationLayers()` only patches layers with `.theta` → skips excluded layers
- `isCheckpointQuantizedLayer()` requires `.scales` → skips excluded layers
- Excluded layers load as regular FP16 weights via `update(parameters:)`

---

## Actual Weight Format (AutoAWQ, not native MLX)

The `z-lab/Qwen3.5-4B-PARO` safetensors use **AutoAWQ packing format**, not native MLX format.

Actual weight key suffixes in the safetensors: `qweight`, `qzeros`, `scales`, `theta`, `pairs`, `channel_scales`, `weight`.

Example for `model.layers.0.mlp.gate_proj`:

| Key suffix | Shape | Type | Notes |
|---|---|---|---|
| `.qweight` | `[2560, 1152]` | I32 | AutoAWQ-packed INT4 weights (reordered) |
| `.qzeros` | `[20, 1152]` | I32 | AutoAWQ-packed zero points |
| `.scales` | `[20, 9216]` | F16 | Quantization scales (transposed vs MLX) |
| `.theta` | `[8, 1280]` | F16 | Rotation angles |
| `.pairs` | `[8, 2560]` | I16 | Channel pair indices |
| `.channel_scales` | `[1, 2560]` | F16 | Per-channel rescaling |

Non-quantized layers (embed_tokens, norms, excluded dynamic layers): just `.weight` in F16.

The Python code's `_convert_autoawq()` function IS required — it converts:
- `.qweight` → `.weight` (unpack, undo [0,2,4,6,1,3,5,7] reorder, repack as MLX sequential uint32)
- `.qzeros` + `.scales` → `.biases` (compute `-scales * zeros`, transpose)
- `.scales` → `.scales` (transpose)

**AWQ/sanitize ordering**: Python does AWQ conversion BEFORE sanitize. The Swift loader matches this order — AWQ first (step 6), then `model.sanitize()` (step 7). This is important because sanitize may remap key prefixes (e.g., stripping `model.` prefix for Qwen3.5's text model keys).

---

## Implementation

### File Structure

```
tesseract/Features/Agent/ParoQuant/
├── RotateQuantizedLinear.swift   # QuantizedLinear subclass: rotation + quantizedMM
└── ParoQuantLoader.swift         # AutoAWQ conversion, layer patching, custom load path
```

The Metal kernel source is an inline string constant in `RotateQuantizedLinear.swift`. No separate file needed — the kernel is compiled at runtime via `MLXFast.metalKernel()`.

### Load Flow (`loadParoQuantModel`)

```
1. Parse config.json
   - Flatten text_config to top-level if present (VLM → text-only)
   - Override model_type to "qwen3_5"
2. Read ParoQuant params (bits, groupSize, krot from quantization_config)
3. Create model via typeRegistry.createModel()
4. EOS token override from generation_config.json
5. Load raw safetensors
6. Convert AutoAWQ → MLX format (BEFORE sanitize)
7. Model-specific sanitization (AFTER AWQ)
8. Patch rotation layers (Linear → RotateQuantizedLinear where .theta exists)
9. Quantize checkpoint-quantized layers (has .scales but no .theta)
10. Load weights into model (verify: .shapeMismatch only — VLM keys may be unused)
11. Quantize IO embedding path (embed_tokens, lm_head)
12. eval(model)
13. Load tokenizer
14. Create processor with messageGenerator
15. Return ModelContainer
```

### Weight Verification

The VLM safetensors may contain vision-related keys (e.g., `visual.*`) that don't match any text model parameter. Python uses `strict=False`. The Swift loader uses `.shapeMismatch` only (no `.noUnusedKeys`) to match this behavior.

---

## Mapping: Python → Swift

| Python (paroquant) | Swift (tesseract) | Notes |
|---|---|---|
| `modules.py:RotateQuantizedLinear` | `RotateQuantizedLinear.swift` | `nn.Module` → `QuantizedLinear` subclass |
| `modules.py:_pack_pairs()` | `RotateQuantizedLinear.packPairs()` | NumPy → MLXArray |
| `modules.py:_rotate()` | `RotateQuantizedLinear.rotate()` | `mx.fast.metal_kernel()` → `MLXFast.metalKernel()` |
| `rotation.metal` | Inline string in `RotateQuantizedLinear.swift` | Verbatim Metal, fix brace escaping |
| `rotation.py:get_rotation_kernel()` | Static cached property | `@lru_cache` → `nonisolated(unsafe) static var` |
| `load.py:_convert_autoawq()` | `ParoQuantLoader.convertAutoAWQ()` | Unpack/reorder/repack int4 weights |
| `load.py:_pack_mlx()` | `ParoQuantLoader.packMLX()` | uint8 → uint32 sequential packing |
| `load.py:_patch_rotation_layers()` | `ParoQuantLoader.patchRotationLayers()` | Walk model tree, swap modules |
| `load.py:load()` | `loadParoQuantModel()` | Custom load path reusing factory components |

---

## Size Comparison

| Model | Quant | Size | Quality (approx) |
|---|---|---|---|
| `Qwen3.5-4B-MLX-8bit` | MLX affine 8-bit | 5 GB | ~98% of FP16 |
| `Qwen3.5-4B-PARO` | ParoQuant INT4 | ~3.8 GB | ~99% of FP16 |
| Standard MLX 4-bit | MLX affine 4-bit | ~2.5 GB | ~95% of FP16 |

ParoQuant INT4 gives better quality than standard INT4 at similar size, and uses ~24% less memory than our current 8-bit default.

---

## Validation Plan

1. `scripts/dev.sh build` — verify compilation
2. Download `z-lab/Qwen3.5-4B-PARO` via model manager
3. Load model — verify `typeRegistry.createModel()` succeeds with flattened config
4. Verify 1-token generation (the `verifyAndStore` smoke test)
5. Run agent conversation — verify tool calling with `.xmlFunction` format
6. Check logs: `ParoQuant config: bits=4, groupSize=128, krot=8`
7. Verify generation params: temp=1.0, topP=0.95 (Qwen3.5 defaults via `forModel("qwen3.5-4b-paro")`)

---

## Review Response Log

### Review 1 (2026-03-09)

| Finding | Severity | Resolution |
|---|---|---|
| `RotateQuantizedLinear` must be `Linear` subclass for `@ModuleInfo` slots | P1 | Fixed: now subclasses `QuantizedLinear` (which subclasses `Linear`) |
| Custom loader duplicates factory path | P1 | Revised: custom load path reuses `typeRegistry.createModel()` and tokenizer loading; only weight conversion and layer patching are custom |
| Rotation params need `@ParameterInfo` or Module-visible storage | P2 | Fixed: stored `let` MLXArray properties are discovered by Module's Mirror reflection — same as `weight` in `Linear` |
| Step 4 may force-quantize layers that should stay FP16 | P2 | Fixed: only quantize layers with `.scales` AND without `.theta`; embed_tokens has no `.scales` so it stays FP16 |
| Model registration uses wrong schema fields | P3 | Fixed: uses `displayName`, `sizeDescription`, `source` per `ModelDefinition` |
| Safetensors are AutoAWQ format, not native MLX | P1 (new) | Fixed: added AutoAWQ→MLX conversion step (port of `_convert_autoawq()`) |

### Review 2 (2026-03-09)

| Finding | Severity | Verdict | Resolution |
|---|---|---|---|
| Config uses `paroquant_config` with `nbit` not `quantization_config` with `bits` | P1 | **INVALID** — actual config.json uses `quantization_config` with `bits`, `group_size`, `krot`. However, the sub-point about hardcoding `(128, 4)` was valid. | Fixed: load path reads `bits`, `group_size`, `krot` from config.json via `readParoQuantConfig()` |
| Custom loader uses wrong API names | P2 | **Partially valid** — EOS override + `messageGenerator`/processor setup were missing | Fixed: pseudocode now mirrors factory exactly |
| `verify: [.noUnusedKeys]` too weak after AWQ conversion | P2 | **Valid** — `.shapeMismatch` exists and would catch bad conversions | Fixed: now uses `[.shapeMismatch]` (dropped `.noUnusedKeys` for VLM compat) |

### Review 3 (2026-03-09)

| Finding | Severity | Verdict | Resolution |
|---|---|---|---|
| `channelScales` won't load — Swift name ≠ checkpoint key `channel_scales` | P1 | **Valid** | Fixed: use `@ParameterInfo(key: "channel_scales") var channelScales` |
| `patchRotationLayers` doesn't pass `krot` | P2 | **Valid** | Fixed: krot inferred per-layer from `weights["\(prefix).theta"].shape[0]` |
| Other PARO models use different schema | P2 | **INVALID** — all 13 published z-lab PARO models use `quantization_config` | No change needed |
| Factory uses `JSONDecoder.json5()` | P3 | **INVALID** — factory uses plain `JSONDecoder()` | No change needed |

### Review 4 (2026-03-12) — Qwen3.5 migration

| Finding | Severity | Resolution |
|---|---|---|
| VLM config needs flattening (`text_config` → top-level) | P1 | Added config flattening with `model_type` override to `"qwen3_5"` |
| AWQ/sanitize ordering mismatch vs Python | P2 | Swapped steps 6/7: AWQ conversion now runs before sanitize |
| `.noUnusedKeys` rejects VLM vision keys | P2 | Relaxed to `.shapeMismatch` only |
| Dynamic exclusion layers need special handling | P3 | No code change — existing loader logic naturally skips FP16-only layers |
