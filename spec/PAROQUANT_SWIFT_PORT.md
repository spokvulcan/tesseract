# ParoQuant Swift MLX Port — Implementation Spec

**Date**: 2026-03-09 (revised 2026-03-09)
**Goal**: Support ParoQuant INT4 models in Tesseract's Swift MLX stack.
**Target model**: `z-lab/Qwen3-4B-PARO` (text-only, `Qwen3ForCausalLM` — already supported architecture).

---

## Executive Summary

ParoQuant adds exactly **one operation** before the standard `quantizedMatmul`: a pairwise Givens rotation of the input activations. The Python MLX implementation is ~150 lines of Python + 75 lines of Metal. All building blocks exist in Swift MLX (`MLXFast.metalKernel`, `quantizedMM`). This is a straightforward port.

### Scope

| What | Status |
|---|---|
| `RotateQuantizedLinear` module (Swift) | To implement |
| Metal rotation kernel (port from Python) | To implement |
| AutoAWQ→MLX weight conversion | To implement |
| Weight loading & layer patching | To implement |
| `Qwen3ForCausalLM` model architecture | Already supported |
| `quantizedMM` (standard INT4 dequant) | Already exists |
| `MLXFast.metalKernel` (custom Metal dispatch) | Already exists |
| Qwen3.5 hybrid attention architecture | Out of scope (separate effort) |

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

For Qwen3-4B (`inputDims=2560`, `krot=8`): ~120KB per layer, ~4.3MB total for 36 layers. Negligible overhead.

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

## Actual Weight Format (AutoAWQ, not native MLX)

**Critical finding**: The `z-lab/Qwen3-4B-PARO` safetensors use **AutoAWQ packing format**, not native MLX format. This was missed in the initial research.

Actual weight key suffixes in the safetensors: `qweight`, `qzeros`, `scales`, `theta`, `pairs`, `channel_scales`, `weight`.

Example for `model.layers.0.mlp.gate_proj`:

| Key suffix | Shape | Type | Notes |
|---|---|---|---|
| `.qweight` | `[2560, 1216]` | I32 | AutoAWQ-packed INT4 weights (reordered) |
| `.qzeros` | `[20, 1216]` | I32 | AutoAWQ-packed zero points |
| `.scales` | `[20, 9728]` | F16 | Quantization scales (transposed vs MLX) |
| `.theta` | `[8, 1280]` | F16 | Rotation angles |
| `.pairs` | `[8, 2560]` | I16 | Channel pair indices |
| `.channel_scales` | `[1, 2560]` | F16 | Per-channel rescaling |

Non-quantized layers (embed_tokens, norms): just `.weight` in F16.

The Python code's `_convert_autoawq()` function IS required — it converts:
- `.qweight` → `.weight` (unpack, undo [0,2,4,6,1,3,5,7] reorder, repack as MLX sequential uint32)
- `.qzeros` + `.scales` → `.biases` (compute `-scales * zeros`, transpose)
- `.scales` → `.scales` (transpose)

---

## Implementation Plan

### Step 1: Port the Metal kernel (~30 min)

**Source**: `paroquant/kernels/metal/rotation.metal` (75 lines)
**Target**: Inline string constant in Swift

The Metal kernel is a JIT-compiled template. The Python version uses `mx.fast.metal_kernel()` with string formatting for template constants. The Swift equivalent is `MLXFast.metalKernel()` which works identically.

```
Python: mx.fast.metal_kernel(name:, input_names:, output_names:, source:)
Swift:  MLXFast.metalKernel(name:, inputNames:, outputNames:, source:)
```

The kernel source ports verbatim — it's already Metal Shading Language. Only the Python `{{ }}` brace escaping needs to become literal `{ }` in the Swift string.

**Metal kernel template** (from Python, cleaned up for Swift):

```metal
// Grid:  (ceil(batch / ROWS_PER_TILE) * half_group, num_groups, 1)
// Threadgroup: (half_group, 1, 1)

constexpr int ROWS_PER_TILE = \(rowsPerTile);
constexpr int MAX_KROT      = \(maxKrot);

const int batch_size  = params[0];
const int hidden_size = params[1];
const int krot        = params[2];
const int group_size  = params[3];

const int half_gs     = group_size / 2;
const int half_hidden = hidden_size / 2;

const int tile_idx  = threadgroup_position_in_grid.x;
const int group_idx = threadgroup_position_in_grid.y;
const int tid       = thread_index_in_threadgroup;

if (tid >= half_gs) return;

// Load rotation coefficients into registers
float cos_vals[MAX_KROT], sin_vals[MAX_KROT];
int   pair_vals[MAX_KROT];

for (int k = 0; k < krot; k++) {
    int idx = k * half_hidden + group_idx * half_gs + tid;
    cos_vals[k]  = float(cos_theta[idx]);
    sin_vals[k]  = float(sin_theta[idx]);
    pair_vals[k] = int(packed_pairs[idx]);
}

// Load activation tile into shared memory (fuse channel scales)
threadgroup float tile[\(maxGroupSize) * ROWS_PER_TILE];

const int ch_lo = group_idx * group_size + tid;
const int ch_hi = ch_lo + half_gs;
float scale_lo = float(channel_scales[ch_lo]);
float scale_hi = float(channel_scales[ch_hi]);

for (int r = 0; r < ROWS_PER_TILE; r++) {
    int row = tile_idx * ROWS_PER_TILE + r;
    if (row < batch_size) {
        tile[tid * ROWS_PER_TILE + r]              = float(x[row * hidden_size + ch_lo]) * scale_lo;
        tile[(tid + half_gs) * ROWS_PER_TILE + r]  = float(x[row * hidden_size + ch_hi]) * scale_hi;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Apply pairwise Givens rotations in-place
for (int k = 0; k < krot; k++) {
    int i_local = pair_vals[k] & 0xFFFF;
    int j_local = pair_vals[k] >> 16;
    float c = cos_vals[k], s = sin_vals[k];

    for (int m = 0; m < ROWS_PER_TILE; m++) {
        float a = tile[i_local * ROWS_PER_TILE + m];
        float b = tile[j_local * ROWS_PER_TILE + m];
        tile[i_local * ROWS_PER_TILE + m] = a * c + b * s;
        tile[j_local * ROWS_PER_TILE + m] = b * c - a * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Write results back
for (int r = 0; r < ROWS_PER_TILE; r++) {
    int row = tile_idx * ROWS_PER_TILE + r;
    if (row < batch_size) {
        out[row * hidden_size + ch_lo] = tile[tid * ROWS_PER_TILE + r];
        out[row * hidden_size + ch_hi] = tile[(tid + half_gs) * ROWS_PER_TILE + r];
    }
}
```

### Step 2: Create `RotateQuantizedLinear` as `QuantizedLinear` subclass (~1 hour)

**File**: `tesseract/Features/Agent/ParoQuant/RotateQuantizedLinear.swift`

Direct port of `paroquant/inference/backends/mlx/modules.py` (~90 lines → ~120 lines Swift).

**Critical design constraint**: `RotateQuantizedLinear` **must subclass `QuantizedLinear`** (which itself subclasses `Linear`). The Qwen3 model declares its projection layers as `@ModuleInfo var wq: Linear`. The `ModuleInfo<T>` setter does `value as? T` — if T is `Linear`, the replacement module must be a `Linear` subclass to pass the cast. `QuantizedLinear` already works because it inherits from `Linear`. Our class must follow the same pattern.

```swift
import Foundation
import MLX
import MLXNN

/// Pairwise Givens rotation + quantized matmul.
///
/// Subclasses `QuantizedLinear` so it can replace `Linear` in `@ModuleInfo` slots
/// via `update(modules:)`. Only overrides `callAsFunction` to insert the rotation
/// step before the standard quantized matmul.
open class RotateQuantizedLinear: QuantizedLinear {

    // Rotation parameters — discovered by Module reflection for update(parameters:)
    let theta: MLXArray                                    // [krot, inputDims/2]
    let pairs: MLXArray                                    // [krot, inputDims] int16
    @ParameterInfo(key: "channel_scales") var channelScales: MLXArray  // [1, inputDims]

    // Cached rotation data (computed once on first forward)
    private var cachedCos: MLXArray?
    private var cachedSin: MLXArray?
    private var packedPairs: MLXArray?
    private var scalesFlat: MLXArray?
    private var _dim: Int = 0
    private var halfGroup: Int = 0
    private var numGroups: Int = 0
    private var _krot: Int = 0

    init(inputDims: Int, outputDims: Int, hasBias: Bool, groupSize: Int, bits: Int, krot: Int) {
        self.theta = MLXArray.zeros([krot, inputDims / 2])
        self.pairs = MLXArray.zeros([krot, inputDims], dtype: .int16)
        self._channelScales = ParameterInfo(wrappedValue: MLXArray.ones([1, inputDims]),
                                            key: "channel_scales")

        // QuantizedLinear.init(weight:bias:scales:biases:groupSize:bits:)
        // provides the direct-arrays initializer — avoids re-quantizing zeros
        super.init(
            weight: MLXArray.zeros([outputDims, inputDims * bits / 32], dtype: .uint32),
            bias: hasBias ? MLXArray.zeros([outputDims]) : nil,
            scales: MLXArray.zeros([outputDims, inputDims / groupSize]),
            biases: MLXArray.zeros([outputDims, inputDims / groupSize]),
            groupSize: groupSize,
            bits: bits
        )
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        ensureCached()
        let shape = x.shape
        let rotated = rotate(x.reshaped(-1, _dim))
        var y = quantizedMM(
            rotated.reshaped(shape), weight,
            scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits
        )
        if let bias { y = y + bias }
        return y
    }

    // ... ensureCached(), packPairs(), rotate() methods ...
}
```

**Why `QuantizedLinear` subclass works**:
- Inherits `weight`, `scales`, `biases`, `bias` from `QuantizedLinear`/`Linear`
- These are discovered by `Module`'s reflection for `update(parameters:)` ✓
- Passes `value as? Linear` cast in `ModuleInfo<Linear>` setter ✓
- `theta`, `pairs` as plain `let` properties map directly to weight keys (names match) ✓
- `channelScales` uses `@ParameterInfo(key: "channel_scales")` to map Swift name → checkpoint key ✓

**How `krot` is resolved**: `patchRotationLayers()` infers `krot` per-layer from `weights["\(prefix).theta"].shape[0]`, matching the Python code: `krot=weights[f"{prefix}.theta"].shape[0]`. It is NOT taken from the global config — this allows for hypothetical per-layer krot variation.

### Step 3: AutoAWQ → MLX weight conversion (~45 min)

**File**: `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift`

Port of `_convert_autoawq()` and `_pack_mlx()` from `load.py`. This is needed because the PARO safetensors use AutoAWQ packing, not native MLX format.

The conversion per quantized layer:
1. **`qweight` → `weight`**: Unpack int32 → 8×uint4, undo [0,2,4,6,1,3,5,7] reorder, transpose, repack as MLX sequential uint32
2. **`qzeros` + `scales` → `biases`**: Unpack qzeros same way, compute `biases = -scales * zeros`, transpose to MLX layout
3. **`scales` → `scales`**: Transpose from `[groups, outputDims]` to `[outputDims, groups]`
4. **`theta`, `pairs`, `channel_scales`**: Pass through as-is (reshape `channel_scales` to `[1, dim]` if needed)

This can be done with MLXArray ops or as a NumPy-style preprocessing step using `vDSP`/`Accelerate`. Since this runs once at model load, CPU performance is fine.

**Detection**: Layers that have both `.qweight` AND `.theta` keys need the full conversion. Layers with just `.weight` (embed_tokens, norms) pass through unchanged.

### Step 4: Weight loading & layer patching (~1 hour)

**File**: `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift` (same file as Step 3)

The load flow closely follows the Python `load()` function. We need a custom load path because:
- Standard `loadWeights()` (Load.swift) doesn't know about AutoAWQ format
- Standard `loadWeights()` uses `verify: [.all]` which would reject unknown keys like `.theta`
- We need to convert weights BEFORE applying them to the model

**Integration point**: Extend `LLMActor.loadModel()` to detect ParoQuant and use a custom load helper. This reuses existing factory components (model creation via `typeRegistry`, tokenizer loading) but handles weight conversion/patching ourselves.

```swift
// In LLMActor.loadModel():
func loadModel(from directory: URL) async throws -> (AgentTokenizer, promptStartsThinking: Bool) {
    let format = Self.detectToolCallFormat(directory: directory)
    let quantMethod = Self.detectQuantMethod(directory: directory)

    if quantMethod == "paroquant" || quantMethod == "paroquent" {
        return try await loadParoQuantModel(from: directory, toolCallFormat: format)
    }

    // ... existing path via loadModelContainer(configuration:) ...
}
```

**Custom load path (`loadParoQuantModel`)**:

This reuses the factory's existing components — `typeRegistry.createModel()` for model instantiation, `loadTokenizer(configuration:hub:)` for tokenizer loading, EOS override from `generation_config.json`, and `LLMUserInputProcessor`/`messageGenerator` setup. Only the weight conversion and layer patching steps are custom.

```swift
private func loadParoQuantModel(from directory: URL, toolCallFormat: ToolCallFormat?) async throws
    -> (AgentTokenizer, promptStartsThinking: Bool)
{
    // 1. Parse config.json — same as LLMModelFactory._load() lines 482-497
    let configData = try Data(contentsOf: directory.appending(component: "config.json"))
    let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

    // 2. Read quantization params from config (do NOT hardcode)
    let paroConfig = readParoQuantConfig(configData)  // → (bits: 4, groupSize: 128, krot: 8)

    // 3. Create model via standard typeRegistry (Qwen3 architecture is unchanged)
    let model = try await LLMModelFactory.shared.typeRegistry
        .createModel(configuration: configData, modelType: baseConfig.modelType)

    // 4. EOS token override from generation_config.json — same as factory lines 508-517
    var eosTokenIds = Set(baseConfig.eosTokenIds?.values ?? [])
    let genConfigURL = directory.appending(component: "generation_config.json")
    if let genData = try? Data(contentsOf: genConfigURL),
       let genConfig = try? JSONDecoder().decode(GenerationConfigFile.self, from: genData),
       let genEos = genConfig.eosTokenIds?.values {
        eosTokenIds = Set(genEos)
    }
    var mutableConfig = ModelConfiguration(directory: directory, toolCallFormat: toolCallFormat)
    mutableConfig.eosTokenIds = eosTokenIds

    // 5. Load raw safetensors
    var weights = loadAllSafetensors(from: directory)

    // 6. Model-specific sanitization
    weights = model.sanitize(weights: weights)

    // 7. Convert AutoAWQ format → MLX format (for layers with .qweight)
    weights = convertAutoAWQ(weights, groupSize: paroConfig.groupSize)

    // 8. Patch rotation layers: swap Linear → RotateQuantizedLinear where .theta exists
    //    krot is inferred per-layer from weights["\(prefix).theta"].shape[0]
    //    (matches Python: `krot=weights[f"{prefix}.theta"].shape[0]`)
    patchRotationLayers(model: model, weights: weights,
                        bits: paroConfig.bits, groupSize: paroConfig.groupSize)

    // 9. Quantize ONLY layers that have .scales but NOT .theta
    //    Rotation layers are already RotateQuantizedLinear; skip them.
    quantize(model: model) { path, module in
        guard weights["\(path).scales"] != nil else { return nil }
        guard weights["\(path).theta"] == nil else { return nil }
        return (paroConfig.groupSize, paroConfig.bits, .affine)
    }

    // 10. Load weights into model — use shapeMismatch to catch bad AWQ conversions
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys, .shapeMismatch])

    eval(model)

    // 11. Load tokenizer — same API as factory line 524
    let tokenizer = try await loadTokenizer(configuration: mutableConfig, hub: defaultHubApi)

    // 12. Create processor with messageGenerator — same as factory lines 532-541
    let messageGenerator: MessageGenerator =
        if let llmModel = model as? LLMModel {
            llmModel.messageGenerator(tokenizer: tokenizer)
        } else {
            DefaultMessageGenerator()
        }
    let processor = LLMUserInputProcessor(
        tokenizer: tokenizer, configuration: mutableConfig,
        messageGenerator: messageGenerator)

    // 13. Assemble ModelContext → ModelContainer
    let context = ModelContext(
        configuration: mutableConfig, model: model,
        processor: processor, tokenizer: tokenizer)
    let container = ModelContainer(context: context)
    // ... store container, create AgentTokenizer, etc.
}
```

**Important**: Step 9 only quantizes layers where `.scales` exist in the checkpoint but `.theta` does NOT. This avoids force-quantizing layers the checkpoint intentionally left in full precision. For Qwen3-4B-PARO, embed_tokens has no `.scales` key (it's plain F16), so nothing gets force-quantized — only the already-quantized layers get their `QuantizedLinear`/`RotateQuantizedLinear` replacements.

### Step 5: Config parsing (~15 min)

The `BaseConfiguration.QuantizationContainer` already skips `quant_method` during decoding (line 118 of `BaseConfiguration.swift`). The `group_size` and `bits` fields parse normally. We need two things:

1. **Detection**: `detectQuantMethod()` reads `quantization_config.quant_method` from `config.json`. We're already doing JSON parsing in `LLMActor` for `model_type` in `detectToolCallFormat()`.

2. **Parameter extraction**: `readParoQuantConfig()` reads `bits`, `group_size`, `krot` from the same `quantization_config` block. These values must NOT be hardcoded — read them from the model's config.json so the loader works with any PARO checkpoint shape.

```swift
struct ParoQuantConfig {
    let bits: Int      // from quantization_config.bits
    let groupSize: Int // from quantization_config.group_size
    let krot: Int      // from quantization_config.krot
}
```

**Note on field names**: The actual published PARO configs use `quantization_config` (standard HF key), with `bits`, `group_size`, `krot`, and `quant_method`. The Qwen3-4B model has a typo: `"paroquent"` instead of `"paroquant"`. Detection must accept both spellings.

No changes to vendored mlx-swift-lm code needed.

### Step 6: Model registration (~10 min)

Add `z-lab/Qwen3-4B-PARO` to `ModelDefinition.swift`:

```swift
ModelDefinition(
    id: "qwen3-4b-paro",
    displayName: "Qwen3-4B PARO (INT4)",
    description: "ParoQuant INT4 — near-FP16 quality at half the size of 8-bit.",
    category: .agent,
    source: .huggingFace(
        repo: "z-lab/Qwen3-4B-PARO",
        requiredExtension: "safetensors"
    ),
    sizeDescription: "~2.7 GB",
    dependencies: []
)
```

---

## File Structure

```
tesseract/Features/Agent/ParoQuant/
├── RotateQuantizedLinear.swift   # QuantizedLinear subclass: rotation + quantizedMM
└── ParoQuantLoader.swift         # AutoAWQ conversion, layer patching, custom load path
```

The Metal kernel source is an inline string constant in `RotateQuantizedLinear.swift`. No separate file needed — the kernel is compiled at runtime via `MLXFast.metalKernel()`.

2 new files, ~350 lines total.

---

## Mapping: Python → Swift

| Python (paroquant) | Swift (tesseract) | Notes |
|---|---|---|
| `modules.py:RotateQuantizedLinear` | `RotateQuantizedLinear.swift` | `nn.Module` → `QuantizedLinear` subclass |
| `modules.py:_pack_pairs()` | `RotateQuantizedLinear.packPairs()` | NumPy → MLXArray or vDSP |
| `modules.py:_rotate()` | `RotateQuantizedLinear.rotate()` | `mx.fast.metal_kernel()` → `MLXFast.metalKernel()` |
| `rotation.metal` | Inline string in `RotateQuantizedLinear.swift` | Verbatim Metal, fix brace escaping |
| `rotation.py:get_rotation_kernel()` | Static cached property | `@lru_cache` → `nonisolated(unsafe) static var` |
| `load.py:_convert_autoawq()` | `ParoQuantLoader.convertAutoAWQ()` | Unpack/reorder/repack int4 weights |
| `load.py:_pack_mlx()` | `ParoQuantLoader.packMLX()` | uint8 → uint32 sequential packing |
| `load.py:_patch_rotation_layers()` | `ParoQuantLoader.patchRotationLayers()` | Walk model tree, swap modules |
| `load.py:load()` | `LLMActor.loadParoQuantModel()` | Custom load path reusing factory components |

---

## Detailed API Mapping

### Metal kernel creation

```python
# Python
mx.fast.metal_kernel(
    name=f"paro_rotate_r{rows_per_tile}",
    input_names=["x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales", "params"],
    output_names=["out"],
    source=_SOURCE.format(ROWS_PER_TILE=rows_per_tile, MAX_GROUP_SIZE=128, MAX_KROT=16),
)
```

```swift
// Swift
MLXFast.metalKernel(
    name: "paro_rotate_r\(rowsPerTile)",
    inputNames: ["x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales", "params"],
    outputNames: ["out"],
    source: metalSource(rowsPerTile: rowsPerTile, maxGroupSize: 128, maxKrot: 16)
)
```

### Metal kernel invocation

```python
# Python
get_rotation_kernel(tile)(
    inputs=[x, packed_pairs, cos, sin, scales_flat, params],
    output_shapes=[x.shape],
    output_dtypes=[x.dtype],
    grid=(ceil(batch/tile) * half_group, num_groups, 1),
    threadgroup=(half_group, 1, 1),
)[0]
```

```swift
// Swift
getRotationKernel(tile: tile)(
    [x, packedPairs, cos, sin, scalesFlat, params],
    grid: (ceilDiv(batch, tile) * halfGroup, numGroups, 1),
    threadGroup: (halfGroup, 1, 1),
    outputShapes: [x.shape],
    outputDTypes: [x.dtype]
)[0]
```

### Quantized matmul

```python
# Python
mx.quantized_matmul(rotated, self.weight, scales=self.scales, biases=self.biases,
                     transpose=True, group_size=self.group_size, bits=self.bits)
```

```swift
// Swift
quantizedMM(rotated, weight, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits)
```

---

## Weight Key Mapping

Actual keys in `z-lab/Qwen3-4B-PARO` safetensors (AutoAWQ format):

For a layer like `model.layers.0.self_attn.q_proj`:

| Key in safetensors | Format | Shape | After conversion |
|---|---|---|---|
| `.qweight` | AutoAWQ packed I32 | `[2560, 320]` | → `.weight` MLX packed U32 |
| `.qzeros` | AutoAWQ packed I32 | `[20, 320]` | → (consumed to compute `.biases`) |
| `.scales` | F16 (transposed) | `[20, 2560]` | → `.scales` F16 `[2560, 20]` |
| `.theta` | F16 | `[8, 1280]` | → `.theta` (pass through) |
| `.pairs` | I16 | `[8, 2560]` | → `.pairs` (pass through) |
| `.channel_scales` | F16 | `[1, 2560]` | → `.channel_scales` (pass through) |

Non-quantized layers:
| Key | Shape | Notes |
|---|---|---|
| `model.embed_tokens.weight` | `[151936, 2560]` F16 | Plain embedding, no quantization |
| `model.layers.*.input_layernorm.weight` | `[2560]` F16 | RMSNorm weight |
| `model.norm.weight` | `[2560]` F16 | Final norm |

Detection: if `weights["\(prefix).theta"]` exists → that layer needs `RotateQuantizedLinear`.

---

## AutoAWQ Conversion Detail

Port of `_convert_autoawq()` from `load.py`. This is the most code-dense part.

### Unpack AutoAWQ int32 → raw uint4

AutoAWQ packs 8 × 4-bit values into each int32 with a **reordered** layout: `[0, 2, 4, 6, 1, 3, 5, 7]`. Must undo this reorder.

```python
# Python reference
_SHIFTS = [0, 4, 8, 12, 16, 20, 24, 28]
_INV_REORDER = [0, 4, 1, 5, 2, 6, 3, 7]

def _unpack_and_reorder(packed):
    raw = ((packed[:, :, None] >> _SHIFTS) & 0xF).astype(uint8)
    return raw[:, :, _INV_REORDER].reshape(packed.shape[0], -1)
```

### Repack as MLX sequential uint32

```python
# Python reference
def _pack_mlx(w):
    w = w.reshape(w.shape[0], -1, 8)  # 8 values per uint32
    p = w[:, :, 0].astype(uint32)
    for i in range(1, 8):
        p |= w[:, :, i].astype(uint32) << (4 * i)
    return p
```

### Full conversion per layer

```python
# For each prefix with both .qweight and .theta:
weight = _pack_mlx(_unpack_and_reorder(qweight).T)  # transpose + repack
scales = scales.T.copy()                              # transpose
biases = (-scales * _unpack_and_reorder(qzeros).astype(float32)).T  # compute + transpose
```

In Swift, this can use Foundation `Data` manipulation or `Accelerate` framework operations on the raw arrays. Since this runs once at load time, using `MLXArray` ops on CPU is also acceptable.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| AutoAWQ unpack/reorder logic has off-by-one | Medium | Validate against Python output on first 2 layers |
| Metal kernel behaves differently in Swift MLX | Low | Same C++ backend, same Metal compiler. Kernel source is identical. |
| `QuantizedLinear` subclass init doesn't match parent | Medium | Use the `init(weight:bias:scales:biases:groupSize:bits:)` initializer |
| Module reflection misses `theta`/`pairs`/`channelScales` | Low | These are stored `let` MLXArray properties — Module's Mirror finds them |
| `update(modules:)` rejects type mismatch | Low | `RotateQuantizedLinear` IS-A `QuantizedLinear` IS-A `Linear` — cast succeeds |
| Performance regression from rotation overhead | Low | Python benchmarks show <10% overhead. Metal kernel is fully parallelized. |

---

## Validation Plan

1. **Weight conversion test**: Convert layer 0 weights, compare output arrays against Python `_convert_autoawq()` output
2. **Rotation test**: Feed known input through `RotateQuantizedLinear`, compare against Python `RotateQuantizedLinear` output
3. **Smoke test**: Generate 10 tokens from the full model, verify non-garbage output
4. **Quality test**: Run agent benchmark suite (S1-S7) with PARO model, compare pass rate to MLX-8bit Qwen3
5. **Performance test**: Measure tok/s vs Qwen3-4B-8bit (expect similar or slightly faster due to INT4 vs INT8)

---

## Size Comparison

| Model | Quant | Size | Quality (approx) |
|---|---|---|---|
| `Qwen3-4B-MLX-8bit` | MLX affine 8-bit | 4.5 GB | ~98% of FP16 |
| `Qwen3-4B-PARO` | ParoQuant INT4 | ~2.7 GB | ~99% of FP16 |
| Standard MLX 4-bit | MLX affine 4-bit | ~2.5 GB | ~95% of FP16 |

ParoQuant INT4 gives better quality than standard INT4 at similar size, and uses ~40% less memory than our current 8-bit default.

---

## Future: Qwen3.5-4B-PARO

The Qwen3.5 PARO model (3.82 GB) additionally requires:
- Hybrid attention architecture (linear + full) — separate implementation effort
- Dynamic quantization exclusions (some layers kept in FP16 via `dynamic` config)
- Vision encoder (can be ignored for text-only use)

Start with Qwen3-4B-PARO (text-only, standard Qwen3 architecture) to validate the ParoQuant integration. Qwen3.5 support can be added once the Qwen3.5 architecture is supported in mlx-swift-lm upstream.

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
| Config uses `paroquant_config` with `nbit` not `quantization_config` with `bits` | P1 | **INVALID** — actual config.json uses `quantization_config` with `bits`, `group_size`, `krot`. Reviewer hallucinated field names. However, the sub-point about hardcoding `(128, 4)` was valid. | Fixed: load path now reads `bits`, `group_size`, `krot` from config.json via `readParoQuantConfig()` instead of hardcoding |
| Custom loader uses wrong API names (`loadTokenizer(from:)`, `JSONDecoder()` wrong) | P2 | **Partially valid** — `JSONDecoder()` is actually correct (factory uses it at line 493), but `loadTokenizer` signature was wrong, and EOS override + `messageGenerator`/`LLMUserInputProcessor` setup were missing | Fixed: pseudocode now mirrors factory lines 508-541 exactly — EOS override, `loadTokenizer(configuration:hub:)`, `messageGenerator`, `LLMUserInputProcessor` |
| `verify: [.noUnusedKeys]` too weak after AWQ conversion | P2 | **Valid** — `.shapeMismatch` exists and would catch bad conversions | Fixed: now uses `[.noUnusedKeys, .shapeMismatch]` |

### Review 3 (2026-03-09)

| Finding | Severity | Verdict | Resolution |
|---|---|---|---|
| `channelScales` won't load — Swift name ≠ checkpoint key `channel_scales` | P1 | **Valid** — Module reflection uses property name, so `channelScales` maps to key `channelScales`, not `channel_scales` | Fixed: use `@ParameterInfo(key: "channel_scales") var channelScales` |
| `patchRotationLayers` doesn't pass `krot` | P2 | **Valid** — `krot` must be inferred per-layer from `weights["\(prefix).theta"].shape[0]`, matching Python's `krot=weights[f"{prefix}.theta"].shape[0]` | Fixed: added note that krot is inferred per-layer, not from global config |
| Other PARO models use `paroquant_config`/`nbit`/`"ParoQuant"` schema | P2 | **INVALID** — verified all 13 published z-lab PARO models (Qwen3, Qwen3.5, Llama, DeepSeek). ALL use `quantization_config` with `bits`/`group_size`/`krot`. Models cited by reviewer (`Qwen3-4B-Thinking-2507-PARO`, `Meta-Llama-3-70B-PARO`) don't exist. | No change needed; scope is `z-lab/Qwen3-4B-PARO` only per user instruction |
| Factory uses `JSONDecoder.json5()` not `JSONDecoder()` | P3 | **INVALID** — `json5` does not appear anywhere in mlx-swift-lm. Factory uses plain `JSONDecoder()` at lines 493, 512. Verified via grep across entire checkout. | No change needed |
