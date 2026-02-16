# Image Generation: FLUX.2-klein-4B on Swift/MLX

> Ported from [mflux](https://github.com/filipstrand/mflux) (Python/MLX) to Swift/MLX.
> Date: 2026-02-13. Status: **builds, not yet runtime-tested**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Package Structure](#package-structure)
- [Model Configuration](#model-configuration)
- [Inference Pipeline](#inference-pipeline)
- [Transformer](#transformer)
- [Text Encoder (Qwen3)](#text-encoder-qwen3)
- [VAE Decoder](#vae-decoder)
- [Scheduler](#scheduler)
- [Latent Creator](#latent-creator)
- [Weight Loading](#weight-loading)
- [App Integration](#app-integration)
- [Build System](#build-system)
- [Layout Conventions (NCHW vs NHWC)](#layout-conventions-nchw-vs-nhwc)
- [MLX Swift API Gotchas](#mlx-swift-api-gotchas)
- [Performance Notes](#performance-notes)
- [Known Issues & Future Work](#known-issues--future-work)
- [Reference Mapping (Python → Swift)](#reference-mapping-python--swift)

---

## Overview

FLUX.2-klein-4B is a **4-billion parameter distilled** image generation model from Black Forest Labs. It produces high-quality images in just **4 denoising steps** (vs. 20-50 for standard diffusion models). The architecture uses a hybrid dual-stream / single-stream transformer with flow matching.

**Why Klein-4B**: Smallest FLUX.2 variant. 4-step distilled inference. Fits in ~8 GB unified memory at bf16 (~3.5 GB at 4-bit if quantized).

### High-Level Flow

```
User Prompt
  → [Qwen3 Tokenizer] → token IDs
  → [Qwen3 Text Encoder (36 layers)] → prompt_embeds [1, seq_len, 7680]

Random Seed
  → [Latent Creator] → packed_latents [1, 4096, 128]  (for 1024×1024)

4× Denoising Loop:
  → [Flux2Transformer: 5 double-stream + 20 single-stream blocks]
  → Flow Match Euler step

Denoised latents
  → unpack → unpatchify → [VAE Decoder] → RGB image [1, 3, H, W]
  → clamp [0,1] → uint8 → CGImage
```

---

## Architecture

### Dual-Stream → Single-Stream Transformer

The FLUX.2 transformer is NOT a standard U-Net. It processes image and text tokens through two separate streams that interact via joint cross-attention, then merges them into a single stream.

```
                ┌─────────────────────────┐
                │    Timestep Embedding    │
                │   + Guidance Embedding   │
                └────────────┬────────────┘
                             │ temb
                             ▼
           ┌──────────── Modulation ─────────────┐
           │                                     │
     ┌─────▼─────┐                       ┌───────▼───────┐
     │ Image Emb  │                       │  Text Emb     │
     │ x_embedder │                       │ ctx_embedder  │
     └─────┬─────┘                       └───────┬───────┘
           │                                     │
           ▼                                     ▼
     ┌─────────────── 5× Double-Stream Block ──────────────┐
     │                                                      │
     │  Image: LayerNorm → Modulate → ──┐                  │
     │  Text:  LayerNorm → Modulate → ──┤                  │
     │                                  ▼                   │
     │                         Joint Attention              │
     │                    (concat Q/K/V, split output)      │
     │                                  │                   │
     │  Image: gate×attn + residual → FFN → gate×FFN+res  │
     │  Text:  gate×attn + residual → FFN → gate×FFN+res  │
     └──────────────────────────────────────────────────────┘
           │                                     │
           └──────── concatenate [text, image] ──┘
                             │
                             ▼
     ┌─────────── 20× Single-Stream Block ──────────────┐
     │  LayerNorm → Modulate                            │
     │  → Parallel Self-Attention + SwiGLU MLP          │
     │  → gate × output + residual                      │
     └──────────────────────────────────────────────────┘
                             │
                             ▼
                    strip text tokens
                             │
                    AdaLayerNorm → proj_out
                             │
                             ▼
                      noise prediction
```

### Key Design Decisions

1. **Modulation over LayerNorm**: Instead of affine parameters in LayerNorm, FLUX uses learned shift/scale/gate from timestep embeddings. LayerNorms all have `affine: false`.

2. **4D RoPE**: Position embeddings use 4 axes (t, h, w, layer) with `axes_dim = [32, 32, 32, 32]`, concatenated to form 128-dim RoPE. This is applied in the complex-interleaved style (not rotate-half).

3. **SwiGLU**: All feed-forward networks use SwiGLU activation: `silu(gate) * up`.

4. **Parallel attention + MLP**: Single-stream blocks fuse Q/K/V and MLP projections into one `Linear(dim → 3×innerDim + 2×mlpHidden)`, then process attention and MLP in parallel.

5. **No bias**: All Linear layers in the transformer use `bias: false`.

6. **RMSNorm on Q/K**: After projection, Q and K go through RMSNorm (eps=1e-5) before attention. This is per-head (dim = head_dim = 128).

7. **bfloat16**: All computation is in bfloat16 except: RMSNorm/GroupNorm internals (float32), RoPE application (float32), attention (float32 Q/K/V for stability via `scaledDotProductAttention`).

---

## Package Structure

```
Vendor/mlx-image-swift/
├── Package.swift
└── Sources/MLXImageGen/
    ├── Flux2Configuration.swift          # All model hyperparameters
    ├── Flux2Pipeline.swift               # Top-level actor: prompt → CGImage
    ├── Transformer/
    │   ├── Flux2Transformer.swift        # Main model (5 double + 20 single blocks)
    │   ├── Flux2TransformerBlock.swift    # Double-stream block (image + text)
    │   ├── Flux2SingleTransformerBlock.swift  # Single-stream block
    │   ├── Flux2Attention.swift          # Joint cross-attention with added_kv
    │   ├── Flux2ParallelSelfAttention.swift  # Fused QKV+MLP, parallel processing
    │   ├── Flux2FeedForward.swift        # SwiGLU FFN
    │   ├── Flux2Modulation.swift         # Learned shift/scale/gate from temb
    │   ├── Flux2PosEmbed.swift           # 4D RoPE position embeddings
    │   ├── Flux2TimestepEmbeddings.swift  # Sinusoidal timestep + guidance embed
    │   ├── AdaLayerNormContinuous.swift   # Adaptive LayerNorm for output
    │   └── AttentionUtils.swift          # Shared QKV processing, attention, RoPE
    ├── TextEncoder/
    │   ├── Qwen3TextEncoder.swift        # Full Qwen3 model (6 classes in one file)
    │   └── Flux2PromptEncoder.swift      # Tokenize + encode + extract hidden states
    ├── VAE/
    │   ├── Flux2VAEDecoder.swift         # Full decoder pipeline
    │   ├── VAEResnetBlock.swift          # GroupNorm→SiLU→Conv→GroupNorm→SiLU→Conv
    │   ├── VAEAttentionBlock.swift       # Mid-block self-attention
    │   └── VAEComponents.swift           # NCHWConv2d, NCHWGroupNorm, Upsample2D
    ├── Scheduler/
    │   └── FlowMatchEulerScheduler.swift # Flow matching with empirical mu
    ├── LatentCreator/
    │   └── Flux2LatentCreator.swift      # Packing, unpacking, grid IDs
    └── WeightLoading/
        └── Flux2WeightLoader.swift       # Safetensors loader with key remapping
```

**Total: 23 Swift files** in the MLXImageGen library.

---

## Model Configuration

All hyperparameters are in `Flux2Configuration.swift` as nested structs with static defaults.

### Transformer Config (Klein-4B)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `patchSize` | 1 | No spatial patching |
| `inChannels` | 128 | Latent dim after packing |
| `outChannels` | 128 | Same as input |
| `numLayers` | 5 | Double-stream blocks |
| `numSingleLayers` | 20 | Single-stream blocks |
| `attentionHeadDim` | 128 | Per-head dimension |
| `numAttentionHeads` | 24 | Total heads |
| `innerDim` | 3072 | = 24 × 128 |
| `jointAttentionDim` | 7680 | Text embed dim (3 layers × 2560) |
| `timestepGuidanceChannels` | 256 | Sinusoidal embedding dim |
| `mlpRatio` | 3.0 | FFN hidden = dim × 3.0 = 9216 |
| `axesDimsRope` | [32,32,32,32] | 4D RoPE, sum=128=headDim |
| `ropeTheta` | 2000 | RoPE base frequency |
| `guidanceEmbeds` | true | Uses classifier-free guidance |

### Text Encoder Config (Qwen3)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocabSize` | 151936 | Qwen3 tokenizer |
| `hiddenSize` | 2560 | Embedding dimension |
| `numHiddenLayers` | 36 | Decoder layers |
| `numAttentionHeads` | 32 | Query heads |
| `numKeyValueHeads` | 8 | GQA: 4× fewer KV heads |
| `intermediateSize` | 9728 | MLP hidden dim |
| `maxPositionEmbeddings` | 40960 | Context window |
| `ropeTheta` | 1,000,000 | Much higher than transformer |
| `rmsNormEps` | 1e-6 | RMSNorm epsilon |
| `headDim` | 128 | = hiddenSize / numAttentionHeads (actually 80 = 2560/32, BUT config says 128) |
| `attentionBias` | false | No bias in Q/K/V/O projections |
| `attentionScaling` | 1.0 | RoPE scaling factor |
| `hiddenStateLayers` | [9, 18, 27] | Which layer outputs to extract |

**Note on headDim**: The config uses `headDim: 128` which is the *transformer's* head dimension. For the text encoder, the actual per-head dimension is `hiddenSize / numAttentionHeads = 2560 / 32 = 80`. The `headDim: 128` in the TextEncoder config controls the Q/K projection dimensions for the text encoder attention, which projects to `numAttentionHeads * headDim` output features. This means Q/K/V projections are `Linear(2560 → 32 × 128 = 4096)` — projecting up from the hidden dimension.

### VAE Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| `inChannels` | 32 | Latent channels |
| `outChannels` | 3 | RGB output |
| `latentChannels` | 32 | Same as inChannels |
| `blockOutChannels` | [128, 256, 512, 512] | Channel progression |
| `layersPerBlock` | 2 | ResNet blocks per up-block |
| `normNumGroups` | 32 | GroupNorm groups |
| `scaleFactor` | 8 | Spatial downscale factor |

---

## Inference Pipeline

**File**: `Flux2Pipeline.swift`

The pipeline is a Swift `actor` (for thread safety, following the same pattern as TTS). Key flow:

### 1. Initialization (`init(modelDirectory:)`)

```swift
public actor Flux2Pipeline {
    public init(modelDirectory: URL, config: Flux2Configuration.Pipeline = .klein4B) async throws
}
```

1. Instantiate all three model components (transformer, text encoder, VAE decoder)
2. Load tokenizer via `AutoTokenizer.from(modelFolder:)` (reads `tokenizer.json`)
3. Load weights from safetensors files in three passes:
   - `Flux2WeightLoader.loadTransformerWeights(from:into:)` — prefix: `"transformer"`
   - `Flux2WeightLoader.loadTextEncoderWeights(from:into:)` — prefix: `"text_encoder"`
   - `Flux2WeightLoader.loadVAEDecoderWeights(from:into:)` — prefix: `"vae"` or `"decoder"`
4. Logging via `NSLog` (vendor code pattern)

### 2. Image Generation (`generateImage(...)`)

```swift
public func generateImage(
    prompt: String,
    width: Int = 1024, height: Int = 1024,
    numSteps: Int = 4,
    guidanceScale: Float = 3.5,
    seed: UInt64 = 0,
    onProgress: (@Sendable (Int, Int) -> Void)? = nil
) async throws -> CGImage
```

**Step-by-step**:

1. **Seed handling**: `seed == 0` means random seed
2. **Encode prompt**: `Flux2PromptEncoder.encodePrompt(...)` → `(promptEmbeds, textIds)`
   - `promptEmbeds`: `[1, seq_len, 7680]` — 3 hidden state layers × 2560
   - `textIds`: `[1, seq_len, 4]` — position coordinates `(t=0, h=0, w=0, token_idx)`
   - `eval()` + `GPU.clearCache()` after encoding (text encoder is large)
3. **Create latents**: `Flux2LatentCreator.preparePackedLatents(...)`
   - Random noise `[1, 128, H/16, W/16]` → packed to `[1, (H/16)×(W/16), 128]`
   - Grid position IDs `[1, seq_len, 4]` — `(t, h, w, layer)` coordinates
   - For 1024×1024: `seq_len = 64×64 = 4096`
4. **Create scheduler**: `FlowMatchEulerScheduler(numInferenceSteps: 4, imageSeqLen: 4096)`
5. **Denoising loop** (4 iterations):
   ```
   for i in 0..<numSteps:
       noisePred = transformer(latents, promptEmbeds, timestep, imgIds, txtIds, guidance)
       eval(noisePred)
       latents = scheduler.step(noise: noisePred, timestepIndex: i, latents: latents)
       eval(latents)
   ```
   - `eval()` after each step forces computation (prevents graph buildup)
   - `GPU.clearCache()` after loop completes
6. **VAE decode**:
   ```
   unpackLatents → unpatchifyLatents → vaeDecoder → clip(0,1)
   ```
   - Unpack: `[1, seq_len, 128]` → `[1, 128, H/16, W/16]`
   - Unpatchify: `[1, 128, H/16, W/16]` → `[1, 32, H/8, W/8]`
   - VAE decode: `[1, 32, H/8, W/8]` → `[1, 3, H, W]`
7. **Convert to CGImage**:
   - `[1, 3, H, W]` → `[H, W, 3]` (transpose)
   - float32 × 255 → uint8
   - `CGImage` with 8 bits/component, 24 bits/pixel, RGB, no alpha

---

## Transformer

### Flux2Transformer (`Flux2Transformer.swift`)

**Main module** containing all sub-components.

**Modules**:
- `pos_embed`: `Flux2PosEmbed` — 4D RoPE
- `time_guidance_embed`: `Flux2TimestepGuidanceEmbeddings` — timestep + guidance
- `double_stream_modulation_img`: `Flux2Modulation(dim, modParamSets=2)` — shift/scale/gate for MSA + MLP
- `double_stream_modulation_txt`: `Flux2Modulation(dim, modParamSets=2)` — same for text
- `single_stream_modulation`: `Flux2Modulation(dim, modParamSets=1)` — single set
- `x_embedder`: `Linear(128 → 3072, bias=false)` — image latent embed
- `context_embedder`: `Linear(7680 → 3072, bias=false)` — text embed
- `transformer_blocks`: 5 × `Flux2TransformerBlock`
- `single_transformer_blocks`: 20 × `Flux2SingleTransformerBlock`
- `norm_out`: `AdaLayerNormContinuous(3072, 3072)`
- `proj_out`: `Linear(3072 → 128, bias=false)` — back to latent dim

**Forward pass**:

1. Scale timestep: if max(ts) ≤ 1.0, multiply by 1000 (normalize to training range)
2. Same scaling for guidance
3. Compute `temb = time_guidance_embed(ts, guidance)` — `[B, 3072]`
4. Embed image: `hs = x_embedder(hidden_states)` — `[B, img_seq, 3072]`
5. Embed text: `ehs = context_embedder(encoder_hidden_states)` — `[B, txt_seq, 3072]`
6. Compute RoPE for image and text position IDs separately, then concatenate:
   ```
   imageRoPE = pos_embed(imgIds)   // [img_seq, 128]
   textRoPE  = pos_embed(txtIds)   // [txt_seq, 128]
   concatRoPE = cat([textRoPE, imageRoPE])  // [txt_seq + img_seq, 128]
   ```
   **Important**: text comes FIRST in the concatenation order.
7. Pre-compute modulation params from temb (one set shared across all blocks of each type)
8. Double-stream loop: each block gets `(hs, ehs)` and returns updated `(ehs, hs)`
9. Merge: `hs = cat([ehs, hs], axis=1)` — text first, image second
10. Single-stream loop: each block processes merged `hs`
11. Strip text: `hs = hs[:, txt_seq:, :]` — keep only image tokens
12. Output: `normOut(hs, temb)` → `projOut(hs)` — `[B, img_seq, 128]`

### Flux2Modulation (`Flux2Modulation.swift`)

Converts timestep embedding to shift/scale/gate parameters.

```swift
Linear(dim → dim × 3 × modParamSets, bias=false)
```

- Input: temb `[B, dim]` after SiLU activation
- Output: `[ModulationParams]` array, each containing shift, scale, gate `[B, 1, dim]`
- `modParamSets=2` for double-stream (MSA + MLP), `modParamSets=1` for single-stream

### Flux2PosEmbed (`Flux2PosEmbed.swift`)

4D Rotary Position Embeddings.

- Input: position IDs `[seq_len, 4]` with columns `(t, h, w, layer)`
- For each axis `i` with dimension `axesDim[i]`:
  - Compute 1D RoPE frequencies: `omega = 1 / theta^(2k/dim)` for `k = 0..dim/2`
  - `freqs = pos[:, i] × omega` → `[seq_len, dim/2]`
  - `cos(freqs)`, `sin(freqs)` → `[seq_len, dim/2]`
- Concatenate all 4 axes: `[seq_len, 32+32+32+32]` = `[seq_len, 128]`

### RoPE Application (`AttentionUtils.applyRopeBSHD`)

Uses **complex-interleaved** format (NOT rotate-half like Qwen3/LLaMA):

```swift
// x: [B, H, S, D] reshaped to [B, H, S, D/2, 2]
// x[..., 0] = real, x[..., 1] = imag
// out_real = real * cos + (-imag) * sin
// out_imag = imag * cos + real * sin
```

This matches FLUX/Diffusers convention. The Qwen3 text encoder uses its own rotate-half RoPE.

### Flux2TransformerBlock — Double-Stream (`Flux2TransformerBlock.swift`)

Each block has:
- `norm1`, `norm1_context`: LayerNorm(dim, affine=false)
- `attn`: Flux2Attention (joint, with addedKVProjDim=dim)
- `norm2`, `norm2_context`: LayerNorm(dim, affine=false)
- `ff`, `ff_context`: Flux2FeedForward (SwiGLU)

Forward:
1. Image: `normHS = (1 + scale) * norm1(hs) + shift`
2. Text: `normEHS = (1 + scale) * norm1_context(ehs) + shift`
3. Joint attention: concat text+image Q/K/V → attention → split output
4. Gated residual: `hs = hs + gate * attn_output`
5. Same pattern for FFN stream

### Flux2SingleTransformerBlock (`Flux2SingleTransformerBlock.swift`)

Simpler: one LayerNorm + modulate → `Flux2ParallelSelfAttention` → gated residual.

### Flux2ParallelSelfAttention (`Flux2ParallelSelfAttention.swift`)

Fuses QKV + MLP projections:

```swift
// Single projection: dim → (innerDim × 3) + (mlpHidden × 2)
// innerDim = 24 × 128 = 3072
// mlpHidden = 3072 × 3.0 = 9216
// Total: 3072 × 3 + 9216 × 2 = 9216 + 18432 = 27648
let proj = toQKVMlpProj(x)  // [B, S, 27648]

// Split into QKV and MLP parts
let qkv = proj[..., :9216]      // → split into 3 × [B, S, 3072]
let mlpHidden = proj[..., 9216:]  // → [B, S, 18432]

// Process attention and MLP in parallel
let attnOut = attention(Q, K, V)      // [B, S, 3072]
let mlpOut = SwiGLU(mlpHidden)        // [B, S, 9216]

// Concatenate and project back
let out = toOut(cat([attnOut, mlpOut]))  // Linear(3072+9216 → 3072)
```

### Flux2Attention — Joint Cross-Attention (`Flux2Attention.swift`)

For double-stream blocks with `addedKVProjDim`:

1. Image: Q/K/V from `to_q/k/v(image)`, RMSNorm on Q/K
2. Text: Q/K/V from `add_q/k/v_proj(text)`, RMSNorm on Q/K
3. Concat: `Q = cat([text_Q, image_Q])`, same for K, V (along seq axis)
4. Apply RoPE to concatenated Q/K
5. `scaledDotProductAttention` (via MLXFast, fp32 internally)
6. Split output: `encoder_out = result[:, :txt_seq]`, `hidden_out = result[:, txt_seq:]`
7. Project: `to_out(hidden_out)`, `to_add_out(encoder_out)`

### Flux2FeedForward (`Flux2FeedForward.swift`)

SwiGLU architecture:

```
Linear(dim → innerDim×2)  →  split  →  silu(gate) × up  →  Linear(innerDim → dim)
```

Where `innerDim = dim × mlpRatio = 3072 × 3.0 = 9216`.

### Flux2TimestepGuidanceEmbeddings (`Flux2TimestepEmbeddings.swift`)

1. Sinusoidal embedding: timestep → 256-dim, then `Linear(256 → 3072) → SiLU → Linear(3072 → 3072)`
2. Same for guidance (optional, controlled by `guidanceEmbeds`)
3. Output: `timestep_emb + guidance_emb` → `[B, 3072]`

Sinusoidal details:
- `freqs = exp(-log(10000) × k/half)` for `k = 0..127`
- `emb = [sin(t×f), cos(t×f)]` → then flip to `[cos, sin]` (flipSinToCos=true)

### AdaLayerNormContinuous (`AdaLayerNormContinuous.swift`)

Used for the output normalization:

```
emb = Linear(dim → dim×2)(SiLU(textEmbeddings))
scale, shift = split(emb)
output = (1 + scale) * LayerNorm(x) + shift
```

---

## Text Encoder (Qwen3)

**File**: `Qwen3TextEncoder.swift` — contains 6 classes in one file.

### Architecture

Standard Qwen3 decoder-only transformer used as a text encoder:

- **Embedding**: `Embedding(151936, 2560)`
- **36 × Qwen3DecoderLayer**: self-attention + MLP with pre-norm
- **Final RMSNorm**: `RMSNorm(2560, eps=1e-6)`

### Qwen3Attention

Grouped Query Attention (GQA):
- 32 query heads, 8 KV heads (4× ratio)
- Head dim: 128 (note: `numAttentionHeads × headDim = 32 × 128 = 4096`, which is larger than hiddenSize=2560)
- Q projection: `Linear(2560 → 4096, bias=false)`
- K projection: `Linear(2560 → 1024, bias=false)` (8 heads × 128)
- V projection: `Linear(2560 → 1024, bias=false)`
- O projection: `Linear(4096 → 2560, bias=false)`
- **Q/K norms**: RMSNorm(headDim=128, eps=1e-6) applied per-head after projection
- RoPE: **rotate-half** style (standard LLaMA/Qwen convention, NOT complex-interleaved)
- KV heads expanded via `repeatKV` for GQA
- Attention computed in float32 via `MLXFast.scaledDotProductAttention`

### Qwen3MLP

Gate-up-down pattern:
```
gate = gate_proj(x)   // Linear(2560 → 9728)
up   = up_proj(x)     // Linear(2560 → 9728)
down = down_proj(silu(gate) × up)  // Linear(9728 → 2560)
```

### Causal Mask

Full causal mask built for all tokens:
- Upper-triangular -inf mask for autoregressive attention
- Padding mask (all 1s since we always process a single sequence)
- Combined: `causalMask + paddingMask` → `[B, 1, seq, seq]`

### Prompt Embedding Extraction (`getPromptEmbeds`)

FLUX uses intermediate hidden states, not the final output:

1. Forward through all 36 layers with `outputHiddenStates: true`
2. Extract layers `[9, 18, 27]` (0-indexed, so these are after the 10th, 19th, 28th layer)
3. Stack: `[B, 3, seq, 2560]`
4. Transpose + reshape: `[B, seq, 3×2560]` = `[B, seq, 7680]`

This is the `jointAttentionDim = 7680` that feeds into the transformer's `context_embedder`.

### Flux2PromptEncoder (`Flux2PromptEncoder.swift`)

Top-level encoding function:

1. Tokenize prompt with Qwen3 tokenizer (max 512 tokens)
2. Call `textEncoder.getPromptEmbeds(inputIds, attentionMask, hiddenStateLayers)`
3. Prepare `textIds`: `[B, seq_len, 4]` with `(t=0, h=0, w=0, token_index)`

---

## VAE Decoder

### Architecture

Standard FLUX VAE decoder. Operates entirely in NCHW internally (with NHWC transposes around MLX conv/norm ops).

```
conv_in:      Conv2d(32 → 512, 3×3)
mid_block:    ResNet(512) → Attention(512) → ResNet(512)
up_block_0:   3× ResNet(512→512) + Upsample2D   // ×2
up_block_1:   3× ResNet(512→512) + Upsample2D   // ×2
up_block_2:   3× ResNet(512→256) + Upsample2D   // ×2
up_block_3:   3× ResNet(256→128)                 // no upsample (final)
conv_norm_out: GroupNorm(32, 128)
conv_out:     Conv2d(128 → 3, 3×3)
```

Input: `[1, 32, H/8, W/8]` → Output: `[1, 3, H, W]`

For 1024×1024: `[1, 32, 128, 128]` → `[1, 3, 1024, 1024]`

### VAEResnetBlock2D (`VAEResnetBlock.swift`)

```
NCHW → NHWC
  → GroupNorm(fp32) → SiLU → Conv2d
  → GroupNorm(fp32) → SiLU → Conv2d
  + skip (optional Conv2d(1×1) if channel mismatch)
→ NCHW
```

- GroupNorm always computed in float32, then cast back to bfloat16
- `pytorchCompatible: true` flag on all GroupNorm instances

### VAEAttentionBlock (`VAEAttentionBlock.swift`)

Single self-attention layer in the mid-block:

```
NCHW → NHWC
  → GroupNorm(fp32)
  → Q/K/V = Linear(channels) with bias
  → reshape to [B, 1, H×W, C] (single head)
  → scaledDotProductAttention(scale=1/√C)
  → to_out (Linear with bias)
  + residual
→ NCHW
```

**Note**: VAE attention uses Linear **with bias** (unlike the transformer which is all bias=false).

### Upsample2D (`VAEComponents.swift`)

Nearest-neighbor 2× upsampling:

```swift
// NCHW: repeat pixels along H and W axes
hs = MLX.repeated(x, count: 2, axis: 2)  // double height
hs = MLX.repeated(hs, count: 2, axis: 3) // double width
// Then conv to smooth
hs → NHWC → Conv2d(3×3, padding=1) → NCHW
```

### NCHWConv2d / NCHWGroupNorm (`VAEComponents.swift`)

Wrapper modules that handle the NCHW ↔ NHWC transpose around MLX's native NHWC ops:

```swift
// NCHWConv2d
func callAsFunction(_ x: MLXArray) -> MLXArray {
    let nhwc = x.transposed(0, 2, 3, 1)
    let out = conv(nhwc)
    return out.transposed(0, 3, 1, 2)
}
```

---

## Scheduler

**File**: `FlowMatchEulerScheduler.swift`

Flow Matching Euler Discrete Scheduler for distilled models.

### Sigma Computation

1. Linear spacing: `sigma[i] = 1.0 - i × (1 - 1/N) / (N-1)` for `i = 0..N-1`
2. Empirical mu: piecewise linear function of `imageSeqLen` and `numSteps`
   ```
   if imageSeqLen > 4300:
       mu = a2 × seqLen + b2
   else:
       mu = interpolate(m10, m200, numSteps)
   ```
   Where `a1=8.74e-5, b1=1.90, a2=1.69e-4, b2=0.457`
3. Exponential time shift: `sigma_shifted = e^mu / (e^mu + (1/sigma - 1)^1)`
4. Timesteps: `sigma × 1000` (scale to training range)
5. Append zero sigma at the end (for final step)

### Euler Step

```
latents = latents + (sigma[i+1] - sigma[i]) × noise_prediction
```

This is a simple forward Euler step in the flow matching formulation.

---

## Latent Creator

**File**: `Flux2LatentCreator.swift`

### Latent Dimensions

For image size `H × W` with `vaeScaleFactor = 8`:

```
latentHeight = H / 16  (factor of 8 for VAE × 2 for patchify)
latentWidth  = W / 16
```

For 1024×1024: `latentHeight = 64, latentWidth = 64`

### Packing (for transformer input)

1. Generate noise: `[B, 128, latH, latW]` via `MLXRandom.normal` with deterministic key
2. Pack: `[B, 128, latH, latW]` → `[B, latH×latW, 128]` (flatten spatial, swap with channel)
3. Grid IDs: `[B, latH×latW, 4]` with `(t=0, h_coord, w_coord, layer=0)`

### Unpacking (for VAE input)

1. Unpack: `[B, seq_len, 128]` → `[B, 128, latH, latW]` (reverse of packing)
2. Unpatchify: `[B, C×4, H/2, W/2]` → `[B, C, H, W]` (reverse 2×2 patchify)
   - Reshape: `[B, C, 2, 2, H/2, W/2]`
   - Transpose: `[B, C, H/2, 2, W/2, 2]`
   - Reshape: `[B, C, H, W]`
   - Result: `[B, 32, latH×2, latW×2]` = `[B, 32, H/8, W/8]`

---

## Weight Loading

**File**: `Flux2WeightLoader.swift`

### Loading Flow

1. Scan model directory for `.safetensors` files
2. Load all arrays with `MLX.loadArrays(url:)`
3. Filter by prefix (`transformer.`, `text_encoder.`, `vae.` / `decoder.`)
4. Strip prefix from keys
5. Apply remapping rules
6. Load into model via `model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)`

### Key Remapping Rules

**Transformer**:
- `to_out.0.weight` → `to_out.weight` (PyTorch Sequential container removal)

**Text Encoder**:
- `model.layers.0.` → `layers.0.` (strip `model.` prefix)

**VAE Decoder**:
- `decoder.` prefix stripped
- `to_out.0.` → `to_out.` (same as transformer)
- **Conv2d weight transpose**: `[O, I, kH, kW]` (PyTorch OIHW) → `[O, kH, kW, I]` (MLX OHWI)
  ```swift
  if key.contains("conv") && key.hasSuffix(".weight") && value.ndim == 4 {
      result[key] = value.transposed(0, 2, 3, 1)
  }
  ```

### Verification

Uses `.noUnusedKeys` verification — will throw if any weight key doesn't match a module parameter. This catches key remapping bugs early.

---

## App Integration

### Files Modified

| File | Change |
|------|--------|
| `tesseract/Core/Logging.swift` | Added `Log.image` category |
| `tesseract/Features/Models/ModelDefinition.swift` | Added `.imageGeneration` category, `flux2-klein-4b` model definition |
| `tesseract/App/DependencyContainer.swift` | Added `lazy var imageGenEngine = ImageGenEngine()` |
| `tesseract/Models/NavigationItem.swift` | Added `.image` case with "photo.fill" symbol |
| `tesseract/Features/Dictation/Views/ContentView.swift` | Added `imageGenEngine` parameter threading |
| `tesseract/App/TesseractApp.swift` | Passes `imageGenEngine` to ContentView |
| `tesseract/Features/Models/ModelsPageView.swift` | Added `flux2-klein-4b` to `isModelLoadedInMemory()` |
| `tesseract.xcodeproj/project.pbxproj` | Added MLXImageGen package reference |

### New Files

| File | Purpose |
|------|---------|
| `tesseract/Features/ImageGen/ImageGenEngine.swift` | `@MainActor ObservableObject` wrapping `Flux2Pipeline` |
| `tesseract/Features/ImageGen/ImageGenContentView.swift` | SwiftUI view with prompt/size/seed controls |

### ImageGenEngine (`ImageGenEngine.swift`)

```swift
@MainActor
final class ImageGenEngine: ObservableObject {
    @Published private(set) var isModelLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var isGenerating = false
    @Published private(set) var loadingStatus: String = ""
    @Published private(set) var currentStep: Int = 0
    @Published private(set) var totalSteps: Int = 0

    private var pipeline: Flux2Pipeline?

    func loadModel(from directory: URL) async throws
    func releaseModel()
    func generateImage(prompt:width:height:numSteps:guidanceScale:seed:) async throws -> NSImage
}
```

- Wraps the `Flux2Pipeline` actor
- Default `numSteps = 4`, `guidanceScale = 3.5`
- Progress callback updates `currentStep` on MainActor (via `Task { @MainActor in ... }`)
- Returns `NSImage` (wrapping CGImage from pipeline)

### ImageGenContentView (`ImageGenContentView.swift`)

- Prompt: `TextField` with `.vertical` axis, 1-3 lines
- Size picker: 512×512, 768×768, 1024×1024 (default: medium/768)
- Seed: text field, defaults to 0 (random)
- Generate button: disabled when model not loaded, generating, or empty prompt
- Progress: `ProgressView(value:total:)` with "Step X/Y" label
- Result: `Image(nsImage:)` with aspect-fit, context menu for Copy/Save
- Error display: red caption text
- Model not loaded: "Download FLUX.2-klein-4B from the Models page to get started"

### Model on Models Page

```swift
ModelDefinition(
    id: "flux2-klein-4b",
    displayName: "FLUX.2-klein-4B",
    description: "4B parameter distilled image generation. 4-step inference, 1024×1024.",
    category: .imageGeneration,
    source: .huggingFace(
        repo: "black-forest-labs/FLUX.2-klein-4B",
        requiredExtension: "safetensors"
    ),
    sizeDescription: "~8 GB",
    dependencies: []
)
```

**Note**: Uses the official bf16 weights (~8 GB). No quantized variant configured yet. If a `mlx-community/FLUX.2-klein-4B-4bit` becomes available, switching to it would halve memory usage.

---

## Build System

### Package.swift

```swift
// swift-tools-version: 6.2
.target(
    name: "MLXImageGen",
    dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "Tokenizers", package: "swift-transformers"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
    ],
    swiftSettings: [
        .unsafeFlags(["-O"], .when(configuration: .debug)),  // Critical for perf
    ]
)
```

**`-O` in debug**: Same pattern as MLXAudioTTS. Without optimization, MLX graph construction is extremely slow in debug builds. This flag ensures reasonable performance even during development.

### Xcode Integration

- Added as local SPM dependency in `tesseract.xcodeproj`
- `MLXImageGen` linked to the app target
- Uses Xcode's folder-based project — new files auto-discovered

---

## Layout Conventions (NCHW vs NHWC)

This is the single most confusing aspect of the codebase. MLX natively uses **NHWC** for convolutions and GroupNorm, but the FLUX model (and its weights) use **NCHW** convention.

### Rules

| Component | Internal Layout | Conv/Norm Layout | Transposes |
|-----------|----------------|------------------|------------|
| **Transformer** | `[B, S, D]` (no spatial) | N/A | None needed |
| **VAE mid/up blocks** | NCHW | NHWC (via transpose) | Every conv/norm call |
| **VAE ResnetBlock** | Receives NCHW, returns NCHW | NHWC inside | 1 transpose in, 1 out |
| **VAE AttentionBlock** | Receives NCHW, returns NCHW | NHWC inside | 1 in, 1 out |
| **Upsample2D** | Receives NCHW | Repeat on NCHW axes, conv in NHWC | Mixed |
| **Flux2VAEDecoder** | Entry: NCHW→NHWC for conv_in | Mid/up blocks in NCHW | Top-level transposes |

### Weight Transposition

PyTorch conv weights: `[O, I, kH, kW]` (OIHW)
MLX conv weights: `[O, kH, kW, I]` (OHWI)

Done during weight loading: `value.transposed(0, 2, 3, 1)`

**Only applies to VAE** — the transformer has no Conv2d layers.

---

## MLX Swift API Gotchas

These were discovered during implementation and are worth remembering:

### 1. GroupNorm parameter naming
```swift
// Wrong: GroupNorm(count: 32, dimensions: 128)
// Right:
GroupNorm(groupCount: 32, dimensions: 128)
```

### 2. MLXArray.full type parameter
```swift
// For concrete types:
MLXArray.full([n], values: MLXArray(Int32(0)), type: Int32.self)

// For DType values:
MLXArray.full([n], values: val, dtype: .bfloat16)

// NOT: type: .int32 (this is DType enum, not HasDType.Type)
```

### 3. Foundation.sqrt vs MLX.sqrt
```swift
import Foundation
// Use Foundation.sqrt for Float scalars:
let scale: Float = 1.0 / Foundation.sqrt(Float(headDim))
// NOT: sqrt(Float(headDim)) — resolves to MLX's sqrt returning MLXArray
```

### 4. ModuleParameters nesting
```swift
// Build nested structure from flat dot-notation keys:
let nested = ModuleParameters.unflattened(weights)
try model.update(parameters: nested, verify: .noUnusedKeys)

// NOT manual nesting — ModuleParameters doesn't expose .value()/.parameters()
```

### 5. Self-capture in init closures
```swift
// Can't use stored properties in closures before init completes:
let dim = self.innerDim  // capture to local first
self._blocks.wrappedValue = (0..<n).map { _ in
    SomeBlock(dim: dim)  // use local, not self.innerDim
}
```

### 6. Conv2d IntOrPair
```swift
// MLX Swift Conv2d accepts plain Int for uniform kernel/stride/padding:
Conv2d(inputChannels: 32, outputChannels: 64,
       kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1))
```

### 7. Vendor code logging
```swift
// Use NSLog in vendor code (not print, not Log.*)
NSLog("[MLXImageGen] Loading weights...")
```

---

## Performance Notes

### Expected Performance (Not Yet Measured)

Based on the Python mflux reference:
- **Python/MLX on M3 Max**: ~4-8 seconds for 1024×1024 (4 steps)
- **Swift/MLX**: Expected to be slower due to graph construction overhead (see TTS performance investigation)
- The transformer has 25 blocks × 4 steps = 100 forward passes, plus text encoding + VAE decode

### Potential Optimization Targets

1. **Text encoder caching**: For the same prompt, `promptEmbeds` can be cached across generations (only seed changes the latents, not the text encoding)

2. **GPU.clearCache() placement**: Currently called after prompt encoding and after denoising. May need tuning based on actual memory pressure.

3. **eval() frequency**: Currently eval after every denoising step. Could potentially batch 2 steps if memory allows (but risks OOM on smaller GPUs).

4. **Quantization**: The model definition currently uses bf16 weights (~8 GB). Using 4-bit quantization would reduce to ~3.5 GB and potentially speed up memory-bound operations.

5. **MLXFast.scaledDotProductAttention**: Already using the fused kernel. This is the main performance-critical operation.

6. **compile()**: Based on TTS experience, `mx.compile()` is unlikely to work due to variable shapes and Swift Int offsets. Don't attempt without careful analysis.

7. **VAE decoder**: Could benefit from the most optimization — 4 upsampling blocks with multiple convolutions and repeated NCHW↔NHWC transposes. Consider keeping everything in NHWC to avoid transposes.

### Memory Considerations

- **bf16 model**: ~8 GB for weights alone
- **1024×1024 latents**: relatively small (~32 MB)
- **Text encoder**: 36 layers × 2560 hidden = significant intermediate activations
- **VAE decoder**: spatial activations at 512 channels = memory-intensive
- **Total peak**: likely 10-12 GB on M3 Max (untested)
- `GPU.clearCache()` between encode and denoise phases helps reclaim text encoder memory

---

## Known Issues & Future Work

### Not Yet Runtime Tested
The implementation compiles but has not been tested end-to-end:
- [ ] Model download from HuggingFace
- [ ] Weight loading (key mapping may need adjustments)
- [ ] Tokenizer loading
- [ ] Image quality verification
- [ ] Memory profiling
- [ ] Performance benchmarking

### Likely Issues to Debug

1. **Weight key mismatches**: The `verify: .noUnusedKeys` check will catch these, but the error messages from MLX can be cryptic. Compare Python `state_dict.keys()` with Swift module paths.

2. **Conv2d weight transpose**: Only applied to keys containing "conv" — if there are linear layers with 4D weights in the VAE, they'll be incorrectly transposed.

3. **Qwen3 tokenizer**: Using `AutoTokenizer.from(modelFolder:)` — this requires `tokenizer.json` and possibly `tokenizer_config.json` in the model directory. If the HF download doesn't include these, tokenization will fail.

4. **Numerical precision**: GroupNorm in float32, attention in float32, but everything else in bfloat16. If images come out as noise, check for precision issues in the VAE decoder or the scheduler step.

5. **Timestep scaling**: The transformer forward pass scales timesteps by 1000 if `max(ts) <= 1.0`. Make sure the scheduler's timesteps are in the right range.

6. **Model loading performance**: Loading 8 GB of safetensors files can take significant time. Consider showing progress during weight loading.

### Future Enhancements

- **Image-to-image**: FLUX supports img2img via partial denoising (start from encoded image instead of pure noise)
- **Inpainting**: With appropriate masking support
- **ControlNet**: If FLUX.2-klein-4B ControlNet adapters become available
- **LoRA**: Fine-tuned style/character adapters
- **Batch generation**: Multiple images from the same prompt with different seeds
- **Progressive preview**: Show intermediate denoised images during generation

---

## Reference Mapping (Python → Swift)

| Python (mflux) | Swift (MLXImageGen) |
|-----------------|---------------------|
| `src/mflux/models/flux2/model/flux2_transformer/transformer.py` | `Transformer/Flux2Transformer.swift` |
| `src/mflux/models/flux2/model/flux2_transformer/transformer_block.py` | `Transformer/Flux2TransformerBlock.swift` |
| `src/mflux/models/flux2/model/flux2_transformer/single_transformer_block.py` | `Transformer/Flux2SingleTransformerBlock.swift` |
| `src/mflux/models/flux2/model/flux2_transformer/attention.py` | `Transformer/Flux2Attention.swift` |
| `src/mflux/models/flux2/model/flux2_transformer/parallel_self_attention.py` | `Transformer/Flux2ParallelSelfAttention.swift` |
| `src/mflux/models/flux2/model/flux2_transformer/feed_forward.py` | `Transformer/Flux2FeedForward.swift` |
| `src/mflux/models/flux2/model/flux2_transformer/modulation.py` | `Transformer/Flux2Modulation.swift` |
| `src/mflux/models/flux2/model/flux2_transformer/pos_embed.py` | `Transformer/Flux2PosEmbed.swift` |
| `src/mflux/models/flux2/model/flux2_transformer/timestep_guidance_embeddings.py` | `Transformer/Flux2TimestepEmbeddings.swift` |
| `src/mflux/models/flux/model/flux_transformer/ada_layer_norm_continuous.py` | `Transformer/AdaLayerNormContinuous.swift` |
| `src/mflux/models/flux2/model/flux2_text_encoder/qwen3_text_encoder.py` | `TextEncoder/Qwen3TextEncoder.swift` |
| `src/mflux/models/flux2/model/flux2_text_encoder/prompt_encoder.py` | `TextEncoder/Flux2PromptEncoder.swift` |
| `src/mflux/models/flux2/model/flux2_vae/decoder/decoder.py` | `VAE/Flux2VAEDecoder.swift` |
| `src/mflux/models/common/vae_components/resnet_block_2d.py` | `VAE/VAEResnetBlock.swift` |
| `src/mflux/models/common/vae_components/attention.py` | `VAE/VAEAttentionBlock.swift` |
| `src/mflux/models/common/vae_components/upsample_2d.py` | `VAE/VAEComponents.swift` |
| `src/mflux/models/common/schedulers/flow_match_euler_discrete_scheduler.py` | `Scheduler/FlowMatchEulerScheduler.swift` |
| `src/mflux/models/flux2/latent_creator/flux2_latent_creator.py` | `LatentCreator/Flux2LatentCreator.swift` |
| `src/mflux/models/flux2/weights/flux2_weight_mapping.py` | `WeightLoading/Flux2WeightLoader.swift` |
| `src/mflux/models/common/config/model_config.py` | `Flux2Configuration.swift` |
| N/A (custom) | `Transformer/AttentionUtils.swift` |
