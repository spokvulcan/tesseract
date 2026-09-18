# ADR-0067: Rotated Ternary Checkpoints load through module substitution on the Qwen3.5 engine

- Status: Accepted
- Date: 2026-09-17
- Relates to: ADR-0006 (amended — the vendor is the frontier surface), ADR-0032 (PARO Checkpoint, the other rotated weight format), map #457 (vision and DFlash2 on one engine)

## Context

Bonsai 2 27B (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`) is Qwen3.8-27B with
every language-model Linear and the token embedding replaced by ternary
weights: stock MLX affine 2-bit, group 128, whose codes only ever take three
values, plus a signed blockwise Hadamard rotation (block 1024) folded into the
weights. The runtime has to rotate activations before every packed matmul and
un-rotate embedding rows after lookup. The config declares its own
`model_type` (`prism_hadamard_qwen35`) beside `base_model_type: qwen3_5`, a
`modules` manifest naming the 402 rotated modules, and the same `text_config`
as the base model. The vision tower ships unrotated FP16; the MTP head is
dropped. Nothing else about the architecture changes.

The stock loader accepts the weights (they carry ordinary `.scales`) and
silently produces garbage, because nothing rotates. Reference runtimes use only
stock ops: `quantized_matmul` and `hadamard_transform`, both present in the
pinned mlx-swift. mlx-vlm merged a Python port the day the pack shipped; it
builds the Qwen3.5 VLM and swaps the manifest modules. PrismML published Swift
layer classes on an unmerged mlx-swift fork branch and no model integration.

## Decision

1. **Module substitution on the existing Qwen3.5 classes.** The vendor
   registers `prism_hadamard_qwen35` in both factories. Each creator decodes
   the manifest, builds the ordinary `Qwen35Model` (text) or `Qwen35` (vision)
   and replaces the manifest modules with `HadamardQuantizedLinear` /
   `HadamardQuantizedEmbedding` placeholders before weights load. The loader's
   quantization pass skips them (they are already `Quantized`) and its
   parameter update fills them. The model that comes out *is* a `Qwen35Model`,
   so the fused decode kernels, prefix-cache support and DFlash2 capture sites
   apply unchanged, and the exact-class guards on projection fusion and the
   DFlash2 head prefix leave rotated layers alone.
2. **The layers live in `MLXLMCommon`**, written against the mlx-vlm reference
   (float32 transform, input dtype out), on stock ops only. No mlx-swift
   change, one upstream PR to mlx-swift-lm.
3. **Signs come from the `.signs` tensors** beside each packed module.
   `hadamard.json` is not read.
4. **The app resolves the Base Architecture** from `base_model_type` when
   present, else `model_type`, and keys every family fact on it. The pack is a
   Qwen3.5-family model to Model Identity, the text-class route, the FLOP
   profile, image keying and the scratch profiles.
5. **Unpacked tensors are cast to the manifest's activation dtype at load.**
   The pack stores its norms, convolution taps and the small gated-delta
   projections in float32 beside float16 packed modules. MLX promotes a
   float16 activation through a float32 norm to float32, so loaded as stored
   the residual stream would run in float32 from the first layer on (twice the
   KV cache, and the fork's fused kernels refuse mixed dtypes). The
   `PrismHadamardQwen35` classes' sanitize pass applies the cast mlx-vlm's
   converter applies to any checkpoint it writes: every floating tensor except
   the packed modules' own and `A_log`, which the recurrence reads in float32.
   The reference gate runs with and without the same cast
   (`--match-app-dtypes`) so the two effects stay separable.
6. **Stock ops first, kernels by measurement.** The rotation's sign flip and
   transform are two extra dispatches per packed matmul; the mlx fork's tuned
   matmul routes are 4-bit only, so 2-bit decode runs the stock kernels. A
   fused signed-Hadamard kernel or 2-bit kernel work is a follow-up justified by
   the bench row, not part of v1.

## Considered / rejected

- **A standalone Bonsai model file** duplicating Qwen3.5: a second copy of the
  tuned engine, exactly what map #457 refuses for the VLM class.
- **A generic loader feature** keyed on any config carrying a `modules`
  manifest, with the type name as an alias: one step away if a second rotated
  pack appears; premature for one.
- **Layers in `MLXNN` (mlx-swift)**, where PrismML put theirs: a second fork
  pin move and a second upstream PR for code that needs nothing from that layer.
- **Reading `hadamard.json`**: a sidecar dependency for data the safetensors
  already carry.
- **Float16 transform** (PrismML's Swift layer): a deliberate drift from the
  reference for an unmeasured gain.

## Consequences

- The GDN input-projection fusion and the attention q|k|v stacking are skipped
  for rotated layers by the existing exact-class guards; the rotated layers pay
  their four projections separately.
- The Qwen3.8 DFlash2 draft pairs with the text-class load by shape and stays
  lossless, but it was distilled against the full-precision target: on Bonsai
  2 27B it accepted 22% of its proposals and decoded at 0.64× the plain rate
  (19.6 vs 30.8 tok/s median, 2026-09-18). Model Identity therefore exposes
  `isRotatedTernaryCheckpoint` (recognized by the `modules` manifest, never by
  name) and the draft load refuses such a target. A draft distilled against
  the rotated pack would lift the refusal; none exists.
- The acceptance gate gains a greedy-parity check against the mlx-vlm
  reference: a wrong sign width or dtype slip yields plausible text, not an
  error.
- `Rotated Ternary Checkpoint` and `Base Architecture` enter the glossary
  (`CONTEXT.md`, Model loading).
