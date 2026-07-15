---
status: accepted
---

# Cap per-image vision resolution — the qwen3_5 ViT is global O(patches²) attention

## Context

A single full-resolution Retina screenshot drove a fresh internal-agent turn to a
**47 GB** peak and **121 s** prefill on `qwen3.6-27b-paro`. The blowup is the
**vision tower**, not the language model: the `qwen3_5` ViT (`Qwen3VL.swift`
`VisionModel`) runs **global attention over every patch** — one `cuSeqlens`, one
`[1, P, P]` mask, one SDPA over `P = total patches`, no windowing (its
`vision_config` declares no `window_size`/`fullatt_block_indexes`). At 30 240
patches that score matrix is `[16 heads, P, P]` bf16 ≈ 29 GB on top of the ~18 GB
model. Cost is O(patches²) and uncapped. The language-model prefill is already
chunked (ADR-0007 phase 2) and backstopped by `VisionPrefixMemoryGuard`, which
models only the LM's `[heads=24, L, L]` matrix — it never saw the tower.

## Decision

Cap each image's processed resolution to a default **2 560 vision-token** budget
(≈ 2.62 MP — what OpenCode reads dense screenshots at), enforced in the **vendored
Qwen3VL processor** (image and video paths) by clamping `maxPixels`. Since
`vision_tokens = pixels / factor²` (`factor = patchSize · mergeSize`, 32 here),
the literal *is* the token budget:

```swift
let maxPixels = processing?.maxPixels
    ?? Self.defaultVisionTokenBudgetPixels(factor: factor, ceiling: config.maxPixels)
// defaultVisionTokenBudgetPixels = min(ceiling, 2560 * factor²)
```

This is the smallest correct fix: it covers both the internal-agent and HTTP
routes (both converge on `processor.prepare`), honors an explicit per-request
`processing.maxPixels`, and is upstreamable — Qwen3VL simply lacked the default
clamp that `Qwen2VL`/`Qwen25VL` (`Qwen25VL.swift:804`) already ship.

The per-image cap does **not** bound a multi-image turn: the tower attends over
all images' patches jointly (block-diagonal masked, but still one `[16, ΣP, ΣP]`
matrix), and the composer admits up to 8 images. So `VisionPrefixMemoryGuard`
also gains a **vision-tower scratch profile** (`vision_config.num_heads`, bf16)
that prices the request's combined patch count and turns the multi-image cliff
into a typed `VisionRejection` instead of an OOM abort.

## Considered and rejected

- **Per-request `Processing(maxPixels:)` plumbing only** (the original handoff
  plan): a no-op — `Qwen3VL.preprocess` read `config.size.maxPixels`, never
  `processing.maxPixels`; and the cache-aware path doesn't build its input via
  `AgentEngine`, so the named plumbing site never ran for it.
- **Window the vision tower**: the model was trained with global ViT attention
  and ships no window structure; imposing one at inference corrupts outputs.
- **Dynamic total budget (budget ÷ N images)**: an image's token count would
  depend on how many *others* are present, so adding an image re-renders the
  prior one — regressing ADR-0007 phase 2 warm image-add and breaking
  multi-image prefix-cache reuse. The fixed per-image cap keeps each pad-run
  stable across turns; the guard handles the multi-image tail.
- **Resize the stored bytes** (e.g. in `ImageIngest`): breaks the
  full-resolution Quick Look viewer and changes the **Image Digest** (the
  prefix-cache key). The cap downscales only what the tower sees.
- **No cap, rely on the guard alone**: turns every full-screen screenshot into an
  error. The common single-image case must *work*.

## Consequences

- Peak for a single full-screen screenshot falls ~47 GB → ~21 GB, prefill
  ~121 s → ~30 s; default fidelity is now ≤ 2 560 vision tokens.
- Prior on-disk snapshots keep their longer pad-runs; a post-cap request builds a
  shorter **Cache Key Path** and simply misses them (never a wrong hit —
  ADR-0007), and the dead snapshots age out. `ModelFingerprint` is unchanged
  because the cap lives in code, not in `preprocessor_config.json`.
- A client wanting higher fidelity sets `processing.maxPixels`; the clamp yields.
- The guard's vision profile is the first GPU-cost model of the tower; eviction
  still prices image positions as plain text tokens, so this changes crash
  safety, not cache economics.

## Amendment (2026-07-15)

The default budget is lowered **2 560 → 1 280 vision tokens** (≈ 1.31 MP),
following review on the upstreamed PR (mlx-swift-lm #398): both cited
precedents — mlx-vlm's `14·14·4·1280` max_pixels (a patch-14 constant ⇒
~1.0 MP ≈ 980 tokens on patch-16 qwen3) and `Qwen25VL.swift`'s
`1280 * factor²` — cap at ~1.0 MP per image, so 1 280 tokens is still the
most generous of the three. Owner A/B'd real workloads at both budgets
(dense-text 5 MP screenshot, UI reads): information parity held except two
single-glyph username misreads in the smallest byline text, judged an
acceptable trade for a default since `processing.maxPixels` still yields.
Everything else in this ADR is unchanged; the numbers above describe the
original 2 560 measurement.
