---
status: accepted
---

# Cap per-image vision resolution — the qwen3_5 ViT is global O(patches²) attention

A single full-resolution Retina screenshot pasted into a fresh internal-agent
chat drove the process to a **47 GB** peak (`phys_footprint_peak`) and a **121 s**
prefill on `qwen3.6-27b-paro`. This records the measured root cause and the
decision to bound it by capping the **Vision Token Budget** (CONTEXT.md) per image.

## What we measured (2026-06-15)

Live repro on current `main`, instrumented through the existing grid log
(`ServerCompletion` "vision grid") and `footprint`:

- The turn took the **cache-aware** path (`route=serverCompatible` → prefix-cache
  path), *not* the single-shot standard path an earlier diagnosis assumed. The
  language-model prefill was already chunked through the windowed
  `prepareContinuation` (ADR-0007 phase 2), with the `VisionPrefixMemoryGuard`
  backstop. So the language model was **not** the source of the blowup.
- The processed grid was `t=1 h=140 w=216` → **30 240 patches → 7 560 vision
  tokens** (`padRun`), exactly what the resize formula predicts for a 7.74 MP
  screenshot. The deferred "one screenshot → ~43 500 pad tokens" anomaly (noted
  in ADR-0007) **did not recur**; the processor math is behaving.
- The transient was **~29 GB** on top of the ~18 GB resident model. That matches
  a single vision-tower attention score matrix `[16 heads, 30 240, 30 240]` bf16
  = 29.3 GB almost exactly, and cross-checks against the OpenCode anchor (2 540
  tokens → 10 160 patches → 3.3 GB + 18 ≈ 21 GB, matching its observed 20.7 GB).
  The cost scales **quadratically** with patch count.

The `qwen3_5` vision tower (`Qwen3VL.swift`) runs **global attention over every
patch of an image**: `vision_config` declares `num_heads: 16`, `depth: 27`, and
**no `window_size` / `fullatt_block_indexes`**; `VisionModel` builds one
`cuSeqlens` and feeds it through all 27 blocks, each materializing a full
`[1, P, P]` mask and one SDPA over `P = total patches`. Unlike Qwen2.5-VL there
is no intra-image windowing to exploit. The vision tower's working set is
therefore O(patches²), uncapped, and invisible to `VisionPrefixMemoryGuard`
(which models only the *language-model* `[heads=24, L, L]` matrix).

## Decision

Cap each image's processed resolution so its vision-token contribution has a
default ceiling of **2 560 tokens** (≈ 2.62 MP; the budget OpenCode is empirically
proven to read dense screenshots at). Enforce it in the **vendored Qwen3VL
processor**, mirroring the clamp its sibling `Qwen25VL` already ships
(`Qwen25VL.swift:804`):

```swift
let maxPixels = processing?.maxPixels ?? min(config.size.maxPixels, 2560 * factor * factor)
```

on both the image and video preprocess paths, with the matching
`processing?.minPixels ?? config.size.minPixels`. `factor = patchSize * mergeSize`
(= 32 here), and `vision_tokens = pixels / factor²`, so the literal *is* the token
budget. This is the smallest correct change: it covers the internal-agent and HTTP
routes uniformly (both converge on `processor.prepare`), it honors an explicit
per-request `processing.maxPixels` (ADR-0008 — a client may ask for more), and it
is upstreamable — Qwen3VL simply lacks the default clamp and the
`processing.maxPixels` honoring that `Qwen2VL`/`Qwen25VL` have.

Separately, extend `VisionPrefixMemoryGuard` with a **vision-tower scratch
profile** (`attentionHeads = vision_config.num_heads`, bf16) checked against the
request's **combined** patch count before the tower runs. The per-image cap does
not bound a multi-image turn — the tower attends over all images' patches jointly
(block-diagonal masked, but one `[16, ΣP, ΣP]` matrix), and the composer admits up
to 8 images, so 3+ large images re-cross the GPU buffer limit. The guard turns
that corner into a typed rejection instead of an OOM abort.

## Considered and rejected

- **Per-request `UserInput.Processing(maxPixels:)` plumbing only** (the original
  handoff plan): a silent no-op here — `Qwen3VL.preprocess` reads
  `config.size.maxPixels`, never `processing.maxPixels`. And the cache-aware path
  (the one actually taken) does not build its input via `AgentEngine`, so the
  named plumbing site never runs for it.
- **Window the vision tower** (cap patches²-cost without downscaling): the model
  was trained with global ViT attention and ships no window structure; imposing
  windows at inference changes the attention math and would corrupt outputs. Not
  available without retraining.
- **Dynamic total budget (budget ÷ N images)**: bounds the whole request but makes
  an image's token count depend on how many *other* images are present, so adding
  an image re-renders the prior one at a new length — regressing ADR-0007 phase 2
  warm image-add and breaking multi-image prefix-cache reuse. The fixed per-image
  cap keeps each image's pad-run stable across turns; the guard handles the tail.
- **Resize the stored bytes** (e.g. in `ImageIngest`): breaks the full-resolution
  Quick Look viewer (PRD #112) and changes the **Image Digest** (the prefix-cache
  key). The cap downscales only what the tower sees; bytes and digest are
  untouched.
- **No cap, rely on the guard alone**: turns every full-screen screenshot into an
  error. The common single-image case must *work*, not just fail safely.

## Consequences

- Default image fidelity drops to ≤ 2 560 vision tokens. The peak for a single
  full-screen screenshot falls ~47 GB → ~21 GB and prefill ~121 s → ~30 s.
- Prior on-disk image snapshots keep their longer pad-runs; a post-cap request for
  the same image builds a shorter **Cache Key Path** and simply misses them (never
  a wrong hit — ADR-0007 keying invariant). The dead snapshots age out by
  eviction. `ModelFingerprint` is unchanged because the cap lives in code, not in
  `preprocessor_config.json`.
- A client (or a future app surface) that wants higher fidelity sets
  `processing.maxPixels` explicitly; the clamp yields to it.
- The guard's vision profile is the first GPU-cost model of the vision tower in
  the app; eviction scoring still prices image positions as plain text tokens
  (ADR-0007 deferred that), so this does not change cache economics, only crash
  safety.
