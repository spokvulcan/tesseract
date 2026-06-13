---
status: accepted
---

# Image identity enters the prefix cache as length-preserving pseudo-tokens

Image-bearing requests were fully ineligible for the prefix cache because the
radix key is the token sequence and a VLM's expanded placeholder run
(`<|image_pad|> × N`) is content-blind — two different same-sized images
tokenize identically, so keying on tokens alone would produce wrong cache hits.
We make image turns cacheable by giving each image an **Image Digest** (SHA-256
over the raw encoded bytes exactly as received) and keying the radix tree on a
**Cache Key Path**: the prepared prompt tokens with each placeholder run
replaced, length-preserving, by pseudo-tokens deterministically expanded from
that image's digest, drawn from a range no vocabulary occupies (negative ints).
The model never sees pseudo-tokens; only the cache does. This preserves the
tree's central invariant — key path index == KV offset — so offset arithmetic,
admission-path proofs, checkpoint planning, trim semantics, and the SSD tier
work unchanged, and one implementation serves both consumers of the machinery
(the HTTP server and the in-process agent chat — noting that the agent edge
must first be *routed onto* it: today internal agent generation submits a nil
prefix-cache conversation on the standard route, so the agent-side
conversation builder and server-compatible routing are part of this work, not
a given).

The key path is built by scanning the *prepared* token sequence for the
family's placeholder-token runs and substituting run *i* with image *i*'s
expansion — N is never re-derived app-side, so "key length == KV length" holds
by construction and cannot drift from the vendored processors. Guards: run
count must equal image count, and an unrecognized placeholder strategy bails to
uncached serving — degrade, never guess.

## Consequences

- The digest→pseudo-token expansion is a frozen pure function (digest + index →
  int): expansions persist inside SSD admission paths across restarts, so
  changing the function invalidates every image-bearing snapshot on disk.
- Identity is exact-byte. A re-encoded or resized variant of the same picture
  is a different image — always a miss, never a wrong hit. Within-conversation
  reuse is what the cache is for, and there bytes are provably stable
  (immutable `ImageAttachment.data` agent-side, verbatim base64 over HTTP).
- `prepare()` must run before cache lookup (the splice scans its output), so a
  fully-cached image conversation still pays CPU image preprocessing per
  request. The vision tower still never runs for cached positions — that is
  the expensive part. Short-circuiting pixel preprocessing for fully-covered
  prefixes is a possible later optimization, keyed by digest.
- Eviction scoring prices image positions as ordinary text tokens — the vision
  tower's recompute cost is not modeled, so image-bearing snapshots are
  systematically undervalued by the flop profile. Accepted for now: the
  failure mode is one cold vision-tower run, eviction policy carries no
  persistence, and folding vision flops into `ModelFlopProfile` later is a
  pure-policy change to make once hit-rate data exists.
- There is deliberately **no settings toggle** for image keying. The
  structural safety net is the Unkeyed Completion bail; shipping is gated by
  a validation ladder in the ADR-0006 style (VLM warmed-cache smoke spike
  first, `HybridCacheCorrectnessRunner` image cases — above all
  different-image-same-size must never hit — `bench.sh` text-path parity, an
  image hit-rate workload), not by a runtime flag.

## Deepened at architecture review (2026-06-11)

The review found the splice-as-one-function shape ships a correctness defect:
two more producers put token paths into the same radix tree, and both render
via the chat template — *unexpanded* space. The planner's last-user re-render
would land its boundary at a wrong KV offset, and the canonical/tool leaf
rebuild re-prefills render-space residuals on top of an expanded-space
boundary — persisting snapshots whose KV does not match their key path. The
existing offset guards are bounds checks, not space checks.

The decision therefore deepens: the splice becomes the **Cache Key Space**, a
per-request value built once after prepare that owns the image table and both
duties — producing the key path *and* translating render-space token sequences
into key space. Every token path that touches the tree passes through it
(lookup/admission, planner boundaries, leaf probes). Supporting decisions:

- The per-family placeholder identity is a **Model Identity** load-time fact
  (`nil` ⇒ family not recognized ⇒ Unkeyed Completion).
- Planner re-renders and leaf probes share one token-only **Conversation
  Render** (family message-forming + template, no pixel work), so probe
  renders cannot drift from prepare's render. The conversation value renders
  through per-message chat messages — the only form that places multi-turn
  interleaved images faithfully.
- Degradation is two-tier: Key Space *construction* failure makes the whole
  request an Unkeyed Completion; a later *translation* failure skips only the
  consuming feature with a typed reason (the LeafSkipReason pattern).
- `ModelFingerprint` adds the preparation-identity files
  (`tokenizer_config.json`, `preprocessor_config.json`,
  `chat_template.json`/`.jinja`, empty-if-absent), so a processor or template
  change partitions the cache instead of leaving stale never-matching paths.

External validation, same review: SGLang's `pad_input_ids` ships this exact
design in production (placeholder runs replaced by image-hash-derived values
in the radix key; their token-pair pattern is our future path for framed
placeholder families); vLLM V1 folds per-image `mm_hash` into block hashes.
Marconi (the paper this cache follows) is text-only but none of its
invariants — exact-match prefixes, admission taxonomy, key-index ==
state-offset — are violated; the phase-1 eviction shortcut converges back to
Marconi's flop-efficiency formula in phase 2.

## Spike results (2026-06-11) — capture shape reshaped

The warmed-cache spike (PRD #72 gate, run on the VLM smoke harness against
Qwen3.5-4B PARO in vision mode) confirmed the keying design and **falsified
the planned capture shape**. Empirical findings, each pinned as a harness
check:

- The vendored VLM container recomputes M-RoPE positions from zero on any
  forward that starts without threaded state: today's restore path is
  positionally wrong on the VLM container even for text-only prompts
  (divergence 2.08), and the upstream `TokenIterator` discards `prepare()`'s
  state, mis-positioning decode after every image prompt (1.12). Both are
  latent today because text traffic runs the LLM container by default.
- **"Single-shot remainder prepare on the warmed cache" is not viable**:
  `prepare()` hardcodes nil state and cannot anchor at a restore offset
  (3.50). No public-API workaround exists, and grafting is impossible anyway
  — Mamba states are strictly sequential (the Marconi premise).
- The correct anchor is reachable app-side with **zero vendor changes**: the
  continuation branch (positions = arange + cache offset + rope delta) is
  selected by seeding the vendor's `"qwen35.ropeDeltas"` state slot through
  the public `LMOutput.Key(String)` API. Seeded restores are **bitwise exact
  (0.0)** — including text turns chaining off a cached image turn.
- The rope delta at any boundary is **reconstructible from the processed
  image grids**: Σ per cached image (max(t, h/m, w/m) − t·h·w/m²). Harvested
  (−56) == reconstructed (−56); no extra snapshot persistence needed.
- Chunk-shape floating-point noise is real (0.21, argmax-stable) with no
  restore involved at all: bitwise equality only holds between runs with
  identical kernel shapes. The existing text gate implicitly satisfied this
  by restoring at prefill-step-aligned offsets; the image gate compares
  bitwise against shape-matched references and argmax-stability against
  production-cold references.

Consequences for the decision:

- The capture shape becomes: **image-add turns serve cold** (full vendor
  `prepare`, correct M-RoPE by construction), the leaf is admitted at end of
  prefill as usual, and **subsequent turns restore with a seeded Position
  Anchor** (offset + reconstructed rope delta). Warm image-bearing remainders
  — and with them per-image remainder pixel slicing — are out of phase 1; the
  lookup clamps hits so that image runs never land in the remainder.
- Generation on the cache-aware VLM path must thread model state end-to-end
  (prefill chunks → prime forward → decode), which the upstream
  `TokenIterator` cannot do; decode is app-owned there.
- The vendor-internal state-key string and the delta-reconstruction formula
  are frozen compatibility facts, pinned by the spike harness checks
  (`prepareStateCarriesRopeDelta`, `textOnlySeededRestoreMatches`,
  `imageLeafRestoreTextRemainderMatches`) so vendor drift fails loudly.

## Implementation notes (2026-06-11)

The image hit/miss cases of the validation ladder landed in the
`PrefixCacheE2ERunner` image scenario rather than the
`HybridCacheCorrectnessRunner` named above: the bitwise tensor harness drives
raw token arrays below the seam where image keying lives, while the E2E
runner exercises the full Server Completion pipeline — including the agent
dispatch path, so different-image-same-size is pinned on HTTP *and* agent
routes in one scenario.

Two accepted degradations surfaced while wiring the decision in, both confined
to rare fallback paths:

- **The thinking-safeguard continuation is degraded on image-bearing
  requests.** The continuation swap re-prefills the stored *real* prepared
  tokens (which is why `fullTokens` stays real and the radix tree keys on
  `keySpace.keyPath` instead) — but it re-forwards them without pixels, so
  image pad tokens hit text embeddings rather than vision features. Accepted:
  the safeguard is an exceptional intervention, and the alternative is
  threading pixel re-preparation through the continuation path for a path
  that should almost never fire.
- **Unkeyed Completion skips decode-time KV quantization.** The cache-aware
  path quantizes the cache once before handing it to the state-threaded
  iterator; the unkeyed path strips `kvBits` entirely instead of replicating
  the upstream iterator's per-step quantization swap. Cost is RAM-only, on a
  degraded path whose cache is discarded after the request anyway.

## Considered / rejected

- **Heterogeneous radix alphabet** (edges as `token | imageDigest`): honest
  modeling, but every consumer of token paths must learn the new alphabet plus
  an explicit key-position→KV-position mapping — much wider blast radius for
  the same observable behavior.
- **Conversation-level digest fields** (like `toolDefinitionsDigest`): wrong
  granularity — image identity becomes all-or-nothing per conversation,
  amputating longest-prefix reuse exactly where it matters.
- **Pixel-normalized or perceptual hashing**: a decode pipeline that must stay
  bit-stable across CoreImage versions, or false hits that silently answer
  about the wrong image. Exact-byte misses cost only what today already costs.
- **Re-implementing patch-count math app-side**: correctness-critical
  duplication that drifts silently when vendored processors update; a wrong N
  corrupts every offset past the image. Worst failure mode in the space.
- **Extending the vendored processor API** to report image-token spans:
  re-diverges the vendor tree that ADR-0006 just un-forked, for something
  derivable app-side.

## Phase 2 (2026-06-13): warm continuation *through* a new image

Phase 1's cut — *"image-add turns serve cold… the lookup clamps hits so that
image runs never land in the remainder"* — is the direct cause of two
production failures with `qwen3.6-27b-paro`:

1. **It never reuses the cache on an image turn.** OpenCode puts the image in
   the newest message, so `minimumWarmOffset` (end of the last image run) sits
   ~20 tokens from the prompt end every time; the prior-turn snapshot is always
   below it, so `PrefillPlanner` (the `snapshot.tokenOffset >= minimumWarmOffset`
   guard) forces **cold** on every image turn.
2. **Cold means one single-shot full-attention pass over the whole prefix.**
   `Qwen35.prepare` ignores `windowSize` (param is literally `windowSize _:`),
   so `[0, minimumWarmOffset)` goes through in one pass. The hybrid model's
   full-attention layers (`full_attention_interval=4`, 16 of 64 layers,
   `num_attention_heads=24`) each allocate a `[24, L, L]` bf16 scratch =
   `24·L²·2` bytes. **Crash cliff ≈ L=25,000 tokens** (30 GB Metal buffer
   limit): a 14 K image turn survives but takes ~98 s and peaks ~42 GB; a 31 K
   coding session + image → 47.9 GB → SIGTRAP; the first crash's 55 K →
   147 GB, matching the MLX error to the byte.

The phase-1 spike concluded *"single-shot remainder prepare on the warmed cache
is not viable… no public-API workaround exists."* **That is correct but was
read too broadly.** It tested only (a) the existing `prepare()` (hardcoded nil
state) and (b) zero-vendor-change tricks (seeding `qwen35.ropeDeltas` → flat
`arange+offset+delta` positions, right for a *text* remainder, wrong for a
*new image's* diverging t/h/w positions). It never tested a method that injects
offset-aware **image** positions, because that needs a vendor change — which
phase 1 was avoiding. Two further phase-1 premises also relax on inspection:
the GatedDeltaNet (Mamba) recurrent state is **position-free and restore-able**
(`MambaCache.copy()` / `state` are public; restore-and-continue from a snapshot
is exact — only *rewind*, i.e. subtracting tokens, is impossible, and we never
rewind); and the full-attention KV cache is already offset-aware.

**Decision.** Image-add turns restore the deepest valid prior snapshot at offset
`P` and continue *through* the new image, chunked. Per ADR-0006's amended vendor
stance (general + upstreamable), the new surface lives in the vendored model:

- **A windowed `Qwen35.prepareContinuation`** (warm cache + remainder tokens +
  remainder image frames/pixels) that honors `windowSize` and loops chunks over
  `[P, end)` internally — crash-safe by construction (scratch bounded to
  `[24, chunk, L]`), and a generally useful "prepare that doesn't OOM on long
  image prompts." Decode stays app-owned (state threaded end-to-end, as today).
- **An offset-aware `getRopeIndex`** — the one genuinely new bit of math: a third
  position branch for *pixels present **and** warm cache*, computing the new
  image's diverging t/h/w positions from the seeded **Position Anchor**
  (cache offset + reconstructed rope delta) instead of resetting to zero
  (`Qwen35.swift:799-802`). No new "position offset" scalar — it reuses the
  Position Anchor already seeded for text remainders.
- App side is small: relax the `PrefillPlanner` guard to allow `P` below the
  image, pass the remainder's image frames to the continuation, keep the
  post-image text tail on the existing app-driven `PrefillExecutor`, capture the
  end-of-prefill leaf as today. The **Cache Key Path** and SSD format are
  unchanged, so no snapshot invalidation.

This supersedes the phase-1 "image-add serves cold" capture shape and the
`minimumWarmOffset` hit-clamp (which becomes a fallback for when no valid
restore exists, not the default).

**Distinguished from the rejected "extend vendored processor API":** that
rejection was about *reporting image-token spans* — derivable app-side. Running
the vision tower, the embedding merge, and an offset-positioned forward is **not**
derivable app-side (the members are `private`/`fileprivate` and need the
weights). And keeping the position/merge math vendor-side honors the
"re-implementing patch-count math app-side" rejection above — a wrong N still
corrupts every offset past the image.

**Validation ladder (gates merge, ADR-0007 style — a wrong position is silent):**
warm-continuation output argmax-stable vs production-cold **and** bitwise vs a
shape-matched reference (chunk-shape FP noise is real per the phase-1 spike);
**replay of the captured real OpenCode requests** (`http-completions/`) warm vs
cold; frozen-fact harness checks for the new vendor surface (offset-image
positions, continuation state) so vendor drift fails loudly; an image hit-rate
workload confirming image turns now restore instead of going cold.

**Folded-in general fix (upstreamed):** `Qwen3VLProcessorConfiguration` decodes
only legacy `min_pixels`/`max_pixels` and ignores the new-style
`size.{longest_edge,shortest_edge}` keys the Qwen3.6 PARO config ships, silently
falling back to defaults — wrong for every Qwen3-VL model on the new format.

**Deferred, not chased:** a separate anomaly where one 2000×1159 screenshot
expanded to ~43,500 pad tokens (vs the provable ~2,268) on a first-turn request
— the processor math cannot produce it, so it needs runtime ground truth, not
more static reading. Cheap instrumentation only (log actual `imageGridTHW` +
pad-run length at prepare); the chunked continuation makes even a 43 K-pad image
non-fatal, so it no longer gates anything.
