---
status: accepted
---

# Image identity enters the prefix cache as length-preserving pseudo-tokens

A VLM's expanded placeholder run (`<|image_pad|> × N`) is content-blind: two
different same-sized images tokenize identically, so a token-keyed radix tree
would serve wrong hits and image turns had to be excluded from the cache
entirely.

**Decision.** Give each image an **Image Digest** (SHA-256 over the raw encoded
bytes as received) and key the radix tree on a **Cache Key Path**: the prepared
prompt tokens with each placeholder run replaced — length-preserving — by
pseudo-tokens deterministically expanded from that digest, drawn from a range
no vocabulary occupies (negative ints). The model never sees pseudo-tokens;
only the cache does. Length preservation keeps the tree's central invariant —
key-path index == KV offset — so all the existing offset arithmetic, admission
proofs, checkpoint planning, trim semantics, and SSD tier work unchanged, and
one implementation serves both consumers (HTTP server and in-process agent).

The key path is built by scanning the *prepared* sequence for the family's
placeholder runs and substituting run *i* with image *i*'s expansion — N is
never re-derived app-side, so "key length == KV length" holds by construction
and cannot drift from the vendored processors. Guards: run count must equal
image count, and an unrecognized placeholder strategy bails to **Unkeyed
Completion** (uncached serving) — degrade, never guess.

## Cache Key Space (deepened, architecture review 2026-06-11)

The splice-as-one-function shape shipped a correctness defect: two other
producers put token paths into the same tree (planner last-user re-render, and
canonical/tool leaf rebuild) and both render via the chat template —
*unexpanded* space — so their boundaries landed at wrong KV offsets and
persisted snapshots whose KV didn't match their key path. The existing offset
guards are bounds checks, not space checks.

So the splice becomes the **Cache Key Space** (`CacheKeySpace.swift`): a
per-request value built once after `prepare` that owns the image table and both
duties — producing the key path *and* translating render-space token sequences
into key space. Every token path that touches the tree passes through it.
Supporting decisions:

- The per-family placeholder identity is a **Model Identity** load-time fact
  (`nil` ⇒ family not recognized ⇒ Unkeyed Completion).
- Planner re-renders and leaf probes share one token-only **Conversation
  Render** (family message-forming + template, no pixel work), so probe renders
  cannot drift from prepare's render.
- Degradation is two-tier: Key Space *construction* failure makes the whole
  request an Unkeyed Completion; a later *translation* failure skips only the
  consuming feature with a typed reason (`LeafSkipReason`).
- `ModelFingerprint` adds the preparation-identity files (`tokenizer_config`,
  `preprocessor_config`, `chat_template`), so a processor or template change
  *partitions* the cache instead of leaving stale never-matching paths.

This is the design SGLang's `pad_input_ids` ships in production
(image-hash-derived values in the radix key) and vLLM V1 folds into block
hashes via per-image `mm_hash`. Marconi (the cache this follows) is text-only,
but none of its invariants are violated.

## Consequences

- The digest→pseudo-token expansion is a **frozen** pure function
  (`ImagePseudoToken.swift`): expansions persist inside SSD admission paths
  across restarts, so changing it invalidates every image-bearing snapshot on
  disk. Golden-value tests pin it.
- Identity is exact-byte. A re-encoded or resized variant is a different image
  — always a miss, never a wrong hit. Within-conversation reuse (immutable
  `ImageAttachment.data` agent-side, verbatim base64 over HTTP) is what the
  cache is for, and there bytes are provably stable.
- `prepare()` must run before lookup (the splice scans its output), so a fully
  cached image conversation still pays CPU image preprocessing per request. The
  vision tower never runs for cached positions — that is the expensive part.
- Eviction scores image positions as ordinary text tokens; the vision tower's
  recompute cost is not modeled. Accepted for now: the failure mode is one cold
  vision-tower run, eviction carries no persistence, and folding vision flops
  into `ModelFlopProfile` later is a pure-policy change.
- **No settings toggle.** The structural safety net is the Unkeyed Completion
  bail; shipping is gated by a validation ladder (ADR-0006 style), not a flag —
  above all, different-image-same-size must never hit.

## Capture shape (spike, 2026-06-11)

The warmed-cache spike (Qwen3.5-4B PARO) confirmed the keying design but
falsified the planned warm-image capture, fixing several vendor facts now
pinned by the VLM smoke harness so vendor drift fails loudly:

- The vendored VLM container recomputes M-RoPE positions from zero on any
  forward that starts without threaded state, and the upstream `TokenIterator`
  discards `prepare()`'s state — both mis-position decode after an image prompt.
  So generation on the cache-aware VLM path must thread model state end-to-end
  (prefill → prime → decode); decode is app-owned there.
- The correct warm anchor is reachable app-side with **zero vendor changes**:
  seed the vendor's `qwen35.ropeDeltas` slot via the public `LMOutput.Key`
  API (`PositionAnchor.swift`) to select the continuation branch (positions =
  arange + cache offset + rope delta). Seeded restores are bitwise exact.
- The rope delta is reconstructible from the processed image grids —
  Σ (max(t, h/m, w/m) − t·h·w/m²) — so no extra snapshot persistence is needed.

**Phase 1 capture shape (now superseded for image-add by phase 2):**
image-add turns serve cold (full vendor `prepare`, correct M-RoPE by
construction); subsequent turns restore with a seeded Position Anchor; warm
image-bearing *remainders* were out of phase 1, the lookup clamping hits so
image runs never land in the remainder.

## Phase 2 (2026-06-13): warm continuation *through* a new image

Phase 1's cold-on-image cut caused two `qwen3.6-27b-paro` production failures:
clients put the image in the newest message, so the snapshot always sat below
`minimumWarmOffset` and every image turn went cold — and cold means one
single-shot full-attention pass over the whole prefix (`Qwen35.prepare`
ignores `windowSize`), whose `[heads, L, L]` bf16 scratch crashes near
L≈25,000 tokens (30 GB Metal buffer limit; a 31 K coding session + image hit
47.9 GB → SIGTRAP).

The phase-1 conclusion "single-shot remainder prepare is not viable" was
correct but read too broadly: it tested only the existing `prepare()` and
zero-vendor-change tricks, never a method injecting offset-aware *image*
positions (which needs a vendor change phase 1 was avoiding). Two phase-1
premises also relax: the GatedDeltaNet (Mamba) state is position-free and
restore-and-continue is exact (only *rewind* is impossible, which we never do),
and the full-attention KV cache is already offset-aware.

**Decision.** Image-add turns restore the deepest valid prior snapshot at
offset `P` and continue *through* the new image, chunked. Per ADR-0006's
amended vendor stance (general + upstreamable), the new surface lives in the
vendored model:

- A windowed `Qwen35.prepareContinuation` that honors `windowSize` and loops
  chunks over `[P, end)` — crash-safe by construction (scratch bounded to
  `[heads, chunk, L]`). Decode stays app-owned.
- An offset-aware `getRopeIndex` (the one genuinely new bit of math): a third
  position branch for *pixels present **and** warm cache*, computing the new
  image's diverging t/h/w positions from the seeded Position Anchor instead of
  resetting to zero. No new scalar — it reuses the anchor already seeded for
  text remainders.
- App side is small: relax the `PrefillPlanner` guard to allow `P` below the
  image, pass the remainder's image frames to the continuation. **Cache Key
  Path and SSD format are unchanged**, so no snapshot invalidation.

This supersedes phase 1's "image-add serves cold"; `minimumWarmOffset` survives
only as the fallback when no valid restore exists, and as the boundary between
the vendor-continued image span and the app-chunked text tail.

This does **not** contradict the rejected "extend vendored processor API":
that was about *reporting image-token spans* (derivable app-side). Running the
vision tower, the merge, and an offset-positioned forward needs the weights and
private members — not derivable app-side — and keeping position/merge math
vendor-side honors the patch-count rejection below.

**Validation ladder (gates merge — a wrong position is silent):** warm output
argmax-stable vs production-cold *and* bitwise vs a shape-matched reference
(chunk-shape FP noise is real); replay of captured real OpenCode requests warm
vs cold; frozen-fact harness checks for the new vendor surface; an image
hit-rate workload confirming image turns now restore instead of going cold.

**Folded-in general fix (upstreamed):** `Qwen3VLProcessorConfiguration` ignored
the new-style `size.{longest_edge, shortest_edge}` keys Qwen3.6 PARO ships,
silently falling back to defaults — wrong for every Qwen3-VL model on the new
format.

## Considered / rejected

- **Heterogeneous radix alphabet** (`token | imageDigest` edges): honest
  modeling, but every consumer must learn the new alphabet plus an explicit
  key→KV mapping — much wider blast radius for the same behavior.
- **Conversation-level digest fields** (like `toolDefinitionsDigest`): wrong
  granularity — image identity goes all-or-nothing per conversation, amputating
  longest-prefix reuse exactly where it matters.
- **Pixel-normalized / perceptual hashing**: a decode pipeline that must stay
  bit-stable across CoreImage versions, or false hits that silently answer
  about the wrong image. Exact-byte misses cost only what today already costs.
- **Re-implementing patch-count math app-side**: correctness-critical
  duplication that drifts when vendored processors update; a wrong N corrupts
  every offset past the image. Worst failure mode in the space.
- **Extending the vendored processor API to report image-token spans**:
  re-diverges the vendor tree ADR-0006 just un-forked, for something derivable
  app-side.

## Accepted degradations (rare fallback paths)

- The **thinking-safeguard continuation** is degraded on image-bearing requests:
  it re-forwards the stored real tokens without pixels (which is why
  `fullTokens` stays real and the tree keys on `keySpace.keyPath`), so image
  pad tokens hit text embeddings. Accepted — the safeguard is an exceptional
  intervention that should almost never fire.
- **Unkeyed Completion skips decode-time KV quantization** (strips `kvBits`
  rather than replicating the iterator's per-step swap). Cost is RAM-only, on a
  path whose cache is discarded after the request anyway.
- A separate anomaly — one 2000×1159 screenshot expanding to ~43,500 pad tokens
  (vs the provable ~2,268) — is **deferred, not chased**: it needs runtime
  ground truth, and the chunked continuation makes even a 43 K-pad image
  non-fatal, so it no longer gates anything. Cheap instrumentation only.
