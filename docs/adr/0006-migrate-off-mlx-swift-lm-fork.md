---
status: accepted
---

# Migrate off the mlx-swift-lm fork: depend on vanilla upstream, re-home the prefix-cache machinery app-side

The app stops depending on the `spokvulcan/mlx-swift-lm` fork and pins vanilla
`ml-explore/mlx-swift-lm` instead. The fork's reason to exist had mostly expired —
ParoQuant (#164), the ToolCallProcessor schema plumbing (#167), and the TokenRing
2D-prompt fix (#168) were all merged upstream (upstream's ParoQuant is in fact
*newer* than the fork's), and the TriAttention stack was already inert per
ADR-0005. What remained fork-only and load-bearing was the prefix-cache machinery:
`HybridCacheSnapshot`, checkpoint-aware chunked prefill
(`prepareWithCheckpoints`, the `GenerateParameters` checkpoint fields), and
`FinalizedKVCacheHandle`. **That machinery moves into the app** (alongside its
only consumers in `Features/Server/`) rather than staying in a fork or being
proposed upstream.

The surprising part for a future reader: bespoke KV-snapshot capture/restore code
lives in the app instead of the inference package. That is deliberate. It is
possible — and only possible — because upstream's public `KVCache` API exposes
everything capture needs (`state`/`metaState` get *and* set, `copy()`, `trim`,
`isTrimmable`, `savePromptCache`/`loadPromptCache`), and
`TokenIterator.init(input:model:cache:parameters:)` accepts a pre-warmed cache.
The app drives chunked prefill itself (splitting chunks at planned checkpoint
offsets — `PrefillPlanner` already computes these), snapshots via `copy()`/`trim`,
and owns the `[KVCache]` array across generation, which makes
`FinalizedKVCacheHandle` unnecessary. The agent path likewise stops using the
package's `generateTask` event stream: it consumes the public raw-token
`generateTokenTask` stream and owns only the event mapping — detokenization
into its own `ToolCallProcessor` instance (in-flight tool-call deltas = text
the processor swallows; EOS recovery via `processEOS(returnBufferedText:)`),
with a `ToolCallDeltaTracker` mirroring the processor's collection states for
the delta stream.

Mechanics: `Vendor/mlx-swift-lm` stays a submodule but its URL changes to
`ml-explore/mlx-swift-lm`, pinned to a vetted recent `main` commit — release tags
lag too far (3.31.3 predates upstream's ParoQuant). The Xcode local-package
reference and `mlx-audio-swift`'s `.package(path:)` are unaffected. The fork repo
is kept as a staging ground for future upstream PRs.

## Accepted costs

- **Image-bearing prompts lose mid-prefill checkpoints past the first image.**
  The VLM's image+text embedding merge happens inside upstream `prepare()`, which
  the app cannot chunk into. Capture degrades to: chunked checkpoints over the
  longest image-free token prefix (where the stable system-prompt checkpoint
  lands), then single-shot `prepare()` for the remainder on the warmed cache,
  then the end-of-prefill leaf. Turn-over-turn radix chaining is preserved
  because each turn's leaf covers the full prompt.
- **TriAttention is deleted with the fork pin** — this supersedes ADR-0005's
  "vendor left inert" clause. The implementation survives in the fork repo's
  branches (`test/paroquant-pr-review` and earlier) if ever revived.
- **Upstream behavior drift** (rewritten tool-call parser #205, prefill
  pipelining #225 replacing the fork's `Memory.clearCache` between chunks, the
  independently fixed hybrid-cache `maybeQuantizeKVCache`) is accepted and gated
  by migration validation: full unit suite, `HybridCacheCorrectnessRunner` +
  `ParoQuantVLMSmokeRunner`, `bench.sh` perf parity, and a scripted hit-rate
  parity workload against a pre-migration fork baseline.

## Considered / rejected

- **Thin fork — rebase onto upstream, keep only the live cache patches.**
  Rejected: shrinks but does not end the fork-maintenance treadmill; every
  upstream sync re-runs the conflict risk on `Evaluate.swift`, the hottest file.
- **Upstream-first — PR the checkpoint/snapshot design to ml-explore and wait.**
  Rejected as a migration *prerequisite*: timeline out of our control and the
  design is Tesseract-shaped. Individual pieces (e.g. tool-call argument
  streaming deltas, OpenAI-parity) may still be proposed later as
  simplifications.
- **Drop the prefix-cache features and use upstream's plain prompt cache.**
  Rejected outright: the tiered radix cache is product priority #2.

## Amended (2026-06-13): the vendor is the frontier-experimentation surface

The original framing ("machinery moves app-side; it is possible *only because*
public API exposes everything") read, in practice, as "never touch the vendored
model." That is **not** the intent and is hereby narrowed. The submodule is
vendored precisely so we can experiment at the frontier of on-device inference —
this is not a stable library that already provides everything we need. The
operating rule is:

- **Prefer vanilla.** If a capability is reachable through public API, do it
  app-side (as the cache machinery already does).
- **Change the vendor freely when public API does not suffice, or when a vendor
  change is more performant / more ergonomic / more correct** — with one
  constraint: the change must be **general and upstreamable** (a capability or
  fix any consumer of `mlx-swift-lm` would want), shaped from the start as an
  upstream PR. We open the PR, keep the change in the submodule pin meanwhile,
  and re-converge on vanilla once merged.
- **Not allowed:** Tesseract-specific hacks that only make sense inside this app.
  Those still belong app-side.

This supersedes the "only public API" reading and refines the "upstream-first as
prerequisite" rejection above: vendor changes no longer *block* on the merge —
they ship in the pin and upstream in parallel.

First instance: **warm image-remainder continuation** (ADR-0007 phase 2). It
needs a general `Qwen35.prepareContinuation` (warm cache + position offset + new
images) and an **offset-aware `getRopeIndex`** (`positionOffset` parameter; today
hardwired to start at 0) — both clean upstream contributions. A latent general
bug surfaced alongside: `Qwen3VLProcessorConfiguration` ignores the new-style
`size.{longest_edge,shortest_edge}` keys (decodes only legacy
`min_pixels`/`max_pixels`), wrong for every Qwen3-VL model on the new config
format — also upstreamed. This revises the "Accepted costs" bullet above
("single-shot `prepare()` for the remainder on the warmed cache"), which
ADR-0007's spike already falsified and replaced with cold image-add turns; phase
2 replaces *that* with a chunked, offset-anchored continuation.
