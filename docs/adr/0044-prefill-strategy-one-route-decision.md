# ADR-0044: The Prefill Strategy is one decision; the raw arms execute it

- Status: Accepted
- Date: 2026-07-18
- Relates to: ADR-0006 (the chunked-prefill driver this routes into),
  ADR-0033 (the sibling move on the server side — derivation rule out, MLX
  glue stays), ADR-0042/0043 (the same policy/performer split elsewhere)

## Context

The chunked-vs-single-shot prefill decision — the crash-relevant one:
bounded chunking through the app driver versus one large single-shot
allocation — was implemented as three hand-written guards:

- `LLMActor.startRawGeneration` checked `ndim >= 2`, `image == nil`,
  `video == nil`, `dim(-1) > step`;
- `LLMActor.buildThinkingContinuationStart` checked `ndim >= 2` and
  `count > step` only — the media legs already missing;
- `ParoParityBenchRunner.runOnce` carried a third copy, written to mirror
  the first.

All three re-derived the `?? 512` step-size fallback, and all three sat
inside `container.perform`, reachable only with a loaded model — the rule
had no tests. `LMInput` also grew an `audio` slot none of the guards knew
about: an audio-bearing 2D prompt would have been fed to the text-only
chunked driver.

## Decision

Extract the route as **`PrefillStrategy`** — a `nonisolated` value with two
cases (`chunked(stepSize:)`, `singleShot`) and three members:

- **`decide(tokenNDim:sequenceLength:hasImage:hasVideo:hasAudio:prefillStepSize:)`**
  — the pure, MLX-free decision table. 2D text-only prompts longer than one
  step chunk; media, 1D shapes (upstream's `TokenIterator` chunks those
  internally), and prompts fitting one step go single-shot. The `?? 512`
  fallback lives here as `fallbackStepSize`, its one home.
- **`decide(for: LMInput, prefillStepSize:)`** — the fact extractor: reads
  the shape facts off the prepared input, so a call site cannot omit a leg.
  The audio leg is new — a no-op today (nothing on `main` feeds audio into
  the raw arms) that closes the latent gap for audio-capable models.
- **`makeIterator(input:model:parameters:)`** — the route executor: the
  Metal-affine glue that warms the cache through `PrefillExecutor` on the
  chunked arm. Stays inside `ModelContainer.perform`, as ADR-0033's
  "derivation out, MLX glue stays" prescribes.

`LLMActor` keeps one shared `makeRawGenerationStart` tail (token-event loop
start + start-value wrap) that both raw arms return through. The parity
bench routes its guard through the same `decide` and keeps its own timed
execution arms — the mirror is now structural, not copied.

## Consequences

- `PrefillStrategyTests` pins the rule as decision tables — the model-class
  leg, each media leg, the strict `>` boundary at the step size, the carried
  step size, and the fallback — with no model, no Metal, no container.
- Drift between the arms is unrepresentable: there is one guard, and the
  extractor derives its facts from the input itself.
- The continuation arm now honors the media legs it silently lacked
  (unreachable today: its input is built token-only by construction).
- Retuning the route — a new media kind, a different boundary — means
  editing one pure value and reading the diff of its table.
