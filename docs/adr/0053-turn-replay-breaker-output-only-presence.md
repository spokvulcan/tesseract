# ADR-0053: Turn Replay Breaker and output-only presence penalty

- Status: Accepted
- Date: 2026-07-20
- Relates to: ADR-0046 (Report-Back), ADR-0052 (report_back in every chat)
- Amended by: ADR-0093 (2026-10-11: the logit seam loses its
  `pathQuantizesKVUpFront` flag; see the amendments at the end)

## Context

On 2026-07-20 a dialogue conversation looped: qwen3.6-35b-a3b-paro replayed
one turn — identical thinking, identical text, identical `track` +
`report_back` calls — ten times in a row, until the owner hit stop. A second
dialogue had done the same the night before (4× on `remember`). The forensics
ruled out the harness: input tokens grew every iteration, so the model saw its
previous turn and both tool results and regenerated the replay anyway
(argument-JSON key order shuffled between copies — live sampling, not an echo).

The trigger is a turn shape the Event Fold era made common: the text already
answers the user, the tool calls are terminal bookkeeping (`track`,
`report_back`, `remember`). Once their results return there is nothing left to
say, and a quantized local model's strongest continuation is its own previous
turn. Every replay re-executed the tools (ten duplicate track records, ten
deposits into the fold's queue) and added another copy to the context,
deepening the attractor. Two defenses existed and both were blind: the
thinking-loop safeguard watches a single `<think>` block, and the presence
penalty — Qwen's recommended loop suppressant, `presence_penalty 1.5` — was
wired to the vendor's 20-token prompt-seeded sliding window, which cannot see
a ~250-token replay. Raising the window would have been worse: the ring seeds
from the prompt tail, so a large window puts the system prompt and history
under a −1.5 blanket.

## Decision

**Turn Replay Breaker.** The agent loop owns a `TurnReplayGuard`: an assistant
turn identical to its predecessor — same trimmed text, same tool calls with
canonically-equal arguments (key order normalized; thinking excluded) — never
re-executes its tools. The first replay commits a corrective error result per
call ("this already executed — its results are above; do not send them
again"), giving the model one differently-worded context to escape through.
A second replay stops the run with a visible `generationError`. Any
interjection (steering, follow-up) resets the chain — "do it again" is a
request, not a replay. Deliberately narrow: paraphrases and same-tool-
different-args turns pass untouched, so retries stay legal.

**Output-only presence penalty.** `AgentLogitProcessors.processor(for:)`
replaces `GenerateParameters.processor()` on every generation path (chunked
prefill, state-threaded decode, single-shot). Presence becomes
`OutputPresencePenalty`: vLLM/OpenAI semantics — additive penalty on tokens
*generated this request*, prompt never enters the ring, window spans the whole
realistic generation (32K ring). Repetition and frequency penalties keep
vendor semantics untouched. The single-shot arm diverts only when `kvBits` is
nil (every agent preset since #252): the explicit-processor iterator init has
no in-iterator cache quantization, and silently changing quantization order
for a theoretical config is worse than keeping the old window there.

Rejected: a round cap on the inner loop (punishes long legitimate tool
chains); deduplicating tool calls run-wide by name+args (breaks read→edit→read
and retry-after-transient-error); stripping prior-turn thinking from the
re-fed context (deviates from the Qwen template contract mid-loop and
invalidates the prefix cache for no proven gain).

## Consequences

- A stuck model costs at most three generations per beat: original, refused
  replay, terminated replay. Side effects (track records, fold deposits,
  memories) happen exactly once.
- The refusal result is model-visible text in the transcript, same as any
  tool error — the entity can and should recover in its next turn.
- Qwen's `presence 1.5` presets now do what the model card means. Decode
  cost is one 32K-max gather/scatter per step, negligible next to a forward.
- Server clients sending `presence_penalty` get OpenAI semantics instead of a
  20-token window — a behavior change in the compatible direction.
- `AgentGenerateParameters.presenceContextSize` still exists but no longer
  drives the main paths; it parameterizes only the vendor window on the
  quantized-KV single-shot carve-out.

## Amendment (2026-07-22)

The "replaces `GenerateParameters.processor()` on every generation path" clause
above was, at ship time, verified only at the pure factory's home — no test
covered the four wiring sites that had to *call* it, and one (single-shot)
re-implemented the `kvBits` carve-out by hand. That is the same wiring-not-math
shape whose earlier lapse was the original inert-penalty bug (issue #405).

The clause is now enforced structurally by one seam, `GenerationLogitProcessor`
(`Features/Agent/GenerationLogitProcessor.swift`): a pure `nonisolated` decider
`resolve(for:pathQuantizesKVUpFront:)` that owns the "which processor does this
generation get" decision, including the `kvBits` carve-out (folded into the
`pathQuantizesKVUpFront` flag). All four sites — single-shot prefill, chunked-
prefill decode, and both state-threaded decode inits — route through it; none
hand-picks a processor. Decision rows and per-path wiring (the state-threaded
iterators asserted to hold the seam's `OutputPresencePenalty`, not the vendor
default) are tested in `GenerationLogitProcessorTests`.

## Amendment (2026-10-11): the seam loses its flag

With the Decode Start (ADR-0093) deciding where each path converts the KV
cache, `GenerationLogitProcessor` keeps two shapes: `processor(for:)`, for an
iterator that holds the processor itself, and `components(for:)`, for an
iterator built from parameters, which keeps the vendor's KV plan. The
`pathQuantizesKVUpFront` flag is gone. The one carve-out it encoded — the
single-shot arm's explicit-processor init runs no KV plan, so with `kvBits` it
keeps the vendor's own penalties (theoretical, #252) — sits in
`PrefillStrategy`, the only place it applies. Which processor each path
decodes with, and what seeds it, is pinned by outcome in `DecodeStartTests`;
`GenerationLogitProcessorTests` keeps the seam's decision rows.
