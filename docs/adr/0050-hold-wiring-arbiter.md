# ADR-0050: The voice hold's async arbitration is a value machine

- Status: Accepted
- Date: 2026-07-19
- Relates to: ADR-0041 (Dual-Path Playback — the hold itself), ADR-0042
  (Voice Session Machine — decides *when* to hold), ADR-0025 (the
  policy/performer split this engine already uses twice)

## Context

`AudioCaptureEngine`'s decomposition stalled one band short. The decision
*to* hold is tested (Voice Session Machine), the keep-vs-rebuild verdicts
are tested (Capture Engine Lifecycle), the duck treatments are tested
(System Audio Duck policy) — but the async arbitration between them ran as
inline mutable state on the engine: `voiceHoldActive`,
`holdWiringInProgress`, `holdWireQueued`, `holdGeneration`, mutated across
`beginVoiceHold` / `endVoiceHold` / `scheduleHoldWiring` /
`commitHoldWiring`, with zero tests reaching any of it.

That arbitration is precisely the discipline that prevents the 2026-07-17
tap-rewire crash class (format-touching calls on a running VP engine): a
detached wiring owns the engine for ~1–2 s; every competing intent — a
device-change re-wire, the hold ending, a press — must either fold behind
it or stale it, and a stale wiring's work must be discarded on the stopped
engine, never raced. The invariant "two wirings never touch one engine at
once" existed only as comments above the fields that implemented it.

## Decision

**`HoldWiringArbiter`** — a pure value machine over the four arbitration
facts (`isHoldActive`, `isWiringInFlight`, `currentGeneration`,
`queuedRebuildFirst`), one transition per driver, each returning a verdict
the engine performs:

- `beginHold()` / `endHold()` — session intent; ending returns
  `leaveDiscardToCommit` when a wiring is in flight (the engine must not
  be touched) or `unwireNow`, and kills the queued request either way.
- `schedule(rebuildFirst:)` — returns `start(rebuildFirst:generation:)`
  when idle, or `folded` while in flight (OR-folding rebuild-first so a
  rebuild demand is never downgraded). Every schedule bumps the
  generation, staling whatever flies.
- `wiringLanded(generation:)` — the landing verdict: `commit` for the
  current generation under an active hold; otherwise
  `discardAndStartNext` (a queued request takes over in the same breath —
  the engine has no ownerless moment) or `discardAndIdle`.

`AudioCaptureEngine` keeps everything else: the graph flags
(`voiceHoldWired`, `holdRenderWired`, the hosted player node), task
handles, and every AVFoundation call (`performHoldWiring`,
`discardHoldWiring`, `adoptBuiltEngine`). The four stored fields are
replaced by the arbiter value plus two computed reads
(`voiceHoldActive`, `holdWiringInProgress`) so the engine's dozen guard
sites are untouched. The `AudioCapturing` interface is unchanged.

This is the third use of the engine's policy/performer template
(lifecycle, duck, now the hold's arbitration) — the difference being that
this policy is stateful, so it is a value machine rather than a stateless
decision function. It stays a struct: state transitions are synchronous
and MainActor-driven; only the wiring work itself is detached.

## Consequences

- The crash-class discipline is a decision table
  (`HoldWiringArbiterTests`): schedule folding, generation staleness, the
  end-mid-wiring discard deferral, the queued handoff, and the rapid
  end→begin cycle around an in-flight wiring — scenarios that previously
  required racing real AVAudioEngine wirings to exercise at all.
- The generation guard has one home. A new driver (say, a future
  route-change teardown) gets its race behavior by calling a transition,
  not by re-implementing bump-and-compare inline.
- The engine's hold sites shrink to gather-verdict-perform, matching the
  shape of its other two policies.
