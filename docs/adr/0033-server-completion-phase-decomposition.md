# ADR-0033: Server Completion decomposes into named phases; plan application deliberately stays inline

- Status: Accepted
- Date: 2026-07-11
- Amended by: ADR-0078 (2026-09-29: the leaf store decides which leaf, a
  Leaf Admission stores it); ADR-0087 (2026-10-03, proposed: the Prefill Plan
  takes back its decisions; see the amendment at the end)
- Relates to: ADR-0015 (actor-confined module), ADR-0016 (Model Session
  seam), ADR-0006 (app-owned prefill), PRD #282 (architecture deepening
  program)

## Context

`ServerCompletion` had grown to ~3,200 lines: one actor-confined module
holding every step of a cache-aware completion inline — tokenization and
cache keying, prefix resolution, checkpoint planning, restore, chunked
prefill, the generation stream drive, leaf admission, and trace
derivation. The module's *interface* was fine (the dispatcher enters
through a handful of entry points, per ADR-0015); the *implementation*
had no internal seams, so every change landed in the same file, the
tokenizer-affine logic was untestable without a loaded model, and the
derivation rules (eviction tallies, leaf-store modes, skip logging)
were smeared through the drive.

## Decision

A cache-aware **Server Completion** is a sequence of six named phases.
Each phase that carries real derivation rules is a `nonisolated` module
with its own tests; the completion module keeps orchestration and the
MLX-touching glue. The phase map:

1. **Request Keying** — `RequestKeyingPhase`: conversation → the
   identities every later phase keys on (prepared input, flat tokens,
   `CachePartitionKey`, the request's **Cache Key Space**), or the
   degrade signal that turns the whole request into an **Unkeyed
   Completion**.
2. **Resolution + plan** — `PrefillPlanner` (pre-existing): boundary
   detection against the key space, then the fold of lookup result +
   checkpoint plan into the request's **Prefill Plan**
   (restore-vs-cold, suffix checkpoint filter, `prefillBaseOffset`).
3. **Plan application** — inline in `ServerCompletion`, by decision
   (below): restore/rewind/cold-start execution and app-driven chunked
   prefill against the live KV cache.
4. **Stream drive** — `ManagedGenerationDriver` (shared with
   `AgentEngine`): safeguard derivation, the generation stream loop,
   cancel bridging, terminal-info re-yield.
5. **Leaf store** — `LeafStorePhase`: leaf-store mode selection,
   structured-leaf capture and admission, skip logging, the
   speculative-prefill seed plan.
6. **Trace accumulation** — `CompletionTraceAccumulator`: the
   terminal-vs-recovered eviction tally paired with its correlated
   diagnostics emission, the restored-offset rule, and the
   `CompletionTraceRecord` derivation.

Constraints the decomposition preserves:

- **ADR-0015 stands.** The phases are implementation structure inside
  the actor-confined module's seam, not new entry points. The
  dispatcher's interface to Server Completion is unchanged.
- **ADR-0016 stands.** Phases that hold non-`Sendable` session values
  (`LMInput`, live `[any KVCache]`) run inside the `withSession` scope
  and are `nonisolated async`, so under approachable concurrency they
  execute on the session's isolation and prepared input never crosses
  it. The Model Session port is still the test seam for
  tokenizer/model-affine phases.
- **Phases return values; the completion module owns effects.** A phase
  hands back a `Result`/`Outcome` value (or ingests into an
  accumulator) and the orchestrator decides what to do — e.g.
  `RequestKeyingPhase` returns `.unkeyed(...)` as data and the caller
  invokes the unkeyed build.

**Plan application stays inline — deliberately.** It was measured
against the deletion test and failed: the restore/rewind/cold execution
has four shapes distinguished mostly by which MLX handles they touch,
and an extraction would need roughly twenty inputs (session, cache,
plan, key space, boundaries, diagnostics, progress, scratch profiles…)
to return ten outputs. That interface is as complex as the
implementation — a shallow module that renames the coupling instead of
hiding it. The real derivation content of plan application already
lives in `PrefillPlanner.plan`; what remains inline is glue over live
MLX state, which is exactly what the completion module is for. Future
reviews should not re-propose this extraction unless the shape changes
(e.g. a second caller appears).

## Consequences

- `ServerCompletion.swift` drops from ~3,200 to ~2,480 lines; the four
  extracted/named phases total ~1,190 lines of `nonisolated`, unit-
  tested modules (`RequestKeyingPhase` 164, `LeafStorePhase` 761,
  `ManagedGenerationDriver` 114, `CompletionTraceAccumulator` 152 —
  plus the pre-existing `PrefillPlanner` 254).
- Leaf-store modes, skip logs, seed plans, eviction tallies, and keying
  degrades are testable without a loaded model, through each phase's
  interface (`LeafStorePhase`-targeted suites already existed and were
  retargeted; `CompletionTraceAccumulatorTests` and
  `ManagedGenerationDriverTests` are new).
- The drive and its safeguards are one implementation shared by the
  agent and the server; a fix to either path lands in both.
- Cross-phase values (`Keyed`, `PrefillPlan`, `LeafStorePhase.Result`)
  are the phase contracts; adding a fact for a later phase means
  extending a value, not threading another parameter through the drive.

## Amendment (2026-10-03): the Prefill Plan takes back its decisions (ADR-0087, proposed)

This takes effect when ADR-0087 is accepted. The premise above, that the real
derivation of plan application lives in `PrefillPlanner.plan`, stopped
holding as the speculation route, the preserve-thinking capture gate, the
maximum advance and the failed-restore re-plan landed inline. ADR-0087 moves
every decision a keyed request makes before the Cache Claim check-out into
the Prefill Plan and keeps this ADR's decision: plan application still
carries the plan out inline, inside the completion module, with no
extraction and no session verb. Once it is built, phase 2 yields every
pre-check-out decision and phase 3 only executes them. Already true today:
there are five shapes, not four, and plan-time predicates choose them; only
their bodies touch MLX. The reopen trigger above still governs plan
application itself, and ADR-0087 does not meet it: plan application keeps one
caller and the same MLX glue.
