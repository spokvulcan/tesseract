# ADR-0078: Every leaf enters the prefix cache through one Leaf Admission; the Snapshot Payload owns the SSD byte form

- Status: Proposed (built; accepted once the loaded-model gates pass, see As built)
- Date: 2026-09-29
- Amends: ADR-0033 (the phase map: the leaf store decides which leaf, a Leaf
  Admission stores it)
- Relates to: ADR-0069 (decisions 3, 5 and 7 hold and become structural),
  ADR-0064 (move or copy at capture), ADR-0016 (the admission runs inside the
  producer's Model Session), ADR-0010 and ADR-0019 (extension, guarantee write
  and supersession stay in the manager), ADR-0068 (view persistence stays in
  the manager)

## Context

A leaf enters the prefix cache in five ordered steps: resolve the extension
base on the MainActor before the Model Session is entered; capture the cache
by move or by copy; check a leased leaf in through its Cache Claim; build the
SSD payload; admit the Snapshot Admission on the MainActor and classify what
it evicted and superseded. Six leaf producers run those steps and sequence
them by hand in three shapes, and checkpoints run a fourth, shorter one:

- the Leaf Store executors (live, direct, boundary, and the backing leaf of a
  think-stripping boundary turn) share one tail, `admitLeaf`
  (`LeafStorePhase+Executors.swift:334-455`), and the phase classifies
  afterwards (`LeafStorePhase.swift:197-201`, `228-231`);
- the Speculative Canonical Prefill captures inside a raw
  `container.perform` and builds a throwaway accumulator to log
  (`SpeculativePrefill.swift:462-538`);
- salvage-on-cancel captures inside the prefill's session, RAM-only, with its
  own throwaway accumulator (`ServerCompletion.swift:3129-3216`);
- checkpoints extract their payload during prefill and are admitted after the
  stream (`ServerCompletion.swift:2033-2042`, `1051-1057`).

Move or copy is decided in four places, one of them the MTP rule behind #497
(`LeafStorePhase+Executors.swift:137-139`). The payload builder and the dtype
wire table live in ServerCompletion (`2814-3084`), so PrefixCacheManager
(`1400`) and SSDSnapshotStore (`1899`) call up into the orchestrator, and the
tests of the SSD byte format go through ServerCompletion statics.
ServerCompletion shrank to 2,479 lines when ADR-0033 carved its phases out.
It is back to 3,218, and prefix-cache features account for 390 of the lines
it regained.

## Decision

1. **One Leaf Admission for every leaf producer.** The Leaf Store phase's
   live, direct, boundary and backing-leaf paths, the Speculative Canonical
   Prefill (both exits) and salvage-on-cancel all store their leaf through it.
   The producer brings the cache to its final state (restore, re-prefill,
   Capacity Compaction) and names whose cache it is: the finished
   generation's, which may be leased; one the producer owns outright; or one
   it only lends.
2. **Two steps, at today's points.** Preparing resolves the extension base on
   the MainActor before the Model Session is entered. Only an admission that
   may reach SSD hops, and it asserts that it runs outside a session scope. A
   RAM-only admission (salvage, a preempted speculative pass, the backing
   leaf) makes no hop and may be prepared anywhere. Admitting runs inside the
   producer's session: capture, check-in, the Snapshot Payload, path
   validation and the Snapshot Admission, one MainActor hop to admit, the
   captured-then-evicted check, and the classification.
3. **Move or copy is the admission's rule on the capture side.** A leased
   live cache moves only for a text-only, unquantized turn whose MTP arm was
   not active. An owned cache moves when every layer class can move. A lent
   cache is copied. Producers name the case and never compute eligibility. The
   claim keeps the restore-side check-out.
4. **The check-in stays a claim step.** The admission takes it for a leased
   leaf, before the payload is extracted (ADR-0069 decision 7) and before
   admission (#498). Capture and admission stay leaf production, outside the
   claim (decision 3). The step is still explicit on the claim (decision 5);
   it moves from the executors into the admission.
5. **The admission classifies once.** It runs the Completion Trace
   Accumulator's pairing of the eviction tally with its log lines, logs the
   supersessions, and returns the tallies. The Leaf Store phase merges them
   into the request's record without logging again. The speculative pass and
   salvage stop building throwaway accumulators.
6. **The Snapshot Payload is its own module.** It holds the payload value,
   Deferred Payload Extraction for whole, extension and view payloads, the
   extension worth-it gate, and the dtype wire table in both directions. The
   SSD store's decoder, the manager's view persistence and demotion extractor,
   the Leaf Admission and the checkpoint path all use it, so no cache tier
   calls into the Server Completion. The container framing stays where it is.
7. **Checkpoints and view persistence stay outside.** A checkpoint's payload
   is extracted during prefill and admitted after the stream, with no
   extension base, check-in or supersession. Its storage intent moves beside
   the Snapshot Admission value, which the Leaf Admission also uses. View
   persistence stays in the manager, and only its payload call changes.
8. **Behavior is preserved.** Every diagnostics event, field and stage label
   stays byte-identical; the stage labels ride into the admission as named
   presets, one per producer. The order of the steps, the number of MainActor
   hops and Model Session entries, and the memory peaks do not change. Known
   irregularities are follow-ups, not part of this change (see Consequences).

## Considered options

- **Hand over a captured snapshot instead of a cache.** The seam would sit
  after capture. It is simpler, but it leaves move or copy and the check-in
  order at every call site, and those are where #497 and #498 happened.
- **Also own the re-prefill executors.** Restore, chunked prefill and
  preemption change for other reasons than storing a leaf does, and the
  admission would have to learn about prefill.
- **One call that enters its own Model Session.** The boundary executor and
  salvage already hold a session when they capture, the session lock is not
  reentrant, and splitting their sessions would add entries.
- **Resolve the extension base inside the session.** That makes one call
  instead of two, but it moves a MainActor wait into the Metal-affine hold.
  It was rejected under decision 8; revisit it if the two steps prove clumsy.
- **Make it a PrefixCacheManager method.** Admitting runs inside the
  producer's session, where a MainActor method cannot run. The manager stays
  the only mutator. The admission reaches it through two members, the
  extension-base query and `admit(SnapshotAdmission)`, which is the sanctioned
  seam and not the retired hydrator's reach-back.
- **Put the payload in SSDSnapshotStore or HybridCacheSnapshot.** The store
  is 2,043 lines of queueing and I/O; the snapshot already owns capture,
  restore and compression. The byte contract needs one home that tests can
  reach without either.
- **Let callers keep classifying.** That is today's shape: every new
  producer is one more chance to forget the classification or run it twice.

## Consequences

- ServerCompletion loses the admission and payload statics
  (`resolveExtensionBase`, `snapshotAdmissionStorage`, `admitStructuredLeaf`,
  `extractSnapshotPayload`, `deferredPayload`, `DeferredLayers`, the dtype
  table, `extractCheckpointAdmissionCandidates`). Salvage keeps its decision,
  `salvageableOffset`, and hands the capture to the admission.
- The Leaf Store executors keep cache preparation: restore, re-prefill,
  compaction and the direct path's guards. Their shared tail, `admitLeaf`,
  goes. `LeafStorePhase.run` loses its per-admission classification.
- Tests follow the modules; nothing is layered on top:
  - The SSD byte-format suites call the Snapshot Payload instead of
    ServerCompletion statics. Their assertions stay as they are.
    `ServerCompletionExtractSnapshotPayloadsTests` takes the module's name,
    and `docs/testing.md` follows.
  - The two source-shape tests in that file,
    `structuredLeafAdmissionStaysWithItsSingleOwner` and
    `mainActorRunClosuresAroundPrefixCacheAdmissionsAreNonSuspending`, pinned
    the old owner. They are rewritten to pin the new one: only the Leaf
    Admission builds a leaf Snapshot Admission, and the MainActor admit
    closures stay non-suspending.
  - New Leaf Admission tests run on the toy Model Session over a real
    manager, with a temp directory where SSD is on. They cover the three
    ownership cases and the move-or-copy rule, a refused or cancelled
    check-in that extracts nothing, and one classification per admission.
    They also pin the wire strings no test pins today: `empty-cache-body`,
    `invalid-path`, `capturedThenEvicted`, salvage's skip reasons, and the
    capture sources `speculativeLeaf`, `speculativePartialLeaf` and
    `cancelledPrefillSalvage`.
  - `CacheClaimMemoryEvidenceTests` drives the real admission. Today it
    re-enacts the check-in-before-extraction order by hand, so the ADR-0069
    memory proof doesn't cover production. `LeafCaptureHandoffTests` retargets
    its `admitStructuredLeaf` case, and `SalvageOnCancelTests` passes a session.
  - The suites that assert on a whole completion's diagnostics lines stay
    unmodified, and they are the parity check for decision 8. They include
    the exit matrix, keyed sequencing, synthesized replay and leaf skip-log
    suites. The executors' `LeafCapture` value stays so the skip-log suite
    still compiles.
- Verification before this is accepted:
  - The prefix-cache unit allowlist in `docs/testing.md`, plus every suite
    touched, plus `SalvageOnCancelTests`, `CompletionTraceAccumulatorTests`,
    `SnapshotDemotionTests` and `SurvivalGateTests`, which the allowlist doesn't
    name. Filter by suite struct: a file-name filter runs no tests and still
    passes.
  - The memory evidence suites, each run on its own.
  - `scripts/dev.sh prefix-cache-e2e` and
    `scripts/dev.sh hybrid-cache-correctness`, which `docs/testing.md`
    requires after any change to ServerCompletion or PrefixCacheManager.
  - The bounded-cache parity run.
  - A before/after diff of the diagnostics JSONL from the same e2e run, with
    ids and timings masked.
- Follow-ups, each its own change with its own evidence:
  1. a handed-off leaf is stored twice, once by the check-in and again by
     `admit`;
  2. `admit` collects `leaseRefusals` that no production code reads, so an
     admission a lease refused still reports that its leaf survived; the
     stale "no production checkout exists yet" comment sits beside it;
  3. the speculative pass and salvage copy caches they own outright; naming
     them owned would move them and drop a full-KV copy;
  4. the speculative pass still enters a raw `ModelContainer` (the deviation
     from ADR-0016); the admission reaches it through the existing
     context-backed session;
  5. an MTP turn copies its leaf, but the report gives no copy reason;
  6. the captured-then-evicted check matches an eviction by offset and
     checkpoint type only, so when an admission evicts a different leaf of
     the same length, the new leaf is reported evicted: its tuner record and
     speculative seed are dropped and a false warning is logged. The new
     eviction test found it.

## As built (2026-09-30)

The branch landed in four commits after the proposal, each green on the
prefix-cache test allowlist:

1. The Snapshot Payload moves into its own file with its builder. The
   manager, the SSD store and the bounded-parity bench call it directly.
2. Storage intent and the checkpoint candidates move beside the Snapshot
   Admission value.
3. The Leaf Admission, with every producer moved onto it. The plan had the
   executors and the other two producers as separate commits. They landed as
   one, because the source-shape checks pin a single owner, which only holds
   once every producer has moved.
4. Salvage's capture source and below-threshold skip are pinned.

How the shape came out:

- `LeafAdmission.prepare`, then `admit(_:in:labels:turn:memory:)` inside the
  producer's session. The cache is `finishedTurn`, `owned` or `lent`.
- The finished turn's case takes whether the turn was text-only and which
  speculative arm ran, not the request's facts: only Request Keying can build
  those. The KV quantization fact comes from the admission's own partition
  key.
- When the claim takes a leased leaf back, the outcome carries the claim's
  own rewind cause, and the Leaf Store phase words it as its skip reason.
- Capacity Compaction and its `capturingLeaf` memory phase stay with the
  Leaf Store executors, the only producers that ever ran them.
- The executors' "captured" line in the unified log is now written after the
  admission returns instead of right after the capture. It is not a
  diagnostics line, and the order of the diagnostics lines is unchanged.

Measured:

- ServerCompletion.swift went from 3,218 to 2,722 lines. The cache tiers
  make no calls into it.
- The memory evidence reproduces ADR-0069's table exactly, now measured
  through the admission rather than a re-enactment: 6,324,240 bytes when the
  payload is extracted first, 2,129,936 when the leaf is checked in first.

Two wire strings can no longer be reached through the interface, so no test
pins them: `empty-cache-body` and `invalid-path`. The admission captures at
the stored length, and a capture never yields an empty body, so both guards
are defensive now. `capturedThenEvicted` isn't pinned either: in the cases
the new tests cover, the Budget Floor kept the newest leaf resident, and the
line fired only through follow-up 6. Every producer's labels, the
capture-stage skip and salvage's labels are pinned.

Loaded-model gates, run on 2026-09-30 on a MacBook Pro with an Apple M3 Max
and 48 GB, macOS 27.0:

- `scripts/dev.sh prefix-cache-e2e` on `qwen3.5-4b-paro`: all 34 checks
  pass, as they do on the commit this branch starts from.
- `scripts/dev.sh hybrid-cache-correctness`: all 12 checks pass, and every
  restore matches bitwise.
- The diagnostics of that e2e run before and after this change match, with
  ids and timings masked. Two kinds of line follow the clock or the
  machine's free memory, so the comparison leaves them out: the periodic
  memory samples (87 before, 77 after) and the budget re-measures, which run
  at most once every 15 seconds (10 and 9). Of the remaining 1,840 lines on
  each side, the 281 written from other tasks (the memory observer's samples
  and the SSD writer's events) match as a set. The other 1,559 match in
  order, except one speculative-prefill `not-idle` skip that trades places
  with the next request's first line; the two come from different tasks.
- The bounded-cache parity run did not reach its checks. It stopped itself
  7 seconds in, at critical memory pressure while loading `qwen3.8-27b`: the
  machine had 19 GiB available at the start, and the archived pass peaked
  at a 23.5 GiB footprint. It needs a rerun with more memory free before
  this is accepted.
