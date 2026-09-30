# ADR-0079: Speculative decoding lives in one Speculation module; both arms read one Speculation Plan

- Status: Accepted (built; loaded-model gates passed, see As built)
- Date: 2026-09-30
- Amends: ADR-0016 (the Model Session sheds its per-drafter members for one
  speculation fact and one plan-taking verb; the amendment's "revisit inside
  the seam only if verbs cluster in practice" has been met)
- Relates to: ADR-0056 (MTP cold path and its leaf-mode amendment), ADR-0057
  and ADR-0061 (DFlash2), ADR-0059 (DFlash2 on the keyed path, the prefill
  split, the KV-quant gate), ADR-0053 (penalties through the app processor),
  ADR-0067 (a Rotated Ternary Checkpoint refuses the draft), ADR-0078 (the
  Leaf Admission's move-or-copy rule for an active MTP arm)

## Context

The app runs two speculative drafters: the MTP head that ships inside a
Qwen3.5-family checkpoint, and the separate DFlash2 draft beside Qwen3.8-27B.
No module owns them. What a drafter is, whether it is loaded and whether one
request uses it are answered in nine files:

- **Residency** sits in `LLMActor`: two boxed drafter fields (`LLMActor.swift:59-70`),
  two loaders that detect, pair, refuse and load (`807-915`), the same pair of
  calls in both load branches (`176-179`, `207-210`), the unload teardown and
  its retention probes (`438-458`, `492-493`), the Models page query
  (`375-380`), and the provider wiring (`305-308`).
- **The Model Session port** carries four members per drafter family: the
  drafter property, the iterator verb, an error type and a default that throws
  it (`ModelSession.swift:54-63`, `137-164`, `180-186`, `196-198`, `247-262`).
  The production adapter holds both drafters (`305-312`), and the provider
  boxes both again (`515-536`).
- **Engagement** is four predicates on two support enums, read by two arms that
  disagree. The server refuses DFlash2 when the request quantizes its KV
  (`DFlash2Support.shouldEngage`); the Raw Generation Start engages anyway and
  clears `kvBits` (`shouldEngageRawArm`, `rawArmParameters`), and the comment
  on the second leaves "reconciling the two" to a Speculation Plan that did
  not exist (`DFlash2Support.swift:196-205`). MTP's predicate
  (`MTPDrafterSupport.swift:54-90`) is read only by the server's cold case.
- **The server** decides DFlash2 before the restore switch, adds its block to
  the maximum advance, returns early into a separate MTP body from the cold
  case, splits the DFlash2 prefill with a free function, and badges the arm
  from the iterator enum (`ServerCompletion.swift:194-250`, `1518-1563`,
  `1651-1683`, `1749-1760`, `1864-1898`, `1982-1987`, `2368-2507`). The two
  arms' iterator factories each spell the ADR-0053 penalty discipline
  (`ModelSession.swift:437-472`, `DFlash2Support.swift:207-249`).
- **Elsewhere**, the token loop names the algorithm by type-checking the
  iterator (`TokenGenerationLoop.swift:246-247`), and the toy session can only
  carry a presence-only MTP drafter that traps if used, so no test runs a
  speculative arm.

Each mode touched 16-28 files when it landed (73adfb44, 39be78dd, 64f12448).
The Models page read the draft's residency off the loaded model's id until
#586, because there was no residency fact to read.

## Decision

1. **One Speculation module owns speculative decoding.** It detects, pairs,
   refuses and loads the drafters a model load may attach, releases them at
   unload, and answers whether a drafter is resident. `LLMActor` holds one
   Speculation value per load and hands it to the session provider. The
   support enums keep the per-family facts the module uses (detection,
   class pairing, geometry, loading, block sizes); the DFlash2 bench also
   uses them.
2. **One Speculation Plan per request, decided once.** The module maps a
   request's facts to a plan or to nothing: which arm runs, the advance
   allowance its rounds need, whether its iterator prefills the whole prompt,
   and where the app's prefill hands over to the iterator. The facts are
   whether the input is text-only, the KV quantization, the temperature, the
   prompt length, whether the turn restores a cached prefix, and which leaf it
   will store (none on the Raw Generation Start). The Server Completion and
   the Raw Generation Start both ask with those facts; neither holds a rule.
3. **The engagement table is today's rules in one place.**
   - Speculation needs text-only input over unquantized KV, on both arms. The
     raw arm used to clear `kvBits` instead of refusing. Nothing in the
     product sets `kvBits` (it is nil since #252), so no request changes arm.
   - DFlash2, when resident, is preferred and engages on every such request,
     warm or cold, whatever leaf it stores (ADR-0059). The app prefills and
     captures up to the split, and the iterator capture-prefills the tail.
   - MTP engages only at temperature 0, on a turn that restores nothing and
     stores a direct leaf, when the single-shot prefill fits the scratch
     budget (ADR-0056 and its amendment). Its iterator prefills the whole
     prompt into an empty cache. The Raw Generation Start stores no leaf, so
     MTP stays off there, as it is today.
   - The allowance is the arm's block size, what `CacheClaim.maximumAdvance`
     adds for the rounds' lookahead.
4. **The Model Session carries one speculation fact and one verb.** The
   session exposes the Speculation it was entered with, and builds a
   speculative iterator from a plan. The plan builds the iterator, so the
   ADR-0053 penalty discipline is written once. The per-family properties,
   verbs, error types and throwing defaults go.
5. **The toy session is the second adapter.** Its provider takes a
   Speculation. A scripted DFlash2 drafter over the toy model, which gains the
   DFlash2 target verbs over its own forward, runs the real vendor iterator,
   so the DFlash2 arm, its split and its leaf are tested on both arms without
   a downloaded model.
6. **Behaviour is preserved.** The same requests engage the same arm, and
   the diagnostics fields, memory phases and log lines keep their names and
   values. The unload still releases MTP before DFlash2, with a memory phase
   after each.

## Considered options

- **Pass the Speculation beside the provider rather than on the session.**
  The session would lose its speculation member, but every consumer would
  have to be handed the value separately, and the provider is the one place
  where "neither consumer can silently drop an arm" holds today.
- **Keep engagement in the support enums and only unify residency.** That
  leaves the kvBits disagreement and the two copies of the iterator factory,
  and the next mode still needs a predicate for each arm.
- **Fold the MTP arm into the keyed path with a split at zero.** It would
  delete the separate whole-prompt body in the Server Completion. It would
  also change what an MTP turn logs (it would gain the lookup event it skips
  today), and neither the toy nor any downloaded checkpoint here can run the
  MTP iterator to check it. It fits better when the Prefill Plan takes back
  its decisions (architecture review candidate 04).
- **Engage MTP on the Raw Generation Start.** The "Greedy (Speculative)"
  preset says it enables MTP, but agent chat has never engaged it. With the
  table in one place this is a one-rule change, but it changes behaviour and
  needs an MTP checkpoint to verify, so it is a follow-up.
- **Move the Leaf Admission's MTP copy rule here.** It is a capture-side fact
  about the arm that ran, recorded on the generation, and ADR-0078 gave it
  one home already.

## Consequences

- `LLMActor` loses its drafter fields and loaders. `ModelSession` loses its
  two drafter properties and two iterator verbs, with their error types and
  throwing defaults, and gains two members. `DFlash2Support` and
  `MTPDrafterSupport` lose their engagement predicates and the iterator
  factory.
- The Server Completion reads one plan: its allowance for the maximum
  advance, whether it prefills the whole prompt (the cold MTP body), and its
  split (the DFlash2 branch of the keyed prefill). The Raw Generation Start
  has one speculative branch for either arm.
- The token loop takes the arm from its caller for its speculation log line.
- Tests follow the module. The engagement tests in `DFlash2SupportTests` and
  `MTPDrafterSupportTests` move to the plan's table, and the presence-only
  MTP drafter rides a Speculation value. New suites run the DFlash2 arm
  through the Raw Generation Start and the Server Completion on the toy.
- Adding a drafter family means a residency loader, a table row and an
  iterator case in one module, plus a session verb only if its iterator needs
  something new from the model.

## As built (2026-09-30)

The branch landed in three commits after the proposal: the module with every
consumer moved onto it and the plan's table tests, the toy runs of the DFlash2
arm, and the glossary and docs.

How the shape came out:

- `Speculation` holds the drafters and answers `plan(for:)`,
  `isResident(_:)`, its memory facts, and the unload with its release probe.
  `SpeculationPlan` carries the arm, `advanceAllowance`,
  `prefillsWholePrompt`, `prefillSplit(checkpointOffsets:executionBaseOffset:promptTokens:)`
  and the one iterator factory. `SpeculativeDecodeIterator` starts the token
  loop for either arm.
- The MTP scratch profile became a load-time fact of the Speculation (priced
  from Model Identity when the drafters load), so neither arm passes it.
- The Server Completion's cold MTP body stays, now driven by the plan
  (`prefillsWholePrompt`). The prefill plan's restore answers
  `restoresPrefix`, so the keyed closure reads the plan without growing its
  branches.
- The unload still releases MTP before DFlash2 with a memory phase after
  each, inside `Speculation.unload()`.

Measured:

- `LLMActor.swift` 916 → 800 lines, `ModelSession.swift` 537 → 443,
  `DFlash2Support.swift` 250 → 150, `MTPDrafterSupport.swift` 177 → 141,
  `ServerCompletion.swift` 2,725 → 2,643; `Speculation.swift` is 477.
- The server, agent and prefix-cache test lists in `docs/testing.md`, plus the
  touched suites: 1,098 cases pass. Breaking the DFlash2 split (handing over at
  the execution base) fails the thinking-turn toy test and the plan's split
  rows.
- The whole unit target in one process fails only in unrelated suites that
  time out under that load; run alone they pass, except
  `MemoryBaselineTests.corpusLoads`, which reads the owner's memory corpus.

Loaded-model gates, run on 2026-09-30 on a MacBook Pro with an Apple M3 Max
and 48 GB, macOS 27.0:

- `scripts/dev.sh prefix-cache-e2e` on `qwen3.5-4b-paro`: all 34 checks pass.
  The load refuses the DFlash2 draft by geometry (64 target layers against
  32) before reading its weights, and the checkpoint ships no MTP head.
- `scripts/dev.sh hybrid-cache-correctness`: all 12 checks pass, every
  restore bitwise.
- The e2e runner on `qwen3.8-27b-paro`, speculation Automatic: all 34 checks
  pass with both drafters resident and DFlash2 engaged, warm restores
  included. The same run on main gives the same speculation log lines for
  every request (rounds, proposed, accepted and emitted tokens) and the same
  check values once timings are masked.

Not verified on a real model: MTP drafting. The only local checkpoint with a
head, the 27B PARO with a grafted bf16 head, traps in the fused RMSNorm
(`residual dtypes differ`) on its first drafted turn, on main as on this
branch, so MTP engagement is covered by the plan's table and the
presence-only drafter.

