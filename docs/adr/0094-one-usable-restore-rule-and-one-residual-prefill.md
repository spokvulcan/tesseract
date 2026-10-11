# ADR-0094: One usable-restore rule decides every restore; one Residual Prefill restores a prefix and extends it

- Status: Accepted (built; loaded-model gates in As built below)
- Date: 2026-10-11
- Builds: ADR-0087 decision 7's named usable-restore rule ("so that a later
  caller which restores a prefix and extends it — the Leaf Store's boundary
  route, the Speculative Canonical Prefill — can reuse it"), ahead of the rest
  of ADR-0087
- Amends: ADR-0069's 2026-10-03 amendment (the failed-restore policy reaches
  the boundary route and the speculative pass), ADR-0078 (closes its
  follow-up 4: the speculative pass enters the Model Session)
- Relates to: ADR-0016 (the session's existing `restore` and `prefill`
  verbs), ADR-0068 (request-local views and their Backing Leaf), ADR-0007
  (the Position Anchor), ADR-0009 (the Speculative Canonical Prefill's
  preemption contract), ADR-0033 (plan application keeps its shape switch)

## Context

Whether a snapshot can seed a prefill was spelled five times: the Prefill
Planner's restore decision, the Leaf Admission Builder (both boundary modes),
the Leaf Store's boundary route after a view's resolve, the Speculative
Canonical Prefill (whose comment said its guards "mirror the canonical-leaf
arm") and salvage-on-cancel. The rule is the same each time: the offset
covers some of the target and not all of it, does not split an image run, and
— for a re-prefill of text — sits at or past the Minimum Warm Offset, which
makes its Position Anchor defined. Two producers then each restored and
extended the prefix their own way. The boundary route restored a view through
its Backing Leaf, reserved the stored path and prefilled the residual through
`session.prefill`; the speculative pass did the same restore in a raw
`container.perform` and ran its own chunk loop on `context.model`, outside the
recorded session verbs (ADR-0078 follow-up 4). Request-local view matching was
written twice. And only the keyed turn dropped a snapshot whose restore threw:
the boundary route and the speculative pass logged (`prefill-threw`,
`restore-failed`) and left it for the next request to fail on again — the gap
ADR-0069's amendment admits.

## Decision

1. **One named rule.** `PrefillPlanner.usableRestore(offset:target:keySpace:residual:)`
   returns the restore point (the offset and its anchor delta, which seeds
   the Position Anchor through its one `initialState(seedsPositionAnchor:)`;
   the keyed plan carries the point whole) or why it is unusable (`coversNothing`, `coversWholeTarget`, `insideImagePrefix`,
   `splitsImageRun`, in that order). The residual says what the prefill past
   it forwards: `mayCarryImages` for the keyed turn, which continues through
   an Image Span (ADR-0007 phase 2), `textOnly` for every re-prefill. All five
   callers read it. The builder maps its reasons to its existing skip
   vocabulary; salvage asks it for the key path plus the one token a resend or
   the next turn adds, which is exactly its old `≤ keyPathCount` guard.
2. **One Residual Prefill.** `ResidualPrefill` (`Features/Server/ResidualPrefill.swift`)
   does everything after the rule, for the boundary route and the speculative
   pass: `restore` brings a snapshot back through its Backing Leaf and
   reserves the whole path's rows; `prefill` runs a range of the path as text
   through the session's `prefill` verb, at the processor's token rank and
   under the caller's eval policy, from the base's seeded Position Anchor or
   the previous range's state;
   `matchingView` is the one request-local view rule. The producers hand the
   resulting cache to their Leaf Admission; ADR-0078 kept re-prefill out of
   the admission, and it stays out. Plan application keeps its shape switch
   (ADR-0033) and reads the same rule.
3. **One failed-restore policy.** `ResidualPrefill.dropUnrestorable` drops the
   snapshot through the manager in one MainActor hop and logs the one
   `restore/unrestorable-snapshot` line; for the keyed turn the manager's drop
   also returns the turn's checkpoint plan made again in that hop (ADR-0087's
   2026-10-11 amendment on the manager's call order). The keyed turn, the boundary
   route and the speculative pass all use it; a failed restore returns a typed
   failure the producer words as before.
4. **The speculative pass is a Model Session client.** It takes a session
   provider instead of the container, and enters one session for the resolve,
   one for the restore, one per chunk and one for the capture, as many entries
   as before.

## Declared behaviour changes

Each is pinned by a test; nothing else changes.

- **A failed restore in the boundary route or the speculative pass drops the
  snapshot**, as the keyed turn's does (`ServerCompletionKeyedSequencingTests
  .aBoundaryThatFailsToRestoreIsDroppedLikeTheKeyedTurns`,
  `SpeculativePrefillPreemptionTests.aBoundaryThatFailsToRestoreIsDropped`).
  The diagnostics on that path gain one line, before the producer's own:
  `skip stage=restore reason=unrestorable-snapshot level=warning offset=…
  checkpointType=… drop=… error=…`, the keyed turn's line. The producer's
  line keeps its stage, reason, level and `error` text
  (`<mode>LeafStore prefill-threw`, `speculativePrefill restore-failed`), and
  the `leafStore` report still reads `skip=prefill-threw`. A request-local
  view reports `drop=notFound`: there is nothing in the tree to drop.
- **The speculative pass's chunks run through `session.prefill`, each
  evaluated checked** (`.checkedSynchronous`; the pass forwards one chunk
  per call, so it gives up no pipelining). A chunk's MLX failure now throws
  inside the checked evaluation and logs `skip stage=speculativePrefill
  reason=prefill-threw level=warning error=…` instead of ending the process
  (`SpeculativePrefillPreemptionTests.aChunkThatThrowsEndsThePassWithNothingAdmitted`).
  Each chunk is one `prefill` verb, which ends, as every `PrefillExecutor.run`
  does, with a checked flush and `Memory.clearCache()`, so the pool is
  returned between chunks (`eachChunkIsOneCheckedPrefill` counts one checked
  verb per chunk). The pass's verbs are recorded on the toy session. The
  boundary route keeps the `.pipelined` evaluation it had: one call over
  the whole residual, whose closing flush is checked.
- Not changed: when a preemption lands. A chunk that passed the loop's check
  runs to its end even if the cancel lands while it waits for its session,
  as the pass's chunks did on `container.perform`; it runs in a detached task
  (the claim's conclusion's way), so the pass's cancellation does not reach
  the prefill's own per-chunk check, and the next check stops the pass
  (`aPreemptionWhileAChunkWaitsForItsSessionStillRunsThatChunk`). An earlier
  build of this ADR broke there and settled one chunk short; review caught
  it before any gate could, since no e2e request preempts at that moment.
- Not observable: the `boundary-splits-image-run` skips of the boundary route
  and the speculative pass are gone. A text-only residual sits at or past the
  Minimum Warm Offset, where the anchor is always defined, so neither could
  fire.

## Considered options

- **Let Leaf Admission re-prefill.** ADR-0078 rejected it: restore, prefill
  and preemption change for other reasons than storing a leaf does.
- **One call that restores and prefills the whole residual.** The speculative
  pass must stop between chunks and hand the Model Session to a preempting
  request within one chunk (ADR-0009), so restore and prefill stay separate
  calls; the boundary route makes both in one session.
- **Salvage through the Residual Prefill.** Salvage captures the cache it
  already holds and restores nothing; it reads the rule only.
- **Fall back to a cold re-prefill after a failed boundary restore.** A leaf
  store is best-effort; dropping the snapshot and skipping keeps the next
  request from paying the same failure, which was the gap.

## Consequences

- `PrefillPlannerTests.theUsableRestoreRule` is the rule's table (text and
  image key spaces, both residuals, every reason). The builder, planner,
  salvage and speculative suites keep their outcome rows unchanged, and
  salvage's pure gate takes a key space.
- Adding a restore-and-extend producer is a `ResidualPrefill.Base` and two
  calls; it inherits the failed-restore policy.

## As built (2026-10-11)

Measured on the machine of ADR-0093's As built (MacBook Pro, Apple M3 Max,
48 GB, macOS 27.0.1), Debug configuration, with ADR-0093's diagnostics method.
The branch's final commit (c25f74a4) carries this ADR, ADR-0069's Turn Restore
amendment and its two declared fixes, and ADR-0087's amendment on the
manager's call order; its gates against `main` (fa438b35):

- `prefix-cache-e2e` on `qwen3.5-4b-paro`: 38 of 38, the check lines and the
  32 decode lines identical; 0 of 1,959 in-order diagnostics lines differ.
  The same held after this ADR alone, and after the Turn Restore with both
  fixes.
- The same e2e on `qwen3.8-27b` with its DFlash2 draft: 38 of 38, the 32
  acceptance lines identical, 0 of 2,020 in-order lines differ. With
  `TESSERACT_E2E_KV_SCHEME=turbo8v4`: 37 of 38 with `main`'s own
  `requestP3` failure on both sides, the 32 lines identical, 0 of 2,018.
- `hybrid-cache-correctness`: 12 of 12, every restore bitwise, the summary
  identical.
- The Completion Trace Log records each run appended (33 per run) carry the
  same restored offsets, hit tokens and rewinds on both sides: no e2e turn
  falls back or plans a hit cold, so neither declared fix shows there, and no
  e2e restore fails, so neither does this ADR's drop.
- `scripts/dflash2-bench.sh`'s run (travel fixture, 192 tokens, width 8,
  `--bench-check`) from each commit's build: `dflash2-compare.py
  --require-identity` reports both arms' streams and round accounting
  identical (145 of 321 drafts accepted in 46 rounds), each matching AR's
  full stream; 22.9 against 22.8 tok/s AR and 73.3 against 73.3 DFlash2, one
  run each. The bench drives the vendor iterators directly, so it guards
  that path; the app's DFlash2 path is the 27B e2e's acceptance lines.
- The other-task lines differ only by the Speculative Canonical Prefill's
  scheduling race (9 or 10 lines per comparison), plus, on 27B, whether the
  last request's one-second-later memory sample lands before the run ends. A
  fresh run of `main` under the later load logged `not-idle` 20 times of
  31, against 25 in the two earlier `main` runs and 20 to 23 on the branch,
  and compares with the final commit both ways, so the race follows the
  machine's load, not the change.

The whole target passes on the final commit (3,653 tests, 7 skipped), as does
`scripts/check.sh`. `ServerCompletion.swift` went from 2,755 lines on `main`
to 2,617; `SpeculativePrefill.swift` from 534 to 520; `ResidualPrefill.swift`
is 193 and `TurnRestore.swift` 98. `PrefixCacheManager.swift` grew from 3,278
to 3,387 with the planned resolution, the re-planning drop and the test
seam's forwarders.
