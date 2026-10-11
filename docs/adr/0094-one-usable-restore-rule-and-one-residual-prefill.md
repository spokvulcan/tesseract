# ADR-0094: One usable-restore rule decides every restore; one Residual Prefill restores a prefix and extends it

- Status: Accepted (built; loaded-model gates in ADR-0093's As built and below)
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
   returns the restore point (the offset and its anchor delta) or why it is
   unusable (`coversNothing`, `coversWholeTarget`, `insideImagePrefix`,
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
   through the session's `prefill` verb, at the processor's token rank, from
   the base's seeded Position Anchor or the previous range's state;
   `matchingView` is the one request-local view rule. The producers hand the
   resulting cache to their Leaf Admission; ADR-0078 kept re-prefill out of
   the admission, and it stays out. Plan application keeps its shape switch
   (ADR-0033) and reads the same rule.
3. **One failed-restore policy.** `ResidualPrefill.dropUnrestorable` drops the
   snapshot through the manager, runs a follow-up against the settled tree in
   the same MainActor hop (the keyed turn's checkpoint re-plan), and logs the
   one `restore/unrestorable-snapshot` line. The keyed turn, the boundary
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
- **The speculative pass's chunks run through `session.prefill`.** A chunk's
  MLX failure now throws and logs `skip stage=speculativePrefill
  reason=prefill-threw level=warning error=…` instead of ending the process;
  each chunk's prefill ends, as every `PrefillExecutor.run` does, with a
  checked evaluation and `Memory.clearCache()`, so the pool is returned
  between chunks; and the pass's verbs are recorded on the toy session.
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
