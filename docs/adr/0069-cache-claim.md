# ADR-0069: The Cache Claim concludes a request's hold on the prefix cache exactly once, inside its GPU lease

- Status: Accepted (as built through #554; the real-model gate's memory rule
  was not met as written, see its result below)
- Date: 2026-09-22
- Amended by: ADR-0081 (2026-10-01: the request's GPU lease is now its **LLM Gate**
  turn; the claim still concludes inside it)
- Relates to: ADR-0064 (the Leaf Lease becomes one part of a claim; its
  pin-table age-out exemption becomes moot), ADR-0019 (the Restore Pin's
  age-out backstop is retired), ADR-0016 (claim steps run inside the Model
  Session; its ban on forced lease release from `completeRequest` becomes
  structural), ADR-0015 (the GPU lease the claim concludes inside), ADR-0033
  (the six phases are unchanged; the claim performs ownership steps only),
  ADR-0018 (the Active-Inference Reserve's lanes), ADR-0009 (Speculative
  Canonical Prefill stays copy-only), ADR-0022 (not pursued; this design
  assumes one generation at a time)

## Context

A keyed request holds three things in the prefix cache: a lane in the
**Active-Inference Reserve**, its **Restore Pins**, and, when it takes the
leaf by **Leaf Handoff**, a **Leaf Lease**. Snapshot Resolution adds the lane
and pins the path, and the check-out takes the lease. Twelve places let them
go: eight exit paths that each rewind the leaf, and four callers that release
the pins and the lane through `completeRequest`. The check-out decides between
handoff and copy and returns a bare copy reason.

The release sits outside the GPU lease on the success path. The drive
finishes the stream before it releases, and delivery does not wait for the
drive on a completed request. The next request can then resolve while the
previous one still holds its pins and lane. The reserve sizes itself by lane
count, so an overlapping lane holds back up to about 22 GB on the 27B at 61k
tokens, and a lane that leaks is never given back. The pin table's age-out at
eight requests is the only backstop, it cannot end a lease (ADR-0064), and it
does nothing for the lane.

The check-out also reports every lease refusal as `pendingFullPayload`,
including a lease refused because the leaf is already leased.

## Decision

1. **One claim per request.** Snapshot Resolution opens a **Cache Claim**
   for every keyed request, a miss included. A miss holds only its lane. The
   manager keeps its pin and lane tables, but only a claim releases their
   entries.
2. **The claim is the check-out.** It decides handoff or copy, including the
   Pending-Payload Wait, and returns a typed outcome: handoff, or copy with
   its reason, its true refusal and what the wait cost. Copy restore, suffix
   prefill and decode stay in plan application and the drive (ADR-0033). The
   claim performs ownership steps only, at quiescent points inside the Model
   Session.
3. **A claim that took the leaf ends only by check-in or Leaf Rewind.** The
   leaf is back before the pins and the lane are let go. Capture and admission
   of a request that did not take the leaf remain leaf production, outside the
   claim. So do the Backing Leaf release and Prefix-View persistence.
4. **Exactly one conclusion, inside the GPU lease.** Delivery awaits the
   drive on a completed request, on the HTTP path and the agent-chat path
   alike. The conclusion does not read `Task.isCancelled`: stream termination
   cancels the drive task on a normal finish too. It is shielded from
   cancellation the way today's rewind is.
5. **One owner at a time.** `start` owns the claim until it hands the claim to
   the drive. A **Speculative Canonical Prefill** pass owns its own claim,
   which can never take a leaf. Each owner concludes through one scoped
   conclusion. The boundary route's rewind and the executors' check-in stay
   as explicit steps on the claim.
6. **A runtime tripwire, not an age-out.** A claim dropped without concluding,
   or concluded twice, is a hard error. Debug builds trap. Release builds log,
   emit telemetry, and let go of the pins and the lane on the MainActor. A
   lease still held stays held, because no exact return is possible, and the
   report names it. The pin table's age-out is removed.
7. **Check-in comes before the extension payload is extracted.** Check-in
   frees the rewind backup before the capture allocates the payload, so on an
   SSD extension turn the check-in peak holds two copies of the recurrent state
   instead of three. Check-in still precedes admission (#498).
8. **Telemetry wire strings stay byte-identical.** One new detail field,
   `copyRefusal`, sits beside the unchanged `copyReason`.

## Considered options

- **Move-only (`~Copyable`) claim handles.** A scratch compile with the app's
  flags rejected them. A captured noncopyable value cannot be consumed inside
  the Model Session closure, and a noncopyable value cannot be its
  `nonSendable:` payload. The runtime tripwire enforces what the type system
  cannot.
- **A conclusion each caller remembers to call** (a `defer` in `start`, a
  call at the drive's tail), or **one handle type per lifecycle state**. Both
  were prototyped against owner scopes. With the first, a forgotten
  conclusion is caught only at runtime. The second makes ten kinds of misuse
  fail to compile, but adds a dozen names and makes the leaf store carry the
  claim's state. Owner scopes conclude on exit, so the common callers carry
  no conclusion code. A copy-only claim type and a release token that only
  the claim can mint keep the most damaging misuses at compile time.
- **Keep the age-out and add the tripwire beside it.** The age-out can end a
  pin while its request still runs and never frees a lane. Once every request
  holds exactly one claim, a leak is a bug to trap, not a quota to trim.
- **Let the claim own capture and admission too.** That would put leaf
  production, and the Cache Account fold, behind the same interface. They
  change for different reasons and are reviewed separately.
- **Release after delivery, as today.** This keeps the overlap that costs a
  reserve lane. Ending the claim inside the lease adds only the drive's
  post-stream tail to the request's own GPU time.
- **Design for batch lanes.** ADR-0022 was reverted. Lanes would need
  several claims per lease and shared leaves, which is a different design.

## Consequences

- The Active-Inference Reserve reads one lane for back-to-back requests, and
  a leaked lane is a trapped bug instead of a permanent loss of headroom.
- A completed request keeps the GPU lease until its drive has concluded the
  claim, not only until its stream ends. The client's last chunk arrives when
  it does today, because the leaf is already stored before the stream
  finishes. The next request waits for the drive's short tail, which today
  runs after the lease is released.
- A copy restore reports why the lease was refused, so `alreadyLeased` no
  longer reads as a payload wait.
- Tests of the claim run on the real manager and tree. The request's exits
  are tested through the Server Completion fixture on the toy Model Session.

## As built (2026-09-22, #554)

The branch follows the PRD's nine steps, each commit green.

**The claim's shape.** `CacheClaim` is a reference type with three owner
scopes. `start` opens it with `withRequestClaim`, which returns a hand-over
token beside its result. The drive redeems the token with `withClaim`. A
speculative pass opens a `CopyOnlyClaim`, whose type has no check-out step.
Each scope concludes when it exits, so no caller carries conclusion code, and
the manager's `release` takes a token only a claim can mint. Snapshot
Resolution takes the claim it opens (`resolve(..., for:)`): a miss adds the
lane, a hit adds the lane and pins the path. `LeafCheckout` is deleted, and
its attempt tests moved to `CacheClaimTests`.

**The conclusion.** It runs in a detached task the owner awaits, so
cancellation cannot cut it short. It enters the Model Session only while a
lease is still held, to rewind it, then releases the pins and the lane in one
MainActor hop and records one `releasingRequest` phase. Because the session
lock is not reentrant, a debug assertion catches a request scope opened
inside a Model Session; both session providers mark the scope for it.

**The tripwire.** It reports six violations: a claim dropped unconcluded, a
hand-over redeemed twice, a claim concluded twice, a second check-out, a step
taken while the claim is handed over but not yet redeemed, and a step after
the conclusion. The `cacheClaimTripwire` event names the violation, the owner,
what it released and any lease still held with its offset and bytes. Only a
claim dropped unconcluded, which nothing else will ever conclude, has its
pins and lane released by the tripwire. Every other violation leaves that to
the claim's own conclusion, so the leaf is still back first. Tests run the
tripwire in a reporting mode that does not trap. It cannot tell owners apart
once the drive holds the claim: a reference that escapes `start` and takes a
step after the hand-over is redeemed goes unreported. Catching that needs a
handle per owner, which the considered options rejected.

**The exits.** The eight exit-path rewinds and the four release calls are
gone. The drive's catch arms no longer rewind; the drive's conclusion does,
when a lease is still held. A check-in that the tree refuses, or that runs
after cancellation, rewinds in the same step and says why. The exit matrix
(`ServerCompletionExitMatrixTests`) covers every exit the PRD lists, and two
of them end differently from the PRD's wording. Decode has no failure exit
(the stream loop never throws), so the failure case is a suffix prefill that
throws after the handoff. A think-stripping turn with a Prefix-View boundary
does not rewind: the boundary route checks the leaf in as its views' Backing
Leaf and stores the canonical leaf from the boundary (ADR-0068). A boundary
turn without one still takes the explicit rewind step.

**Delivery.** HTTP completion delivery awaits the drive on `.completed`, and
cancels and awaits it on every other outcome, an undelivered finish included.
The internal agent-chat route does the same: it waits for a completed drive
before ending its stream, and cancels and waits for a failed one before
reporting the failure.

**The parity bench.** The bounded-cache parity bench takes and rewinds its
leaf through `CheckedOutLeaf`, the same code the claim's check-out and Leaf
Rewind run, compaction included, but outside any claim: it checks the tree and
the move, not the claim's lifecycle.

**`copyRefusal`.** It is on the `lookup` and `leafStore` events, and as
`restoreCopyRefusal` on the `restored` phase of `requestMemory`. Every lease
refusal still takes the Pending-Payload Wait's path, as when they all read
`pendingFullPayload`, so the wait's behaviour is unchanged.

### The memory items

The model-free evidence (`CacheClaimMemoryEvidenceTests`, run alone) measured
the MLX peak around each step on synthetic caches whose recurrent state is
4 MiB:

| Item | Before | After |
| --- | ---: | ---: |
| 1. Peak of the check-in step on an SSD extension turn | 6,324,240 B | 2,129,936 B |
| 1. Peak of a refused check-in | | 0 B, no payload extracted |
| 2. Compaction transient, 8 layers | 9,437,216 B | 1,179,654 B |
| 3. Held across a check-out | | 4,194,312 B, the recurrent backup; 0 after rewind |
| 5. Peak across an SSD hit's restore, 8 MiB leaf | 8,388,624 B (copy) | 4,194,312 B (handoff) |

Item 1 saves exactly one recurrent state (4,194,304 B). Item 2's transient
is one layer's replacement (1,179,648 B). Item 3 also stops copying the
leaf's token path, about 0.5 MB at 61k tokens. Item 5 saves the second
resident leaf, about 4.1 GB at 61k tokens on the 27B; the SSD restore
harness shows the hit making no restore call at all.

Item 6 kept today's rule. The measurement and the rule were registered before
any number was read
([`benchmarks/allocation-profile/2026-09-22/`](../../benchmarks/allocation-profile/2026-09-22/README.md)).
Of 45 compactable leaves with a next turn, 4 (0.09) grew because compaction
took capacity the next turn would have used, under the registered 0.20. The
threshold, the one 256-row step of spare rows and both compacting exits stay.
A toy decode checks the kept rule through the Server Completion module.

### What the tests do not cover yet

- No exit-matrix case reaches a boundary turn without a Prefix-View boundary,
  the one think-stripping case that still takes the explicit rewind step.
- The body refusals `emptyBody` and `quantizedLayer` and the manager's
  `compressing` and `bodyReplaced` have no claim-level test.
- A leaf loaded from SSD has no other owner by construction, so the refused
  case the PRD names is covered only by the resident pending-payload test.
- The SSD memory checks (backing addresses, the kept SSD ref, the peak) run
  at the claim's interface, where they are observable. The Server Completion
  SSD harness checks only that the hit makes no restore call.

### The real-model gate (2026-09-23)

Run under the plan the owner approved, with one change the owner approved
after the first attempt: every campaign ran on its own new SSD cache
directory, since the shared cache still held #553's run of the same prompts
and would have warm-started every campaign. The plans, the pass rule and the
results are in
[`benchmarks/allocation-profile/2026-09-22-gate/`](../../benchmarks/allocation-profile/2026-09-22-gate/README.md),
next to #553's baseline. No resource stop fired, and the lowest available
memory was 9.8 GiB.

- Restore modes, copy reasons and leaf sources matched main step for step,
  every sample of the reserve reported one lane, the lease count was 0 after
  every request, parity passed its 18 checks and the e2e run its 22.
- The memory rule was not met as written. 22 of the PR's 69 readings sit above
  both main runs, 9 of them within 10 MB. The largest are 0.21 to 0.23 GiB of
  peak footprint in the PR's second profile run at three steps where its
  first run was at or below main; main's own runs differ by up to 0.73 GiB.
- After a cancelled generation, the PR settles about 1.1 GiB lower than main
  (17.37 and 17.60 GiB against 18.54 and 18.73), because the conclusion clears
  MLX's buffer cache after the rewind's compaction.
- The PR's second probe completed four of its seven requests: it was stopped
  from outside, then stopped again when the diagnostics sink rotated. No SSD
  hit occurred in any campaign, so the SSD-loaded handoff rests on the
  model-free evidence above.

Whether the memory readings above main are noise is the owner's call.

## Amendment 2026-10-03: a restore that yields no cache runs the turn cold

A planned restore can end with no cache. The check-out answers copy, or finds
nothing to check out, and the copy fails: `HybridCacheSnapshot.restore` throws
when it cannot rebuild a layer (corrupt or truncated persisted data, or a
Prefix-View Checkpoint whose Backing Leaf is missing or no longer fits it), and
the restore verb treats that as a cache miss. Until now only the cache was
the miss. Plan application still sliced the input to the suffix, kept the
restore offset as the execution base along with the warm checkpoint filter,
and prefilled the suffix into a new, empty cache. The model saw the suffix
alone, at positions from zero, and answered without the conversation.
Checkpoints captured on the way were labelled at the restore offset plus their
position in the suffix. Admission checks only that an offset lies inside the
path, so those snapshots could be admitted and restored by later requests. The
fault was found by reading the code, not in a recording.

1. **A restore that yields no cache re-plans the turn cold.** The Prefill
   Planner's cold plan replaces the warm one. It is the rule `plan` already
   answers with when nothing is restorable (`PrefillPlanner.coldPlan`), not a
   second one. The whole prompt prefills from zero with execution base 0, the
   cold checkpoint filter and transient boundaries apply, and an image-bearing
   Cache Key Space takes the cold image route anchored at zero. The turn's
   maximum advance is re-priced from zero, since its cache grows from empty.
2. **The Speculation Plan is kept.** It was decided before the check-out with
   a restored prefix, so it is DFlash2 or nothing, and DFlash2 serves a cold
   turn as it serves a warm one. Asking again with nothing restored could
   engage MTP, whose single unchunked prefill captures no checkpoints and no
   boundary snapshots, on the one turn that has to recapture what failed to
   restore. A fallback turn engages no arm its plan did not, and its
   `restoring` phase has already reported that plan.
3. **The wire strings stay.** `restoreMode` still reads `failedCopy`
   (decision 8). One new detail field says the turn fell back:
   `restoreFallback=cold` on the `lookup` event and on the `restored` phase of
   `requestMemory`, written only on such a turn, so every other line stays
   byte-identical. A fallback turn reports `skippedPrefillTokens=0`, and its
   handle reports no cached tokens.
4. **The snapshot that failed is dropped.** Left in the tree, it would make
   every request that resolves to it pay the same failed restore and run cold,
   and checkpoint planning would keep counting it as stored, so a lost system
   checkpoint would never be captured again. The fallback turn drops it on the
   MainActor (`PrefixCacheManager.dropUnrestorableSnapshot`), but only while
   the node at its path still holds that very body and no Leaf Lease holds it.
   The RAM body always goes. A `RestoreError` means a layer's class or
   metaState cannot be rebuilt, and the SSD copy holds the same bytes, so its
   backing is deleted before its ref is discarded, and a chain-prefix point
   the body was composed through is cleared. Any other failure, such as a view
   whose Backing Leaf no longer fits it, keeps the SSD copy for the next hit to
   hydrate. After a drop the turn plans its checkpoints again against the
   settled tree before it takes the cold plan, so a checkpoint the drop took
   is captured anew on the same turn. The turn logs
   `skip stage=restore reason=unrestorable-snapshot` with the offset, the
   checkpoint type, what went (`drop=body`, `bodyAndSSDCopy`, `notFound` or
   `leased`) and the error. The Speculative Canonical Prefill and the Leaf
   Store's boundary route restore stored snapshots too; they still only log a
   failure, and the next foreground request that resolves to the snapshot
   drops it.

`ServerCompletionRestoreFallbackTests` runs keyed requests through the Server
Completion fixture with the toy session's `restore` made to throw. A text-only
turn restored by copy and a restore planned below a new image each check that
the turn fed the whole prompt from position zero into a new cache, that every
checkpoint it captured carries the offset its cache held, and that every body
the cache holds afterwards reads back the path it is stored under; the text
case also checks that the next turn restores what the fallback turn stored. A
thinking-off direct-leaf turn with a resident MTP drafter that traps if
engaged checks that the fallback keeps its Speculation Plan. A failed restore
of a system checkpoint checks that the body is gone, that the same turn
captures a new one and that the next request restores it; with an SSD tier
behind it, a `RestoreError` also removes the old copy from the manifest and
any other error keeps it.
