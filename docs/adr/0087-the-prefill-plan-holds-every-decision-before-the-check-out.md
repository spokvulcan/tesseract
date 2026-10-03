# ADR-0087: The Prefill Plan holds every decision made before the Cache Claim check-out; plan application carries it out inline

- Status: Proposed (accepted once the change is built and the loaded-model
  gates pass, the bar ADR-0078 set)
- Date: 2026-10-03
- Amends: ADR-0033 (the phase map, and the premise behind keeping plan
  application inline; the inline decision itself stands), ADR-0079 (where the
  keyed path asks for its Speculation Plan, and when the MTP fold it deferred
  fits)
- Relates to: ADR-0069 and its 2026-10-03 amendment (the cold fallback after
  a failed restore, whose rule this ADR gives a home), ADR-0007 (Cache Key
  Space and Position Anchor; its rejection of app-side patch-count math
  holds), ADR-0014 (the vision guard prices the forwarded image), ADR-0016 (no
  phase verbs on the Model Session), ADR-0044 and ADR-0055 (a pure decision
  value an executor carries out), ADR-0056 and its 2026-08-18 amendment (the
  whole-prompt MTP arm forfeits the transient boundaries), ADR-0059 (the
  DFlash2 split), ADR-0064 and ADR-0069 (the Cache Claim decides handoff or
  copy), ADR-0068 (Transient Boundaries are request-local views), ADR-0070
  (one extractor reads the Keyed Request), ADR-0078 (the behaviour bar),
  ADR-0083 (where each route converts to the KV Scheme)

## Context

ADR-0033 kept plan application inline because "the real derivation content
of plan application already lives in `PrefillPlanner.plan`", and asked
future reviews not to re-propose extracting it. The inline decision was
right. The premise has stopped being true.

Since 2026-07-11 the decisions a keyed request makes before it touches MLX
have landed in `makeHTTPPrefixCacheGeneration` instead of the planner: the
speculation route (73adfb44, 39be78dd, 64f12448, 36347365), and the
preserve-thinking capture gate and the maximum advance (b1f0dca3; the advance
became a reserve input in 6cc03e1b), placed around a check-out that 86117457
moved into the Cache Claim. The failed-restore fix (#619) added one more: a
cold re-plan inside the restore arm. The function grew from 602 lines to 887.

For this ADR every decision in plan application was inventoried and checked
against the code. Today the Prefill Plan decides restore-or-cold, the suffix
checkpoint filter and transient survival. Plan application decides the rest,
from values that were all in scope when the plan was made:

- which of five shapes runs: a warm text remainder, a warm restore through a
  new image, a cold image prefix with a text tail, the MTP whole-prompt
  diversion, and cold text. ADR-0033 counted four, "distinguished mostly by
  which MLX handles they touch". In fact plan-time predicates choose them, and
  only their bodies touch MLX;
- where prefill starts, held as two values under four names: the cached-token
  count (`prefillBaseOffset`, and a copy the trace derives from the lookup
  reason) and the text-tail start (`executionBaseOffset`, handed to salvage
  as `restoreBaseOffset`). The two differ whenever the prefill forwards an
  image;
- the image span, where the Position Anchor is seeded, the preserve-thinking
  gate and the merge of transient boundaries into the capture map, the
  Speculation Plan and its split, the maximum advance, and the prefill eval
  policy.

The sampled history has five fixes to how a decision was made (eb615212,
89e2da4e, ddbbec38, d732376a, 9b6bc1f3). Each was a decision known before
execution that was fed a wrong input or worked out again by a consumer, and
none was in MLX execution. Four of the five sat in plan application; the
fifth sat in the planner. Which shape runs can only be tested by driving a
whole completion through `ServerCompletionFixture` and reading the recorded
verbs, the cached-token count, the fed tokens and log strings. The
warm-through-image shape has one such case, always with a single image, and
the MTP diversion has none.

## Decision

1. **Phase 2 yields every decision; phase 3 carries them out inline.** The
   Prefill Planner returns one `PrefillPlan`, together with the Speculation
   Plan it asked for (decision 4). The plan holds every decision a keyed
   request's prefill makes before the Cache Claim check-out:
   - its **Cache Opening**: cold, restore, or whole prompt;
   - its prefill, which is one run from a starting cache: how many key-path
     tokens that cache covers, what it forwards (an **Image Span** then the
     **Text Tail**, or text only), where the Position Anchor is seeded, the
     eval policy, the **Capture Schedule**, the **Decode Handover**, the
     **Maximum Advance**, and how many planned checkpoints a cold Image Span
     drops (for the log line that reports them).

   Plan application stays in `ServerCompletion`. It switches exhaustively,
   once each, over the Cache Opening, the forward and the Decode Handover.
   The forward is bound to the prepared input before `beginPrefill`, so the
   ADR-0014 guard still prices its image before the cache is allocated, and
   the MLX code inside the error scope switches over that bound value. This
   moves decisions, not execution: there is no extraction and no Model
   Session verb (ADR-0016). ADR-0033's reopen trigger for plan application
   itself ("unless the shape changes (e.g. a second caller appears)") is not
   met: plan application keeps one caller and the same MLX glue.

2. **The plan is a closed value, not a bag of fields.** The Cache Opening,
   the forward and the Decode Handover are enums whose cases carry exactly
   what their executor arm consumes. An image forward has no handover slot,
   and a whole-prompt arm is never a prefill's handover, so the type encodes
   two rows of Speculation's engagement table: an image-bearing request never
   speculates, and MTP never engages on a restored prefix. The plan is
   Sendable and Equatable and holds offsets, ranges and enums only: never a
   snapshot, a token array, an `LMInput` or a drafter. This follows the
   Completion Route, the Prefill Strategy and Live Leaf Capture's live case,
   whose executors have no field to work anything out from. It leaves the
   struct of fields that today's Prefill Plan and the Leaf Capture Plan use,
   around which decisions drifted back into their executors.

3. **An interface of plain values.** The production entry takes five values:
   the Keyed Request, the prefill boundaries, the resolved restore offset,
   the checkpoint plan and the session's Speculation. One extractor reads the
   request's facts off the Keyed Request (ADR-0044; ADR-0070 decision 4), and
   tests build those facts directly (ADR-0070 amendment). Any rule that asks
   whether the request is text-only reads the Keyed Request's fact (ADR-0070
   decision 6), never the Cache Key Space's identity; the key space supplies
   offsets and image runs only. The planner reads about fifteen facts and
   returns about nineteen across eleven types, close to the twenty inputs and
   ten outputs ADR-0033 called shallow. The difference is that every input
   and output is a plain value, not a live handle or an effect sink, so the
   whole pre-check-out decision is one table, and the planner takes over
   about a dozen rules that live in plan application today.

4. **The planner asks Speculation once.** After it decides restore or cold,
   the planner asks the session's Speculation once (ADR-0079 decision 2) and
   folds the answer in: the arm and split into the Decode Handover, the
   allowance into the Maximum Advance, a whole-prompt arm into the Cache
   Opening. Engagement stays in `Speculation.plan(for:)`, and the
   `SpeculationPlan` type is unchanged. Plan application's switches read
   none of its members: they hand it to iterator construction, which checks
   its arm against the Decode Handover, and to the unchanged whole-prompt
   body.

5. **The keyed eval policy is a plan decision**, recorded here for the first
   time. A prefill evaluates each chunk synchronously inside the MLX error
   scope, so a failure throws, exactly when it forwards an Image Span. It
   pipelines otherwise, where an MLX failure is fatal. "Otherwise" includes a
   restore that already covers every image of an image-bearing request, whose
   cache holds those images: the case the policy's own rationale says should
   be checked. That stays a named test row and a follow-up.

6. **The cold fallback is the planner's cold rule, asked again after the
   drop.** ADR-0069's 2026-10-03 amendment fixed the rule: a restore that
   yields no cache runs the turn cold under the request's one Speculation
   answer, drops the snapshot that failed, and plans its checkpoints again
   against the settled tree, so a checkpoint the drop took is captured anew.
   The drop and the re-read are effects on the tree, so the fallback cannot be
   computed before the check-out. The value the planner returns therefore
   offers the cold rule as a pure function of a checkpoint plan, with the
   request's facts, boundaries and Speculation answer already inside it.
   Plan application performs the drop and the re-read, then asks that
   function for the fallback prefill. It works nothing out itself, and it
   never asks Speculation again. This is the staged form the Snapshot
   Resolution Ladder uses for outcomes only execution knows.

7. **Image Span facts are key-space facts.** The Cache Key Space carries each
   image's patch count: the product of the grid the processor reported
   (t·h·w), never recomputed. The run lengths are still scanned from the
   prepared sequence, so ADR-0007's rejection of app-side patch-count math
   stands. The Image Span then travels as a token range, image indices, the
   pixel rows to skip and the anchor. The usable-restore rule and the Image
   Span rule become named planner rules, so that a later caller which
   restores a prefix and extends it (the Leaf Store's boundary route, the
   Speculative Canonical Prefill) can reuse them. `LMInput` and the anchor
   state are built inside the session by one bind step (ADR-0033, ADR-0016).

8. **What stays out of the plan.** Keyed or unkeyed (Request Keying,
   ADR-0007). Resolution and the checkpoint plan, which are inputs. The
   check-out, its outcome and the restore mode: the Cache Claim decides
   handoff or copy from live tree state, and plan application performs the
   copy (ADR-0064's 2026-09-22 amendment that gives the check-out to the
   Cache Claim; ADR-0069 decision 2). Iterator construction, KV Scheme
   conversion, cache adoption and the penalty seed, which a planned Decode
   Start (one module that converts the KV cache once and starts decode for
   every generation path) will own. Telemetry, admission, cancellation and
   salvage. And the positions of the three vision refusals: the ADR-0014
   vision-tower guard, the chunked-vision backstop and the
   anchored-continuation capability check.

9. **A strict pure refactor.** The bar of ADR-0078 decision 8 and ADR-0069
   decision 8 applies: byte-identical diagnostics, and the same steps,
   MainActor hops, Model Session entries and memory peaks. Three pure
   computations move up to the planner call: asking Speculation for its
   plan, the maximum advance and the split. None of them hops to the
   MainActor, logs, checks cancellation or runs inside a timed window, so the
   counted step sequence does not change. Every oddity the inventory found
   (about a dozen) stays as a named test row, in the planner's table or, for
   the execution-side ones, in the fixtures. Changing one is its own
   follow-up.

## Considered options

- **Extract plan application, or add a session verb.** Rejected again for
  ADR-0033's and ADR-0016's reasons. The decisions move; the execution stays.
- **A flat five-case shape enum with a recursive cold fallback.** It repeats
  the text tail in four cases, and its fallback re-plans from scratch, which
  asks Speculation a second time: one request could then run under two plans,
  and a failed restore on an MTP-resident turn would divert into the
  whole-prompt body after its check-out.
- **Carry a precomputed fallback prefill in the restore opening.** It was the
  first design. It cannot see the checkpoint that the snapshot drop frees, so
  plan application would have to patch it, which is the re-derivation this
  ADR removes.
- **A struct of optional fields, the shape named by which are set.** Invalid
  combinations, such as an image forward with a speculative handover, would
  type-check.
- **A staged three-step ladder for every turn** (plan, check out, fold the
  outcome). It leaves the Speculation request assembled in plan application,
  where two of the sampled bugs lived, and turns one call into three. The
  staged form earns its place only on the failed-restore path (decision 6).
- **Make the Speculation Plan an Equatable value, or carry its terms in the
  plan.** The first edits an accepted module (ADR-0079); the second lets code
  outside Speculation build an engagement Speculation never decided. The arm,
  the split and the folded advance already pin what tests need.
- **Fix the oddities while restructuring.** Each one changes behaviour or
  wire output, so each lands later as a one-row change with its own check.

## Consequences

- `PrefillPlannerTests` becomes the decision table: one value row per shape,
  opening, handover and named oddity, with no cache or snapshot fixture.
  Speculative rows construct the toy drafters but never evaluate them. The
  fallback rule gets rows of its own, including a dropped system checkpoint
  captured anew. Fixtures keep one case per executor arm, checking that the
  arm does what the plan says.
- Adding a Cache Opening or a Decode Handover costs one case, one planner
  branch, one row and one executor arm. A new forward kind costs two executor
  arms that the compiler links through the bound value, because the ADR-0014
  guard must price the bound image before the cache is allocated.
- The MTP fold ADR-0079 deferred becomes a planner-row change plus the
  deletion of the whole-prompt opening and its body. It stays deferred until
  MTP drafting is verified on a real model and the telemetry an MTP turn
  would gain is accepted.
- Decode Start consumes the Decode Handover and the eval policy; other
  generation paths can build a Decode Handover of their own.
- Four release preconditions in the pure planner, each on an input that
  Speculation's table or the Cache Key Space makes unreachable today: a
  Speculation answer on an image forward, a whole-prompt arm as a prefill's
  handover, a whole-prompt opening on an image-bearing key space, and a
  restore below the Minimum Warm Offset with no image left to forward. They
  trap before the check-out, where today the same input fails later or runs
  silently.

## Verification

The change lands in slices: test-only characterization of what plan
application feeds the Model Session; patch counts in the Cache Key Space; the
plan, unwired; a DEBUG-only shadow that asserts the plan equals today's
inline derivation on every path, including the failed-restore fallback; then
the switch. The last slice runs the loaded-model gates in `docs/testing.md`
on a quiet GPU: `prefix-cache-e2e`, `hybrid-cache-correctness`, the 27B
DFlash2 end-to-end run diffed against `main`, and the image scenario on
`bonsai-2-27b`. MTP stays verified only by Speculation's engagement table and
the Prefill Plan's rows, as ADR-0079 records.
