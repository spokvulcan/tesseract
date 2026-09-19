# Warm Body parity gate — pre-registration, 2026-09-19

Issue #528, part of #520. **Status: NOT RUN; owner execution pending.**
This document defines the decision before collecting loaded-model results.
It does not authorize a run. No loaded-model, long-context or model-reload
workload may run on the Mac used to prepare this change.

## Decision and scope

Warm Bodies remain opt-in (`warmCompressionEnabled = false`). Enable the
8-bit default only when every fidelity and timing check below passes on the
frozen loaded-model verification workload without a resource-stop breach.
Four bits is a separate experiment under the identical gate; it becomes an
option only after passing and is not selected as the default by this plan.
Failure, incomplete warm-8/control evidence or an invalid comparison leaves
the default off. A passing RAM result does not authorize #531: SSD-hydrated
Warm Bodies must pass again before that Stored Form can ship.

The existing [Canonical-Echo Fidelity corpus gate](../../../docs/testing.md#canonical-echo-fidelity-gate-corpus-mode)
is tokenizer-only: it neither restores caches nor loads model weights.
Its threshold is **zero mismatched boundaries, with a nonempty boundary set**.
Run it on the actual per-arm loaded replay recordings. Running it repeatedly
on an unchanged historical corpus cannot demonstrate warm-restore parity.
Also compare actual greedy output token IDs against the matched fp16 arm;
require zero mismatched token sequences. Lossy cache bytes and logits need
not be bitwise identical; generated token parity is not inferred from them.

## Freeze before execution

Commit this pre-registration first. Before any measured or exploratory
loaded run, the owner must complete and commit a run manifest based on
[owner-plan.template.json](owner-plan.template.json), approve it explicitly,
and record both the pre-registration commit and manifest commit in results.
Null fields and empty lists mean **not executable**, not automatic defaults.
Commit any protocol change as a new pre-registration before collecting its
results. Preserve failed and stopped attempts; do not select thresholds,
fixtures, sample counts or exclusions after seeing performance or fidelity.

Freeze the app/vendor revisions, benchmark instrumentation patch revision,
Release binary checksum, host/RAM/OS/MLX details, model/tokenizer/template
checksums, private corpus checksum and ordered case IDs. Freeze the model's
live KV dtype (fp16), group size 64, affine quantization, prompt/output caps,
greedy sampling (temperature 0), speculation policy and fixed seed wherever
applicable. Any drafter, vision or model-specific auxiliary state must match
between arms and be included in the resource estimate. Do not quantize the
live cache or use an already-quantized KV partition as the fp16 control.

Select the existing loaded-model verification workload before running. Include
its bounded direct-leaf continuation, planned view, transient boundary view,
tool continuation, think-stripping boundary and cancel/resend cases where
supported. Record each expected restore offset, view offset and boundary kind.
Any unsupported case must be identified in the manifest and restricts the
result's claimed coverage; silently dropping a case cannot produce a pass.
A bounded preflight is an execution-readiness check, not a substitute for the
frozen verification workload or evidence for untested long contexts.

## Arms and execution readiness

| Arm | Resident attention body | Live restore | Required diagnostic evidence |
| --- | --- | --- | --- |
| fp16 control | Owned full body | Restore by copy | Same offset and request suffix; no handoff, SSD hydration or re-prefill |
| warm-8 | 8-bit affine, group size 64 | Dequantize to fresh fp16 | Direct `source=warm`, `copyReason=warmBody`; view `source=view`, `backingLeafForm=warm`, `copyReason=checkpoint` |
| warm-4 experiment | 4-bit affine, group size 64 | Dequantize to fresh fp16 | Same requirements as warm-8, plus recorded quantization metadata proving 4 bits |

For the control view, require `backingLeafForm=ownedBody`. For both warm view
arms, slice packed rows to the view offset before dequantizing; preserve the
view's own whole-state checkpoint. Force comparable restore-by-copy behavior
for the direct fp16 control, and verify it from the cache diagnostics rather
than comparing a warm copy against Leaf Handoff. Pin equivalent Backing Leaf
geometry and reset each case's cache/index state before the next repetition.
Do not hold all three bodies concurrently: create one arm from identical
prefill inputs at a time, evaluate it, release preparation-only owners, and
settle before timing. Reconstruct the source from the same inputs for each
arm; never derive 4 bits from an already quantized 8-bit body.

**Current execution gap:** production compression is fixed at 8 bits, and the
existing corpus test and loaded cache correctness runners do not run this
three-arm Warm Body campaign or separately time dequantization. This PR adds
no executable runner and no 4-bit product setting. Before execution the owner
must provide/review the experiment instrumentation at the existing Model
Session and snapshot seams, freeze its revision, and verify the actual arm
selection and timing boundaries. Existing commands in `docs/testing.md` are
not a ready-to-run #528 gate. Do not claim that toy tests or the existing
bitwise uncompressed-cache gate fill this gap. A new test seam would require
returning to the owner before implementation.

## Fidelity, timing and memory protocol

Use one isolated validation process, one model resident at a time, no ordinary
HTTP clients, a private output directory and no production SSD cache. Run arms
serially with identical settings. Freeze a separate case for any speculation
policy instead of pooling different policies. Reconstruct the same starting
cache and replay the exact same request/suffix for every arm. Include actual
model-generated continuations in each arm's recorded corpus; tool calls use
predeclared synthetic results and are never executed. Retain raw output IDs
and boundary verdicts privately, with checksums in the public scalar report.

For each case, do one untimed warmup per arm, then six measured blocks, one
for each order: `(fp16,8,4)`, `(fp16,4,8)`, `(8,fp16,4)`, `(8,4,fp16)`,
`(4,fp16,8)`, `(4,8,fp16)`. This gives six measured repetitions per arm per
case. No statistical outlier removal or retry-to-pass. A resource stop ends
the campaign; a fidelity/timing failure is retained and cannot be replaced
by another sample. Technical invalidation (wrong offset/form, missing trace,
concurrent request) makes the campaign inconclusive, not passing. Any rerun
requires a newly recorded owner decision and retains the previous attempt.

Measure end-to-end TTFT on a monotonic clock from request submission through
the first emitted token/delta, using the same client and observer for every
arm. Also record server restore and first-token intervals separately. Exclude
prefill preparation, compression and model loading from the timed hit; include
lookup, restore and continuation work. Verify that every measured request
actually takes the intended restore path. Full re-prefill, a shallower restore,
Leaf Handoff or SSD hydration invalidates that paired observation.

Measure the dequantization allowance independently for each warm arm and case
on the matching quantized attention prefix. Evaluate/synchronize inputs before
timing, then time the vendor dequantize conversion through evaluated fresh
fp16 outputs and device completion. Include output allocation. Exclude source
quantization, whole-state copying, prompt prefill and token generation.
Use the same six repetitions and an identical synchronization convention for
the control timing; record the observer overhead. Do not use the whole warm
restore interval as the allowance, which would include unrelated work and
make the comparison circular. For a view, measure its prefix, not its full
descendant. Release these measurement outputs before the TTFT observations.

For each case and warm bit width independently, require:

- Canonical-Echo Fidelity mismatches = 0, with nonzero checked boundaries and
  the same boundary-kind coverage as fp16. Preserve `noPath`/advisory counts;
  an increase versus fp16 is inconclusive and must be explained before a pass.
- Every measured greedy output sequence equals its matched fp16 sequence.
- Median paired excess `median(TTFT_warm[i] - TTFT_fp16[i])` is no greater than
  `median(dequantize_warm[i])`. No added tolerance or cross-case pooling.
  Publish all six samples and p95 as context, plus both sides of the inequality.
- Every required observation and resource sample is present, no configured
  stop was breached, and all restored live attention remains fp16.

Collect the existing request memory timeline and an external process/system
sampler every 250 ms. Record settled resident body bytes, whole-state bytes,
active/cached/peak MLX bytes, peak/settled physical footprint, minimum available
memory, swap change, pressure and sampling timestamps. Preserve model-loading
and compression peaks separately from the timed hit. Sampled stop thresholds
are abort conditions, not guaranteed peak ceilings; report observed lifetime
peaks as well. Logical bytes and `nbytes` do not establish physical ownership
or imply that view and backer allocations can be added without overlap.

## Owner resource approval

Use a suitable Apple Silicon host, not the preparation Mac. The prior
[loaded-model owner plan](../../leaf-checkout/2026-09-12/README.md#pending-loaded-model-plan--requires-owner-approval)
proposes at least 96 GiB RAM and 40 GiB initially available, a 4k-token preflight
and a 128-token output ceiling, stopping at 56 GiB process footprint, below
16 GiB available memory, or 1 GiB additional swap. These are proposed bounds,
not approval or a guarantee that a chosen model/workload fits. The owner must
record explicit numeric bounds appropriate to the selected host and model,
including prompt length, campaign/request deadlines and pressure stop level.

Before launch, validate the approved manifest, expected model-resident plus
cache/conversion peak and available disk space. Stop on any threshold breach,
critical or unknown pressure, missing memory samples or process disconnect.
Cancel once; allow at most five seconds for quiescence, then terminate only
the isolated validation process. Do not retry, reload models automatically,
raise thresholds or progress to larger contexts after a stop. Archive the
partial evidence and mark the campaign stopped. Private inputs remain outside
the repository; commit scalar results and checksums only.

## Results and decision ledger

No loaded-model results were collected for this pre-registration. `PENDING`
means unmeasured; it is not zero, a pass, or an estimate from compression bits.
Populate one table per registered case after the owner run, with links to
all raw scalar samples and the frozen manifests.

| Arm | Fidelity mismatches / boundaries | Greedy token mismatches | TTFT median / p95 (ms) | Dequantize median (ms) | Paired excess median (ms) | Body bytes | Peak MLX / footprint (bytes) | Minimum available / swap growth (bytes) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 copy | PENDING | reference | PENDING | n/a | reference | PENDING | PENDING | PENDING | NOT RUN |
| warm-8 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | NOT RUN |
| warm-4 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | NOT RUN |

After a valid pass or failure, append the decision and evidence links to
[ADR-0068's as-built section](../../../docs/adr/0068-prefix-view-checkpoints-and-compressed-warm-tier.md#as-built).
A valid warm-8 pass authorizes a separately reviewed default change; otherwise
leave it off and record why. A warm-4 pass authorizes consideration as an option
only. Missing warm-4 results keep that experiment pending; they cannot validate
warm-4 or substitute for warm-8 evidence. #531 remains owner-blocked until the
8-bit RAM gate passes, and then requires its own SSD-hydrated replay before
quantized Stored Form enablement. This pre-registration does not close #528.
