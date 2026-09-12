# Leaf checkout and rewind — bounded evidence, 2026-09-12

Implementation for [#480](https://github.com/spokvulcan/tesseract/issues/480),
starting from `fef2ee55ab6b65a302ca5e60fbe45dfe3d87e3e5` (capture handoff plus
tree-side leases). The revised code moves eligible resident leaf objects into
the request and returns them on success or rewind. It also removes unused
preserve-thinking, text-only transient boundary captures. No model weights
were loaded for these measurements.

## What was measured

The isolated `LeafCheckoutMemoryEvidenceTests` run uses real MLX attention and
Mamba caches: 4,096 rows, head dimension 64, two float32 attention arrays
(2,097,152 bytes), plus 64 bytes of recurrent state. One process performs
24 success/cancel/failure cycles, each advancing two tokens before check-in
or rewind. It deliberately retains all retired request owners, then clears
the tree and checks that the attention object deallocates.

| Observation | Captured result |
| --- | ---: |
| Independent copied-restore allocation | 2,097,232 B |
| Active MLX allocation at checkout, min/max over 24 turns | 64 / 64 B |
| Independent recurrent rewind state | 64 B |
| Returned owners retaining any cache or rewind state | 0 / 24 |
| Tree leases after each return | 0 |
| Logical final leaf, after eight successful two-token extensions | 2,105,408 B |
| Process active MLX, first checkout baseline / final return | 426,594,496 / 426,725,568 B |
| Process footprint, first baseline / final return | 1,224,361,856 / 1,296,304,072 B |
| Sampled system swap | 0 B |

The extra 131,072 active MLX bytes match one 256-row attention capacity step
(256 × 64 × 2 × 4 bytes). Logical trimming does not shrink that backing.
The process-footprint increase includes the test host, allocator and other
runtime work; it is not explained solely by logical leaf growth, and no
controlled settling interval was measured. The physical identity and
retired-owner assertions establish release; this run does not establish
steady-state process footprint or cancellation cleanup latency for a loaded
model. The cache-capacity lifetime is tracked in [#501](https://github.com/spokvulcan/tesseract/issues/501).

`small-cache.json` preserves all 72 phase rows. `small-cache-diagnostics.log`
contains scalar request/lease events from the same isolated run. It predates
the final addition of `leafStore source=rewind`; the production synthesized
replay asserts that final event. Logical component counts overlap and must
not be added to estimate physical memory. No prompts or generated content
are included.

## Capture-only baseline versus checkout

The [September 8 report](../../capture-handoff/2026-09-08/README.md) and
[paired table](../../capture-handoff/2026-09-08/comparison.md) remain the
production baseline. Its detailed archive SHA-256 is
`2944ddec10464fb1ad2d367ff6a2a7dd9079237180293b4a5152164414b00c23`.

| Component | Preserved capture-only evidence | Checkout implementation / current evidence |
| --- | --- | --- |
| Warm copied restore at ~46k | About 3.19 GB allocation | Eliminated for eligible movable leaves; proven by small-cache identity and 64 B checkout allocation, not yet measured at 46k |
| Transient boundary helpers | About 3.19 GB at 46k; 11.61 GB in the matched 93k pair | Not captured for preserve-thinking text-only identity requests; loaded allocation reduction pending |
| Leaf capture at ~46k | 156.685 → 1.805 ms; 3.16 GB allocation removed by #478 | Capture move retained; no new loaded timing measurement |
| Recurrent rewind state | No checkout backup | Independent deep copy, exact state-slot metadata/lengths/padding; 64 B in the bounded fixture |
| Pending full payload | May alias resident attention arrays | Forces copied restore until materialization detaches/releases arrays; full payload bytes are not a second physical KV copy |
| Detached extension payload | Already independent after #474 | Does not prevent handoff; pending payload bytes still consume memory |
| Historical 93k peak | 50.2 GB lifetime peak on September 6 | Reduction by at least one leaf remains unverified |
| Peak and settled retained process footprint | Not uniformly improved by capture alone | No comparable production sample yet |
| ActiveInferenceReserve | Existing capture-copy factor | Unchanged; separate measurement-led re-pricing |

## Replay and correctness

The tokenizer-only corpus gate consumed 85 recordings, found 66 replayable
boundaries, registered all 66 and resolved all 66 next-request prefixes.
There were zero fidelity rejections, zero undecodable paths and no rejected
registrations. The longest emitted path was 92,759 tokens. The slowest
CPU-only leaf/index phase was 86.3 ms. `corpus-summary.log` preserves the
per-boundary scalar outcomes.

There were no live `trace-*.jsonl` files beside these recordings. The corpus
harness uses synthetic cache state and reports `source=live`; it does not
exercise checkout, prove production handoff per recording, or establish the
HTTP post-EOS tail bound. The real small-cache server replay separately
proves partial-turn cancel → rewind → resend hit and warm-prefill cancellation
with lease/registry release. Rewind tests replace recurrent state and cross an
attention growth allocation, then compare original offset, bytes and
metadata. Eligibility cases include system/branch/window/untrimmable/rotating,
quantized, image and pending full payload paths. Existing lease suites cover
pressure and writer exclusion. Quantized two-turn sequencing uses actual
quantized cache objects and caught/fixed retention of the pre-quantization
array (which produced incorrect second-turn output).

## Validation and review

The full unit target ran once: 2,874 passed, 15 skipped, one failed out of
2,890 tests. The failure was the old telemetry assertion expecting `copy`
on an eligible warm restore; it was updated to require `handoff` and verify
lease acquisition/release. The final focused run passed all 26 tests across
checkout, synthesized replay and request-memory suites. The real quantized
cache regression passed in the earlier 23-test focused run. The 15 optional
or environment-gated skips remain skips, not passing loaded-model evidence.
See `validation.json` for exact scope. Source compilation/typechecking passed
in each green Xcode run, and the final Release build passed. Formatting and
documentation references pass.
SwiftLint reports size/complexity and directive/comment warnings, no errors.

Parallel standards and spec reviews found a quantized-object retention bug
and a startup-rewind mode mismatch; both were fixed and regression-covered.
No verified ownership or rewind defect remains from those reviews. The
loaded-model acceptance gaps below remain open.

## Reproduction

Follow `docs/testing.md` for quitting/relaunching the app and Xcode flags.
Run `LeafCheckoutMemoryEvidenceTests` alone for memory evidence. Its JSON
marker is `LEAF_CHECKOUT_EVIDENCE=`. Run `EmittedPathReplayCorpusTests` with
`TEST_RUNNER_TESSERACT_FIDELITY_CORPUS` pointing to the private recordings and
`TEST_RUNNER_TESSERACT_FIDELITY_MODEL` pointing to the tokenizer directory.
These inputs and temporary Xcode result bundles are not evidence artifacts.
The scalar exports here are durable; no private model or corpus files are
included. `manifest.json` provides SHA-256 checksums.

## Pending loaded-model plan — requires owner approval

The previous large-model work preceded a user-reported shutdown/crash. No
45k/75k/93k run or reload experiment was attempted on this 48 GiB Mac.
Do not treat the following plan as authorization.

1. Use a suitable Apple Silicon host with at least 96 GiB RAM, at least
   40 GiB initially available, and no other inference process. Run one
   validation process at a time. Use the existing collector and private
   corpus; preserve model/drafter/config hashes and matching cache plans.
2. Preflight at 4k tokens, one cold and one warm turn, maximum output 128,
   temperature 0, DFlash2 on, unquantized KV. Sample process footprint and
   system memory every 250 ms. Cancel and stop the session if footprint
   reaches 56 GiB, available memory falls below 16 GiB, or system swap grows
   by 1 GiB from the preflight baseline. If cancellation does not finish
   within five seconds, terminate only that isolated validation process.
3. Only after the preflight passes, run one matched cold/warm pair at each
   of 45k, 75k and 93k, in increasing order. Stop the entire campaign on a
   threshold breach; do not retry or automatically reload. Capture-only and
   checkout builds run serially with identical explicit settings and the
   same controlled setup, recording cache-plan differences.
4. Include a bounded tool continuation and cancel/resend case, including
   cancellation during cleanup. Record signal → quiescence → lease/registry
   release and sample after SSD drain plus a five-second settling interval.
   Record output parity, cache offsets, actual restore/leaf sources and copy
   reasons, active/cached MLX, peak and retained process footprint, backup
   bytes, pending payloads and checkpoint lifetimes.
5. Measure the <20k post-EOS HTTP tail (<150 ms), verify one live attention
   owner on eligible turns, and compare the 93k peak to both the capture-only
   run and the historical 50.2 GB observation. A different host makes the
   historical process comparison contextual; physical ownership and matched
   same-host before/after results are the decisive mechanism evidence.

Issue #480 remains open for these loaded-model gates. The design ADRs are
Accepted as built; that status is not a claim that the outstanding
performance acceptance criteria passed. Required follow-ups:
[index persistence #499](https://github.com/spokvulcan/tesseract/issues/499),
[optional lineage #500](https://github.com/spokvulcan/tesseract/issues/500),
[warm-prefill/DFlash2 and retained-capacity profiling #501](https://github.com/spokvulcan/tesseract/issues/501),
[generation-prompt-only partitioning #502](https://github.com/spokvulcan/tesseract/issues/502).
Issue #466 was already closed as superseded by #471.
