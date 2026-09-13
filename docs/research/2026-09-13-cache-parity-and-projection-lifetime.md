# Bounded cache parity and projection lifetime

Follow-up to the [three allocation investigations](2026-09-13-allocation-investigations.md),
authorized by the maintainer to run a bounded production correctness gate,
prepare a PR and investigate the remaining projection-weight overlap.
[Evidence and reproduction](../../benchmarks/allocation-parity/2026-09-13/README.md)
include both loading attempts, source patches, model/binary hashes and test logs.

## Production cache and logit parity

The new `--bench-bounded-cache-parity` mode of `HybridCacheCorrectnessRunner`
uses exactly 2,048 deterministic tokens, a leaf at 1,024 and one final sentinel
token. It keeps unquantized KV and the production Qwen3.8-27B affine 4-bit target
plus 4-bit DFlash2 load. MLX work stays within exclusive model-container access.
The gate calls production prefill, snapshot, checkout/rewind, payload, SSD writer
and hydration paths. Its unique SSD directory is removed after a writer flush;
ordinary SSD metadata is untouched.

`CacheStateBytes` copies logical state to independent host evidence. Equality
covers layer/tensor counts, cache class, offsets, metadata, shapes, dtypes and
raw bytes. It does not compare allocation capacity beyond logical state. Three
model-free regressions verify signed-zero, structural/type/offset sensitivity
and independence from later mutation. This opt-in correctness observer evaluates
and copies arrays; it is separate from scalar production telemetry.

The passing run completed **18 checks**:

| Boundary | Verified |
| --- | --- |
| Copy capture/restore and moved snapshot restored by copy | Exact 64-layer prefix state |
| Copied continuation | Exact final state and logit bytes against cold prefill |
| Leaf checkout | Original request/tree relinquish their cache owners; exact prefix state |
| Handoff continuation | Exact final state and logits against cold prefill |
| Growth then Leaf Rewind | Empty request owner and recurrent backup, exact original state; resend state/logits match cold prefill |
| Full SSD write and restore | Exact prefix state and continuation state/logits |
| Extension SSD write and chain restore | Exact extended state; a subsequent sentinel's state/logits match an in-memory reference |

Prefix state: **221,052,928 B**; final 2K state: **288,161,792 B**.
Each logit vector is 496,640 B. Copied, handed-off, rewound and SSD-restored
continuations share SHA-256
`c0a17c60abbf5e80131e6d2175329ea0895c5effdd7ef1a91bac0b2ea4dd32b4`.
Full and suffix payloads reached the borrowed-chunk writer; encoding staging
was 21,213 B and 26,992 B respectively.

The 48 GiB Mac completed the capture in 45.72 seconds. All 180 pressure samples
were normal; maximum sampled footprint was **25,189,139,064 B (23.46 GiB)** and
system swap growth was zero. Loading took 6.96 seconds. The 250 ms sampler,
32 GiB footprint stop, 2 GiB additional swap stop, critical/unknown pressure
stop and ten-minute deadline bound the experiment; sampled stops are not peak
ceilings. This correctness run holds extra reference/evidence copies and does
not establish production latency or memory savings.

### Stopped attempt and loader correction

The first attempt used the existing harness's bare `AgentEngine`, which has no
`SettingsManager` and defaults to `.automatic` speculation. Unlike the preceding
server captures, it began loading the MTP head as well. The guard terminated it
at critical pressure during `modelMTPLoadBegin`, before parity checks:
15.35 seconds, maximum sampled footprint 30,678,717,728 B, system swap growth
1,215,299,584 B. This was a configuration-mismatched load, not a failed cache
comparison or an optimization regression.

The bounded mode now loads through `LLMActor` with explicit `.dflash2`, matching
the previously verified server load without changing user preferences. The
passing run records target/DFlash2 loading, no MTP load, and all model-file
hashes match the preceding production captures. The default full correctness
matrix and recorded-replay mode retain their existing behavior.

This gate does not exercise speculative decoding, network cancellation timing,
45k/75k/93k contexts, other model/layout families or the full #480 matrix. The
earlier HTTP captures cover observed DFlash2 engagement and real disconnects;
those scopes must not be conflated with this exact-byte gate.

## Remaining projection-weight overlap

The previous production capture retained the same active-MLX high-water mark
after clearing reusable buffers. Ranked explanations were: (1) the flattened
module list retains replaced projection objects until traversal ends; (2) lazy
concatenation graphs retain old tensor inputs despite module release; (3)
compiled traces retain old weights. An incremental visitor should lower overlap
if (1) dominates; otherwise changing traversal alone would be insufficient.

The source supplies a precise owner: `stackSameInputProjections(in:)` iterates
`model.modules()`, which returns a strong flat array of every descendant.
Blocks replace original projections with placeholders, but that array remains
alive across all blocks. `stackedQuantizedLinear` evaluates new concatenations
before replacement, so overlap within one block is still required.

The minimized fixture has eight real `Qwen3NextMLP` blocks (the MLP used by the
Qwen3.5 family), dimensions 512/1,024, group size 64, 4-bit quantization, seed
506 and input `[1, 4, 512]`. It loads no production weights. The initial control
used the public production traversal for both arms; its lower-overlap assertion
failed. The treatment then changed only traversal to `Module.visit`, retaining
the same folding methods and trace invalidation. Before/after output **bytes**
remained identical. The green measurement and final opt-in rerun both show:

| Small-fixture observation | Production flat traversal | Incremental visitor prototype |
| --- | ---: | ---: |
| Projection bytes folded | 5,242,880 | 5,242,880 |
| Active-MLX peak above the post-stacking value | 5,242,880 | 655,360 |
| Output bytes equal to unstacked reference | Yes | Yes |

Peak above the final value is derived as `beforeBytes + peakIncreaseBytes -
afterBytes`. Both arms ended at 432,369,872 active bytes in every green run.
Incremental-arm entry values varied, so the raw entry-relative increments
differ; all raw counters are preserved. Absolute
hosted-process MLX counters include unrelated owners. These bracketed live-byte
observations are neither process footprint nor predicted GiB savings on 27B.

This confirms avoidable flat-list retention in the fixture: overlap falls from
all eight projection pairs to one pair. The test is opt-in and must run alone
because the peak counter is process-global. The visitor remains **a test-only
prototype**; production traversal is unchanged. Its next gate is a matched
loaded-model comparison of target/draft stacking counts, exact weights/outputs
and compiled-trace lifetimes. This small MLP does not establish trace retention
in the complete target/drafter or guarantee lower sampled loading footprint.
