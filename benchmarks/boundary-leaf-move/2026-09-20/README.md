# Boundary leaf capture by move — session-log audit and A/B

Refs [#501](https://github.com/spokvulcan/tesseract/issues/501), part of
[#520](https://github.com/spokvulcan/tesseract/issues/520) phase 4.

Two parts: what the 2026-09-20 agent session's logs say about where the
memory goes, and a controlled A/B of the one change that came out of it.

## Part 1 — the session audit

Source: the unified log for `app.tesseract.agent`, 03:07–03:57 UTC on
2026-09-20, the interactive session that ran conversations `D474F284`
(weather, then shoes) and its neighbours. 1,731 `requestMemory` samples over
42 requests, Qwen3.8-27B 4-bit with DFlash2, unquantized KV, 48 GB M3 Max.

Constant floor: 14.09 GB of target weights plus 1.04 GB of DFlash2 draft
weights. Everything below is on top of that 15.13 GB.

### Where the process peak is

Peak `processFootprintBytes` rises with the conversation, from 19.4 GB at
550 prompt tokens to **32.11 GB at 67k**, against 9.7 GB of system swap by
the end of the session. Every request's peak sample is at end of turn, and
on the requests that peak hardest the phase is `preparingPayload`, which
follows the leaf capture.

The peak is not the SSD writer. `ssdPayloadArrayBytes` is `0` on every
request in this session — nothing was written — so the no-copy writer of
[#541](https://github.com/spokvulcan/tesseract/issues/541), which is in this
build, has nothing to do with it.

The peak correlates with exactly one field:

| `leafCaptureMode` | requests | peak over decode steady |
| --- | --- | --- |
| `handoff` | 32 | +0.06 to +0.53 GB |
| `copy` | 10 | +1.53 to **+8.19 GB** |

and every `copy` request is a `path=boundary` leaf store with
`boundary=think-stripping-user-boundary`.

### Why

A think-stripping template re-renders a finished stop-turn once the next
user message arrives, so that turn cannot key its leaf on what the model
fed. `LiveLeafCapture` sends it down the boundary path: restore the boundary
snapshot, re-prefill the canonical residual, capture. The capture was a full
deep copy of the re-prefilled cache — a second resident copy of the whole
conversation's KV, 4.45 GB at 69k tokens.

In an agent session this is not a rare path. Tool-call turns render verbatim
and take the fast path; the turn where the agent answers the user is a
stop-finish turn and takes the boundary path. It was 10 of 42 requests here,
and the global capture-mode tally over the window is 165 copies to 467
handoffs.

The copy costs three times over, one leaf each:

1. **The capture transient.** Traced on request `FB6E0140` (67,077 prompt
   tokens): active MLX sits at 24,589 MB through decode, the live leaf is
   handed off for free, then `capturingLeaf` jumps to 29,757 MB — exactly
   the 4,454 MB `leafSnapshotArrayBytes` — and `preparingPayload` reaches
   30,602 MB before everything drops to 21,656 MB.
2. **The next turn's restore.** The copy leaves a `.copied` body where
   check-out needs `.moved`, so the following request is refused with
   `copyReason=immutableBody` and restores by copy instead — another full
   KV. `immutableBody` is the dominant restore-copy reason in the window
   (98 samples) and `restoreMode` is `copy` 176 times against `handoff` 33.
3. **The Active-Inference Reserve.** It doubles a lane while the most recent
   leaf store captured by copy, so a boundary turn also cuts the RAM-tier
   ceiling by a leaf — the subtraction [#238](https://github.com/spokvulcan/tesseract/issues/238)
   is about.

### What the audit found already fixed

Every boundary turn in the session also logged
`stage=boundaryBackingLeafRelease reason=not-releasable`, keeping the live
backing leaf and the canonical leaf resident together — at 69.5k tokens two
bodies one token apart. That is [#551](https://github.com/spokvulcan/tesseract/issues/551),
fixed in `59b2b2e6`, which landed after this build was compiled (binary
02:24 local, commit 05:58 local). No action; the session predates the fix.

### What the audit found small

`requestFullAttentionUnusedArrayBytes` — the retained growth capacity
[#534](https://github.com/spokvulcan/tesseract/issues/534) is about — is
8–9 MB on a 4.2 GB attention body here, because the geometric growth and
prompt reservation of [#542](https://github.com/spokvulcan/tesseract/issues/542)
are in this build. Compaction has no measurable headroom on this workload;
#534's threshold should be set from a cancelled-long-generation trace, not
from this one.

## Part 2 — the change and its A/B

`LeafStorePhase.captureStructuredLeafFromBoundary` now moves its restored
cache into the leaf instead of deep-copying it. The cache is request-private
by construction — `restore` deep-copies and evaluates every layer out of the
tree — so ADR-0064's one-owner rule permits the move. See the ADR-0064
amendment of 2026-09-20.

### Workload

One growing conversation over `/v1/chat/completions`, 6 rounds, each
appending ~18k characters of `CONTEXT.md` and the model's reply, reaching
25.7k prompt tokens. `reasoning_effort: low`,
`chat_template_kwargs: {"preserve_thinking": false}` and no `reasoning_content`
echoed back, which is what puts every turn on the think-stripping boundary
path. `max_tokens` 320, temperature 0. Fresh app launch per arm, identical
script, same model and settings. Driver: `longctx.py` in this directory.

### Result

| | before (`main`, 59b2b2e6) | after (`7b32a697`) |
| --- | --- | --- |
| boundary captures by copy | 5 — mean **+1,208 MB**, max **+1,752 MB** | **0** |
| active MLX allocated across the capture | +1,208 MB mean | **0 MB** (n=8 moves) |
| peak active MLX at 25.7k tokens | 21.81 GB | **20.42 GB** |
| peak active MLX, mean over the 6 turns | 19.68 GB | 19.35 GB |
| peak process footprint at 25.7k | 22.90 GB | 22.69 GB |
| restores refused `immutableBody` | 4 | **0** |
| boundary `leafStore` source | `boundary` ×5 | `handoff` ×4 |

The saving is one leaf per boundary turn and scales with the leaf: 1.2 GB at
26k tokens here, and the session logs measured the same copy at 2.97 GB at
44k and 4.45 GB at 69k.

Process footprint moves less than active MLX because MLX's buffer pool
absorbs a freed transient; the allocator-level number is the one that shows
the copy going away. No latency claim is made from this run — round times
overlap between arms and each arm is a single trial.

### Loaded-model gates, same model

- `hybrid-cache-correctness`: 12/12 PASS, including
  `movedLeafRestoredByCopyMatchesBitwise`.
- `prefix-cache-e2e`: 23/23 PASS, Overall PASS. On the e2e's own (180 MB
  leaf) workload the capture allocates 0 MB across all 58 captures, where
  `main` allocated +183.9 MB mean on its 14 boundary captures — the same
  mechanism at a scale small enough that process peak does not move.

### Limits

One trial per arm, one machine, one model, text-only, no images and no
quantized-KV partition. The quantized fallback is covered by a unit test
(`canCaptureMovingAgreesWithWhatAMoveActuallyTakes`), not by a loaded-model
run. Peaks are sampled lower bounds from the phase telemetry, not continuous
maxima. The 69k figures in Part 1 are observations of the unfixed build, not
an A/B.

Per the 2026-09-08 capture-handoff note, the long-context experiments were
kept bounded: one process at a time, a 38 GB stop threshold in the driver,
no repeated model reloads, and no run past 26k tokens.
