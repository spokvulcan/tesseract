# Retained attention capacity after a cancelled generation — #501 profile

Refs [#501](https://github.com/spokvulcan/tesseract/issues/501) and
[#534](https://github.com/spokvulcan/tesseract/issues/534), part of #520
phase 4. Plan: [plan.json](plan.json), committed before the run; driver
`scripts/cancelled_generation_profile.py`; app commit `3671ebcc`.

## What is measured, fixed before running

The 2026-09-20 boundary-leaf-move audit found only 8–9 MB of unused
attention capacity on ordinary turns with the #542 growth policy, and said
#534's threshold must come from a cancelled long generation. This run
produces that trace. At three prompt sizes (~8k, ~33k, ~66k tokens, built
by pasting `CONTEXT.md` slices) the driver runs, in order: a short growing
turn, a long streaming generation cancelled by the client after a fixed
number of generated tokens (300, 1024, 1024), and a short follow-up turn.
For each cancelled request the new `leafRewind` fields report the rewound
cache's full-attention array bytes, the lease offset's logical bytes and
their difference (the capacity the leaf retains); for each ordinary turn the
`capturingLeaf` sample's `requestFullAttentionUnusedArrayBytes` reports
the check-in remainder. `requestMemory` gives the phase timeline (active
and cached MLX, footprint) and the settled sample after release.

Decision rule for #534, fixed now: the compaction threshold is the smaller
of a quarter of the body and 256 MB **unless** the measured retained
capacity after a cancelled generation is below 64 MB at every size, in
which case compaction is not worth its copy and #534 is closed with the
numbers. If the retained capacity exceeds 256 MB at the 33k or 66k size,
256 MB is the threshold; otherwise the threshold is the smallest measured
retained capacity that exceeded a quarter of its body, rounded down to a
power of two, or 64 MB if none did.

## Bounds (48 GiB host)

| Bound | Value |
| --- | --- |
| Minimum initial available | 20 GiB |
| Expected peak | 28 GiB (weights 15.1 GB + ~4.4 GB body at 66k + checkout/rewind transients) |
| Footprint stop | 34 GiB |
| Minimum available stop | 6 GiB |
| Swap growth stop | 1 GiB |
| Pressure stop | level 4 or unknown |
| Request / campaign deadline | 400 s / 3600 s |

One process, DFlash2 on (the app default), unquantized KV, reasoning effort
low with `preserve_thinking: false` (the think-stripping boundary path,
as in the audit). No retry after a stop.
