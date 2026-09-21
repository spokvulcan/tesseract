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

## Results (run 2, 2026-09-21 08:29–08:39 UTC)

App commit `3671ebcc`, plan commit `9a238a92`, driver `dd8a097d` (the first attempt, `run1`, stopped after 50 s when the diagnostics sink rotated at 8 MB; the collector now follows the rotation and run 2 saw 0 rotation(s)). No resource stop: peak footprint 27.98 GB (stop 36.5 GB), minimum available 8.96 GB, pressure at most 2 (warning during model load only), swap growth 0 MB. Raw records (`requests.json`, `events.json`, `os-samples.json`) stay in the private results directory; [summary.json](summary.json) carries the per-request scalars.

### Retained capacity after a cancelled generation (Leaf Rewind)

| Step | Lease offset (tokens) | Generated before cancel | Body logical bytes | Array bytes after rewind | **Retained capacity** | Share of body |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p8k-cancel | 6993 | 300 | 458.3 MB | 508.4 MB | **50.1 MB** | 10.9% |
| p32k-cancel | 30538 | 1024 | 2001.3 MB | 2118.6 MB | **117.2 MB** | 5.9% |
| p64k-cancel | 60878 | 1024 | 3989.7 MB | 4106.9 MB | **117.2 MB** | 2.9% |

The retained capacity is set by the generated length and the growth granules, not the body: 300 tokens at 8k leave 768 rows (50.1 MB, 256+512), 1024 tokens at 30k and 61k leave 1792 rows (117.2 MB, 256+512+1024) — 65,408 bytes per row on this model (16 full-attention layers).

### Retained capacity at ordinary check-in

| Step | Prompt tokens | Check-in samples (`requestFullAttentionUnusedArrayBytes`) |
| --- | ---: | --- |
| p8k-grow | 6995 | 7.1 MB, 16.6 MB |
| p8k-after | 7012 | 45.1 MB, 16.6 MB |
| p32k-grow | 30540 | 64.4 MB, 16.6 MB |
| p32k-after | 30557 | 112.4 MB, 16.6 MB |
| p64k-grow | 60880 | 4.9 MB, 16.6 MB |
| p64k-after | 60897 | 111.9 MB, 16.6 MB |

Two `capturingLeaf` samples per turn on the think-stripping boundary path: the live leaf checked in as the boundary's Backing Leaf, then the canonical leaf. Ordinary check-ins retain 5–17 MB (the prompt reservation's granule remainder). The `-after` turns show the lifetime the issue asks about: the rewound leaf's retained capacity is inherited by the next turn's live leaf (45.1 MB at 8k, 112.4 MB at 30k, 111.9 MB at 61k) and only the canonical leaf's copy drops it — without compaction the capacity lives as long as the leaf is checked out and returned.

### Phase-correlated memory

| Step | Prompt / cached | Peak active MLX | Peak footprint (sampled) | Settled active MLX | Settled footprint | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p8k-grow | 6995 / 0 | 17.40 GB | 19.68 GB | 16.52 GB | 18.32 GB | 42.9 |
| p8k-cancel | — / — | 16.92 GB | 18.61 GB | 16.56 GB | 17.73 GB | 13.6 |
| p8k-after | 7012 / 6993 | 17.50 GB | 19.07 GB | 16.52 GB | 18.52 GB | 3.0 |
| p32k-grow | 30540 / 7010 | 22.14 GB | 23.91 GB | 18.07 GB | 21.25 GB | 132.0 |
| p32k-cancel | — / — | 18.51 GB | 21.79 GB | 18.17 GB | 19.35 GB | 53.3 |
| p32k-after | 30557 / 30538 | 20.65 GB | 23.72 GB | 18.07 GB | 21.30 GB | 3.4 |
| p64k-grow | 60880 / 30555 | 26.50 GB | 27.98 GB | 20.05 GB | 23.30 GB | 203.8 |
| p64k-cancel | — / — | 20.51 GB | 23.76 GB | 20.15 GB | 21.34 GB | 60.1 |
| p64k-after | 60897 / 60878 | 24.63 GB | 27.96 GB | 20.05 GB | 23.36 GB | 4.5 |

Weights: 14.09 GB target + 1.04 GB DFlash2 draft (15.13 GB floor). The largest transient on this build is the boundary turn's second full-size cache at check-in (active MLX 20.35 → 24.36 GB at 61k, a +4.0 GB body-sized step during the canonical re-prefill/capture), not retained capacity; that path is [#552](https://github.com/spokvulcan/tesseract/issues/552)'s and is outside #534's scope. Cancelled requests settle back to the pre-request active MLX within the 5 s settle window.

## Decision for #534 (per the rule fixed above)

Retained capacity exceeded 64 MB at two of three sizes (117.2 MB at 30k and 61k), so compaction is implemented. No size exceeded 256 MB and none exceeded a quarter of its body, so the threshold is **64 MB**, applied as the smaller of a quarter of the body and 64 MB (the issue's shape, with 256 MB replaced by the measured value). At the measured points compaction frees 117 MB per cancelled long generation and leaves ordinary check-ins (5–17 MB) untouched.

## Not covered

No 75k/93k contexts, no capture-only-build comparison, and no tail-latency claim: the profile targets the capacity lifetime #534 needs. The 2026-09-20 boundary-leaf-move A/B and session audit remain the reference for the boundary-turn transient.
