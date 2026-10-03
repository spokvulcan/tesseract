# TurboQuant KV cache on qwen3.8-27b (2026-10-02/03)

Refs [#603](https://github.com/spokvulcan/tesseract/issues/603). Related:
[#252](https://github.com/spokvulcan/tesseract/issues/252) (8-bit KV, measured
and dropped), [#259](https://github.com/spokvulcan/tesseract/issues/259) (live
KV unquantized, quantized only in storage).

## Verdict, after the decode kernel rework (2026-10-03)

**Both bars pass once the vendor's decode kernels are reworked.** The rework is
two commits on the fork branch `perf/turboquant-gqa-decode`, which this change
pins (see "Decode kernel rework" and the fork ledger,
`docs/mlx-swift-lm-fork.md`). With it, TurboQuant decodes within about 2% of
bf16 at every length (−0.8% to +2.2%), its KL is no higher, and two decodes
from one cache give the same tokens.

| Bar (#603, "to confirm") | turbo8v4 | turbo0v4 | Verdict |
| --- | --- | --- | --- |
| KL in the range the vendor calls healthy | 0.8–1.3e-3 mean, 98.8–99.8% top-1 | 0.9–1.2e-3, 98.6–99.8% | **pass** |
| At most ~10% decode loss at 32K | **+0.6%** | **−0.8%** | **pass** |

| decode vs bf16 | 8K | 32K | 64K |
| --- | ---: | ---: | ---: |
| turbo8v4, as shipped | −21% | −51% | −64% |
| turbo8v4, reworked | +0.7% | +0.6% | +2.2% |
| turbo0v4, as shipped | −25% | −56% | −68% |
| turbo0v4, reworked | +1.4% | −0.8% | +0.7% |

The memory findings below stand: the KV cache is 2.53× smaller and the run
peak doesn't move. #603's build-out list is what remains before the app can use
it: the KV scheme as a request fact and partition key, prefix-cache capture,
restore and SSD round trip, `copy()`, and DFlash2 and MTP support. DFlash2
still requires the plain cache, and in production it is worth far more than
either scheme's speed difference. Until that lands, a TurboQuant request with
DFlash2 resident would not get TurboQuant at all: the Speculation Plan refuses
only on `kvBits`, so DFlash2 would engage and decode over the unquantized cache.
The vendor change should go upstream first, per the fork rules.

## Verdict on the vendor as shipped (2026-10-02)

**No-go as the vendor ships it.** The quality bar holds and the speed bar fails
by a factor of five. Two more results matter for the build-out plan in #603:
the run peak doesn't move, and TurboQuant decode isn't reproducible run to run.

| Bar (#603, "to confirm") | turbo8v4 | turbo0v4 | Verdict |
| --- | --- | --- | --- |
| KL in the range the vendor calls healthy | 0.9–1.5e-3 mean, 98.8–99.6% top-1 | 1.0–1.6e-3, 98.6–99.8% | **pass** |
| At most ~10% decode loss at 32K | **−51%** | **−56%** | **fail** |

- **Quality.** Mean decode-time KL is about 1e-3 nats at every length. That is
  two to three times the KL of a benign change (re-chunking the prefill: 4.9–6.2e-4).
  It is 3–5× under the vendor's turbo0v4 figure (0.005 on Qwen2.5-7B) and 25–45×
  under its turbo4 figure on Mistral-7B (0.040), both of which the vendor calls
  healthy. Every teacher-forced top-1 miss is a tie or a near-tie in the reference
  (top-two margins of 0–0.375 nats for turbo8v4 and turbo0v4; 20 of the 56
  misses across all arms are exact ties).
- **Speed.** Decode loses 21% at 8K, 51% at 32K and 64% at 64K for turbo8v4,
  and a little more for turbo0v4. The loss grows with context. The cost sits in
  the vendor's decode path, not in the scheme: the cache is smaller, but every
  step copies all of it and decodes each block once per query head. See "Why
  decode is slow".
- **Memory.** The KV cache is 2.53× smaller (25,856 against 65,536 bytes per
  token, exactly the layout's arithmetic), so a 64K cache holds 1.87 GB instead
  of 4.50 GB. Neither peak falls: the run peak is the bf16 prefill's (17.1 /
  21.1 / 26.5 GB in every arm), as #252 found for 8-bit KV, and the conversion
  briefly holds the bf16 and raw copies together, so turbo8v4's decode-phase
  peak ends 0.38 GB above bf16's at 64K.
  The win is the steady-state cache, which is the prefix cache's RAM tier
  (#603's motivation), not a request's peak.
- **Reproducibility.** Two decodes from the same restored cache disagree for
  every TurboQuant arm at every length (9 of 9 pairs); bf16 and affine8 never
  do (6 of 6). The cause is a data race in the vendor's value-encode kernel,
  confirmed by a patched rerun in which every pair reproduces (12 of 12; see
  "Non-determinism"). Until that fix lands, #603's gate "a restored saved state
  continues token-for-token like the same state continued live" cannot pass.

Worth reopening when the vendor's TurboQuant decode kernels are fixed. The
four changes under "What it would take" are upstreamable, and by the code the
three decode changes should bring decode back near bf16. (They did; see the
verdict above.)

## Setup

- MacBook Pro 16-inch, Apple M3 Max (16-core CPU, 40-core GPU), 48 GB, macOS
  27.0 (26A428). Release build, launched with `open` (nice 0), app bootstrap
  skipped so no other model shares the process.
- `qwen3.8-27b` (`mlx-community_Qwen3.8-27B-4bit`, affine 4-bit weights, bf16
  activations). 16 of 64 layers carry a KV cache (4 KV heads × 256, 24 query
  heads). The unquantized baseline is **bf16**, the checkpoint's dtype, not
  fp16; the bytes are the same.
- Plain decoding: drafters not loaded, no prefix cache, greedy, no penalties.
  Thinking on (the template's default, effort `xhigh`).
- Prompt: the 80 files of `docs/adr/` joined in name order (600,782 bytes,
  SHA-256 `d23190e6…a8888`), cut to the target length, then a question that
  asks for every record in order. The thinking it produces walks the records,
  so decode attends across the whole prompt. Prompts are nested prefixes of one
  text: 8,193, 32,769 and 65,537 tokens.
- Arms: `fp16` (the reference), `turbo8v4`, `turbo0v4`, and for context
  `turbo8v3` (the vendor's recommended preset) and `affine8` (the scheme #252
  dropped). Every TurboQuant arm converts all 16 attention layers, which the
  harness checks layer by layer; boundary-layer protection does not engage for
  these schemes.
- Source: `d67ab03a` plus the uncommitted harness (`TurboQuantBenchRunner.swift`
  SHA-256 `030c74ef…3e70`), vendor `mlx-swift-lm` at `7d8e38e`.

Harness: `--turboquant-bench` (`tesseract/Features/Agent/Benchmark/TurboQuantBenchRunner.swift`).
Per context it prefills once on the unquantized cache. Prefill is bf16 in
every scheme, since the vendor converts the cache after the last prompt token.
It snapshots that cache and restores a copy for every pass:

- **Quality pass, per arm.** It forwards the last prompt token on bf16,
  converts the cache with the vendor's own `applyKVCacheConfiguration`, then
  decodes 512 tokens one forward at a time, as decode runs. The bf16 arm records
  its greedy stream and log probabilities; every other arm is forced along that
  stream and scored by KL(bf16 ‖ arm) and argmax agreement. Step 0 comes from
  the bf16 cache in every arm, so its KL must be exactly 0, and it was. This is
  the negative control #252 needed.
- **Noise floor.** It re-prefills at chunk 768 instead of 1024 and scores bf16
  against the reference the same way.
- **Speed pass, per arm and round.** It runs the chunked Prefill Strategy route
  (`PrefillExecutor.makeIterator`, the vendor `TokenIterator` converting from
  the `kvCache` parameter) for 512 tokens. Steady-state tok/s counts from the
  second token on; the one-time switch-over is reported apart. Two rounds,
  reversed (ABBA). The bf16 rounds must reproduce the reference stream, and they
  did.

```bash
scripts/bench.sh quick --model qwen3.8-27b --turboquant-bench \
  --bench-corpus "$PWD/docs/adr" --bench-contexts 8192,32768,65536 \
  --bench-max-new 512 --bench-runs 2 --bench-noise-floor-step 768 \
  --bench-schemes fp16,turbo8v4,turbo0v4,turbo8v3,affine8
scripts/turboquant_summary.py benchmarks/turboquant/2026-10-02/sweep.json
```

Raw report: [`sweep.json`](sweep.json) (every per-step KL, every token stream,
the reference text); log: [`sweep.log`](sweep.log). The run took 57 minutes;
harness checks 18/18.

## Results

### Quality

KL(bf16 ‖ arm) in nats over decode steps 1–511. Top-1 is the share of those
steps where the arm's argmax is the reference token; the parenthetical counts
the misses. The last column is where each speed round's free-running greedy
stream first leaves the reference. One flipped near-tie is enough to send a
thinking stream elsewhere (affine8 leaves at 38 at 64K, the noise floor's first
miss there is at step 49), so the teacher-forced columns are the measure.

| context | arm | KL mean | KL median | KL p99 | KL max | top-1 | free-running stream leaves the reference at |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8,193 | turbo8v4 | 1.53e-03 | 3.36e-05 | 1.71e-02 | 2.40e-02 | 99.6% (2/511) | 15, 448 |
| 8,193 | turbo0v4 | 1.59e-03 | 4.69e-05 | 1.72e-02 | 2.00e-02 | 99.8% (1/511) | 341, 448 |
| 8,193 | turbo8v3 | 3.66e-03 | 1.53e-04 | 3.07e-02 | 5.76e-02 | 98.8% (6/511) | 15 |
| 8,193 | affine8 | 5.58e-04 | 7.61e-06 | 5.63e-03 | 1.16e-02 | 100.0% (0/511) | never |
| 8,193 | noise floor | 6.24e-04 | 7.67e-06 | 6.79e-03 | 8.07e-03 | 100.0% (0/511) | – |
| 32,769 | turbo8v4 | 8.97e-04 | 1.15e-05 | 9.49e-03 | 1.37e-02 | 99.4% (3/511) | 274, 94 |
| 32,769 | turbo0v4 | 9.88e-04 | 1.31e-05 | 1.00e-02 | 1.75e-02 | 98.8% (6/511) | 263, 94 |
| 32,769 | turbo8v3 | 2.69e-03 | 3.71e-05 | 2.62e-02 | 4.36e-02 | 99.2% (4/511) | 38, 94 |
| 32,769 | affine8 | 5.10e-04 | 3.68e-06 | 6.16e-03 | 8.83e-03 | 99.8% (1/511) | 263 |
| 32,769 | noise floor | 4.90e-04 | 2.62e-06 | 6.23e-03 | 1.67e-02 | 99.8% (1/511) | – |
| 65,537 | turbo8v4 | 1.36e-03 | 3.35e-05 | 9.95e-03 | 6.36e-02 | 98.8% (6/511) | 14, 38 |
| 65,537 | turbo0v4 | 1.48e-03 | 3.50e-05 | 1.41e-02 | 6.53e-02 | 98.6% (7/511) | 17, 49 |
| 65,537 | turbo8v3 | 3.53e-03 | 1.36e-04 | 3.52e-02 | 4.70e-02 | 98.2% (9/511) | 7 |
| 65,537 | affine8 | 5.52e-04 | 6.04e-06 | 4.90e-03 | 7.92e-03 | 98.8% (6/511) | 38 |
| 65,537 | noise floor | 5.41e-04 | 6.92e-06 | 5.14e-03 | 1.01e-02 | 99.2% (4/511) | – |

- The value bits set the error. turbo0v4 keeps bf16 keys and is no better than
  turbo8v4, and dropping to 3-bit values (turbo8v3) more than doubles the KL.
  8-bit keys cost about what re-chunking the prefill costs (affine8 sits on the
  noise floor).
- The error does not grow with context: 8K, 32K and 64K are within 1.7× of
  each other in every arm.
- These are the kernels as shipped, race included (below). With the race fixed
  the TurboQuant KL is 2–23% lower; see "Non-determinism".

### Decode speed

Median of the two rounds. The rounds agree to within 0.1 tok/s at 32K, and at
64K for bf16, turbo8v4 and turbo0v4. The noisy pairs are 8K bf16 (20.47 / 22.06),
8K turbo8v4 (16.36 / 17.18) and 64K affine8 (13.57 / 12.76); none of them
changes a verdict. The per-token column is the inverse.

| context | arm | tok/s | vs bf16 | ms/token | switch-over ms (first token) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8,193 | bf16 | 21.27 | – | 47.0 | 81 (37) |
| 8,193 | turbo8v4 | 16.77 | −21.1% | 59.6 | 145 (87) |
| 8,193 | turbo0v4 | 15.85 | −25.5% | 63.1 | 132 (70) |
| 8,193 | turbo8v3 | 16.69 | −21.5% | 59.9 | 144 (87) |
| 8,193 | affine8 | 20.76 | −2.4% | 48.2 | 93 (53) |
| 32,769 | bf16 | 19.60 | – | 51.0 | 96 (45) |
| 32,769 | turbo8v4 | 9.60 | **−51.0%** | 104.2 | 370 (266) |
| 32,769 | turbo0v4 | 8.70 | **−55.6%** | 115.0 | 302 (194) |
| 32,769 | turbo8v3 | 9.33 | −52.4% | 107.2 | 374 (267) |
| 32,769 | affine8 | 17.26 | −11.9% | 57.9 | 147 (98) |
| 65,537 | bf16 | 16.50 | – | 60.6 | 118 (52) |
| 65,537 | turbo8v4 | 5.97 | −63.8% | 167.4 | 648 (482) |
| 65,537 | turbo0v4 | 5.31 | −67.8% | 188.2 | 597 (420) |
| 65,537 | turbo8v3 | 5.68 | −65.6% | 176.1 | 657 (485) |
| 65,537 | affine8 | 13.17 | −20.2% | 76.0 | 268 (188) |

The switch-over is the one-time conversion plus the first step on the
converted cache. TurboQuant compresses the whole prefill there: about 0.43 s at
64K on top of the bf16 arm's first token. affine8 reproduces #252's shape on
this model (−2% at 8K, −12% at 32K, −20% at 64K).

### Memory

| context | arm | KV bytes/token | vs bf16 | KV cache after decode GB | decode-phase peak GB | run peak GB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8,193 | bf16 | 65,536 | 1.00× | 0.74 | 16.04 | 17.10 |
| 8,193 | turbo8v4 | 25,856 | 2.53× | 0.39 | 15.99 | 17.10 |
| 8,193 | turbo0v4 | 41,216 | 1.59× | 0.52 | 15.99 | 17.10 |
| 32,769 | bf16 | 65,536 | 1.00× | 2.35 | 17.88 | 21.10 |
| 32,769 | turbo8v4 | 25,856 | 2.53× | 1.02 | 17.96 | 21.10 |
| 32,769 | turbo0v4 | 41,216 | 1.59× | 1.54 | 17.76 | 21.10 |
| 65,537 | bf16 | 65,536 | 1.00× | 4.50 | 20.24 | 26.47 |
| 65,537 | turbo8v4 | 25,856 | 2.53× | 1.87 | 20.62 | 26.47 |
| 65,537 | turbo0v4 | 41,216 | 1.59× | 2.89 | 20.21 | 26.47 |

KV bytes per token are the attention layers' live state over their offset,
read from the realized cache: 16 layers × 4 heads × (1024 B bf16, 404 B
turbo8v4, 644 B turbo0v4). "KV cache after decode" is the process's resident
growth over the pass, including step padding and the fixed 154 MB of
GatedDeltaNet state. The run peak is the shared prefill's. The decode-phase
peak is the pass's peak less the resident snapshot. The same is true of
turbo8v3 (23,808 B/token) and affine8 (34,816); see the raw report.

## Why decode is slow

TurboQuant's cache is smaller than bf16's, but its decode path does more work
per step, and the excess grows with context. Excess over bf16 per token:
12.6 ms at 8K, 53 ms at 32K and 107 ms at 64K for turbo8v4, about 1.65 µs per
token of context. By reading the vendor
code (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/TurboQuantKernels.swift`,
`TurboQuantKVCache.swift`, MLX's custom-kernel input handling), ranked by
estimated cost:

1. **The pass-1 flash kernel** (`turboFlashPass1AffineKSource`,
   `turboFlashPass1RawKSource`). Its grid has one simdgroup per *query* head,
   so each of the 6 query heads sharing a KV head decodes the same K/V block
   again: 6× the decode work and the cache reads. Lanes stride the head
   dimension (`d = lane + 32·i`), so every token takes 8 scalar loads per array
   per lane instead of vector loads. The value-codebook lookup indexes a
   per-thread array. The vendor already has the fix shape: the symmetric
   schemes' NR0 kernels decode once for several query rows. The raw-K and
   affine-K kernels lack it. This is most of the excess.
2. **A copy of the whole compressed cache on every step.** The kernels are
   built with `ensureRowContiguous: true` and handed `buf[..., ..<T, ...]`
   slices of buffers allocated in 256-token steps. With 4 KV heads such a slice
   is not row-contiguous, so MLX copies every input of every attention layer
   on every step. That is about 1.7 GB per token at 64K for turbo8v4 (2.7 GB
   for turbo0v4), roughly 11 and 18 ms. bf16 attention takes the strided
   slice as is.
3. **Pass 2 merges T/64 partial results serially** in one simdgroup per query
   head (1,024 dependent steps at 64K), then un-rotates the values with a dense
   256×256 matrix-vector product per head.
4. **The compiled whole-step decode schedule is lost** for any quantized
   cache. affine8 loses it too and is only 2.4% slower at 8K, so it costs at
   most that.

The per-token encode of the new K/V is small and does not grow with context.

## Non-determinism

The value encoder (`fusedEncodeWHT`, `TurboQuantKernels.swift`) runs its
Walsh-Hadamard butterfly in threadgroup memory. From stage 5 on, a thread's
partner is in another simdgroup, and each thread overwrote its own slot
without a barrier between reading both slots and writing. Its partner may then
read the new value instead of the old one. The encoder runs on the whole
prefill at compression and on every decoded token, so the stored values can
differ between runs from the same input.

[`wht-encode-race-fix.patch`](wht-encode-race-fix.patch) reads both slots,
places a barrier, then writes. A rerun with the patch applied (8K and 32K, three
rounds per arm, no noise-floor control, otherwise the same protocol;
[`race-fix.json`](race-fix.json), [`race-fix.log`](race-fix.log)) confirms it is
the cause:

| | unpatched sweep | patched |
| --- | --- | --- |
| TurboQuant rounds that reproduce each other | 0 of 9 pairs | **12 of 12 pairs** |
| bf16 and affine8 rounds that reproduce each other | 6 of 6 | 6 of 6 (bf16) |
| turbo8v4 KL mean, 8K / 32K | 1.53e-3 / 8.97e-4 | 1.23e-3 / 8.75e-4 |
| turbo0v4 KL mean, 8K / 32K | 1.59e-3 / 9.88e-4 | 1.23e-3 / 8.91e-4 |
| turbo8v4 decode vs bf16 at 32K | −51.0% | −51.0% |

The race costs a little quality: one scored pass each, so treat the KL drop
as direction, not size. It costs no speed. The fix is part of the decode kernel
rework below, with a vendor test (`testFusedEncodeWHTIsDeterministic`) that
fails on every one of its 19 repeat encodes without the barrier and passes
with it.

## Decode kernel rework (2026-10-03)

The vendor changes, two commits on the fork branch `perf/turboquant-gqa-decode`,
pinned by this change (the encode race fix, then the rest). The runs below used
the working tree that became them; later edits touched comments, tests and an
unused tuning parameter only:

- **One decode of each block for several query heads.** A new pass-1 kernel
  (`turboFlashGQAPass1RawKSource`, `turboFlashGQAPass1AffineKSource` in
  `TurboQuantKernels.swift`) runs one threadgroup per (KV head, group of query
  heads, token block). Each token's K and V are read and decoded once per group
  and scored against all of the group's heads. Each lane owns 8 contiguous
  dimensions, so its K slice is one contiguous load and its 4-bit values one
  packed word. The codebook sits in threadgroup memory, and the softmax skips
  the rescale on the tokens where the running maximum doesn't move.
- **Three heads per group, not six.** Holding all six heads' queries and
  accumulators in one lane costs more in registers than decoding the block a
  second time. At 64K, one group of six takes about 1.85× as long as two
  groups of three for turbo8v4 (1.5× for turbo0v4).
- **No per-step copy.** The kernel reads the whole step-padded buffers in place,
  by their row strides, so MLX no longer copies the cache on every step.
- **A parallel merge.** Pass 2 runs one threadgroup per query head: one thread
  weighs each block, one thread merges each dimension, and the threadgroup
  applies the inverse value rotation together. The serial merge and the
  one-simdgroup dense rotation are gone.
- **The encode race fix** (above).
- **The compiled decode schedule.** Qwen3.5 now runs TurboQuant decode through
  its compiled segments (`supportsUntracedDecodeAttention` in
  `AttentionUtils.swift`). Attention already ran there untraced, between
  segments, through the same `compressedAttention` route.
- Shapes the kernel does not serve fall back to the per-query-head kernels:
  `dim / 32 · valueBits > 32`, a key group that splits a lane, or a repeat
  above 8. Setting `TURBO_FLASH_GQA=0` forces that fallback for A/B runs.

Kernel time, one decode step through 16 attention layers (microbench,
[`microbench.txt`](microbench.txt); defaults: 3 heads per group, 2 simdgroups,
at most 64 blocks). The reworked figures are the file's `hps=3 nsg=2
maxBlocks=64` rows. Its unlabelled `turbo8v4 GQA` / `turbo0v4 GQA` and
`ratio vs SDPA` lines were timed under the earlier defaults (six heads per
group, 4 simdgroups, 128 blocks):

| context | bf16 SDPA | turbo8v4 as shipped | turbo8v4 reworked | turbo0v4 reworked |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 2.55 ms | 9.6 | 2.3 | 2.0 |
| 32K | 7.6 | 34.8 | 6.7 | 6.0 |
| 64K | 13.8 | 71.1 | 12.7 | 11.7 |

The whole per-layer step (append the token, then attend), 16 layers: bf16
2.6 / 8.1 / 15.0 ms at 8K / 32K / 64K; turbo8v4 2.9 / 7.7 / 13.6; turbo0v4
2.4 / 6.8 / 12.3.

End to end on qwen3.8-27b, the sweep's protocol for bf16, turbo8v4 and turbo0v4
without the noise-floor control (two ABBA rounds, 512
tokens; [`gqa-kernel.json`](gqa-kernel.json), [`gqa-kernel.log`](gqa-kernel.log);
harness checks 12/12):

| context | arm | decode tok/s | vs bf16 | rounds | KL mean | top-1 | rounds reproduce |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 8,193 | bf16 | 19.34 | – | 18.61 / 20.07 | – | – | yes |
| 8,193 | turbo8v4 | 19.48 | +0.7% | 18.91 / 20.05 | 1.29e-3 | 99.8% | yes |
| 8,193 | turbo0v4 | 19.61 | +1.4% | 19.01 / 20.21 | 1.24e-3 | 99.8% | yes |
| 32,769 | bf16 | 19.00 | – | 18.68 / 19.32 | – | – | yes |
| 32,769 | turbo8v4 | 19.12 | +0.6% | 18.99 / 19.24 | 8.22e-4 | 99.2% | yes |
| 32,769 | turbo0v4 | 18.85 | −0.8% | 18.70 / 18.99 | 9.37e-4 | 99.6% | yes |
| 65,537 | bf16 | 16.54 | – | 16.54 / 16.55 | – | – | yes |
| 65,537 | turbo8v4 | 16.91 | +2.2% | 16.89 / 16.94 | 1.19e-3 | 98.8% | yes |
| 65,537 | turbo0v4 | 16.67 | +0.7% | 16.93 / 16.40 | 1.22e-3 | 98.6% | yes |

Notes on this run:

- **The machine was loaded.** It was swapping: 4.6 of 5 GB of swap in use,
  the window server busy. Read the arms against each other within a round. The
  absolute bf16 tok/s here are 9% under the morning sweep at 8K, 3% under at
  32K and level at 64K.
- **One-time costs that allocate are inflated.** The switch-over at 64K read
  1.9 s for both TurboQuant arms here (first token 1.7 s), against 0.6 s
  (first token 0.42–0.48 s) in the morning sweep. bf16's own prime step was
  3.5× slower in this run as well, and the cache-level microbench (the
  `switch-over` lines of `microbench.txt`) puts the compression of 16 layers at
  64K at about 150 ms. Re-measure the switch-over on a quiet machine before
  quoting it.
- **The as-shipped column isn't in this run.** It comes from the sweep above,
  same machine and protocol.

Vendor tests: the new `TurboQuantGQAFlashTests` cover these cases:

- the GQA kernel against the per-query-head kernel, within 1e-4, for bf16 and
  8-bit affine keys, every head split and nine lengths up to 20,000 tokens,
  from step-padded buffers;
- decode through the cache across a buffer growth, against exact attention;
- TurboQuant on Qwen3.5's compiled segments, against the unquantized cache;
- encode determinism.

These suites pass with the change: `TurboQuantIntegrationTests`,
`TurboFlashAttention`, `TurboQuantKVCache`, `KV-cache configuration`,
`CompiledDecodeWeightUpdateTests` and `Qwen35CompiledDecodeLifecycleTests`
(35 XCTest and 44 Swift Testing cases). The microbench is opt-in:

```bash
cd Vendor/mlx-swift-lm
TEST_RUNNER_TURBOQUANT_DECODE_BENCH=1 xcodebuild test -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' -skipPackagePluginValidation \
  -configuration Release ENABLE_TESTABILITY=YES \
  -only-testing:MLXLMTests/TurboQuantDecodeMicrobench
```

### Draft upstream PR description

For the owner to read, edit and post (the vendor's PR template asks for the
checklist and AI-usage disclosure, which are the owner's to write):

```markdown
## Proposed changes

TurboQuant's raw-K and affine-K decode (`turbo0vN`, `turbo8vN`) ran slower than bf16 attention and lost more with context: turbo8v4 decoded 21% slower at 8K, 51% at 32K and 64% at 64K on a 27B model with 24 query heads over 4 KV heads (head dim 256), although its cache is 1.6–2.5× smaller.

Three things cost the time. The pass-1 kernel ran one simdgroup per query head, so every head sharing a KV head decoded the same block again. It received `..<T` slices of 256-row step-padded buffers, which are not row-contiguous with several KV heads, so MLX copied the cache on every step. And pass 2 merged the blocks serially in one simdgroup per head.

This adds a GQA decode kernel for those modes. A threadgroup decodes each block once for a group of query heads (three by default; six heads' state in one lane costs more in registers than a second decode). Lanes own contiguous dimensions, the buffers are read in place by row stride, and pass 2 merges and un-rotates in parallel. Shapes it does not serve keep the existing kernels, and `TURBO_FLASH_GQA=0` forces them.

It also fixes a data race in the fused WHT value encoder (cross-simdgroup butterfly stages wrote a slot their partner had not read yet), which made the stored codes, and so greedy decoding, differ between runs from the same input. And it lets Qwen 3.5 decode TurboQuant through its compiled segments, where attention already runs untraced.

Kernel time for one decode step through 16 attention layers (M3 Max):

| tokens | bf16 SDPA | turbo8v4 before | turbo8v4 after | turbo0v4 after |
|---|---|---|---|---|
| 8K | 2.55 ms | 9.6 | 2.3 | 2.0 |
| 32K | 7.6 | 34.8 | 6.7 | 6.0 |
| 64K | 13.8 | 71.1 | 12.7 | 11.7 |

End to end, decode against bf16 KV: turbo8v4 +0.7% / +0.6% / +2.2% at 8K / 32K / 64K (was −21% / −51% / −64%), with unchanged decode-time KL.

Tests: `TurboQuantGQAFlashTests` (kernel parity with the per-head kernel, cache decode against exact attention, compiled-segment decode, encode determinism) and an opt-in microbenchmark, `TurboQuantDecodeMicrobench`.
```

## What remains

The vendor change is carried on the pin (fork ledger, "TurboQuant GQA decode")
and goes upstream next, per `docs/mlx-swift-lm-fork.md`. The draft above covers
both commits; drop its race-fix paragraph if the fix is filed on its own. Then
the rest of #603's list:

- the KV scheme as a request fact and part of the prefix cache's partition key,
  and a Speculation Plan that refuses on that scheme, not only on `kvBits`;
- TurboQuant in DFlash2's verify and commit and in Qwen3.5's verify pass;
- `copy()`;
- capture, restore and SSD round trip.

Re-measure with this harness before any of it lands in the app.
Two vendor items are left on the table: an in-kernel Walsh-Hadamard transform
in place of the dense inverse rotation, and an incremental compression during
prefill, which would remove the one-time switch-over.

## Notes on the record

- The issue's 64K compaction figure: the code default is 80,000 tokens
  (`SettingsCatalogue.swift`), as `docs/model-parameters.md` says. The 64K row
  here covers the issue's ceiling either way.
- A free-running greedy stream is a weak quality measure on a thinking turn:
  one flipped tie sends it elsewhere (at 64K, affine8 leaves the reference at
  38), and the TurboQuant arms' streams also diverge from each other (the
  race). The teacher-forced columns carry the quality verdict.
- At 2K (the smoke run) the noise floor was exactly 0: this dense stack is
  chunk-invariant there. From 8K on, re-chunking costs about 5e-4 nats.
- The runs carried the uncommitted harness on `d67ab03a` (SHA-256 above). The
  committed file differs from it only in comments, formatting and two
  error-case names.
