# ADR-0083: A TurboQuant KV Scheme is a request fact; DFlash2 verifies over it and the prefix cache stores it

- Status: Accepted (built; loaded-model gates in As built)
- Date: 2026-10-03
- Relates to: #603 (the measurement), ADR-0059 and ADR-0079 (the Speculation
  Plan and its KV gate), ADR-0064 (Leaf Handoff), ADR-0068 (Stored Form),
  ADR-0070 (Request Keying), ADR-0078 (Leaf Admission)

## Context

#603 measured TurboQuant KV on Qwen3.8-27B: 4-bit values in a rotated
codebook with 8-bit affine keys (`turbo8v4`) or bf16 keys (`turbo0v4`). With
the vendor's GQA decode kernel the cache is 2.53x or 1.59x smaller than bf16,
decodes within about 2% of bf16 at 8K to 64K, and stays inside the KL bar
(`benchmarks/turboquant/2026-10-02/README.md`). Nothing in the product could
use it:

- Only `kvBits` (affine) was a request fact. The Speculation Plan refused
  speculation on `kvBits` alone, so a TurboQuant `kvCache` would have engaged
  DFlash2 over the unquantized cache.
- DFlash2 builds round N+1 before round N commits: a verify pass writes its
  rows at a lazy position past the committed offset, and the next pass reads
  them. `TurboQuantKVCache` keyed every write, growth, compression and read
  to its host offset, so it could not hold those rows.
- The prefix cache knew no TurboQuant class, so capture returned nil and
  every leaf of such a turn was lost.

## Decision

1. **The KV Scheme is a request fact.** `AgentGenerateParameters.kvScheme`
   comes from the KV Cache Compression setting, only for a model the schemes
   were measured on (`KVScheme.supports(modelID:)`). It rides
   `GenerateParameters.kvScheme` and enters the cache partition key, the
   partition digest (a tagged field appended only when set, so existing
   partitions keep their directory names) and the SSD partition meta. It is
   the partition's Stored Form.
2. **Prefill stays unquantized; the cache converts once after prefill.** The
   keyed standard path converts in `quantizeKVCache` before its iterator. A
   DFlash2 turn passes the scheme to the vendor iterator, which converts just
   before the final prompt position (its own capture prefill must run
   first) and exposes the converted array; the Server Completion adopts it,
   and a Leaf Handoff's owner adopts it too, so the leaf the turn stores is
   the array the iterator advanced.
3. **DFlash2 verifies over TurboQuant.** The vendor cache gains positioned
   rows: a verify pass encodes its rows and writes them at the lazy position
   into the compressed buffers, growth keeps every row, and `commitRows`
   moves the offset. Attention runs on a multi-query kernel shaped like
   mlx's GQA-packed MMA verify kernel, dequantizing each 32-key block once
   into threadgroup memory under a device-side position mask. Qwen 3.5's
   verify and the iterator's commit go through one `DFlash2AttentionCache`
   protocol that `KVCacheSimple` and `TurboQuantKVCache` both implement. The
   Speculation Plan keeps DFlash2 for a scheme and refuses MTP (its head and
   the vendor MTP round staging were not built for it).
4. **The prefix cache stores TurboQuant layers as sliceable attention.** The
   vendor's state gives per-row norms a trailing unit axis, so every
   token-indexed array slices on axis -2 like a plain layer; the key group
   size joins the metaState. Capture, restore (with the state count checked
   against the key mode), prefix views, SSD segments, Leaf Handoff and Leaf
   Rewind (a trim) all treat it like a plain attention layer. Warm Body
   compression and the drain's opportunistic compression skip TurboQuant
   bodies and partitions.

## Consequences

- Every body a scheme partition keeps is in its Stored Form. `capture`
  compresses a full-precision attention layer's copy (mid-prefill
  checkpoints are taken before the live cache converts) and records a
  prefix view under the TurboQuant class, so a branch point captured during
  the full-precision prefill restores from the turn's compressed leaf. A
  cache a leaf takes by move is converted and compressed first. SSD segment
  chains therefore join only matching layers.
- A warm turn in a scheme prefills its suffix against the compressed prefix,
  so a warm turn and a cold one need not produce byte-identical greedy text;
  their difference is TurboQuant's own error. In the e2e runs below the
  `turbo8v4` warm and cold texts matched in full and the `turbo0v4` ones for
  their first 27 characters.
- Changing the setting moves requests to another partition; the old one ages
  out through stale-partition GC.
- The affine-key verify kernel costs about 5 ms more per DFlash2 round than
  bf16 SDPA at 32K (the raw-key one is at parity); that is the first item for
  the optimization loop.

## As built

Measured on Qwen3.8-27B 4-bit, Apple M3 Max 48 GB, greedy, 256 tokens,
`scripts/dflash2-bench.sh --bench-kv-scheme` (the AR arm converts through its
plan, DFlash2 inside its prefill):

| prompt | arm | bf16 | turbo8v4 | turbo0v4 |
|---|---|---|---|---|
| 7.4K | AR tok/s | 21.3 | 21.3 | 20.6 |
| 7.4K | DFlash2 ms/round (acceptance) | 76.7 (34.2%) | 77.2 (32.4%) | 77.9 (33.8%) |
| 29K | AR tok/s | 17.1 | 17.2 | 13.7 (one run, see below) |
| 29K | DFlash2 ms/round (acceptance) | 83.9 (31.4%) | 89.0 (31.4%) | 75.7 (31.4%) |

Single runs, not ABBA; the 29K turbo0v4 AR figure ran third on a warm
machine and disagrees with #603's ABBA measurement (within 1% of bf16), so
read it as noise until re-measured. At 7.4K the 16 attention layers hold
192 MB under turbo8v4 and 306 MB under turbo0v4 against 484 MB at bf16.
DFlash2's stream follows the AR stream of the same scheme for 30 tokens at
7.4K (bf16: all 256) and 54 at 29K (bf16: also 54): the verify kernel stages
dequantized K and V in the activation dtype, the decode kernel works in f32.

`TESSERACT_E2E_KV_SCHEME=turbo8v4` and `=turbo0v4 scripts/dev.sh
prefix-cache-e2e --bench-model-id qwen3.8-27b` both pass every check (35),
DFlash2 resident and engaging on the text turns (27–62% acceptance): stable
prefix and leaf hits, Leaf Handoff, branch-point views and their survival
under a budget cut, the SSD restart, demotion and hydration, and the image
scenarios. No plain full-attention layer remained after prefill.
