# Bounded production parity and projection lifetime

See the [report](../../../docs/research/2026-09-13-cache-parity-and-projection-lifetime.md)
for exact assertions, loader diagnosis, lifetime hypotheses and limits.

| Capture | Outcome |
| --- | --- |
| `production/` | Stopped at critical pressure during unintended MTP loading, before parity checks; preserves the original bare-engine loader. |
| `production-dflash2/` | Corrected production-equivalent target/DFlash2 load; all 18 exact-byte checks passed in 45.72 seconds. All 180 OS samples normal; 23.46 GiB maximum sampled footprint and zero additional system swap. |

Each capture includes its actual runner/source patch, environment, binary hash,
complete model-file hashes, OS samples, selected scalar diagnostics and outcome.
The passing capture also includes the correctness report and log. Its temporary
full/extension SSD files were flushed and removed. Inputs are the deterministic
2K benchmark fixture; no private conversation content was used.

`validation/` preserves complete compressed test logs, readable summaries, exact
test commands and raw-log hashes, derived projection measurements, both Release
build summaries and final source provenance. Three byte-comparison tests passed.
The projection experiment failed its bound when both arms used the production
traversal, then passed with the incremental visitor. The final opt-in test also
passed alone. These add four distinct test functions to the preceding 107.

The projection result is a **small-fixture prototype**, with production traversal
unchanged: eight 4-bit MLP blocks, 5 MiB of folded projections, 5 MiB versus
0.625 MiB peak active memory above the final value, and identical output bytes.
The report explains entry-counter variation and the remaining loaded-model gate.

Verify the archive from its directory:

```bash
shasum -a 256 -c SHA256SUMS
```

The following command prints the current plan unless `--run` is included.
Run from the repository root with the ordinary app closed, a Release binary
and a new output directory; restore the ordinary app afterward:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/bounded_cache_parity.py \
  --app '/absolute/path/to/Tesseract Agent.app/Contents/MacOS/Tesseract Agent' \
  --output /absolute/path/to/new-evidence --run
```

Fixed bounds: 2,048 prompt tokens, leaf at 1,024, one final sentinel, unquantized
KV, 250 ms sampling, 32 GiB footprint, 2 GiB additional system swap, critical or
unknown pressure stop, ten-minute deadline and one child process. Starting
pressure must be normal. These sampled abort conditions do not guarantee a
peak ceiling. The gate loads DFlash2 but does not run speculative decoding;
the earlier HTTP evidence covers that separate path. It does not run the full
16K correctness matrix or the 45k/75k/93k acceptance campaign.

The lifetime fixture must run alone with the opt-in environment flag described
in [testing documentation](../../../docs/testing.md). Its global allocator
counter must not be compared while other model or test work runs concurrently.
