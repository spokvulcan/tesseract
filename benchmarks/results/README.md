# Agent-benchmark results — read before comparing runs

The `bench_*.json` files in this directory are **local artifacts and are not
tracked** (see `.gitignore`). This README is tracked, because the rules for
reading them have to survive a fresh clone even though the data does not.

Written by `scripts/bench.sh` → `BenchmarkRunner`. Schema:
`tesseract/Features/Agent/Benchmark/BenchmarkReport.swift`.

## Runs that must be excluded: the pre-rotation era

**Any PARO result dated between 2026-03-17T10:41Z and 2026-03-29T10:37Z is
invalid and must not appear in any comparison or trend.**

During that window the ParoQuant rotations were pre-baked into the quantized
weights. That makes them a **no-op** — rotating the weights and then
dequantizing cancels out, so the quantization-friendliness the rotations exist
to provide is lost. The effect on the numbers is large and in the flattering
direction: **~+23% throughput**, with degraded output quality (the 9B dropped to
5/14 scenarios passed at an 11% duplicate rate).

Read at face value, these runs make late March look like a peak the project
later regressed from by ~18%. It is the reverse: the drop on 2026-03-29 is a
**correctness fix**, and every number after it is the honest one.

- Flagged by `liang2kl`, the ParoQuant author, in review of
  [ml-explore/mlx-swift-lm#164](https://github.com/ml-explore/mlx-swift-lm/pull/164):
  *"It is not appropriate to pre-rotate the weights. The purpose of the
  rotations is to make the weights more quantization-friendly. If you rotate
  the weights then dequantize them the rotation will have no effects at all."*
- Introduced in vendored `mlx-swift-lm` `37bc9c1` (2026-03-17T10:41Z),
  removed in `f77058c` (2026-03-29T10:37Z).

Affected files carry a top-level `validity` block
(`BenchmarkValidity` in `BenchmarkReport.swift`). **Filter on it:**

```bash
jq -r 'select(.validity == null) | .aggregate.avgTokPerSec' benchmarks/results/bench_*.json
```

Non-PARO models in the same window are unaffected — the bug was in the
ParoQuant path only.

Results are never deleted or edited when they turn out to have measured the
wrong thing. They are records of what the machine did. Annotate and filter.

## Attribution: `metadata.sourceRevision`

Since 2026-07-27 each report records the git revision it was built from
(`a73a7aa5`, or `a73a7aa5-dirty` if the working copy was not clean).
`bench.sh` supplies it via `--bench-source-revision`; the app cannot derive it,
because a launched `.app` has no reliable path back to the repo.

**Reports written before 2026-07-27 have no `sourceRevision`.** For those, a
before/after comparison can establish *that* something changed but never
*what* — treat any attribution to a specific PR as correlational. A
`-dirty` suffix means the tree was not reproducible; the run is evidence about
a machine state, not about a commit.

## Comparing runs at all

Three things make a naive `avgTokPerSec` comparison misleading:

1. **Thermals.** The M3 Max throttles under sustained load, and a single sweep
   is long enough to do it to itself. Runs have been observed starting at 107
   tok/s and decaying to 52 within 41 turns while another held ~101 flat
   throughout. Compare the **median of the first ~7 turns** across runs, which
   holds thermal state roughly constant, rather than the run aggregate. See
   `benchmarks/experiments-ledger.md` trap 2.
2. **Cohort.** Match on `metadata.modelName`, `metadata.sweepLabel`,
   `metadata.promptProfile` and `aggregate.totalScenarios`. The filename's hex
   suffix is the parameter hash, not a build identifier. Note that
   `Qwen3.5-4B PARO (INT4)` and `Qwen3.5-4B PARO` are the **same model** — a
   display-name-only change in `ac564c9a` (2026-04-01), same id, same repo.
3. **Sampling.** The sweep runs at temperature 1.0, so `passedScenarios` and
   `overallToolAccuracy` are noisy. Historical range on one unchanged model is
   3–7 of 14 and 62.9–91.4%. Do not read a single run's quality delta as a
   signal.

### What this benchmark cannot measure

It is an end-to-end agent benchmark: turn latency is dominated by decode and
tool execution. Work on the request-tokenization path (experiments-ledger
C24–C31) saves ~32 ms on a turn whose non-decode overhead is ~2400 ms — about
1.3%, against a run-to-run spread on that metric of roughly ±20%. Such work is
real but **structurally invisible here**; gate it with the direct instrument
(`bench.sh --tokenize-cache-bench`) instead.
