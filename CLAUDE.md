# CLAUDE.md

## What this is — and why

Tesseract Agent — a fully offline AI assistant for macOS (macOS 26+,
Swift 6.2 / SwiftUI), everything running locally on Apple Silicon via
MLX.

## Docs (read before touching the area)

- Architecture → `ARCHITECTURE.md`
- Tests & suites → `docs/testing.md`
- Model numbers (context, output length, sampling) → `docs/model-parameters.md`
- Decisions & domain → `CONTEXT.md`, `docs/adr/`

## Experiments and benchmarks: keep the loop tight

Many cheap, trustworthy measurements per hour beat one slow end-to-end run.

- **Smallest harness first.** Measure a kernel or cache change in a vendor
  microbench (e.g. `TurboQuantDecodeMicrobench`: synthetic K/V at any
  context, seconds per run). Load the real model only to confirm the
  finished change.
- **Prefill once.** Loaded-model prefill dominates wall time (PARO: about
  35 s for 8K tokens, per arm). Capture the prefilled cache once and
  restore it per arm, as `--turboquant-bench` does with
  `HybridCacheSnapshot`. `--dflash2-bench` re-prefills every arm and
  `--bench-check` adds a whole AR arm: run only the arms the question
  needs, on the shortest prompt that shows the effect.
- **Build once, run many.** App tests: `scripts/test.sh [--no-build]
  [suite…]`. It runs the built `.xctestrun`, skipping the 7–10 s project
  load of every `-scheme` call (a suite takes ~3 s), and prints failures
  with their `#expect` details. Iterate with `--no-build`; run the whole
  target once, before committing. The vendor fork's tests:
  `scripts/vendor-test.sh [--no-build] [suite…]`. Read a script's source
  before passing it flags: `scripts/bench.sh` builds and runs on any
  argument.
- **A/B in one build.** Put both variants behind a temporary env switch,
  alternate them (ABAB, at least four runs), compare medians, and keep a
  reference arm (bf16 SDPA) in every run to catch GPU clock drift. Runs from
  different builds or hours drift 10–20% on this machine.
  `MLX_KERNEL_PROFILE` serializes the GPU: read it for relative per-kernel
  cost only.
- **Quiet GPU.** Check `ps -Ao pcpu,comm -r | head` before measuring;
  animated wallpapers, video and other GPU apps skew results.
- **Watch long runs.** Run anything over two minutes in the background with
  a timeout, note its expected duration, poll its output, and kill it when it
  overruns. A subagent with no new tool call for five minutes is stuck: stop
  it and do the work directly. The app's logs:
  `scripts/dev.sh log-show [minutes] [pattern]`, which returns at once.
- **Failures in unrelated suites:** run the same suite on the base commit
  and diff the failure lists before chasing them.

## Agent skills

### Issue tracker

Issues and PRDs live as GitHub issues in `spokvulcan/tesseract` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
