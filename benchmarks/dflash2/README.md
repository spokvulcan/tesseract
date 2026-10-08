# DFlash2 experiment loop

Run one DFlash pass instead of the full six-pass app benchmark:

```sh
scripts/dflash2-bench.sh \
  --bench-prompt-file benchmarks/dflash2/travel.txt \
  --bench-json /tmp/dflash-before.json
```

The fast wrapper defaults to the frozen short travel fixture (82 input
tokens). Use `--bench-prompt-file benchmarks/dflash2/summary.txt` for the
original 5,976-token summary workload. `bench.sh` makes path arguments
absolute (the app runs from `/`).

After a code change, run the same command with another JSON output path.
When changing only prompts, block widths, or output lengths, add `--no-build`.
It reuses this checkout's last Release build; it labels the source as
`reused-binary` rather than attributing that binary to the current source tree.
All `bench.sh` entrypoints share a lock, so two benchmarks cannot compete for
the GPU or overwrite the same log. A harness that fails writes the error to
the log as a `[harness]` line, and the script exits nonzero.

Useful options:

| Option | Purpose |
| --- | --- |
| `--bench-runs 3` | Repeat in the same process, sharing loaded weights and compiled traces. |
| `--bench-max-tokens 768` | Examine a longer continuation. Default: 192. |
| `--bench-blocks 3,5,8` | Compare widths sequentially, sharing model loading. |
| `--bench-check` | Add one AR pass and compare **every generated token**, on every run. |
| `--bench-round-timings` | Include individual round widths, acceptance and latency in JSON. |
| `--bench-draft-policy fc8` | Experiment with drafter precision: `4bit`, `8bit`, `unquantized`, `fc8`, `selector8`, `fc-selector8`. Target weights are unchanged. |
| `--bench-json /tmp/run.json` | Save timings, acceptance, prompt identity and complete token streams. |
| `--bench-vision` | Load the vision class (it runs the text class's engine, ADR-0089), to compare the two classes on one prompt. |
| `--bench-image PATH` | Attach an image to the prompt; implies `--bench-vision`. DFlash2 hands the image to the target, which prefills through it, and speculates over the text after it. |

Fast mode does not run AR or claim to check identity unless `--bench-check`
is supplied. Use it periodically and after changes to numerical kernels,
sampling, cache updates, or acceptance. A failed full-stream check exits
nonzero; the JSON report is saved before checking so divergence is inspectable.
The original repeated AR/DFlash benchmark remains available through
`scripts/bench.sh quick --model qwen3.8-27b --dflash2-bench`.

Compare an experiment with a saved baseline without another AR pass:

```sh
python3 scripts/dflash2-compare.py /tmp/dflash-before.json /tmp/dflash-after.json
```

Add `--require-identity` to compare complete before/after streams from two
saved reports, without generating another AR baseline. This is separate from
comparing DFlash against AR. Saving tokens adds no GPU synchronization;
the iterator already returns each token to the CPU.
The comparator rejects mismatched prompts, lengths or model identifiers,
and flags changed acceptance so a different trajectory is not mistaken for
a faster verification pass. For final claims, repeat runs in alternating
before/after order and inspect both acceptance and milliseconds per round
([Before and after a change](#before-and-after-a-change)).

Keep long prompts in a frozen file. The original full runner's default summary prompt is assembled
from live repository documentation, so editing the docs changes its output
and acceptance. These short fixtures are iteration workloads, not full
GSM8K, HumanEval or MT-Bench evaluations. The token loop uses a fixed length;
inspect long runs for EOS before treating every token as answer text.

For a numerical divergence, `--bench-logits-at N --bench-logits-file PATH`
records the top five logits and preceding eight tokens at generated position
N (zero-based). Create the file first. A speculative row may describe a
rejected history; compare only records with identical preceding tokens.
This diagnostic synchronizes the GPU and must be omitted from timing runs.

The small replay component can be tested independently with a Python MLX
environment:

```sh
python benchmarks/dflash2/replay_microbench.py
```

It reads the recorded vendor baseline commit, checks exact recurrent states,
and alternates the two kernels. Run it without another GPU workload. Its
component timings must not be reported as whole-model generation speedups.
See [FINDINGS.md](FINDINGS.md) for this investigation and saved results.

## The speed ruler

`scripts/dflash2-ruler.sh` measures the 500/100 goal in one Release run and
writes one JSON report (ledger, session 2026-10-08):

```sh
scripts/dflash2-ruler.sh --bench-check --bench-json /tmp/ruler.json
```

Decode runs first: each of `travel`, `summary`, `math` and `code` prefills
the production way (the app driver's pipelined 1,024-token chunks up to the
speculative split, then the DFlash2 iterator's capture prefill of the tail)
and decodes 512 greedy tokens at block 8. Then cold prefill is timed on
`prefill-2k.txt`, `prefill-8k.txt` and `prefill-32k.txt` (exactly 2,048 /
8,192 / 32,768 templated tokens), first chunk to first sampled token, with a
digest of the cache's bits so two builds can be shown to prefill
identically. Every timed run waits `--bench-cooldown` seconds (default 30)
first: sustained load throttles this machine's GPU by ~16%.

| Option | Purpose |
| --- | --- |
| `--bench-fixtures summary,code` / `none` | Decode these fixtures only |
| `--bench-prefill prefill-8k.txt` / `none` | Prefill these prompts only |
| `--bench-runs N` | DFlash2 runs per fixture |
| `--bench-check` | `TokenIterator` AR reference per fixture after the timed runs, and the same AR teacher-forced along run 0's stream |
| `--bench-round-timings` | Each round's milliseconds and accepted drafts |
| `--bench-kv-scheme turbo8v4` | The app's KV Cache Compression (attention layers compress once prefill ends); default bf16 |
| `--bench-lattice DIR` | Dump the drafter's lattice at every anchor (offline policy replay) |

`MLX_*` and `DFLASH2_*` variables reach the app (`bench.sh` forwards them
through `open --env`), so an env-switch A/B runs as two arms of one build.
`scripts/dflash2-ruler-report.py A=a1.json A=a2.json B=b1.json B=b2.json
--require-identity` prints medians per arm, deltas and stream identity.

Identity. The baseline's own stream leaves AR's argmax at bf16 ties (two
logits within 0–2 ulps), so `--bench-check` reports DIVERGED on every
fixture at 512 tokens. The forced check says which positions those are: at
each one the record holds the stream's token, AR's argmax and the gap in
bf16 ulps. Every change must reproduce the first arm's streams exactly,
tree verification included, under bf16 and turbo8v4 (each tree row reads
its own key where a chain block holds it; ledger G11, G12). Moving round
boundaries can still change a verify pass's key partitions, which depend
on its length: 2,048-key span buckets in the two-pass kernel and the
one-pass/two-pass switch at 1,024 keys under bf16, 512-key buckets in
turbo8v4's verify kernel. A stream that parts from the first arm's for
that reason, at a tie, passes `--require-identity` when every forced
departure is at most 2 ulps (ledger G9). Under turbo8v4 it cannot: the
compressed cache leaves bf16 AR's argmax by more than ties.

## Before and after a change

Bench the base commit and the change on one prompt, from two Release builds:
a worktree at the base keeps its own build, and `TESSERACT_BENCH_APP` points
`--no-build` at it. The base binary must know every flag you pass.

```sh
git worktree add --detach ../tesseract-gate-main main   # once
git -C ../tesseract-gate-main submodule update --init --recursive
(cd ../tesseract-gate-main && scripts/dev.sh build release)
scripts/dflash2-bench.sh --bench-check --bench-json /tmp/dflash-after.json
TESSERACT_BENCH_APP=../tesseract-gate-main scripts/dflash2-bench.sh --no-build \
  --bench-check --bench-json /tmp/dflash-before.json
python3 scripts/dflash2-compare.py --require-identity \
  /tmp/dflash-before.json /tmp/dflash-after.json
```

To move the base, check out the new commit in the worktree, update its
submodule and rebuild. For a speed claim, alternate the two arms (ABAB) and
compare medians.
