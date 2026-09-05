# DFlash2 experiment loop

Run one DFlash pass instead of the full six-pass app benchmark:

```sh
scripts/dflash2-bench.sh \
  --bench-prompt-file "$PWD/benchmarks/dflash2/travel.txt" \
  --bench-json /tmp/dflash-before.json
```

The fast wrapper defaults to the frozen short travel fixture (82 input
tokens). Use `--bench-prompt-file "$PWD/benchmarks/dflash2/summary.txt"` for
the original 5,976-token summary workload.

After a code change, run the same command with another JSON output path.
When changing only prompts, block widths, or output lengths, add `--no-build`.
The script reuses the recorded Release app path; it labels the source as
`reused-binary` rather than attributing that binary to the current source tree.
All `bench.sh` entrypoints share a lock, so two benchmarks cannot compete for
the GPU or overwrite the same log.

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
before/after order and inspect both acceptance and milliseconds per round.

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
