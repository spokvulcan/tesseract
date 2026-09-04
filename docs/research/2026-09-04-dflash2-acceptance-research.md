# Why DFlash2 accepts ~28% of drafts on the canonical bench — and 63% on math

Date: 2026-09-04 · Status: measured · Companion to ledger R56/R57, ADR-0058.

## The question

The canonical DFlash2 bench (docs-summary prompt, Qwen3.8-27B 4-bit +
`incoai/Qwen3.8-27B-DFlash2`, block 8) accepts 21.6% of drafts today and
28.4% on average over eight near-identical prompts (R56). The drafter is
advertised as accepting "more than half" of its drafts. Is the port broken,
or is the number expected?

## What the primary sources actually claim

Both the DFlash paper (arXiv 2602.06036 §3.1) and the inco.ai DFlash2 model
card report **acceptance length τ** = completion tokens per verification
step, **including the bonus token**. The fraction of drafts accepted is
(τ − 1)/(B − 1). The model card's numbers for this exact drafter (bf16
target on an H200, SGLang, temperature 1.0 / top-p 0.95 / top-k 20, `xhigh`
reasoning, block 8 = 7 drafts, max 4096 new tokens):

| task | τ | drafts accepted |
| --- | --- | --- |
| GSM8K | 5.46 | 63.7% |
| MATH-500 | 5.28 | 61.1% |
| HumanEval | 4.39 | 48.4% |
| MBPP | 4.79 | 54.1% |
| MT-Bench (open-ended prose) | 4.10 | 44.3% |

So "above 50%" holds for math and (barely) code; it already fails on the
model card's own prose task. The paper's open-ended rows are the floor of
every table: Alpaca τ 3.73 at block 10 = 30% of drafts (Table 4, LLaMA-3.1-8B).
Two further published effects stack on top of content:

- **Thinking traces draft worse.** Paper Table 2, Qwen3-4B on MATH-500,
  greedy: τ 7.84 with thinking disabled → 5.74 with thinking enabled (−27%).
- **A 4-bit target costs acceptance.** mlx-dspark (DFlash2, Qwen3.8-27B,
  M4 Pro, greedy): 8-bit target 64.7% → 4-bit 59.1%. Aryagm/dflash-mlx
  (Qwen3-4B, block 16): bf16 target 83.7% → 4-bit 52.8%. Drafter precision
  does not matter (our R53 agrees: +1.3 points from 4-bit to bf16).

The "acceptance should be above 50%" rule is not in any DFlash source; it is
the generic speculative-decoding rule of thumb (glukhov.org), and DFlash2 nets
2.67× on MT-Bench at 44% because its verify step is cheap relative to the
tokens it recovers.

No source publishes acceptance on summarization or long-context prompts.

## Cross-stack measurement on this machine

Same target (4-bit, group 64), same drafter (4-bit at load), greedy, block 8,
192 new tokens. "Swift" = the app bench (`--dflash2-bench --bench-blocks 8f`,
`DFLASH2_BENCH_PROMPT_FILE` for math/code); "Python" = the z-lab reference
`dflash/model_mlx.py` (mlx 0.32) driven by `acceptance_probe.py` on the
identical prompt bytes and chat template.

| prompt | Swift accepted | Swift τ | Python accepted | Python τ |
| --- | --- | --- | --- | --- |
| canonical docs, chat template (thinking on) | 115/532 = 21.6% | 2.53 | 111/552 = 20.1% | 2.40 |
| docs variant 6 | 133/406 = 32.8% | 3.31 | 129/423 = 30.5% | 3.10 |
| canonical docs, raw text (no template) | — | — | 105/518 = 20.3% | 2.43 |
| canonical docs, chat, thinking off | — | — | 122/478 = 25.5% | 2.78 |
| math (GSM8K-style, two problems) | 158/252 = **62.7%** | 5.33 | 156/241 = 64.7% | 5.49 |
| code (HumanEval-style LCS) | 140/357 = 39.2% | 3.76 | 143/334 = 42.8% | 4.00 |

Output identity MATCH on every Swift arm. The two stacks land within 2–4
points of each other on every prompt (their greedy trajectories differ by
kernel numerics, so exact counts cannot match), and both reproduce the model
card on math (5.33/5.49 vs 5.46). **The port is at spec.**

Speed on the short prompts (147 and 98 prompt tokens, so no 6K KV stream in
the round): math **75.2 tok/s** (3.31× over AR 22.7), code 53.7 tok/s
(2.41×). The 60 tok/s line is crossed on the drafter's home content with the
stack as it stands.

## Why the canonical prompt sits at 20–30%

Three published effects, all present at once:

1. **Content class.** A one-paragraph summary of an architecture document is
   open-ended prose, the floor class (MT-Bench 44%, Alpaca 30% in the
   sources) — and a 6K-token technical document makes the continuation
   less predictable than MT-Bench's short chats.
2. **Thinking trace.** The 192-token budget never leaves the model's
   planning preamble ("We need answer user's request… Need one paragraph…");
   the paper measures a ~27% τ penalty for reasoning traces on the same
   dataset. The raw and thinking-off conditions above (20–26%) show the
   answer text itself is no easier on this document.
3. **4-bit target.** A further ~5 points on this model per mlx-dspark's
   8-bit vs 4-bit measurement.

Per-position anatomy on the canonical roll (R56): first-draft hit rate 0.69,
target inside the drafter's top-2 candidates 0.79 and top-16 0.93, 31% of
rounds accept nothing. The drafter's candidate lattice often contains the
right token; the single selected path does not — the signature of
unpredictable prose, not of a broken drafter.

## Consequences

- The canonical docs prompt is the adversarial arm by design (ADR-0058
  "known limits"); it should not be read as the drafter's typical rate.
- A content suite (math / code / chat / docs) with the prompt hash on each
  line is the ruler that matches the published tables; the bench now takes
  `DFLASH2_BENCH_PROMPT_FILE` and `--bench-prompt-variants N` for that.
- Raising acceptance on prose remains a drafter-training question (R30/R53);
  nothing in the inference stack is leaving acceptance on the table.

## Reproduction

- Python: `research/dflash-venv/bin/python research/acceptance_probe.py`
  (conditions: canon-chat canon-raw var6-chat canon-nothink math-chat
  code-chat); prints the prompt sha256 (`cd4da088…` for the canonical
  prompt). `research/` is unversioned (gitignored, like `bench_dflash.py`
  and the reference checkout it drives).
- Swift: `"<Release app>/Contents/MacOS/Tesseract Agent" --bench-model-id
  qwen3.8-27b --dflash2-bench --bench-blocks 8f` with
  `DFLASH2_BENCH_PROMPT_FILE=<prompt.txt>`; quit the running app first.
