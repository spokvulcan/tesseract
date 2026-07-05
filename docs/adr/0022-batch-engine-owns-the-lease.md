---
status: accepted
---

# Batch inference: a lane engine owns the GPU lease

Today the second concurrent completion waits FIFO on the GPU lease up to 60 s,
then 503s — the primary workload this hurts is single-user subagent fan-out
(OpenCode-style parallel workers). The structural constraint that shapes any
fix: all MLX/Metal work must stay on the single Metal-affine actor (the oMLX
command-buffer race), so batch concurrency is logical lanes multiplexed onto
one executor, never threads. Decided in the 2026-07-05 grilling of #151.

## Decision

The **Batch Engine** owns the GPU lease whenever any lane is live. Completions
(HTTP requests *and* the internal Agent Run — anything that is a text
completion on the loaded LLM) submit to the engine instead of acquiring the
lease themselves; the engine acquires once when going non-idle, drives every
lane's prefill and decode from one step loop, and releases when the last lane
drains. `InferenceArbiter` slot semantics survive unchanged; TTS and model
loads see one long-running lease consumer.

- **Lane Admission**: lanes = `min(hard cap 4, headroom)` where headroom is
  the existing `N × activeInferenceReserve` ceiling arithmetic — on a big
  model N degenerates to 1 by arithmetic, not policy. The waiting queue keeps
  today's 60 s → 503 + `Retry-After` contract, now as an admission timeout.
  Queue order is longest-prefix-match (SGLang cache-aware), aged to strict
  FIFO after ~10 s so cold-prefix requests cannot starve. The cap is a
  constant, not a user Setting; the bench may override it.
- **Step loop**: one prefill chunk (smaller when lanes are decoding) alternates
  with decode steps, bounding a decode lane's stall at one chunk; one prefill
  at a time, admission order. Prefill+decode fusion is the known follow-up if
  batched decode wins the spike.
- **Boundary Yield**: at step/chunk boundaries the engine yields the lease to
  waiting slot-preserving consumers (TTS), lanes pausing as plain data —
  bounded TTS latency instead of waiting for pool idle, which continuous
  admission would make unbounded.
- **Admission Freeze**: a request naming a different model, or an
  image-bearing request, ages to queue head, freezes new admissions, and runs
  solo after the pool drains. Image lanes stay solo until the overlapping-image
  stress run is clean on the paged structure *and* the parked MLX buffer-
  lifetime investigation renders its verdict — overlapping image requests are
  the workload that historically produced the InvalidResource SIGABRT.
- **Speculative Canonical Prefill** stays idle-only (zero lanes, empty queue;
  first admission preempts). Background-lane promotion waits for #152's
  skip-reason investigation.

## Considered options

- *Per-step lease sharing* (lanes as independent tasks, lease per decode
  step): structurally precludes batched matmul — lanes must step together in
  one forward pass to share the weights read — so it would pre-decide the
  decode-shape measurement in the wrong direction.
- *Concurrent actor interleaving* (N `ServerCompletion`s interleaving at
  suspension points): no scheduling control, no batched forward pass, and the
  closest shape to the oMLX Metal race.

## Consequences

- The decode-shape question (true batched matmul vs interleaved round-robin)
  stays open by design: both are step functions inside this skeleton, so the
  spike's verdict (pre-registered: batched ships iff ≥1.8× aggregate tok/s at
  N=4 on the small model, stable, per-lane latency ≤1.5× single-lane) swaps
  one function, not the architecture.
- A stray request for a different model can stall a fan-out burst for one
  freeze window (bounded by the longest in-flight completion) — accepted in
  exchange for keeping today's model-switch contract.

## Decode-shape verdict (2026-07-06)

**Interleaved round-robin ships; batched matmul NOT justified.** On
qwen3.5-4b (Release, M-series, `--batch-lane-bench`): batched `[N, 1]`
decode at N=4 clears the aggregate bar (2.50× at ctx 2048, 1.89× at
ctx 8192, vs the 1.8× threshold) but fails the per-lane latency bar at
both context points — 1.60× and 2.11× vs the ≤1.5× cap. Per the
pre-registration, the latency bar is not negotiable after the fact, so
batched decode stays a follow-up question (it is one step function away
if a future revisit re-weighs latency or shrinks the batched round).
Report: `tmp/tesseract-debug/benchmark/batch-lane-curves/`.
