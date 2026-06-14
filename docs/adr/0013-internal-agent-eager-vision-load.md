---
status: accepted
---

# The internal chat agent loads the vision variant eagerly for vision-capable models; the composer toggle becomes a global opt-out

This extends ADR-0008 (HTTP loads the vision variant for vision-capable models,
always) to the in-app chat agent, decided in the 2026-06-14 image-handling UX
grilling (PRD: #112). The internal agent already shares the HTTP
cache-aware **Server Completion** path; the only difference was load policy —
chat resolved vision from the composer toggle (`.fromSettings`, default off)
while HTTP used vision-if-capable. We make chat load the vision variant from
turn one for any **Vision-Capable Model**, and retire the per-turn composer
toggle in favour of a single global Settings opt-out ("Use vision models when
available", default on).

The trigger: the old toggle defaulted off because of a believed "~3.4× slower
VLM text prefill". That number is stale — it predates the ADR-0006 chunking
work, which routes text-only prompts through the app's chunked prefill driver on
both containers. Measured 2026-06-14 on Qwen3.6-27B PARO (16,413-token cold
prefill via `--prefill-step-benchmark`): vision 79.3 s vs text 79.9 s (vision
marginally *faster*); warm 20.9 s vs 21.2 s. The only real cost of holding the
vision variant is ~+1 GB RAM (the vision tower). With the prefill penalty gone,
eager loading is strictly better than the alternatives for the stated
constraints — no toggle friction, and an image-add turn warm-continues (ADR-0007
phase 2) instead of swapping and re-prefilling.

Considered and rejected:

- *Lazy swap on first image* (keep the text variant, reload to vision when an
  image is attached) — the swap is a full ~27B reload that tears down the warm
  RAM prefix cache; even though the SSD tier preserves the cached prefix across
  the swap (the cache partition key is vision-mode-agnostic — same model
  fingerprint for both containers), it pays a multi-second reload at the worst
  possible moment: the instant the first screenshot is pasted into a long
  session. Eager loading never swaps.
- *Keeping both containers resident* — they wrap the same ~27B language-model
  weights, so a second container doubles them. Only LLM + TTS are co-resident,
  never LLM + VLM.
- *Leaving the per-turn toggle* — friction for a default that should just work,
  and confusing now that the HTTP server ignores it (ADR-0008).

Consequences:

- Vision-capable models hold the vision tower (~+1 GB) for pure-text sessions
  too. The global opt-out exists for users who want the text-only container; the
  HTTP server still ignores it.
- The stale "~3.4× / ~390 tok/s" comments (`SettingsManager.swift`,
  `LLMActor.swift`, `ParoQuantLoader.swift`) are corrected as part of this work,
  so the rationale is not silently contradicted in-tree.
- A non-vision-capable selected model has no vision variant: image affordances
  are hidden and an image paste/drop prompts a switch to a vision-capable model.
