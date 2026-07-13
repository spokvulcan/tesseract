# ADR-0037: One checkpoint, two roles — quantized 1.7B VoiceDesign serves both tiers

- Status: Accepted (under overnight delegation, 2026-07-13 — owner review pending; two written contingencies below)
- Date: 2026-07-13
- Relates to: map #334, tickets #336 (model family), #338 (budgets), #339 (benchmarks + ear verdicts), #341 (this decision); ADR-0036 (inference substrate)

## Context

The map chartered "two model tiers by design": a quality tier for long-form read-aloud and a fast tier for the companion. The tiers were a *means* — the actual requirements are the two budget sets (#338) and one coherent product voice. The benchmark ticket (#339) measured every candidate on the M3 Max baseline and the owner recorded ear verdicts (`research/model-bench-339/listening.md`):

- **Quality tier**: 8-bit VoiceDesign ≈ bf16 by ear ("almost indistinguishable"; adherence holds); 6-bit "acceptably good". bf16 breaks the ≤ 3 GB envelope outright (4.5–4.7 GB RSS) — quantized shipping is confirmed by measurement.
- **Fast tier: no Design-then-Clone candidate passes cleanly.** 0.6B-Base-ICL: pronunciation problems + stray artifacts — unusable. Both x-vector modes: "awful" — ruled out (machine metrics predicted it: clipping + speaker-embedding drift). 1.7B-Base-ICL: clean and voice-consistent but prosody-flat — usable, unsatisfying, and at 3.26 GB RSS it breaks the envelope anyway. 0.6B-CustomVoice-8bit: "quite good for its size" — a viable *fixed-timbre* fallback that cannot inherit the designed voice.
- **The speed rationale for a small fast tier collapsed**: 0.6B is only 1.22× faster per step than 1.7B (the size-independent code-predictor + streaming-decode floor dominates), and **1.7B-VD-6bit (48.5 steps/s) matches 0.6B-Base-8bit (49.0)**. Quantized 1.7B-VD meets *every fast-tier budget*: warm TTFA 123 ms vs ≤ 300 ms, RTF 0.26 vs ≤ 0.5, cold ≈ 1.8–1.9 s vs ≤ 2 s.
- **A seed does not pin a voice** — same seed yields different voice realizations across precisions and run lengths. Voice identity must come from reference/anchor machinery, which also means a two-checkpoint product would speak with two voices unless the anchor chain spans checkpoints — exactly the fragile Design-then-Clone path the ear test just failed.

## Decision

**Both roles are served by a single resident checkpoint: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign` quantized.** "Quality tier" and "fast tier" survive as *role configurations* (TTS parameters, anchor policy, session defaults), not as separate models.

- **Precision policy**: 8-bit is the preferred shipping precision (the owner's recorded ear verdict); 6-bit is the envelope-compliant floor (2.84–2.86 GB peak RSS — the only 1.7B config that fits ≤ 3 GB by RSS; 8-bit measured 3.23–3.24 GB, ~8 % over). **Measurable gate, no re-decision needed**: after v2's memory-lifecycle work lands, measure 8-bit peak on the long-form scenario under the #344 memory definition — if it fits the envelope, ship 8-bit for both roles; otherwise ship 6-bit for both roles. One precision ships for everyone; both checkpoints are already staged.
- **Voice identity**: the VoiceDesign description is the product's voice-identity input for both roles; per-session consistency is pinned by the 48-step voice-anchor machinery (ported and mechanically validated in #337). Design-then-Clone is **retired** as the companion path.
- **Role deltas the spec must expose**: none at the model level. Per-role configuration only — anchor policy (companion: persistent pinned anchor; read-aloud: per-session anchor built from segment 1) and TTS parameter defaults.

## Considered / rejected

- **Fast tier = 1.7B-Base-ICL (Design-then-Clone)**: best clone by ear but prosody-flat ("not really listenable"), breaks the memory envelope, needs a second checkpoint plus the clone chain. Strictly dominated.
- **Fast tier = 0.6B-CustomVoice-8bit**: retained as the *written fallback*, not the pick — fixed timbre can't inherit the designed voice.
- **Quality at 8-bit + fast at 6-bit (two quants, one model)**: voice realizations differ across precisions at the same seed, so the product would audibly change voice between roles; reintroduces switching and double residency for no budget win.
- **Per-machine adaptive precision (6-bit on 16 GB floor, 8-bit on large RAM)**: two shipped checkpoints, voice differs across a user's machines, doubles validation surface. Rejected for one-precision-ships-for-everyone.

## Consequences

- Tier *switching* vanishes: #344 reduces to load / keep-warm / unload of one model plus LLM coexistence. The switch-not-stack residency clause of #338 is satisfied trivially.
- Companion and read-aloud share warm state — a companion exchange right after a read-aloud pays no model switch.
- The 682 MB codec + never-quantized parts are shared, not duplicated.
- Two contingencies for the owner's morning listen (staged samples + checklist):
  1. **Anchored-voice consistency** (the one listen #339 didn't run): if anchored VD drifts voice across independent companion-style utterances → fall back to 0.6B-CustomVoice-8bit for the fast role (accepting fixed timbre), 1.7B-Base-ICL as last resort.
  2. **Extended 6-bit listening** (only if the gate forces 6-bit): if long-form 6-bit reveals fatigue-level degradation vs 8-bit → invoke #338's written contingency ("the ear test outranks the memory number") and escalate 8-bit's 8 % envelope overage as a budget amendment instead.

## Accepted costs

- The companion runs a 1.7B model where a 0.6B might have sufficed — justified: the step-time floor makes the speed delta 1.22×, all latency budgets pass with ≥ 2× headroom, and it buys one coherent voice with zero switching.
- Long-form quality is capped at quantized 1.7B-VD (bf16 reference-only) — already accepted at #338 and confirmed by ear at #339.
