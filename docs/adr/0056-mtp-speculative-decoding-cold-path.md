# ADR-0056: MTP speculative decoding adopts the vendor iterator on the cold path

- Status: Accepted
- Date: 2026-08-17
- Relates to: ADR-0007 (state-threaded token iterator), ADR-0016 (ModelSession
  port), ADR-0053 (output-only presence penalty — every generation path routes
  penalties through the app's logit processor), tesseract PR #431 (vendor
  re-pin that brought `MTPSpeculativeTokenIterator` into reach)

## Context

Qwen3.5-family checkpoints can ship a **multi-token-prediction head** — 15
tensors under the `mtp.*` prefix: a 1-layer full-attention drafter that
consumes the target's last hidden state and proposes tokens the target then
verifies in a single batched call. The vendored mlx-swift-lm implements this
end to end (`MTPSpeculativeTokenIterator`, `MTPDrafterModelFactory`, the
Qwen3.5 text/VLM draft models). Greedy speculative decoding is **lossless by
construction**: the target accepts a draft only where it matches its own
argmax, so output must be token-identical to non-speculative greedy decoding.

Three facts shaped the integration:

1. **The vendor's MTP prompt prefill is unchunked by design.** The drafter's
   private 1-layer KV cache is prefilled over the *shifted* prompt and needs
   one target hidden row per prompt token, so the target's prompt pass must be
   single-shot — it cannot ride the app's chunked `PrefillExecutor`, and it
   cannot start from a radix-cache-restored checkpoint (no hidden states are
   stored, only KV).
2. **The Qwen3.5 drafters are greedy-only and shallow.** They declare
   `requiresGreedySampling = true` (the iterator silently passes through at
   temp ≠ 0) and clamp `maximumBlockSize = 2` — one drafted token per round,
   because the hybrid target's recurrent linear-attention state can rewind at
   most one position in place. The family ceiling is therefore a 2× cut in
   target calls.
3. **Distributed quants strip the head.** Every mlx-community/unsloth quant
   surveyed ships without `mtp.*`; the official bf16 checkpoints keep it. A
   quantized target can still speculate because the vendor loader quantizes
   only modules with `.scales` tensors — a bf16 head grafted beside 4-bit
   weights loads natively (`scripts/graft_mtp_head.py` performs the graft).
   One landmine, fixed in the vendored fork: the Qwen3.5 `sanitize` used
   `mtp.*` **presence** as its raw-HF-checkpoint signal and shifted every
   RMSNorm weight by +1 on seeing it. That proxy held for official releases
   (raw, ship mtp) vs community quants (pre-shifted, strip mtp), but a
   grafted quant is *pre-shifted and ships mtp* — the double shift turned
   every target forward to garbage while the drafter (whose norms really are
   raw, from the official shard) loaded correctly. The fork now keys the
   shift on the conv1d weight layout alone, the direct observation of the
   checkpoint's convention; upstreamable.

## Decision

Adopt the vendor iterator **on the cold/unkeyed decode path only**, behind a
pure engagement policy, with the drafter loaded eagerly beside the target.

- **Detection + loading** (`MTPDrafterSupport`): at model load, a cheap
  header-only scan for `mtp.*`; when present (and the "Speculative Decoding
  (MTP)" setting is on, default on), the drafter is loaded from the same
  checkpoint directory. Drafter selection keys on **the class of the model
  instance that actually loaded** (`drafterPairing(for:)`), never on config
  shape or the app's vision intent — the app force-loads the text target from
  VLM-shaped checkpoints in non-vision mode, and the generic loader itself
  falls back VLM → LLM when the VLM factory throws (legacy-layout checkpoints
  like the Qwen3.8-27B community quant, `language_model.*`/`vision_tower.*`
  naming, load as the text model even in vision mode). Each drafter
  `fatalError`s on the other family's target, so intent-keyed pairing crashes;
  instance-keyed pairing cannot. An unknown family, a missing head, or a
  failing drafter load never fails the model load — speculation just stays
  off.
- **Engagement** (`MTPDrafterSupport.shouldEngage`, pure + unit-tested):
  speculate iff a drafter is loaded ∧ temperature == 0 ∧ the request is
  text-only with an identity key space ∧ the whole-prompt single-shot score
  matrix fits a 4 GiB scratch budget (the unchunked-prefill knob; ≈ 9K prompt
  tokens on the 27B profile). The decision sits in `ServerCompletion`'s
  restore switch, `case .cold` — warm restores keep the ordinary
  state-threaded path by construction.
- **ADR-0053 compliance**: penalties are stripped from the iterator's
  sampling parameters and the app's logit processor is injected through
  `GenerationComponents`, so the presence penalty applies identically on both
  arms.
- **The speculative arm still seeds the cache.** `makeMTPGeneration` returns
  a *keyed* generation record; post-generation leaf capture from the final
  cache proceeds as on the normal path, so turn 2 restores warm (and, per the
  policy above, decodes non-speculatively).
- **Opt-in surface**: the `Greedy (Speculative)` sampling preset (temp 0,
  output-only presence penalty kept) is how the agent chooses MTP; block size
  is the vendor default 4 (clamped to 2 by today's drafters); per-round
  telemetry (rounds, proposed, accepted, acceptance rate, target calls) is
  logged after every speculative generation.

## Consequences

- Greedy parity, as measured: on the **VLM target class the gate is byte-exact**
  — three cold prompts on Qwen3.5-2B, MTP on vs off, token-identical outputs
  (407/394/385 tokens each), proving the iterator machinery (accept/reject,
  hybrid-cache rewind, ADR-0053 penalty routing) is exactly lossless when the
  forward numerics are uniform. On the **text target class** the same
  experiment diverges at a near-tie ~50–70 tokens in (2B bf16 and 27B 4-bit
  alike), with both sides fully coherent: the MTP arm's forwards are
  single-shot prompt + S=2 verify while plain decode is chunked prompt + S=1
  steps, and the text implementation's kernel schedules for those shapes are
  not bitwise-identical (the fused decode paths were ruled out by a
  kill-switch A/B — plain decode is bit-stable under them). This is the
  standard batched-verify numerics caveat of speculative decoding, not an
  acceptance bug; output quality is indistinguishable.
- Measured lift: Qwen3.5-2B (bf16) ≈60–63% acceptance, ~1.6 tokens per target
  call, ≈1.15× wall-clock — modest because a small target makes the drafter's
  own forward proportionally expensive. Qwen3.8-27B (4-bit + grafted bf16
  head) **75–87% acceptance, ~1.8–1.9 tokens per target call, ≈2.0× measured
  wall-clock decode (15.3 → 29.7–31.1 tok/s)** — at the family's 2× ceiling
  (the drafters clamp `maximumBlockSize = 2`). The lift grows with target
  size; the head stays useful after 4-bit quantization of the target.
- Cache-restored turns never speculate. Lifting that needs drafter-cache
  persistence alongside the radix leaf (and a chunked MTP prefill upstream) —
  filed as a follow-up issue rather than built.
- Temp > 0 speculation needs an acceptance sampler upstream — also filed,
  not built.
- PARO checkpoints (today's default agents) ship no head and are unaffected;
  the feature lights up only for checkpoints that carry `mtp.*`, official or
  grafted.

## Amendment (2026-08-18): engagement narrowed to leaf modes that store from `finalCache`

"The speculative arm still seeds the cache" above holds only for the
`directLeaf` mode. The unchunked MTP prompt prefill forfeits the transient
boundary snapshots (`makeMTPGeneration` returns nil for both), and two leaf
modes are *synthesized from those boundaries*: a thinking template's
canonical leaf and a tool-call turn's direct-tool leaf. On those modes the
post-generation store skips (`no-canonical-restore-boundary`), the partition
never seeds, MTP re-engages on the next cold turn, and the conversation
prefills the whole prompt every turn forever — the ~2× decode lift silently
costs 100% of the prefix cache (observed live on qwen3.8-27b OpenCode
sessions, 60s+ prefill per turn).

`shouldEngage` therefore also requires the predicted
`HTTPLeafStoreMode` — from the same `selectHTTPLeafStoreMode` classification
the leaf store itself uses, with defined tools conservatively standing in
for emission (tool *emission* is unknowable at engagement time) — to be
`directLeaf`. In practice this parks the MTP arm for the thinking-template Qwen3.5+
checkpoints over agent traffic until a boundary-preserving MTP prefill
exists — the cache is the product's first priority and beats a 2× decode
lift on any multi-turn workload.

(2026-08-24: superseded for the DFlash2 arm by ADR-0059 — the app-owned
prefill keeps every boundary snapshot and the DFlash2 iterator warm-starts
behind it, so that arm engages on all keyed leaf modes, warm or cold. This
amendment continues to govern the MTP arm.)
