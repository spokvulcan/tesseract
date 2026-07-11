# ADR-0034: Proofread Pass — own co-resident MLX model, skip-when-busy

- Status: Accepted
- Date: 2026-07-12
- Relates to: map #283 (dictation overlay redesign), ticket #288; ADR-0025
  (always-armed capture — the "instant press-to-talk" bar the pass must not
  lower); GPU lease arbitration (CONTEXT.md)

## Context

Dictation commits the regex-post-processed Whisper transcription verbatim.
Misheard words, missing punctuation, and filler artifacts all reach the
target app, and a garbage take (mumbling, crosstalk) injects garbage. The
redesign program (map #283) adds a **Proofread Pass**: an LLM polish stage
between transcription and commit that corrects transcription errors or
rejects an unintelligible take — with the overlay narrating what it did.

Three ways to run that model on-device:

1. **Apple's system models** (Foundation Models framework / Writing Tools):
   zero download, ANE-resident, but the API surface is Apple-curated —
   prompt control is limited, availability varies by machine and OS
   settings, and the output contract (especially a reliable REJECT channel)
   is not ours to pin. Offline guarantees are implicit, not contractual.
2. **The agent's own LLM** via the existing `LLMActor`: no extra memory,
   but dictation would queue behind the GPU lease — an agent turn can run
   for minutes, and a dictation commit that waits even seconds is broken.
   It also drags the full prefix-cache machinery into a 40-token task.
3. **A second, small MLX model owned by the pass** (Qwen3.5-0.8B, 4-bit,
   ~0.5 GB): our prompt, our parser, our latency budget — co-resident with
   the agent model.

A second co-resident MLX model is architecturally safe here: there is no
single-container assertion in the stack, and the TTS `SpeechEngine` already
co-resides with the agent LLM. The one process-global hazard is MLX's
`Memory.cacheLimit`, which the agent's `LLMActor` tunes for its own
workloads.

## Decision

The Proofread Pass runs on **its own MLX model** — `ProofreadModel`, an
actor loading `mlx-community/Qwen3.5-0.8B-4bit` outside the agent's
`LLMActor`/prefix-cache machinery — under three locked policies:

- **Skip-when-busy.** The pass *reads* the Inference Arbiter's GPU-lease
  state (`isGPULeaseHeld`) and skips when held; it never queues on the
  lease. Dictation while the agent or server is generating commits the raw
  text immediately.
- **Fail-open, always.** Every skip and failure path — setting disabled,
  model not downloaded, GPU busy, load failure, model error, budget overrun
  (4 s) — commits the regex-cleaned raw text. The pass can only improve a
  commit, never gate one. An acceptance guard (edit-share and length-ratio
  caps) extends fail-open to the model's *output*: a reply that rewrote
  rather than corrected is discarded in favor of the user's words.
- **Hands off `Memory.cacheLimit`.** The process-global MLX cache knob
  belongs to the agent's `LLMActor`; the 0.8B pass rides whatever limit is
  in force.

The pass lives in `VoiceCaptureSession` (behind an injected closure), so
dictation and agent **Voice Input** both gain it. The reply contract is
plain corrected text with a single `REJECT: <reason>` escape hatch — a
no-think 0.8B holds a text contract far more reliably than JSON. The model
keeps one persistent KV cache trimmed to the fixed system prompt's prefix
between passes, guarded by the container's serial `perform` executor.

## Consequences

- ~0.5 GB extra unified memory while the proofread model is resident. It is
  prewarmed at launch (a first-press load would blow the latency budget),
  unloaded together with the agent's offload path, and invisible to the
  agent's `loadedSlots` bookkeeping — offload handles it explicitly.
- Correction quality is capped by a 0.8B model; the acceptance guard means
  the failure mode is "left it unchanged", not "rewrote my words". Prompt
  and threshold tuning stay open follow-ups on map #283.
- An ANE port (same pass, Core ML backend, zero GPU contention) is ticket
  #293 — this ADR fixes the *policy* surface it would slot under, not the
  backend.
- The skip-when-busy read means a dictation issued mid-agent-turn is never
  proofread — accepted: raw-commit latency outranks polish, always.
