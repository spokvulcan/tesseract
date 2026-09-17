---
status: accepted
---

# iOS: the phone runs the agent on-device, with its own memory, from a second target

Design for [#515](https://github.com/spokvulcan/tesseract/issues/515), the
iOS v1 PRD. Grilled with the owner on 2026-09-17. Relates to ADR-0035 (living memory),
ADR-0037/0039 (TTS checkpoint and residency), ADR-0032 (Prepared
Checkpoint), ADR-0047 (the Mac agent left the sandbox), and the architecture
note "defer Agent package extraction".

## Context

Tesseract is a 468-file single macOS target with no platform conditionals.
The owner wants an iPhone app: chat with the agent, text-to-speech, voice
input later. Three shapes were weighed:

1. **Phone as a client of the Mac.** An agent-level API on the existing HTTP
   server; same memory, same 27B model, no MLX on the phone. Cheapest for
   chat. Needs a reachable Mac, so no offline use away from home.
2. **Phone runs inference on-device via MLX.** The Mac's default agent tier
   (Qwen3.5-4B PARO) plus the embedder, memory, tools, skills, and a small
   TTS. Offline anywhere. Requires the vendored MLX stack to build and run
   on iOS.
3. Both, phased.

A build spike on 2026-09-17 settled the feasibility unknown: `MLXLLM` and
`MLXAudioTTS` from the vendored fork (mlx-core 0.31.1 + custom kernels,
mlx-swift 0.31.6) build unmodified for the iOS 27 device SDK and the
simulator SDK, producing an iOS metallib. The fork's JIT-compiled kernels
(`qmv_wide`, `affine_qmm_mma8`, fastmath) are validated only at first run on
a device, not at compile time.

## Decision

**1. The phone runs the agent on-device.** Offline inference is the
product's point on the phone as on the Mac; a phone that needs the Mac is
not Tesseract. It runs the full agent — memory, tools, skills, slash
commands — not a stripped chat. The Mac-as-server shape is not built, not
even as a fallback. Floor: iPhone 16 Pro (8 GB); larger models are a 12 GB
tier. Model: the Mac's default `qwen3.5-4b-paro`, shipped as a pre-published
Prepared Checkpoint so the phone never converts. TTS: Qwen3-TTS 0.6B
CustomVoice 8-bit co-resident with the LLM (a **Preset Voice**, not the
Mac's **Pinned Voice**), with `AVSpeechSynthesizer` while it loads. Context
window fixed per tier (32k / 64k), thinking off by default, RAM prefix tier
only, no SSD tier, no HTTP server.

**2. The phone's memory is its own, built merge-ready.** Two independent
living-memory stores; no sync in the MVP; iCloud never (bytes would leave
the device). Every episode is stamped with its device of origin from day one
so a later LAN union-merge of the immutable episodic layer (ADR-0035) is a
query, not a migration. Sleep consolidation runs foreground-opportunistically
(idle chat, free GPU lease) with a manual trigger; the background-task path
is a gated experiment until Metal-in-background is verified on device. The
proactive Companion loop does not ship on the phone.

**3. A second app target now; package extraction later.** `tesseract-ios`
shares source membership with the Mac target: clean directories in both, iOS
replacements for the handful of AppKit views, `Platform/` excluded. No
`#if os(...)` in the engine. The `TesseractCore` package extraction —
reversing the recorded deferral — happens once the target boundary has
exposed the real seam, with its own ADR.

## Considered / rejected

- **Phone as Mac client**: cheapest chat, but it cannot meet
  offline-away-from-home; and the server exposes raw completions, so parity
  would have needed a new agent API anyway.
- **Standalone chat without memory, tools, skills**: nothing in the
  architecture forces it — the engine, conversation store, tools, and skill
  bundles are already platform-clean (7 of 148 Agent files touch AppKit, all
  views).
- **iCloud memory sync**: contradicts "no cloud" as the README states it.
- **1.7B VoiceDesign on the phone**: 2.8 GB peak RSS (ADR-0037) cannot
  co-reside with the LLM on 8 GB; kept as a 12 GB-tier experiment with an
  LLM/TTS swap through the arbiter's slots.
- **Package extraction first**: guesses the seam across 123k lines and
  re-opens the app-target vs package execution-convention trap.
- **One multiplatform target with conditionals**: spreads `#if os` through
  the 41 files that import AppKit.

## Consequences

- The Mac's episodes gain a device stamp in `meta` now, ahead of any phone
  build.
- The Model Fetching port gains a background-URLSession adapter
  (Wi-Fi-only default, cellular override, resumable); the Mac adapter is
  untouched.
- The speech engine's Voice gains a second kind (Preset Voice) beside the
  anchored Pinned Voice.
- CI gains a build-only iOS device job; the fork checkout is 117 MB and a
  full DerivedData 2.6 GB.
- Development is device-only: MLX does not run on the simulator's Metal.
- Read-aloud keeps playing after screen lock but generation pauses
  (audio background mode + lookahead), pending one device test.
- Distribution: TestFlight first, App Store once the 8 GB floor has held in
  daily use; bundle id `app.tesseract.agent` reused.
- Phone vocabulary: **Voice Input**, never dictation — iOS has no
  cross-app injection flow.

## Accepted costs

- Two Jarvises that diverge until the merge is built.
- A different voice on the phone than on the Mac.
- A 2.3 GB first-launch download with an explicit onboarding step.
- The JIT kernels are unproven on A18 until the first milestone — a
  throwaway app that loads the 4B and generates one token — runs on the
  owner's iPhone 16 Pro Max.
