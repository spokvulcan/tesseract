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

## Amendment 2026-09-24 — v1 fitted to the phone (#515)

The PRD was checked against `main` and Apple's documentation, and the owner
settled four open points. Decisions 1–3 stand; these details change.

- **The 12 GB tier moves past v1.** It can't be measured on the owner's 8 GB
  phone, and the 9B is an 8.6 GB download. Every supported phone gets the
  8 GB configuration, 12 GB phones included; the **Device Tier** keeps the
  seam, so a 12 GB value is one addition later. The 64k window and the
  VoiceDesign experiment go with it.
- **A small SSD prefix tier on the phone**, replacing "RAM prefix tier only,
  no SSD tier". The Budget Floor keeps only in-flight restores and the latest
  leaf; `.system` chains are protected by the SSD tier (ADR-0019). RAM only
  would re-prefill the system prompt for every new conversation and after
  every relaunch iOS forces.
- **The voice yields; the window doesn't shrink.** On 8 GB the estimated set
  (~2.3 GB weights, ~1.07 GB KV at 32k, 0.35 GB embedder, and the 0.6B voice,
  1.97 GB on disk) is past the ~5 GB the app can expect. The window stays at
  32k. The neural voice loads only when there is room, and read-aloud uses the
  system voice otherwise. This replaces "co-resident with the LLM".
- **No background GPU work.** Apple grants background GPU time only to
  continued-processing tasks on iPads with M3 or later, and on no iPhone. There
  is nothing for the gated `BGProcessingTask` experiment to verify, so the
  path is dropped rather than built behind a flag.
- **A backgrounded reply pauses.** The **Foreground Gate**, owned by the
  arbiter, closes when the scene leaves the active state. A reply stops after
  its current token with its KV held and resumes on return; the partial reply
  is saved first in case iOS ends the app. The arbiter also gains a TTS-only
  unload, so "the GPU lease and arbiter are unchanged" no longer holds for the
  arbiter; the GPU Lease Queue is unchanged.
- **Episode Origin names a store, not a device.** Each memory store takes a
  random id at creation, stamped into every episode it writes. Unstamped
  episodes are the Mac's, from before stamping. The consequence "the Mac's
  episodes gain a device stamp now" hasn't been done; it is slice 2 of the
  PRD.
- **Figures corrected.** The 4B's Prepared Checkpoint is 3.86 GB on disk. The
  published artifact leaves out the vision encoder, which brings it to about
  3.2 GB. The "2.3 GB" above is its resident size, once the loader has
  quantized the embeddings. The first run needs the model and the embedder,
  about 3.5 GB, and the voice's ~2 GB follows.

### Amendments this makes

- **ADR-0032.** The Prepared Checkpoint gains a published mode, for an
  artifact made on one machine and loaded on another:
  - The manifest doesn't depend on the source files' modification times.
  - The artifact loads with no originals present.
  - A bad artifact fails with a typed error, and the app re-downloads it,
    because there is nothing to re-convert from.

  "Self-heal, never fail" stands for local artifacts. The rejection of
  fully-final parameters stands too: the published artifact is the same
  prepared form. The weight-identity contract changes in published mode.
  `ModelFingerprint` skips Prepared Checkpoint file names, and on the phone
  that file is the only weights file, so there the published manifest's
  identity joins the fingerprint. Without it the SSD tier's partition key
  would hash no weights at all.
- **ADR-0018.** On the phone the ceiling reads its headroom from
  `os_proc_available_memory()`, the per-app limit iOS enforces. The kernel
  buckets and the Working-Set Bound are Mac readings. The band, the floor and
  the reserve's shape are unchanged. Device Tier supplies the starting budget
  and the output cap the reserve prices growth from.
- **ADR-0039, decision 5.** The trigger that unloads the voice at critical
  memory pressure was listed but never wired on the Mac. On the phone, Phone
  App Bindings wires it, and the engine still doesn't know the app. There, an
  unload hands the rest of the utterance to the system voice.
