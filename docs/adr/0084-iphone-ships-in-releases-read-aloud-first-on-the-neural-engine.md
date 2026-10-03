# ADR-0084: The iPhone app ships in four releases, and the first reads aloud with Qwen3-TTS on the Neural Engine

- Status: Accepted (slice 1 of #515 sets the budgets and the weight precision)
- Date: 2026-10-03
- Supersedes: ADR-0066 decisions 1 and 2, and the details its 2026-09-24
  amendment fitted to the agent (Device Tier sizing, the Foreground Gate, the
  voice yielding to memory, the phone's SSD prefix tier, the published LLM
  checkpoint). ADR-0066 decision 3 stands.
- Amends: ADR-0075 (on the phone, the talker and code predictor run on the
  Neural Engine too; the Mac is unchanged)
- Relates to: [#515](https://github.com/spokvulcan/tesseract/issues/515)
  (release 1's PRD), ADR-0071 (first-party Qwen3-TTS), ADR-0074 (the parity
  method), ADR-0076 (the Reader), ADR-0077 (Word Timing)

## Context

ADR-0066 made the phone a second home for the whole agent: Qwen3.5-4B on the
GPU, its own living memory, tools and skills, with read-aloud as one feature
that gave way to the LLM when memory ran short. On 2026-10-03 the owner
re-planned the phone around what is most useful there first, and named the
requirement that rules it: the phone must not get hot or drain the battery,
so the voice should run on the Neural Engine, and in real time.

Four facts shaped the decision.

- **The GPU stops when the screen locks.** iOS refuses GPU work from a
  backgrounded iPhone app. A GPU voice reads only as far as its buffer once the
  phone is in a pocket. From iOS 27 the Neural Engine is restricted too: any
  background use needs the Background Inference entitlement
  (`com.apple.developer.background-tasks.continued-processing.inference`,
  [release notes](https://developer.apple.com/documentation/ios-ipados-release-notes/ios-ipados-27-release-notes)).
  Apple doesn't say whether a read-aloud app can have it.
- **Qwen3-TTS 0.6B fits the Neural Engine.** Argmax's open-source TTSKit runs
  this checkpoint on CPU + Neural Engine through Core ML on iOS. It reaches
  real-time factor 0.63 on an Apple M4 Mac
  ([TTSKit #513](https://github.com/argmaxinc/argmax-oss-swift/pull/513)), and
  an A19 iPhone keeps up with 1.25× playback
  ([TTSKit #506](https://github.com/argmaxinc/argmax-oss-swift/pull/506)).
  Nothing is published for the owner's A18 Pro.
- **Its cost is memory traffic.** Each 80 ms frame runs the 28-layer talker
  once and the 5-layer code predictor 15 times, reading 1.66 billion weights,
  about 1.7 GB at 8 bits. The Neural Engine makes the arithmetic cheaper, not
  the traffic. The levers are fewer calls, state that stays in place, and
  fewer bits per weight. Keeping a KV cache in place on the M3's Neural Engine
  cut a Whisper decoder pass from 8.4 to 4.6 ms and its power from 1.5 W to
  0.3 W ([WhisperKit](https://arxiv.org/html/2507.10860)).
- **ADR-0075 left the door open.** It kept the talker and code predictor on MLX
  because they were estimated to run 2 to 3 times slower on the M3 Max's Neural
  Engine. Its condition for revisiting was keeping the GPU free mattering more
  than TTS speed. On a phone it does.

## Decision

**1. Four releases, each with its own PRD.**

1. Read-aloud ([#515](https://github.com/spokvulcan/tesseract/issues/515)).
2. A plain chat with an on-device model: no agent tools, no memory, and web
   access if the phone can do it well.
3. Dictation.
4. The Companion, as research only until its phone UX is clear.

The app is Tesseract from release 1 and keeps the bundle id
`app.tesseract.agent`, so every release updates the same app.

**2. Release 1 reads aloud with Qwen3-TTS 0.6B CustomVoice, and nothing
else.** There is no language model, no memory and no agent. The checkpoint's
nine speakers are Preset Voices. `AVSpeechSynthesizer` reads while the neural
voice is unavailable, and it is never a voice identity.

**3. The whole voice runs on the Neural Engine, never on the GPU.** The talker,
the code predictor and the codec are Core ML models loaded with
`.cpuAndNeuralEngine`, and the compute plan must put every op on the Neural
Engine. That is ADR-0075's rule, extended from the codec to the whole model. The
CPU reads text embedding rows from disk and supplies sampling noise. The graphs
take what TTSKit and WhisperKit measured:

- the code predictor's 15 passes are one call per frame, with sampling inside
  the graph;
- the talker's KV cache stays in place, in Core ML's stateful models;
- activations are fp16 and weights 8-bit, or 4-bit if 4-bit passes a listening
  test.

The talker also returns the Alignment Head's attention row, so Word Timing
works as on the Mac (ADR-0077).

**4. We write the graphs, through public Core ML.** `MLProgramBuilder` grows from
the codec to the whole model and builds it on the device from the checkpoint,
as ADR-0075 does. If compiling on the phone proves too slow or too
memory-hungry, the built models are published to the owner's Hugging Face org
instead. TTSKit is a reference we read, not a dependency, the stance ADR-0071
takes toward upstream code.

**5. Battery and heat are budgets that gate the voice.** #515 holds the
starting values:

- at most 5 % of the battery per hour of listening, screen off;
- thermal state nominal or fair through that hour;
- real-time factor at most 0.67, so 1.5× playback never runs dry;
- first audio within 300 ms.

When the phone gets hot, a Thermal Policy hands reading to the system voice.
Slice 1 measures the budgets and the weight precision on the owner's iPhone
16 Pro Max, and this ADR is amended with what the owner confirms.

**6. Reading with the screen locked depends on Background Inference.** If Apple
grants the entitlement and it is enough for an app kept running by background
audio, synthesis continues with the screen locked. If not, synthesis runs far
ahead while the app is in front. Reading then stops at a sentence boundary
when that buffer runs out, and a notification says why.

## What this changes in earlier ADRs

- **ADR-0066, decision 1** (the phone runs the full agent, and a plain chat was
  rejected) is superseded. Release 1 has no language model. Release 2 is the
  plain chat ADR-0066 rejected, because the owner chose a smaller second step.
  Whether the agent comes to the phone is a later release's decision.
- **ADR-0066, decision 2** (the phone's own memory, stamped for a later merge)
  waits for the agent. Episode Origin isn't built now.
- **ADR-0066, decision 3** (a second target, shared source, no `#if os`, the
  package extraction deferred) stands. Release 1's target takes only what
  read-aloud needs.
- **ADR-0066's amendment** fitted the agent to an 8 GB phone. None of its
  details are built for release 1: Device Tier, the Foreground Gate, the voice
  yielding to memory, the SSD prefix tier and the published LLM checkpoint.
  Release 2's PRD decides which of them return.
- **ADR-0066's phone vocabulary**, "Voice Input, never dictation", assumed iOS
  has no way to put text into other apps. A custom keyboard is such a way, so
  release 3's PRD settles the name.
- **ADR-0075**: on the phone, the talker and code predictor join the codec on
  the Neural Engine. The Mac keeps the 1.7B voice on MLX, with its codec on the
  Neural Engine.

## Considered and rejected

- **The agent first** (ADR-0066 as amended). It needed a 3.5 GB download before
  chat worked and an 8 GB floor. Its read-aloud was slice 9 of 10, gave way to
  the LLM for memory, and stopped soon after the screen locked. The owner
  wants the most useful piece on the phone first. Read-aloud alone should need no
  increased memory limit, fits any phone that keeps up in real time, and ships
  sooner.
- **Qwen3-TTS on the GPU through our MLX code** (the 2026-09-24 plan). It
  can't read past its buffer with the screen locked, and the owner wants the
  voice off the GPU for heat. Slice 1 still runs the same graphs on the GPU, so
  the choice is checked against numbers.
- **Kokoro-82M.** It is non-autoregressive, with one pass per sentence. It runs
  at real-time factor 0.08 on an iPhone 16 Pro
  ([speech-swift](https://github.com/soniqo/speech-swift/blob/main/docs/benchmarks/ios-coreml.md)),
  so it would keep the Neural Engine busy about an eighth as long. The owner
  set it aside, for four reasons:
  - none of its code is ours;
  - it speaks neither Russian nor German;
  - its reference phonemizer falls back to espeak-ng, which is GPL;
  - the Mac speaks Qwen3-TTS.

  If Qwen3-TTS can't meet the budgets even at 4 bits, the owner chooses
  between relaxing them and reopening this.
- **Private Neural Engine APIs** (the reverse-engineered maderix/ANE path,
  already rejected for the Mac in ADR-0075).
  - App Review guideline 2.5.1 allows public APIs only
    ([guidelines](https://developer.apple.com/app-store/review/guidelines/)).
  - App Store Connect scans every upload for non-public symbols, TestFlight
    builds included.
  - Hiding them is "trying to trick the review process", which costs the
    developer account. That is the same account that signs the Mac's
    notarized releases.
  - ADR-0075 put the saving at about 0.25 ms per call, under 1 % of a frame
    here.

  Development builds on the owner's own phone may use them for research, and
  nothing that ships does.
- **Core AI** (iOS 27). It is Apple's new framework for custom models, and its
  ahead-of-time compilation would shorten preparing the voice on the phone. It
  isn't used in release 1, for three reasons:
  - the app supports iOS 26;
  - our builder writes Core ML programs;
  - Core AI can only prefer the Neural Engine, while Core ML can rule the GPU
    out ([SpecializationOptions](https://developer.apple.com/documentation/coreai/specializationoptions)).

  #515's third slice looks at it again.
- **TTSKit's prebuilt Core ML models as a dependency.** They are graphs we
  can't change: no Alignment Head output, and no say in state layout or
  precision.

## Consequences

- #515 is release 1's PRD. Its first slice runs on the owner's phone, and its
  results amend this ADR.
- The Mac doesn't change.
- `MLProgramBuilder` grows from one fixed codec graph to transformer ops:
  attention over carried state, RMSNorm, RoPE and gathers. Each is pinned by a
  tiny model compiled on the CPU, as the codec is.
- `Voice` gains the Preset Voice case.
- Episode Origin, Device Tier, the Foreground Gate and the published LLM
  checkpoint wait for the release that needs them.

## Accepted costs

- The phone gets no chat until release 2 and no memory until later still.
- Qwen3-TTS keeps the Neural Engine busy far longer than a small model would,
  so the budgets may need relaxing.
- Reading with the screen locked rests on an entitlement Apple doesn't
  document for read-aloud apps.
- The phone's voice differs from the Mac's, as ADR-0066 already accepted.
