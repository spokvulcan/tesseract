# LFM3-Audio measured: the recognizer and the voice, against Whisper and Qwen3-TTS

**Date:** 2026-10-28 · **Scope:** step 1 of the order in [the voice-pipeline note](2026-10-28-lfm3-audio-voice-pipeline.md) (§6, "Before any app code: measure"). LFM3-Audio-2B as the Mac's speech recognizer against the app's WhisperKit (`whisper-large-v3-turbo`), and as its voice against the app's Qwen3-TTS. No app code changed.
**Method:** a throwaway Swift harness on the fork at `595b243`, `Vendor/tesseract-speech` at `259325f6` and WhisperKit (argmax-oss-swift 1.0.0, the app's pin), built Release with xcodebuild. It ran on the M3 Max (48 GB, macOS 27.0.1, Xcode 27.1) with the owner's Tesseract Agent app open the whole time at about 85% CPU; GPU use was 0–6% before each run. The harness, scripts, texts, raw results and WAVs are in the worktree's git-ignored `research/lfm3-audio-2026-10-28/` (§7). No clip holds the owner's data.

---

## In short

- **Accuracy: close to Whisper, a little behind on real speech.** On 114 public-domain read-speech clips LFM3-Audio 8-bit makes 3.70% word errors against Whisper's 2.66%: 1.0 point more, 95% interval 0.2 to 1.8. On 60 technical dictation sentences voiced by Qwen3-TTS they tie (2.35% against 2.21%), and both caught all 96 technical terms. 4-bit recognizes as well as 8-bit.
- **Idle, it is faster.** An 11 s take comes back in 0.29 s at 8 bits against Whisper's 0.79 s (medians of four). Over the replay, 8–14 s takes: 0.45 s at 8 bits, 0.31 s at 4 bits, 0.71 s for Whisper.
- **During an agent prefill in the app's process, it stalls.** With Qwen3.8-27B prefilling 8,192 tokens in the same process, the 11 s take came back after 26.8 s (24.7–28.2 over four rounds), against Whisper's 6.0 s. MLX's lock makes it take turns with the prefill's 4.85 s chunks. In a process of its own it took 2.2 s, against Whisper's 4.1 s. Beside a decode the two are close: 2.1 s against 1.5 s.
- **With no speech, it invents text.** Digital silence became a whole made-up sentence at 8 bits, and a phrase looped to the 1,024-token cap at 4 bits (6.1 s). The app's −55 dBFS silent-capture skip keeps silence away from either recognizer; noise above it became "Thank you." from both.
- **The voice: as fast as Qwen3-TTS, first audio sooner, fully intelligible.** First audio 43 ms against 90 ms for the app's pinned voice (58 ms on a session's first line); both run at a real-time factor of 0.14. Both recognizers transcribe both voices' ten lines without an error, and LFM3-Audio ended its speech by itself in all 26 renders. Whether it sounds good, and stays one person, needs the owner's ears: the blind-listen WAVs are in `research/lfm3-audio-2026-10-28/listen/`.
- **Recommendation:** do not wire LFM3-Audio in as the recognizer in the app's process. The prefill stall rules it out, and accuracy gives no reason to switch. In its own process it would be fast, but that is a bigger build, worth deciding only after replaying the owner's own takes. The voice passes on speed and intelligibility; the owner's blind listen decides whether the synthesizer (Slice 2) comes next (§5).

## 1. Method

### 1.1 Clips

| Set | Clips | Audio | Source |
| --- | --- | --- | --- |
| LibriSpeech dev-clean, reader 1272 | 73 | 481 s | public-domain LibriVox reading via LibriSpeech (CC BY 4.0): Hugging Face's `hf-internal-testing/librispeech_asr_dummy` sample, 9.2 MB |
| LibriSpeech test-clean, reader 6930 | 20 | 165 s | `sanchit-gandhi/librispeech_asr_dummy`, 3.2 MB |
| LibriSpeech test-other, reader 7902 | 20 | 110 s | the same, the harder split, 1.9 MB |
| JFK inaugural | 1 | 11 s | public domain, from WhisperKit's test resources (already on disk) |
| Technical sentences | 60 | 262 s | 30 dictation-style lines written for this test, rendered by the app's Qwen3-TTS in two designed voices (a young American man, a middle-aged British woman) |
| Long takes | 4 | 294 s | 62–85 s each: consecutive utterances of one reader, 0.5 s apart |
| No speech | 4 | 40 s | 10 s each of digital silence, white noise at −45 dBFS, pink noise at −30 dBFS, 50 Hz hum at −35 dBFS |
| The voices under test | 20 | 107 s | the ten Companion lines of §4 in each voice, for intelligibility |

Every clip is a 16 kHz mono float WAV, and both recognizers get the same samples. The app's resample from the device rate happens before this and is not timed.

### 1.2 The two recognizers

- **Whisper** is the app's `WhisperKitSpeechRecognizer`, option for option: the audio encoder on the GPU (`.cpuAndGPU`), prewarm, greedy, `language: "en"` with the prefill prompt, `noSpeechThreshold` 0.6, VAD chunking, and the text of every chunk joined.
- **LFM3-Audio** is the adapter the note sketches (§2.1): a `VLMModelFactory` container, a system turn `Perform ASR in English.` and a user turn holding only the clip, greedy decoding, at most 1,024 tokens, through `MLXLMCommon.generate`. The stop tokens come from the pack's `generation_config.json`.
- Each recognizer is warmed with a silent second and one clip, then makes one pass over every clip.

### 1.3 Scoring

Text is normalized with Whisper's English normalizer (transformers 4.51.3, with no spelling map), and the word error rate is counted over each set: errors over reference words. Differences come with a paired bootstrap over clips (2,000 resamples). A spacing-insensitive variant counts a split or joined compound as right ("swift ui" for "SwiftUI", "mantelboard" for "mantel board"). For the technical sentences, 96 technical terms (`ci`, `json`, `swiftui`, `kubernetes`, `provisioning profile`…) are checked in each transcript with spaces ignored.

### 1.4 Latency under load

One process holds Qwen3.8-27B (4-bit MLX, the app's text-only load with its projections stacked and the MLX buffer pool capped at 512 MB), LFM3-Audio 8-bit and Whisper, as the app would if LFM3-Audio were its recognizer. The agent prefills an 8,192-token prompt in the app's 1,024-token chunks through the fork's standard prefill (`TokenIterator`, `PrefillParameters(stepSize: 1024)`), or decodes 400 tokens. The 11 s JFK clip is transcribed idle, 2.4 to 3.7 s into a prefill, and 1.2 to 1.6 s into a decode. Four rounds alternate the recognizers and run each load alone too. A second experiment puts the agent and the recognizer in two processes (§3.3).

### 1.5 The voices

- **LFM3-Audio:** `LFM3AudioGenerator` with `output: .speech` on the 8-bit pack, the card's sampling (temperature 1.0, top-k 128, repetition penalty 1.1), the system turn `Perform TTS in American English using a female voice.` and the line as the user turn. The frames go into one `LFM3AudioCodec` stream: the first frame alone, then five at a time, the note's guess (§3.2). First audio is timed from the prompt's preparation to the first decoded samples.
- **Qwen3-TTS:** the app's Voice Engine, `SpeechEngine` over `Qwen3Synthesizer` on VoiceDesign 1.7B at the shipped 6 bits, its Neural Engine codec, default `TTSParameters`, the `.companion` profile (a pinned session: its first line becomes the Reference Take). The voice is v2-listen's narrator, "A calm, warm female narrator with a clear, steady tone.", because the owner's own designed voice sits in the app's container, which the harness cannot read. First audio is timed from `speak` to the first audio event.
- Each voice renders a warm-up line first, then the ten Companion lines at fixed seeds; then a 95-word passage, whole and one sentence at a time.

## 2. Recognition accuracy

| Set | Clips | Reference words | Whisper | LFM3 8-bit | LFM3 4-bit |
| --- | --- | --- | --- | --- | --- |
| Read speech (LibriSpeech + JFK) | 114 | 2,029 | 2.66% (2.56%) | 3.70% (3.41%) | 3.60% (3.31%) |
| · dev-clean, reader 1272 | 73 | 1,169 | 3.59% (3.42%) | 4.53% (4.37%) | 4.45% (4.28%) |
| · test-clean, reader 6930 | 20 | 443 | 0.90% (0.90%) | 2.71% (1.81%) | 2.71% (1.81%) |
| · test-other, reader 7902 | 20 | 395 | 2.03% (2.03%) | 2.53% (2.53%) | 2.28% (2.28%) |
| · JFK | 1 | 22 | 0% | 0% | 0% |
| Technical sentences (Qwen3-TTS) | 60 | 680 | 2.21% (0.44%) | 2.35% (0.59%) | 2.35% (0.59%) |
| Long takes, 62–85 s | 4 | 745 | 2.42% (2.15%) | 3.22% (3.22%) | 3.49% (3.49%) |
| The two voices' lines | 20 | 288 | 0% | 0% | 0% |

Word error rate; in parentheses, spacing-insensitive.

- **Against Whisper**, with 95% intervals: read speech +1.03 points [+0.24, +1.81] at 8 bits and +0.94 [+0.20, +1.74] at 4 bits; technical sentences +0.15 [−0.89, +1.31], so no measurable difference; long takes +0.81 [+0.54, +1.07].
- **What LFM3-Audio gets wrong that Whisper gets right** is mostly names and near-homophones: Raoul → "Rao", Brion → "Brian", "thrust" → "trust", "starlit" → "starlet", "thought" → "said", Risdon Graeme → "Risden Graham". Four of its errors are two of LibriSpeech's old spellings ("to day", "to morrow"), which Whisper writes as the references do.
- **Error kinds** on read speech (substitutions / deletions / insertions): Whisper 42/6/6, LFM3 8-bit 63/11/1. Whisper's insertions include a phantom "i am not" at the end of a clip.
- **Technical terms:** 96 of 96 for every recognizer. The spacing errors split between them: LFM3 wrote "swift ui" and "tesseractspeech", Whisper "web socket".
- **Long takes** decode in one pass with no chunking, in 2.3–3.1 s at 8 bits and 1.6–2.1 s at 4 bits, against Whisper's 3.1–4.0 s.
- **4 bits are enough for recognition**, as the note guessed (§4): no set is measurably worse than 8 bits.
- **The technical set is easy.** Speech from a TTS model is clean, and both recognizers are near-perfect on it. It shows the vocabulary is covered, not how either copes with the owner's voice in a room.

### No speech

| Clip | Whisper | LFM3 8-bit | LFM3 4-bit |
| --- | --- | --- | --- |
| Digital silence | "Thank you." | "And so, if you look at the data, you can see that the number of people who are in the workforce is increasing." | "And then we have the second one, which is the one that we've been talking about, …" looped to 1,024 tokens, 6.1 s |
| White noise, −45 dBFS | "Thank you." | "Thank you." | "Thank you." |
| Pink noise, −30 dBFS | "." | "Thank you." | "Thank you." |
| 50 Hz hum, −35 dBFS | "." | "Thank you." | "Thank you." |

The app never sends the first row to a recognizer: `CaptureLevel` skips a take whose loudest 20 ms stays at or below −55 dBFS. The other three pass that check. There Whisper writes "Thank you." or a full stop, and LFM3-Audio "Thank you." The runaway loop is the case for capping the adapter's tokens by the clip's length rather than at 1,024.

## 3. Recognition latency

### 3.1 Idle

| Recognizer | Median, takes up to 20 s (n = 192, median 4.7 s of audio) | 8–14 s takes (n = 22) | 62–85 s takes | Real-time factor |
| --- | --- | --- | --- | --- |
| Whisper | 0.57 s | 0.71 s | 3.1–4.0 s | 0.12 |
| LFM3-Audio 8-bit | 0.28 s | 0.45 s | 2.3–3.1 s | 0.056 |
| LFM3-Audio 4-bit | 0.16 s | 0.31 s | 1.6–2.1 s | 0.032 |

Whisper's 0.71 s matches the dictation note's 0.65–0.75 s.

### 3.2 Beside the agent, in one process

The 11 s clip, median of four rounds (range):

| Recognizer | Idle | 2.4–3.7 s into an 8,192-token prefill | 1.2–1.6 s into a decode |
| --- | --- | --- | --- |
| Whisper | 0.79 s (0.71–0.83) | 5.97 s (5.42–6.06) | 1.54 s (1.45–1.61) |
| LFM3-Audio 8-bit | 0.29 s (0.24–0.31) | **26.83 s (24.65–28.23)** | 2.06 s (1.83–2.22) |

- **LFM3-Audio waits behind the prefill.** Alone, each 1,024-token chunk of the 27B takes 4.85 s (median of 32). A transcription released early in a prefill came back 25–28 s later, 10–12 s before the prefill ended. Its own evaluations take turns with the prefill's through MLX's process-wide lock and the one GPU stream (ADR-0081), and each turn waits for the chunk in flight. That is ADR-0081's "a speech frame can wait behind one prefill chunk", paid many times over by the 30 or so evaluations of one transcription. A shorter prefill should hold a take up for about what is left of it, which this run did not time. The text was right every time.
- **Whisper slows too, to about 6 s.** Its encoder runs on the GPU through Core ML, outside MLX's lock, but the GPU is busy with the prefill. The dictation note timed Whisper beside a decode only; beside a decode this run gets its 1.45 s again (1.54 s median).
- **Beside a decode, the gap is small:** 2.06 s against 1.54 s. Decode steps are short, so turns interleave finely.
- **The agent barely notices.** Its prefill took 36–42 s alone and 37–43 s with a transcription inside it, which is within the drift between rounds; its decode ran at 19–23 tokens/s either way.
- Round 3's LFM3-Audio prefill arm overlapped the pitch analysis of §4 on the CPU. It sits inside the spread of the other rounds, whose own prefills alone varied from 36 to 42 s, and without it the median is 25.85 s.

### 3.3 In two processes

The note's first fix for the lock is to run LFM3-Audio in its own process (note §2.4). So one process prefilled the agent's 8,192 tokens over and over, with 4 s pauses, while a second transcribed the clip every 1.5 s, first with LFM3-Audio and then with Whisper:

| Recognizer, in its own process | Started inside a prefill | Started between prefills |
| --- | --- | --- |
| Whisper | 4.07 s (2.73–4.16, n = 12) | 0.66 s (n = 4, one ran into the next prefill: 3.99 s) |
| LFM3-Audio 8-bit | **2.19 s** (0.45–2.31, n = 14) | 0.26 s (n = 2) |

- **Out of the agent's process, the stall is gone.** 2.2 s during a prefill, against 26.8 s in the same process. The GPU is still shared, so it is about 8× its idle time, but it no longer waits for chunks.
- **The agent did not slow down:** its four prefills took 37.1–37.9 s with a transcription running every few seconds beside them.
- **Whisper is slow during a prefill either way:** 4.1 s in its own process, 6.0 s in the agent's. Its encoder is a heavy GPU job on a GPU the prefill keeps busy, and MLX's lock was never what held it up.

## 4. The voice

| Voice | First audio, a session's first line | First audio, lines 2–10 (median, range) | Real-time factor (median, range) | Audio, 10 lines |
| --- | --- | --- | --- | --- |
| LFM3-Audio 8-bit | 45 ms | 43 ms (42–44) | 0.140 (0.138–0.143) | 49.2 s |
| LFM3-Audio 4-bit | 40 ms | 39 ms (38–39) | 0.117 (0.114–0.120) | 49.7 s |
| Qwen3-TTS 6-bit, the app's voice | 58 ms | 90 ms (89–115) | 0.142 (0.136–0.145) | 57.4 s |

- **First audio.** LFM3-Audio's first sound is one frame decoded alone (1,440 samples, 60 ms of audio). Qwen3-TTS's 58 ms on a session's first line is close to ADR-0097's 54 ms; its later lines are pinned to that line's take, and the longer prompt costs about 30 ms. LFM3-Audio's figure leaves out the engine's event stream, which Qwen's includes, so the gap is a little smaller than it looks. Both are far below what a listener notices.
- **The passage.** LFM3-Audio read the 95-word passage in one prompt: 27.4 s of audio, first audio 59 ms, real-time factor 0.133, and it ended by itself. One sentence at a time, first audio was 42–48 ms per sentence. Qwen3-TTS through its engine: 31.4 s, 58 ms, 0.135.
- **Memory.** MLX's peak allocation: LFM3-Audio 2.61 GB at 8 bits and 1.74 GB at 4 bits, weights and codec included. Qwen3-TTS 1.84 GB, with its codec on the Neural Engine outside that count; the catalog's precision gate (ADR-0037) measured its whole peak at 2.88 GB over a long read.
- **Intelligible.** Whisper and LFM3-Audio transcribe both voices' ten lines with no error (§2).
- **Ended by itself** in all 26 LFM3-Audio renders, the longest 342 frames. None reached its frame cap.
- **One person?** A crude hint, not a verdict: per line, the median pitch (pYIN) and the mean MFCCs. LFM3-Audio's ten lines stay closer in pitch than Qwen3-TTS's (171–194 Hz, standard deviation 8 Hz, against 182–225 Hz and 16 Hz) but differ more in timbre (median MFCC distance from the voice's average line 17.9 against 12.0). Each LFM3-Audio line starts from the prompt alone, while Qwen's are pinned to one Reference Take, so a timbre spread is what the note's §3.5 unknown would look like. Ears decide.

### The blind listen

`research/lfm3-audio-2026-10-28/listen/` holds the ten lines (`line01-A.wav` … `line10-B.wav`), the passage read whole (`passage-A.wav`, `passage-B.wav`) and the passage one sentence at a time, joined with 0.2 s gaps (`passage-A-by-sentence.wav`, `passage-B-by-sentence.wav`). All are 24 kHz mono WAVs, as the models wrote them, with no loudness matching. A coin flip assigned A and B; the answer is in `KEY-open-after-listening.json`, and the folder's README lists the texts and what to listen for. Two questions only ears can answer: which voice the owner would rather hear every day, and whether "female" stays one person from line to line (the note's §3.5 unknown).

## 5. Recommendation

The note's order was: measure, then ASR for dictation, then the synthesizer in its own target, and stop if step 1 or 3 fails.

- **ASR: do not build Slice 1 as sketched.** In the app's process, a take released early in an 8,192-token prefill came back 25–28 s later, against Whisper's 6 s. Prefills that long happen every day: the Morning Plan spends 34 s on its prompt (ADR-0081), about as long as this test's, so the owner would hit the stall. In a process of its own it drops to about 2 s, better than Whisper does today, and idle it answers a 10 s take in 0.45 s where Whisper takes 0.71 s. But accuracy gives no reason to switch: it is a point behind on real speech, mostly on names, level on technical vocabulary, and it writes text into silence. A helper process is a much bigger slice than the note's adapter, with its own load, lifecycle and Model Residency row behind a process boundary. Build it only if the owner's own takes justify it. So the next step is the replay this run left out by design: the latest 100 Capture Dump takes, scored against the Correction Pairs with technical words apart (note §6 step 1).
- **TTS: the owner's ears decide.** Speed, first audio and intelligibility pass, and no render ran on. If the owner prefers it blind and hears one person across the lines and the by-sentence passage, go on to Slice 2: the synthesizer in `LFM3Speech` and a v2-listen mode with peak memory. That still gives up VoiceDesign, the Pinned Voice and word timing (note §3.5–3.6). If the owner prefers Qwen3-TTS, or hears the voice change, stop here.
- **If both go ahead,** the recognizer needs a process of its own, so a shared speech-side load (Slice 4) would move the voice into that process too. Design the two slices together.
- **The owner's goal**, one model that hears them and answers in its own voice, is the speech-to-speech design (2026-10-27 note §7.3), not these two adapters. The lock carries over: such a talker in the agent's process would stall behind every 27B prefill as the recognizer did. Whether it keeps up in a process of its own is unmeasured.

## 6. What this did not measure

- **The owner's own dictation.** The note's step 1 replays the latest 100 Capture Dump takes and scores them against the Correction Pairs, technical words apart. This run used no owner data, so read speech and TTS-voiced technical sentences stand in for it. The real test is the owner's voice, microphone and room.
- **Unseen speech.** LibriSpeech is a standard training corpus, and either model may have heard these readings; Whisper writing "to day" as the references do hints at it. Read the read-speech rates as a floor.
- **The Live Preview**, which re-decodes a growing window without segments (note §2.3), and how often the Learned Words fire on LFM3-Audio's text.
- **The app's own prefill path.** The agent prefill here is the fork's standard chunk loop, which pipelines chunks with `asyncEval`. The app's prefix-cache prefill (`HybridCacheSnapshot`) has its own loop, and its chunks may hold the GPU differently.
- **PARO.** The agent was the uniform 4-bit Qwen3.8-27B. `CLAUDE.md` puts the owner's PARO load at about 35 s for an 8K-token prefill, against 36–42 s here, so its chunks should hold the GPU about as long.
- **The voice during a prefill.** Qwen3-TTS already shares the agent's process and lock, and rides through a chunk on about 8 s of lookahead (ADR-0081). LFM3-Audio's voice would sit in the same place; how it fares there is unmeasured.
- **Parity of processor-built prompts** with mlx-vlm (the note's fork-maturity row). The ASR prompts here come from `LFM3AudioProcessor`, and their output was sane.
- **A quiet machine.** The owner's app used about 85% of one core throughout; each figure here is a median or comes from alternating runs.

## 7. Reproduce

The harness is a SwiftPM package outside the app (`research/lfm3-audio-2026-10-28/harness/`; it links the worktree's `Vendor/` packages by path). Build it with xcodebuild so MLX's metallib lands next to the binary, as for v2-listen:

```bash
xcodebuild -scheme lfm3lab -destination 'platform=macOS' -derivedDataPath dd -configuration Release -skipPackagePluginValidation -skipMacroValidation CODE_SIGNING_ALLOWED=NO build
```

The modes (`asr`, `tts-lfm3`, `tts-qwen`, `contend`, `prefill-loop`, `asr-loop`) are listed at the top of its `main.swift`. `scripts/` holds the clip builder, the scorer and the run scripts, `texts/` the sentences, and `results/` every transcript and timing as JSON.
