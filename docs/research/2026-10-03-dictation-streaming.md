# Dictation: can it stream without losing accuracy?

**Date:** 2026-10-03
**Question:** The owner picked Catch, the Dictation concept whose overlay shows the words while they are spoken. Can Tesseract's dictation stream in real time, and what does streaming cost in accuracy and in the wait after the key comes up?

**Method.** Read-only copy of the owner's Capture Dump on 2026-10-03: the last 100 takes (2026-09-27 to 2026-10-03, 21 minutes, median 8.9 s, longest 80 s), 97 of them with their Correction Pair. Each take was played against a simulated recording clock on this Mac (M3 Max, 48 GB, macOS 27.0.1): a decode that starts when *t* seconds of audio exist shows its text at *t* plus its measured decode time, so every number below is what the overlay would have shown, when. The decodes ran through the app's own WhisperKit revision (1.0.0), model (large-v3-turbo), decoding options and resampler. Streaming schemes simulated: the app's existing Live Partial pump (the trailing 12 s re-decoded, 300 ms after the last decode lands) and WhisperKit's own `AudioStreamTranscriber` rule (decode from the end of the last confirmed segment once a second of new audio exists; all but the last two segments become confirmed; on release only the tail is decoded). Apple's on-device `SpeechAnalyzer` (macOS 27, `SpeechTranscriber` and `DictationTranscriber`, both already installed for en-US) ran on the same takes, whole-file for accuracy and fed at speaking speed for timing. There are no hand references for these takes: where two systems disagree, Parakeet TDT 0.6B v2 (another model family) decides the region, and a region it hears differently from both stays undecided. Fillers are dropped and case and punctuation ignored when counting words. Per-take transcripts stay outside the repository, and so do the replay tools and their outputs.

---

## 1. Findings

### 1.1 Streaming costs no accuracy as long as the pasted text is the full pass

The text the app pastes is one full-take decode after release. A Live Preview only changes what the overlay shows, so the pasted text stays exactly as accurate as today. (The replay reproduces the app's own stored text exactly in 87 of 97 takes; the rest differ in a word or two, the same spread as two runs of the replay.)

### 1.2 Pasting Whisper's streamed text loses accuracy on long takes

WhisperKit's `AudioStreamTranscriber` rule confirms early segments from decodes that had not heard what came next. Up to 20 s it rarely confirms anything and the final matches the full pass in 75 of 82 takes; over 20 s it matches in 3 of 18. Overall 2.6% of words differ, and where they differ Parakeet sides with the full pass on 24 words and with the streamed text on 10 (20 undecided). The errors are real words lost or changed at segment boundaries, not only punctuation. What it buys is a shorter wait after a long take (1.26 s for the tail against up to 3 s), which the GPU encoder (1.6) gets without the loss.

### 1.3 Apple's recognizer is fast and streams natively, but is less accurate on this owner's speech

`SpeechTranscriber` differs from Whisper in 11.6% of words; where they differ Parakeet sides with Whisper on 113 words and with Apple on 40 (91 undecided). It does worst on technical words ("backend" as "sepicant", "KV cache" as "qvc", "HTML" as "a stream", "team lead" as "kimuid") and keeps fillers (154 "uh" and "um" in 100 takes; the app's cleanup removes none). `DictationTranscriber` is worse (174 against 24). Fed at speaking speed, its words appear 0.82 s behind the voice (p90 1.24 s), it updates every second, its first words come after 2.0 s, and its final text is ready 0.08 s after the audio ends. It mishears the owner's terms differently from Whisper, so a "heard → meant" rule learned from Whisper's text would not fire on its preview.

### 1.4 A Whisper preview on the Neural Engine runs 2 s behind and makes the paste wait

Every decode costs about 1.25 s, 1.08 s of it in the encoder, whatever the audio length (the encoder always runs a 30 s window). The app's Live Partial pump (the trailing 12 s re-decoded 300 ms after each decode lands) shows a word a median 2.04 s after it is spoken (p90 2.82 s), updates every 1.53 s, shows its first words after 2.35 s, and 20% of words appear only after release. The recognizer is busy 67% of the take. The encoder cannot be cancelled once running, so a preview in flight at release delays the final pass by 0.44 s on average (p90 1.07 s, in 72 of 100 takes). The confirmation rule as a preview is no faster (1.97 s behind, updates every 1.29 s, first words after 3.17 s).

### 1.5 The key comes up half a second after the last word

From Whisper's word timestamps, the owner releases the key a median 0.52 s after the last word ends (p10 0.30 s, p90 0.81 s), and the audio level before release is not silence. Starting the final pass at a pause (0.3 s after the last frame above the noise floor) therefore starts at release in the median take and saves nothing; its text matched the full pass in 90 of 100 takes either way.

### 1.6 The encoder on the GPU halves the wait and the lag

With `audioEncoderCompute: .cpuAndGPU` the encoder takes 0.40 s instead of 1.08 s (after a one-time 4.3 s shader compile at load). The full pass lands 0.65 s after release (p90 1.21 s) against 1.29 s (p90 1.63 s). A preview (25 takes) shows a word 1.04 to 1.12 s behind the voice (p90 about 1.65 s), updates every 0.9 to 1.0 s, shows its first words after 1.67 s with the pump's pacing, and makes the paste wait 0.08 to 0.11 s on average; the recognizer is busy 45 to 56% of the take. The text matches the Neural Engine's in 88 of 100 takes and differs in 13 of 2,111 words, against 14 between two Neural Engine runs; where GPU and Neural Engine disagree Parakeet sides with the GPU on 6 words and the Neural Engine on 2.

### 1.7 With the 27B model generating, the GPU still wins for dictation, at a cost to the generation

With Qwen3.8-27B (4-bit, MLX) decoding beside the replay: alone it runs 21.8 tokens/s; beside Whisper on the GPU, 14.8 tokens/s while Whisper decodes (about 15% averaged over a take, since a GPU preview decodes about 45% of the time); beside Whisper on the Neural Engine, 21.2 tokens/s. Whisper slows too: on the GPU a preview decode takes 1.36 s and the full pass 1.45 s (0.75 s on the same takes with the LLM idle); on the Neural Engine 1.97 s and 1.96 s (1.44 s idle), most likely because the two share memory bandwidth. Dictation is faster on the GPU whether the LLM is idle or not.

## 2. Recommendation

1. **Stream in the overlay, paste the full pass.** No accuracy cost; the pasted text is the same decode as today.
2. **Use Whisper for the preview, not Apple's recognizer**, so the preview mishears the way the final does and learned words flip live. Decode from the end of the last confirmed segment so the overlay can show the whole take at a bounded cost, paced like the existing pump (from 0.6 s of audio, 300 ms after each decode lands).
3. **Move the encoder to the GPU.** It halves the wait after release (0.65 s against 1.29 s) and the preview lag (about 1 s against 2 s) with the same text. The cost is a generation running at the same time, about 15% slower over a take, in line with ADR-0081. If that matters in practice, keep a second encoder on the Neural Engine for while the LLM Gate is held (about 1.2 GB more memory).
4. **Do not paste streamed text**, from WhisperKit's rule or from Apple's recognizer, and do not start the final pass at a pause.

The design that uses this is the Catch PRD, #612.
