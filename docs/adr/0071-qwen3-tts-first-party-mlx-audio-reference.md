# ADR-0071: Qwen3-TTS inference is first-party code; mlx-audio-swift becomes a reference

- Status: Accepted
- Date: 2026-09-26
- Supersedes: ADR-0036
- Relates to: ADR-0037 (one checkpoint), ADR-0038 (engine boundary),
  ADR-0066 (iOS second target), ADR-0072 (Reference Take)

## Context

ADR-0036 re-vendored upstream mlx-audio-swift v0.1.3 as a full 1:1 tree, with
a ledger of every divergence, so a later sync would be a clean three-way
re-port. That only pays if upstream keeps improving the code we run, and it
stopped:

- Upstream has not touched its Qwen3-TTS model since June 14, before our base.
  None of its 25 commits since v0.1.3 touch it, and v0.1.3 is still its latest
  release.
- The Qwen3-TTS fixes that do land go into Python mlx-audio. The end-of-speech
  sampler fix (#567) came from there; the Swift port never took it.
- The full tree cost us every day. It was 416 files and 122K lines of Swift,
  of which the app compiled about 55K (15 TTS model families, 13 codecs) to
  run one 4.7K-line model. It loaded the codec's encoder, 225 MB of F32
  weights that only voice cloning from reference audio uses. And it carried
  two couplings nobody chose: the app downloaded models through the vendor's
  resolver, and the vendor's manifest was the only declarer of the
  swift-transformers fork pin.

Meanwhile the owner wants to keep optimizing Qwen3-TTS inference, on the Mac
and later on the phone (ADR-0066), which means changing this code often.
ADR-0036 named the exit for this case: absorb the code we already ship.

## Decision

The Qwen3-TTS model code is a first-party target, `Qwen3TTS`, in
`Vendor/tesseract-speech` beside the engine. It keeps the talker, the code
predictor, the configuration, the speech tokenizer's streaming decoder, and
the VoiceDesign and CustomVoice generation paths. The rest of mlx-audio-swift
is deleted: the other model families, the codec encoder, the batch decoder,
the speaker encoder, voice cloning from reference audio, and the Hub download
code.
`THIRD_PARTY.md` records the provenance and the MIT license.

- No fork and no submodule. A fork earns its keep when upstream keeps changing
  the code we carry. Here it would add a second repo and a pin bump to every
  experiment, with nothing to pull.
- Upstream is something we read. Python mlx-audio's `qwen3_tts`, Qwen's
  official implementation and soniqo/speech-swift are the references; a fix
  from them is ported by hand when it earns it, like any other change.
- The move was checked bit for bit. The v2-listen harness rendered
  byte-identical audio before and after (three long-form segments and seven
  companion lines, seed 42, q6), at the same speed. Peak RSS fell from
  2.87 GB to 2.65 GB because the encoder no longer loads.
- The app owns its model download code and declares its own swift-huggingface
  dependency, which its LLM loaders already import. tesseract-speech now
  declares the swift-transformers fork pin, so the lockstep mlx-swift pin
  lives in two manifests instead of three.
- The patch ledger retires with the tree. With no upstream to re-port onto, a
  divergence is just our code; the behaviors that matter keep their tests in
  `Qwen3TTSTests`.

## Consequences

- The bench-339 and spike-smoke tools went with the tree; v2-listen is the
  harness.
- Upstream PRs for the portable fixes (the warm-cache mask fix, seed, the EOS
  sampler fix) are goodwill now, not maintenance relief.
- Loading refuses Base checkpoints, which the engine never ran.
- If the model code should ever become a published library, `git subtree
  split` can lift the target into its own repo with its history.
