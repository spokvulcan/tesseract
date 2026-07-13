# Adding a New Model

Use the codebase as the source of truth. The fastest path is to find the
closest existing model and follow its structure rather than treating this file
as a full implementation spec.

## 1. Start from a similar model

- TTS: `Sources/MLXAudioTTS/Models/`
- STT: `Sources/MLXAudioSTT/Models/`
- STS: `Sources/MLXAudioSTS/Models/`
- Codecs: `Sources/MLXAudioCodecs/Models/`

Pick the nearest match in task and architecture, then mirror its layout,
naming, and loading flow.

## 2. Match the right protocol

Implement the protocol for the model's task:

- TTS: `SpeechGenerationModel`
- STT: `STTGenerationModel`
- Codec: `AudioCodecModel`

Keep the public API aligned with similar models already in the repository.

## 3. Keep configuration and loading Swift-native

- Decode `config.json` into a `Codable` + `Sendable` config type.
- Map JSON fields with `CodingKeys` when needed.
- Load weights from HuggingFace in `fromPretrained()`.
- Add any key renames or tensor reshaping needed to match the Swift model.

## 4. Register the model

Factory registration is required.

- Add the model type to the appropriate enum.
- Update the corresponding factory switch in TTS/STT/STS/codec loading.
- Make sure HuggingFace `model_type` resolution reaches the new case when
  applicable.

## 5. Publish weights outside this repository

Do not commit model weights here. Publish them on HuggingFace, preferably under
[`mlx-community`](https://huggingface.co/mlx-community) when that fits the
model.

Include the model repo in the PR description.

## 6. Add validation

- Add tests for the main generation path.
- Add streaming tests when the model supports streaming.
- Add the model to `README.md` if it is user-facing.

## PR checklist

- [ ] Model lives in the correct module directory
- [ ] Model conforms to the correct protocol
- [ ] Config loading and weight loading work from HuggingFace
- [ ] Factory registration is complete
- [ ] Tests cover the main inference path
- [ ] `README.md` is updated when needed
