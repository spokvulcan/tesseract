# Headless TTS (Console)

This repository includes a console-first headless TTS workflow for Qwen3-TTS:

- Script: `scripts/tts-headless.sh`
- Backend executable: `Vendor/mlx-audio-swift` product `mlx-audio-swift-tts`

## Why this wrapper exists

Running the raw SwiftPM executable can fail with:

`Failed to load the default metallib`

The wrapper handles this by staging `default.metallib` next to the binary before launch.

## Build

```bash
scripts/tts-headless.sh build
```

This builds the headless CLI and ensures `default.metallib` is available for runtime.

## Run

```bash
scripts/tts-headless.sh run \
  --text "Hello from headless mode." \
  --output /tmp/headless-tts.wav
```

Notes:

- If `--model` is omitted, the script defaults to:
  - `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- Relative `--output` and `--ref_audio` paths are converted to absolute paths.

## Smoke test

```bash
scripts/tts-headless.sh smoke
```

This generates a short WAV (default: `.artifacts/headless-smoke.wav`) and fails if no file is produced.

## Performance/profile run

```bash
QWEN3TTS_PROFILE=1 scripts/tts-headless.sh run \
  --text "Profile this generation path." \
  --output /tmp/headless-profile.wav \
  --max_tokens 256
```

## Troubleshooting

If metallib resolution fails:

1. Build the macOS app once so Xcode emits `default.metallib`:
   - `xcodebuild build -project tesseract.xcodeproj -scheme tesseract`
2. Or set an explicit path:
   - `export MLX_DEFAULT_METALLIB=/absolute/path/to/default.metallib`

Optional override for alternate DerivedData root:

- `export DERIVED_DATA_DIR=/custom/DerivedData`
