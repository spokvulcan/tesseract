# Tesseract

A personal AI assistant that runs entirely on your Mac. No cloud. No accounts. No telemetry. Every model runs locally on Apple Silicon — your data never leaves your device.

## Features

- **Personal AI Assistant** — An on-device LLM agent that remembers you across conversations. Helps you achieve your goals, build habits, and stay on track. Sets reminders. Learns your preferences. Talk to it by voice or text — everything stays on your Mac.
- **Dictation** — Push-to-talk voice-to-text. Hold a hotkey, speak, release. Your words are transcribed with WhisperKit and typed into whatever app you're using.
- **Text-to-Speech** — Natural speech synthesis powered by Qwen3-TTS. Real-time, consistent voice with voice anchoring for long-form content.
- **Image Generation** — Create images from text using FLUX.2 diffusion models. *(In development)*
- **100% offline** — Powered by open models via MLX. Works without internet after the initial model download.

## Requirements

- macOS 26+
- Apple Silicon (M1 or later)

## Development

```bash
scripts/dev.sh dev     # Build, kill running app, launch new build
scripts/dev.sh log     # Tail app logs
scripts/dev.sh clean   # Clean build artifacts
```

See [CLAUDE.md](CLAUDE.md) for architecture details, build commands, and contribution guidelines.
