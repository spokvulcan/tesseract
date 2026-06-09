# Tesseract

A personal AI assistant that runs entirely on your Mac. No cloud. No accounts. No telemetry. Your data never leaves your device, and you give up nothing for it. Tesseract is living proof that frontier AI runs on hardware you already own. We intend to keep pushing that frontier.

## Features

- **Personal AI Assistant** — An on-device tool-calling LLM agent that remembers you across conversations. Helps you achieve your goals, build habits, and stay on track. Sets reminders. Learns your preferences. Talk to it by voice or text. You can trust it with your whole life precisely because it never leaves your Mac.
- **Inference Server** — The foundation the agent stands on, and a product in its own right. An OpenAI-compatible `/v1/chat/completions` server with tiered RAM + SSD radix prefix caching delivers cache hit rates no other on-device stack matches. Wire it into any OpenAI-compatible agent harness (opencode, OpenClaw, or your own) as a fully local backend for coding agents.
- **Dictation** — Push-to-talk voice-to-text. Hold a hotkey, speak, release. Your words are typed into whatever app you're using. Fully offline.
- **Text-to-Speech** — Natural, consistent speech synthesis with voice anchoring for long-form content. State-of-the-art quality, entirely on device.
- **100% offline** — Powered by open models via MLX. Works without internet after the initial model download.

## Requirements

- macOS 26+
- Apple Silicon (M1 or later)

## Development

```bash
scripts/dev.sh dev          # Build Debug, kill running app, launch new build
scripts/dev.sh dev-release  # Same, with the Release configuration
scripts/dev.sh log          # Tail app logs
scripts/dev.sh clean        # Clean build artifacts
```

Run `scripts/dev.sh` with no arguments for the full command list. See
[CLAUDE.md](CLAUDE.md) for working conventions, [ARCHITECTURE.md](ARCHITECTURE.md)
for design, and [CONTEXT.md](CONTEXT.md) for domain vocabulary.
