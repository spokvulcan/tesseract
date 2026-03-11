# App Store Metadata

## App Name

Tesseract Agent

### Name Research

**Rationale:**

- "Tesseract" is taken by multiple apps on the Mac App Store (VM manager, puzzle games, journal app, workforce management)
- The hyphen makes it unique, distinctive, and avoids naming conflicts
- Lowercase style fits modern app branding (bear, craft, things)
- "Tess" can serve as the name for the future local AI assistant

**Availability (Feb 2026):**

| Platform         | Status             | Notes                          |
| ---------------- | ------------------ | ------------------------------ |
| Mac App Store    | Clear              | No apps named "tesse-ract"     |
| thetesseract.app | owned by me        | and will be used for the app   |
| USPTO Trademark  | Needs manual check | Search at uspto.gov/trademarks |

---

## Subtitle (30 chars max)

On-Device Intelligence

## Promotional Text (170 chars max)

AI that lives on your Mac. Dictation, text-to-speech, image generation, and more — powered by open models, processed entirely on-device. No cloud. No accounts. Just you.

## Description

Tesseract Agent brings intelligence to your Mac without sending a single byte to the cloud. Every model runs locally on Apple Silicon — your data never leaves your device.

Voice

Press a hotkey, speak, release. Your words are transcribed with Whisper and typed directly into whatever app you're using. Push-to-talk means it only listens when you ask it to.

Speech

Natural text-to-speech powered by Qwen3-TTS. Hear any text read aloud with consistent, high-quality voice synthesis — generated in real time, entirely on-device.

Image Generation

Create images from text descriptions using on-device diffusion models. No waiting for a server, no usage limits, no content filtering by a third party.

What's Next

Local LLM chat and a general-purpose AI agent — all running privately on your Mac.

Built Different

- 100% offline — works without internet after model download
- No accounts, no subscriptions, no telemetry
- Apple Silicon optimized with Core ML and MLX
- Open models you can inspect and replace
- Your data stays on your Mac. Period.

## Keywords (100 chars max)

dictation,voice,tts,speech,ai,offline,privacy,whisper,image,generation,local,llm,productivity,mac

## Category

Primary: Productivity
Secondary (optional): Utilities

## Support URL

https://thetesseract.app/support

## Marketing URL (optional)

https://thetesseract.app

## Privacy Policy URL

https://thetesseract.app/privacy

## App Review Notes

Tesseract Agent uses:

- Microphone access for recording voice dictation.
- Accessibility permission to capture the global hotkey and to simulate paste into the active app.
- Network access for one-time model downloads from Hugging Face.

All AI inference runs locally on-device. There is no account login, no telemetry, and no data collection.

## App Review Test Instructions

1. Launch the app and complete onboarding.
2. When prompted, grant Microphone access.
3. When prompted, grant Accessibility access in System Settings > Privacy & Security > Accessibility.
4. Open TextEdit.
5. Press and hold Option+Space to record, speak a short sentence, and release.
6. The transcription is typed into TextEdit.

Default hotkey: Option+Space (customizable in Settings).

## Export Compliance

This app does not use encryption beyond Apple-provided system frameworks.
