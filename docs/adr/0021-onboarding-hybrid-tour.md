---
status: accepted
---

# Onboarding is a hybrid cinematic tour that carries first-run setup

The dictation-era onboarding (a 5-step sheet: welcome, language, mic,
accessibility, done) never mentioned the agent, the server, TTS, or the fact
that the app is useless until a multi-GB model lands on disk. The rewrite had
three candidate identities — pure feature tour, pure setup wizard, or a hybrid
— and we chose the hybrid: a chaptered cinematic tour whose runtime hides the
model download's head start and whose chapters host the setup that belongs to
them. Decided in the 2026-07-05 grilling; vocabulary in `CONTEXT.md` →
Onboarding tour.

## Decision

- **Hybrid tour.** Six user-paced Chapters (Welcome → Agent → Dictation →
  Voice → Server → Ready) in a dedicated hidden-title-bar Welcome Window — the
  only window on first launch; the main window appears at the Handoff. The
  HIG's "don't let large downloads hinder onboarding" is satisfied by starting
  the download on the first screen and never blocking a Chapter on it.
- **Close = permanent skip.** Closing the Welcome Window mid-tour sets
  `hasCompletedOnboarding` exactly like finishing; the tour is never re-shown
  on a later launch (HIG: skippable, non-repeating) and remains relaunchable
  from Settings, replaying the same state-aware flow — no separate tour mode.
- **Downloads survive skip.** A skip does not cancel in-flight model
  downloads; cancel lives on the Models page.
- **Hardware-aware zero-UI model pick.** The agent tier is auto-selected from
  physical RAM (mapping derived from the model catalog), shown transparently
  with size and rationale plus a "Change" link — no picker step.
- **Whisper-first download ordering** (Whisper → Voice Engine → LLM), so the
  small models land early enough that Try-it moments — real dictation and TTS
  inside the tour — can unlock mid-tour on a first run.
- **Permissions in-context.** Microphone and Accessibility are requested
  inside the Dictation chapter via priming cards (user-initiated, deep-linked,
  never gating Continue) — mic copy covers all voice features (the agent
  composer shares `AudioCaptureEngine`), Accessibility copy is hotkey-scoped
  (only `HotkeyManager` needs it).
- **No raster or generated imagery.** All visuals are code (live tesseract
  projection, mesh gradients, embedded real app components as the demos) so
  the tour is resolution-independent and theme-aware; no Lottie/Rive asset
  pipeline.

## Considered options

A pure cinematic tour was rejected because it ends in an app that still can't
do anything; a pure setup wizard was rejected because it wastes the one moment
the user is guaranteed to be watching; a model-picker step was rejected as a
settings screen in a tuxedo; generated raster imagery was rejected as
fixed-resolution, theme-blind, and cheaper-looking than native drawing.

## Amendment — 2026-07-10

Two owner decisions during the face-lift (map #211, ticket #247's PR):

- **Seven chapters.** An Appshot chapter (PRD #170 — the double-Command
  frontmost-window capture staged into the agent composer) joins the tour
  after The Agent: Welcome → Agent → Appshot → Dictation → Voice → Server →
  Ready. It is scripted like the Agent chapter — a real capture needs Screen
  Recording (asked lazily on first use) and would summon the main window over
  the tour — with the demo strings built by the real `AppshotController`
  label builders.
- **The app icon replaces the live tesseract projection.** The code-drawn 4D
  wireframe read as cheap; the welcome hero and the ambient corner indicator
  are now the app icon (one deliberate exception to "no raster imagery" — it
  is the app's own identity asset). The corner percent text alone carries the
  ambient download progress.
