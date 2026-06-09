# CLAUDE.md

## What this is

Tesseract Agent — a privacy-focused, fully offline AI assistant for macOS.
All inference runs locally on Apple Silicon via MLX (WhisperKit ASR, Qwen3TTS,
an on-device tool-calling LLM agent). macOS 26+, Swift 6.2 / SwiftUI.

## Working here

- **Swift / SwiftUI / macOS work:** invoke the `build-macos-apps:*` skills,
  and follow them over any local pattern. The architecture is an evolving MVP
  mid-refactor — do **NOT** treat the current structure as prescriptive.
- **Build & run:** `scripts/dev.sh dev-release`. `ls scripts/` for the rest.
- **New files:** Xcode synchronized groups pick up anything added under
  `tesseract/` or `tesseractTests/` automatically — never edit `project.pbxproj`.
- **Never use `print()`** — use the `Log` enum (`Core/Logging.swift`).
- **Commits:** follow Conventional Commits.

## Docs (read before touching the area)

- Architecture → `ARCHITECTURE.md`
- Tests & suites → `docs/testing.md`
- Decisions & domain → `CONTEXT.md`, `docs/adr/`
- After editing docs, run `scripts/check-docs.sh` (also a CI gate).

## Agent skills

### Skill-first

Before any **work** request (not chit-chat or quick lookups), ALWAYS open your reply with
`Skill check: <skill> — <why>` or `Skill check: none`. Scan the available skills every time;
prefer one even when the task seems doable directly.

- Obvious or cheap → invoke, announce, proceed.
- Ambiguous or heavy → propose + why, then wait.
- Already inside a skill → skip.

### Issue tracker

Issues and PRDs live as GitHub issues in `spokvulcan/tesseract` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
