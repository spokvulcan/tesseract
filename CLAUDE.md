# CLAUDE.md

## What this is — and why

Tesseract Agent — a fully offline AI assistant for macOS (macOS 26+,
Swift 6.2 / SwiftUI), everything running locally on Apple Silicon via
MLX.

## Docs (read before touching the area)

- Architecture → `ARCHITECTURE.md`
- Tests & suites → `docs/testing.md`
- Model numbers (context, output length, sampling) → `docs/model-parameters.md`
- Decisions & domain → `CONTEXT.md`, `docs/adr/`

## Agent skills

### Issue tracker

Issues and PRDs live as GitHub issues in `spokvulcan/tesseract` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
