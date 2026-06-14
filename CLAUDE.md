# CLAUDE.md

## What this is — and why

Tesseract Agent — a fully offline AI assistant for macOS (macOS 26+,
Swift 6.2 / SwiftUI), everything running locally on Apple Silicon via
MLX. Nothing ever leaves the Mac — no cloud, no telemetry. The stack,
in priority order:

1. **The agent** — an on-device tool-calling LLM assistant you can
   trust with your whole life precisely because it is local.
2. **The inference server** — the foundation the agent stands on, and
   a product in its own right: an OpenAI-compatible
   `/v1/chat/completions` server whose tiered RAM + SSD radix prefix
   cache delivers hit rates no other on-device stack matches. A good
   server is a good agent.
3. **Dictation** — push-to-talk speech-to-text typed into any app:
   the everyday convenience, fully offline.
4. **Text-to-speech** — natural long-form voice synthesis: the
   nice-to-have, held to state-of-the-art on-device quality.

We are at the frontier of what's possible on-device — Tesseract is the
living proof — and the only real failure is not pushing it. When
weighing decisions:

- **Be ambitious in everything** — never write an idea off as "too
  ambitious for local hardware"; aim at the frontier in every decision,
  product and code alike.
- **Constraints are the design material** — one GPU, tight RAM,
  sandboxed, offline: design within them, not around them.
- **Knowing the user is first-class** — persistent memory of goals,
  habits, and preferences is the product, not a feature.

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

### Issue tracker

Issues and PRDs live as GitHub issues in `spokvulcan/tesseract` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
