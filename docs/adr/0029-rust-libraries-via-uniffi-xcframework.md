# Rust libraries join the stack via UniFFI, shipped as committed XCFrameworks

Status: accepted

Some capabilities Tesseract wants have no serious Swift implementation but a
best-in-class Rust one. The first case is transcript rendering for the Tool
Panels redesign (PRD #200): syntax highlighting (`syntect` — the engine behind
`bat`/`delta`, Sublime-grammar based, ~190 languages, fully offline) and diff
computation with word-level inline emphasis (`similar`, by Sentry's Armin
Ronacher). The Swift alternatives are materially worse: Highlightr is
unmaintained; its fork HighlighterSwift runs highlight.js inside JavaScriptCore;
Splash is Swift-language-only; Neon/tree-sitter means one grammar package per
language and a heavy integration.

**Decision:**

- **Rust libraries are an accepted part of the stack** for niche capabilities
  where the Rust ecosystem is clearly ahead of Swift's. Offline-first is
  non-negotiable, as everywhere in Tesseract — no Rust crate that phones home.
- **Bridging is UniFFI-based** (via `cargo-swift` or equivalent): a small Rust
  crate exposes a narrow, purpose-built interface; UniFFI generates the Swift
  bindings; the product is a static-library **XCFramework** wrapped in a Swift
  Package.
- **The built XCFramework is committed to the repo.** Day-to-day builds
  (`scripts/dev.sh`, CI) stay pure-Xcode and require no Rust toolchain. Only
  regenerating the framework (rare: crate change, dependency bump) needs
  `rustup`/`cargo`, via a dedicated script in `scripts/`.
- **One crate per capability, narrow FFI.** The Rust side does the heavy
  lifting and returns plain data (e.g. styled spans as ranges + color roles);
  Swift owns all rendering, theming, and UI. No UI, no I/O policy, no state on
  the Rust side.

**Consequences:**

- Binary size grows by the crate + grammar assets (a few MB for syntect).
- The repo gains a second language, quarantined to rarely-touched crates with
  a script-owned build; contributors without Rust installed are unaffected.
- Committed binary artifacts mean the crate source and its framework can drift
  — the regeneration script must be the only way the framework is produced,
  and crate changes must land together with the regenerated framework.
- Licenses so far are MIT (`syntect`, `similar`) — compatible. Each new crate
  needs the same check.
