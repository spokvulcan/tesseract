---
status: accepted
---

# Releases: Release Please cuts the version; a dispatched CI build ships a signed, notarized DMG

This records the design decided in the 2026-07-03 release-pipeline grilling.
The repo had no release infrastructure at all — no tags, no build CI, a static
`MARKETING_VERSION = 1.0` in the pbxproj — and the goal is downloadable
releases with maximum automation and minimum standing maintenance for a solo
developer whose PRs are written by coding agents.

The decision, end to end:

- **Release Please owns versioning.** It watches `main` (Conventional
  Commits), maintains a rolling **Release PR** with the changelog and next
  semver, and merging that PR is the release act: tag, GitHub Release, and a
  dispatched build follow mechanically. Releasing is one deliberate click,
  which is the right amount of ceremony for a solo dev. First release is
  **1.0.0** (via `release-as`, removed afterwards).
- **Squash-only merges + a Conventional-Commits PR-title check.** Release
  Please is only as truthful as the commit log; with agents authoring PRs, the
  PR title is the one human-checked line, and it becomes the `main` commit.
  A non-conventional squash title (e.g. `e7feacdf`) silently vanishes from
  the changelog — the title lint makes that impossible.
- **The pbxproj never learns the version.** CI injects
  `MARKETING_VERSION=<tag>` and `CURRENT_PROJECT_VERSION=<run number>` as
  xcodebuild arguments; `version.txt` (Release Please's file) is the sole
  source of truth. Local builds show the pbxproj dummy — irrelevant, users
  only ever run CI builds.
- **Full Developer ID signing + notarization in CI.** Certificate lands in a
  temp keychain from a base64 secret; `notarytool` authenticates with an App
  Store Connect API key (not an Apple ID app-specific password — no 2FA
  surface, no password rotation coupling); the DMG is stapled. Users
  double-click; Gatekeeper stays quiet.
- **Release assets:** `Tesseract-<v>.dmg`, an unversioned `Tesseract.dmg`
  (stable `releases/latest/download/…` URL), and `Tesseract-<v>-dSYMs.zip`
  (no crash-reporting service exists; public symbols are the only path to
  readable stack traces).
- **Full CI on every PR** — Release-config build, the test suite, SwiftLint —
  because the repo is public and macOS runners are free. The release build
  gates on those checks being green for the released commit, so a release
  cannot be cut from a `main` that doesn't build or pass tests.
- **The build workflow is dispatched explicitly** by the Release Please
  workflow. Releases created with the default `GITHUB_TOKEN` do not emit
  events that trigger other workflows, and the alternative — a long-lived
  PAT — is a standing credential with repo-wide blast radius on a public
  repo. `permissions: actions: write` + one `gh workflow run` is cheaper.

Standing risk accepted: the Developer ID private key lives in GitHub Actions
secrets on a public repository. Secrets are unavailable to fork PRs, but any
workflow change merged to `main` runs with access — workflow-file diffs are
the one category the human must always read.

## Considered / rejected

- **osaurus's flow (manual tag push + release-drafter).** No — it is proven
  (we studied `build-and-release.yml` in detail and copy its signing recipe),
  but a hand-pushed tag is a human step that can drift (wrong base commit,
  forgotten changelog), and release-drafter's label-based notes duplicate
  what Conventional Commits already encode. Release Please replaces both
  with the PR-merge gesture we already use for everything else.
- **Release Please editing the pbxproj** (`extra-files` generic updater).
  No — every release PR would regex-patch `project.pbxproj`, the one file
  this repo forbids editing by hand, and a silent mismatch corrupts the
  project. Build-time injection has one moving part, at build time.
- **Unsigned or sign-only artifacts.** No — an unsigned app greets its first
  user with a Gatekeeper block and a right-click ritual; for a product whose
  pitch is "trust it with your whole life," the download must be boring.
- **PAT-authenticated Release Please** to make tag events trigger builds
  natively. No — standing credential, manual rotation, bigger blast radius;
  explicit dispatch does the same job with the ephemeral token.
- **Tests kept out of CI** (documented local preflight instead). Initially
  chosen for cost reasons, reversed on the fact that public-repo Actions are
  free. The residual cost is flake maintenance on virtualized-Metal runners;
  the policy is to quarantine an individually flaky suite, never to retreat
  from CI testing wholesale.
- **Sparkle auto-update, Homebrew cask, Mac App Store variant.** Deferred,
  not rejected. Sparkle needs an opt-in privacy design first (a periodic
  update-check phones home; the product story is "nothing leaves the Mac"),
  Homebrew's official cask has notability bars a new repo may not clear, and
  the MAS variant is kept open by what already holds: App Sandbox + Hardened
  Runtime stay on, and release scripts stay parameterized by signing method.
