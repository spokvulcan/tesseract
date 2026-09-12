# Releasing

How Tesseract ships. Decision record: `docs/adr/0017-release-please-signed-notarized-releases.md`.

## The flow (no human steps except one merge)

1. PRs land on `main` via **squash merge only**; the PR title must be a
   Conventional Commit (enforced by `pr-title.yml`) because it becomes the
   commit on `main` and, from there, the changelog line.
2. `release-please.yml` keeps a **Release PR** open with the next version and
   accumulated changelog. **Merging that PR is the release.** (Release PRs
   and their merge commits change only version files and skip CI — the code
   was already tested when it landed on `main`.) Every conventional commit
   type counts as release-worthy — `docs:`/`refactor:`/`ci:` alone still
   produce a Release PR (all types are non-hidden via `changelog-sections`
   in `release-please-config.json`; release-please's default would release
   only on `feat`/`fix`/`deps`).
3. On merge, Release Please creates the tag (`vX.Y.Z`) and the GitHub
   Release, then dispatches `release-build.yml`.
4. `release-build.yml` requires CI (`build-release`, `test`, `lint`) to be
   green for the release's code state (the tagged commit's parent — normally
   already finished, so the gate is instant), then: archives with Developer ID signing,
   verifies the signature, packages a DMG, notarizes + staples it, and
   uploads to the Release:
   - `Tesseract-<version>.dmg`
   - `Tesseract.dmg` — stable URL:
     `https://github.com/spokvulcan/tesseract/releases/latest/download/Tesseract.dmg`
   - `Tesseract-<version>-dSYMs.zip` — keep these; they are the only way to
     symbolicate a user's crash report.

The version is injected at build time (`MARKETING_VERSION` from the tag,
`CURRENT_PROJECT_VERSION` from the run number). The pbxproj's own version is
a dummy — never bump it by hand; `version.txt` is the source of truth and
Release Please maintains it.

## CI builds on pull requests

`CI` runs SwiftLint automatically on PRs. The macOS jobs, `build-release`
and `test`, are skipped by default; both run automatically when code lands
on `main`. Version-only changes still skip CI as described above.

- **Build from a PR:** add the `run-build` label in the PR sidebar. This
  requests one Release build and Debug test run of the PR's merge commit,
  including for fork PRs. New commits do not automatically request another
  run; remove and re-add the label when you want to build and test again.
  Other labels do not request builds or tests, and label changes do not
  rerun lint. GitHub may require approval for fork runs.
- **Run full CI manually:** open **Actions → CI → Run workflow**, select
  the branch, and click **Run workflow**. The CLI equivalent is
  `gh workflow run ci.yml --ref BRANCH`. This runs the build, tests, and
  lint. The Actions button becomes available after this workflow reaches
  `main`; use the PR label for branches in forks.
- **Merge requirements:** keep `build-release` and `test` out of required
  status checks in branch protection and rulesets. Both are optional before merge;
  the release pipeline still requires `build-release`, `test`, and `lint`
  to pass for the released code on `main`.

## One-time setup (before the first release)

### 1. Developer ID Application certificate

If you don't have one yet: Xcode → Settings → Accounts → Manage
Certificates → "+" → **Developer ID Application** (or via
developer.apple.com → Certificates).

Export it: Keychain Access → My Certificates → right-click the
"Developer ID Application: … (5RBTC2MNY8)" entry → Export as `.p12` with a
strong password. Then:

```bash
base64 -i DeveloperID.p12 | gh secret set MACOS_CERTIFICATE_BASE64
gh secret set MACOS_CERTIFICATE_PASSWORD   # the .p12 password
openssl rand -base64 24 | gh secret set KEYCHAIN_PASSWORD
gh secret set APPLE_TEAM_ID --body "5RBTC2MNY8"
```

### 2. App Store Connect API key (for notarization)

appstoreconnect.apple.com → Users and Access → Integrations →
App Store Connect API → Team Keys → generate with the **Developer** role.
Download the `.p8` (single chance!), note the Key ID and the Issuer ID shown
on that page:

```bash
gh secret set ASC_API_KEY_ID       # e.g. ABC123DEFG
gh secret set ASC_API_ISSUER_ID    # UUID from the page header
gh secret set ASC_API_KEY_P8 < AuthKey_ABC123DEFG.p8
```

## Secrets inventory

| Secret | Contents |
| --- | --- |
| `MACOS_CERTIFICATE_BASE64` | base64 of the Developer ID Application `.p12` |
| `MACOS_CERTIFICATE_PASSWORD` | password of that `.p12` |
| `KEYCHAIN_PASSWORD` | arbitrary; protects the throwaway CI keychain |
| `APPLE_TEAM_ID` | `5RBTC2MNY8` |
| `ASC_API_KEY_ID` | App Store Connect API key ID |
| `ASC_API_ISSUER_ID` | App Store Connect issuer ID |
| `ASC_API_KEY_P8` | contents of the `.p8` private key |

Security note (ADR-0017): these live on a public repo. Fork PRs cannot read
them, but **any workflow change merged to `main` runs with access — always
read workflow-file diffs yourself before merging.**

## Operations

- **Re-run a failed release build:**
  `gh workflow run release-build.yml --ref vX.Y.Z -f tag=vX.Y.Z`
  (idempotent — assets upload with `--clobber`).
- **Skip a commit from the changelog:** it can't be skipped retroactively;
  fix the PR title before merge. That is the entire reason `pr-title.yml`
  exists.
- **"A required agreement is missing or has expired" (HTTP 403):** Apple
  published an updated developer agreement and blocks notarization
  account-wide until it's accepted — nothing on our side broke (this killed
  the v1.3.0 build one day after v1.2.0 notarized fine). The Account Holder
  signs in at <https://developer.apple.com/account> and accepts the pending
  agreement (also check App Store Connect → Agreements), then re-runs the
  workflow. Two guards exist: `release-build.yml` runs
  `scripts/release/notary-preflight.sh` before the archive so this fails in
  seconds, and `notary-health.yml` runs the same check weekly and opens an
  issue so it's caught before release day.
- **Notarization rejected:** the workflow prints the `notarytool log`
  output; the usual causes are a missing hardened runtime or an unsigned
  nested binary (both are caught earlier by `verify-signing.sh`).
- **`setup-xcode` can't find a matching version:** the runner image moved;
  adjust `XCODE_VERSION` in `ci.yml` + `release-build.yml` (kept as a `^`
  range so this should be rare).
- **A test suite is flaky on runners** (virtualized Metal): quarantine that
  one suite (skip trait + tracking issue) — do not remove tests from CI
  wholesale (ADR-0017).
- **Local dry run of the packaging steps** (uses your login keychain, no
  ephemeral keychain needed):
  ```bash
  APPLE_TEAM_ID=5RBTC2MNY8 scripts/release/archive.sh 0.0.0-local 1
  scripts/release/verify-signing.sh
  scripts/release/package.sh 0.0.0-local
  ```

## The agent is not sandboxed (ADR-0047)

The agent ships **non-sandboxed** — `ENABLE_APP_SANDBOX = NO`, no
`com.apple.security.app-sandbox` entitlement — because an ambient assistant
must read other processes (notifications via Accessibility, the screen next),
which the App Sandbox forbids. Hardened Runtime stays on, so notarization and
Developer-ID signing are unchanged; this is only viable *because* distribution
is Developer ID, not the App Store (the Store mandates the sandbox). The Mac
App Store is therefore closed to the agent — `SIGNING_METHOD=app-store` is
reserved for the future *server*, a contained product that can be sandboxed.

## Deferred by design (ADR-0017)

Sparkle auto-update (needs an opt-in privacy design first), Homebrew cask, and
the Mac App Store variant — now a *server*-only future (`scripts/release/archive.sh`
already takes `SIGNING_METHOD=app-store`), not an agent path (see ADR-0047).
