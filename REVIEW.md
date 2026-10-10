# Review instructions

Scoping rules for `/code-review` (CI and local). Keep findings high-signal:
these reviews gate a solo developer's merges — verified bugs matter, volume
does not.

## Severity

Reserve **Important** for:

- Logic bugs that break behavior, lose data, or corrupt cache state
- Concurrency: actor-isolation violations, races, deadlocks. The build sets
  `SWIFT_DEFAULT_ACTOR_ISOLATION=MainActor`; protocols satisfied by actor
  adapters must be `nonisolated protocol` (see `ARCHITECTURE.md` → Actor
  Isolation).
- Prefix-cache / snapshot correctness — anything that could serve stale or
  mismatched KV state
- Tests that fail in a parallel run (`docs/testing.md` → Parallel runs share
  one test host): a sleep or a count of `Task.yield()`s standing in for "the
  work finished" before a positive check, when the work leaves the main actor
  (an actor hop, a `Task.sleep`, a detached task); a suite that computes for
  seconds without `@Suite(.cpuBound)`. SwiftLint already rejects wall-clock
  budgets and `for … where … { await Task.yield() }` waits.
- Security and sandbox escapes (PathSandbox, entitlements)
- Assertions that pass on wrong output: `contains(fixtureText)` over generated
  text (a prompt, a summary, a render) holds with garbage around the fixture,
  as a summary prompt's debug-described assistant turns once did. Pin the
  whole line.

Everything else — naming, style, structure preferences — is a nit.

## Cap nits

At most 5 nits per review; summarize the rest as a count.

## Skip entirely

- `assets/`, `build/`
- Lock/manifest churn (`*.resolved`)

## Vendor

`Vendor/` is code we own. `tesseract-speech` and `tesseract-highlight` are
in-tree packages: review them like the app. `mlx-swift-lm` is our fork of the
model engine, pinned as a submodule. A PR that moves its gitlink carries every
fork commit in between, while the PR diff shows only the two `Subproject
commit` lines. Review `git -C Vendor/mlx-swift-lm diff <old>..<new>` (`log`
for the commits) at the same bar as app code, under the fork's own
`Vendor/mlx-swift-lm/CLAUDE.md` code standards. The fork's carry branches have
no CI of their own: this review and the `vendor-test` CI job are its gate.

## Documentation drift

Flag it if the PR renames, moves, or deletes a module that `ARCHITECTURE.md`
names; introduces domain vocabulary that `CONTEXT.md` lacks; changes a test
workflow documented in `docs/testing.md` or `docs/prefix-cache-testing.md`; or
moves the vendor pin without an entry in `docs/mlx-swift-lm-fork.md`.

## Decided trade-offs

Don't re-litigate decisions recorded in `docs/adr/` — flag only genuine
violations of them.
