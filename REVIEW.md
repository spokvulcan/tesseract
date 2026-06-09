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
- Security and sandbox escapes (PathSandbox, entitlements)

Everything else — naming, style, structure preferences — is a nit.

## Cap nits

At most 5 nits per review; summarize the rest as a count.

## Skip entirely

- `Vendor/` and `TriAttention/` (inert vendor fork)
- `.claude/skills/` (vendored, pinned by `skills-lock.json`)
- `assets/`, `build/`
- Lock/manifest churn (`skills-lock.json`, `*.resolved`)

## Documentation drift

Flag it if the PR renames, moves, or deletes a module that `ARCHITECTURE.md`
names; introduces domain vocabulary that `CONTEXT.md` lacks; or changes a test
workflow documented in `docs/testing.md`.

## Decided trade-offs

Don't re-litigate decisions recorded in `docs/adr/` — flag only genuine
violations of them.
