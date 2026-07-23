# mlx-core (Cmlx) fork & pin scheme

How tesseract consumes a **writable mlx-core** for the Cmlx
inference-optimization loop (roadmap: `docs/mlx-core-optimization-roadmap.md`,
experiments: `benchmarks/experiments-ledger.md`). Established 2026-07-23.

## The pin chain

```
tesseract.xcodeproj
  └─ local packages: Vendor/mlx-swift-lm (submodule), Vendor/mlx-audio-swift,
     Vendor/tesseract-speech — each pins, in lockstep (SwiftPM cannot mix two
     revision-based requirements for one package):
       url: https://github.com/spokvulcan/mlx-swift
       revision: <exact commit on branch pin-tesseract>
  └─ spokvulcan/mlx-swift @ pin-tesseract
       = ml-explore/mlx-swift @ 0bb916c (the 0.31.6 tag) + ONE commit
         (54ca1ec): .gitmodules provenance edit only, zero source diff
       └─ submodule Source/Cmlx/mlx  → https://github.com/spokvulcan/mlx
            branch pin-tesseract, gitlink @ ce45c525 (== upstream mlx v0.31.1)
       └─ submodule Source/Cmlx/mlx-c → ml-explore/mlx-c @ 0726ca9 (untouched)
```

Pin revision as of scheme creation: `54ca1ec7cf9601c39809720725211afe601cfdd5`
(provenance-only). Every ACCEPTED Cmlx experiment adds one commit on
`spokvulcan/mlx` `pin-tesseract`, one gitlink-bump commit on
`spokvulcan/mlx-swift` `pin-tesseract`, and moves the three Package.swift pins
to that new commit.

**Corrected fact** (the roadmap previously said "Cmlx tracks ml-explore/mlx @
dc43e62d"): `dc43e62d` is an mlx-**swift** revision seen in a stale DerivedData
checkout, not an mlx revision. The actual mlx-core the app builds is
`ce45c52505c8158ea48d2a54e8caae05efd86bfe` (tag `v0.31.1`), recorded as the
`Source/Cmlx/mlx` gitlink of mlx-swift `0bb916c`; verified in the live build
DerivedData (`git ls-tree 0bb916c Source/Cmlx/` and the resolved checkout).

## Why this shape

- The Cmlx sources reach the build only as a **git submodule of mlx-swift**
  (`Source/Cmlx/mlx`); there is no lighter seam. Forking `ml-explore/mlx`
  alone is not enough — `.gitmodules` lives in the mlx-swift repo, so
  `spokvulcan/mlx-swift` carries exactly one provenance commit.
- Exact-revision pins (not branches) keep historical tesseract commits
  reproducible and match the existing lockstep discipline.
- The forks' pin branches are **append-only**: never force-push them, never
  delete — old tesseract commits' pins must stay fetchable (same rule as the
  mlx-swift-lm fork's old pin branches).
- Changes to mlx sources must stay general and upstreamable (PR-shaped for
  `ml-explore/mlx`), per ADR-0006's fork rules extended one level down.

## Working copies

- `~/projects/mlx` — clone of `spokvulcan/mlx` (remote `upstream` =
  ml-explore/mlx), branch `pin-tesseract`. **Source of truth** for Cmlx edits.
- `~/projects/mlx-swift` — clone of `spokvulcan/mlx-swift` (remote `upstream`
  = ml-explore/mlx-swift), branch `pin-tesseract`.
- The live build tree:
  `~/Library/Developer/Xcode/DerivedData/tesseract-*/SourcePackages/checkouts/mlx-swift/`
  (the app-target DerivedData, not a worktree's). `scripts/bench.sh` builds it
  in place; mid-iteration edits here need no re-resolution.

## Per-iteration workflow (one hypothesis per iteration)

1. Save the current Release `.app` as the A/B baseline (`/tmp/...`).
2. Edit Cmlx sources **in the DerivedData checkout's submodule** (fast loop).
   Gotcha: SwiftPM checkouts are **read-only** (`-r--r--r--`) — `chmod u+w`
   the files first (a later `xcodebuild -resolvePackageDependencies` or the
   revert in step 4 restores them). The Edit tool is workspace-scoped and
   cannot touch DerivedData — patch via shell.
3. Build + measure Release-only (`scripts/bench.sh`, `scripts/parity-ab.sh`,
   ABBA, nice 0, serialized GPU; parity gate `--paro-parity-bench`
   token-identical on both PARO models for anything numeric).
4. Verdict:
   - **REJECTED** — restore the checkout:
     `git -C <DD>/checkouts/mlx-swift/Source/Cmlx/mlx checkout -- .` (plus
     `git clean -fd` for new files). Nothing else was touched.
   - **ACCEPTED** — port the diff verbatim to `~/projects/mlx`
     (`pin-tesseract`), commit (Conventional Commits), push. In
     `~/projects/mlx-swift`: advance the `Source/Cmlx/mlx` gitlink to the new
     commit, commit, push. Update the three Package.swift pins to the new
     mlx-swift commit (the Vendor/mlx-swift-lm one is a commit on its
     `pin-upstream-mlx-swift` branch per `docs/mlx-swift-lm-fork.md`; the
     other two are in-tree edits). Commit in tesseract: pins + gitlink +
     ledger entry.
   - After an accepted re-pin: `xcodebuild -resolvePackageDependencies`
     re-syncs the DerivedData checkout; verify the port with
     `git -C <DD>/checkouts/mlx-swift/Source/Cmlx/mlx diff ce45c525` — it must
     equal the accepted diff exactly.
5. Tree clean between iterations (tesseract + Vendor submodule).

## macOS/SwiftPM builds JIT the Metal kernels — no instantiation plumbing

On this platform `Package.swift` excludes `nojit_kernels.cpp` and the
`kernels/` dir; `jit_kernels.cpp` generates Metal source at runtime from
template definitions (e.g. `get_gather_qmm_kernel` substitutes tile params
into the template and caches by kernel name). Consequences:

- Tile-geometry / template-param changes need **host-side edits only** —
  no `instantiate_*` lines, no metallib rebuild; the kernel is regenerated
  on first dispatch with the new name.
- **Kernel-body edits have two homes**: the canonical
  `Source/Cmlx/mlx/mlx/backend/metal/kernels/*.{h,metal}` AND the checked-in
  JIT string copies `Source/Cmlx/mlx-generated/*.cpp` (verbatim string of
  the kernel source with `#line` markers — SwiftPM builds the JIT from
  these, it does not regenerate them). Edit both consistently, or
  regenerate via `tools/update-mlx.sh`'s cmake step (`make <kernel>` under
  `mlx/backend/metal`, then copy `build/mlx/backend/metal/jit/*` into
  `mlx-generated/`). Verify a hand edit by diffing the `.h` against the
  string body.
- Standalone SwiftPM probes can't find `default.metallib` at init; copy the
  app's `mlx-swift_Cmlx.bundle/.../default.metallib` next to the probe
  binary as `mlx.metallib` (colocated-library fallback). It's only needed
  to satisfy device init — JIT covers the templated kernels.

## Re-converging on vanilla

When an accepted change merges upstream (`ml-explore/mlx`), drop it from
`spokvulcan/mlx` `pin-tesseract` on the next re-pin: rebase the branch onto
the new upstream base the app moves to, keeping only unmerged carries — the
same re-convergence rule as `docs/mlx-swift-lm-fork.md`.
