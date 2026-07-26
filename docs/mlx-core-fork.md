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
       revision: <exact commit on the current pin branch>
  └─ spokvulcan/mlx-swift @ pin-tesseract
       = ml-explore/mlx-swift @ 0bb916c (the 0.31.6 tag) + one provenance
         commit (54ca1ec: .gitmodules only, zero source diff) + one
         gitlink-bump commit per accepted Cmlx change
       └─ submodule Source/Cmlx/mlx  → https://github.com/spokvulcan/mlx
            branch pin-tesseract = upstream mlx v0.31.1 (ce45c525) + the
            accepted Cmlx commits, append-only
       └─ submodule Source/Cmlx/mlx-c → ml-explore/mlx-c @ 0726ca9 (untouched)
```

The branch tips move with every accepted experiment; the **current** pins are
whatever the three `Package.swift` files record (the diagram describes the
structure, not a snapshot). Every ACCEPTED Cmlx experiment adds one commit on
`spokvulcan/mlx`, one gitlink-bump commit on `spokvulcan/mlx-swift`, and moves
the three Package.swift pins to that new commit.

## Pin branch history

The pin branches are append-only, so a base change gets a **new dated branch**
rather than a force-push. Never delete an old one — historical tesseract
gitlinks point into them.

| Branch (both forks) | Base | Status |
| --- | --- | --- |
| `pin-tesseract` | mlx v0.31.1 `ce45c525` / mlx-swift 0.31.6 `0bb916c` | **current** |
| `pin-tesseract-2026-07-27` | mlx main `973e27f8` / mlx-swift main `09051ed` | built, blocked — see below |

Scheme creation started from `54ca1ec` (provenance-only) on the original
`pin-tesseract` branch.

**Historical note.** The roadmap once said "Cmlx tracks ml-explore/mlx @
dc43e62d"; `dc43e62d` is an mlx-**swift** revision seen in a stale DerivedData
checkout, not an mlx revision. The mlx-core the app builds is
`ce45c52505c8158ea48d2a54e8caae05efd86bfe` (tag `v0.31.1`).

## Why mlx-core is still on v0.31.1 (attempted 2026-07-27)

mlx-core sits at v0.31.1 while upstream has moved 248 commits on (v0.32.0
shipped 2026-07-07). The move to upstream main was attempted, **builds green,
and is pushed as `pin-tesseract-2026-07-27` in both forks** — but it cannot
ship yet. The blocker is below; read it before re-attempting.

### The blocker: thread-local command encoders

mlx made streams and their Metal command encoders **thread-local**
(ml-explore/mlx#3281, 2026-03-25 — "Make each thread have its own default
stream"; #3348, 2026-04-01 — "Make CommandEncoder thread local"). Both landed
days after v0.31.1, and both are in v0.32.0.

`metal::get_command_encoder(Stream s)` now looks up a `thread_local` encoder
map, falls back to a global map, and otherwise throws. `gpu::new_stream`
registers **thread-locally**; only `gpu::new_thread_unsafe_stream` registers
globally.

mlx-swift caches one process-wide `Stream` (`Device.defaultStream`, resolved
from the `_tlDefaultDevice` TaskLocal). Tesseract's Swift-concurrency runtime
evaluates arrays on whatever thread a task resumes on, so any hop away from
the stream's creating thread throws:

```
Fatal error: There is no Stream(gpu, 0) in current thread.
  at mlx-c/mlx/c/transforms.cpp:73
```

Reproduced on `LeafHomeGuaranteeTests`, `CheckpointCaptureTests`,
`HybridCacheSnapshotTests`, `LeafAdmissionBuilderTests`. **Attribution is
settled**: the crash reproduces with the tesseract Cmlx carries fully reverted
(pure upstream `973e27f8`), and disappears when `gpu::new_stream` is patched to
register in the global encoder map instead of the thread-local one. Our carries
are not involved.

The proper fix is `new_thread_unsafe_stream`, and it is **not reachable from
Swift**: mlx-c binds no such entry point, and the open mlx-c regeneration PR
(ml-explore/mlx-c#121, "Bump to MLX 0.32.0") does not add one either. Closing
the gap means an mlx-c binding plus mlx-swift adopting it for its default
streams — two upstreamable contributions, and the natural next step.

### The port that is ready and waiting

Everything else about the move works. Four changes take mlx-swift to mlx main,
all general and upstreamable, all on `pin-tesseract-2026-07-27`:

- **mlx-c `0726ca9` (v0.6.0) → `fba4470b`.** `mlx::core::fft::rfft2/rfftn`
  gained an `FFTNorm` parameter after v0.31.1; the old bindings cannot call
  them (12 compile errors in `mlx-c/mlx/c/fft.cpp`).
- **`Package.swift` excludes `mlx/mlx/distributed/jaccl/lib`** — vendored
  jaccl transport sources added upstream, whose headers are not on the
  include path (`fatal error: 'jaccl/ring.h' file not found`). The existing
  excludes only covered `jaccl/*.cpp`, not the new nested dir.
- **`tools/update-mlx.sh` generates `gemv` and `steel_gemm_segmented_nax`** —
  two JIT kernels mlx added since v0.31.1. Without them the link fails on
  `mlx::core::metal::gemv()` / `::steel_gemm_segmented_nax()`. The kernel
  list in that script is hand-maintained: **a new upstream JIT kernel is a
  link error, not a build error, and it surfaces only when linking the
  example executables.**
- **`Source/MLX/FFT.swift` threads the new `mlx_fft_norm` argument** through
  its 20 transform call sites (`MLX_FFT_NORM_BACKWARD` = the upstream
  default, so behavior is unchanged). `fftshift`/`ifftshift` do not take it.

`Source/Cmlx/mlx-generated` and the framework headers were regenerated from
the new mlx tree by `tools/update-mlx.sh`.

### What the carries do on the new base

C1, C13 and C8+C9 come across for free: the branches behind mlx#3918/#3919/
#3920 were already rebased onto upstream main when they were filed, and all
three are host-side C++ only (`quantized.cpp`, `fast.cpp`, `ops.cpp`,
`transforms.cpp`), so `mlx-generated` needs no kernel-body mirroring.
`pin-tesseract-2026-07-27` carries exactly those three.

**C4, C5, C6, C7 do not rebase.** Upstream merged `DeviceStream` into
`CommandEncoder` with thread-local encoders, which is exactly the struct C4
patches — the rebase conflicts on `device.cpp`/`device.h` and cannot be
resolved mechanically. They need **re-implementation**, not a rebase; the
ledger numbers (C4 +2.5–5% MoE decode and −13.5% dense 8K peak, C5 +3.9% MoE
8K, C6 +3.1–4.7% MoE, C7 +3.7–5.9% MoE) stand as the targets. The
commit-accounting machinery survives upstream as
`CommandEncoder::buffer_ops_` / `buffer_sizes_`, so the re-port has a home.
C6 needs a re-check first — upstream mlx#3869 removed the regex it targeted.

C7's Swift surface (`GPU.setCommitLimits`) rides the same carries, so
`LLMActor.loadModel`/`unloadModel` lose their per-model commit policy on the
new base until C4+C7 are re-ported.

### Re-attempt checklist

1. Land the mlx-c `new_thread_unsafe_stream` binding and the mlx-swift
   default-stream adoption (or wait for upstream to).
2. Rebase `pin-tesseract-2026-07-27` in both forks onto the then-current
   upstream tips; re-run `tools/update-mlx.sh` (check the JIT kernel list
   against `mlx/backend/metal/kernels/` for new additions).
3. Re-port C4/C5/C7 onto `CommandEncoder`, re-check C6 against mlx#3869, and
   re-measure each per the ledger A/B protocol before re-accepting.
4. Restore the two `GPU.setCommitLimits` call sites in `LLMActor`.

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
  ml-explore/mlx), branch `pin-tesseract` (the branch the app builds; the
  upstream-main port is on `pin-tesseract-2026-07-27`). **Source of truth**
  for Cmlx edits.
- `~/projects/mlx-swift` — clone of `spokvulcan/mlx-swift` (remote `upstream`
  = ml-explore/mlx-swift), same two branches.
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
     equal the accepted diff exactly. **This check only catches resolution
     failures — after re-resolve the checkout equals the pin by
     construction, so it can NOT prove the pin matches the text that was
     benched.** A port typo ships silently through it (C13 shipped a
     rename that made the kernel uncompilable exactly this way — review
     round 2026-07-24).
   - **Clean-build confirmation (mandatory):** rebuild Release from the
     re-pinned tree and re-run the touched path once (one smoke round of
     `parity-ab.sh` against the pre-experiment baseline, or at minimum a
     bench leg that dispatches the changed kernel). Only then is the
     experiment's accepted state considered ported. C4 had this step and
     was fine; C13 skipped it and shipped broken.
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
