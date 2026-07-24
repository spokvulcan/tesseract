# gather-sweep — the Cmlx probe rig

The standalone SwiftPM rig behind the C1/C3/M4/M5(C13) probe evidence in
`benchmarks/experiments-ledger.md`: kernel-isolation timing (one lazy
graph per measurement, 32 disjoint weight sets, in-process ABBA) and the
bitwise gates that proved the fused-kernel replications
(`m4-fused-kernel.metal`, `m5-fused-kernel.metal`, and the fused
causal-softmax body now living in `mlx/fast.cpp`). Preserved here because
it is the instrument that re-proves the bitwise claims; the working copy
in `/tmp/gather-sweep` does not survive reboots.

- `Sources/gather-sweep/main.swift` — current rig (M5/C13 sections).
- `main.swift.gqmm-backup` / `main.swift.m4-backup` — the C1 gather_qmm
  B/E sweep and the M4 fused rotate+dequant rig, kept verbatim.
- `Package.swift` depends on the local clone `/Users/owl/projects/mlx-swift`
  (branch `pin-tesseract`) by absolute path — adjust if your clone lives
  elsewhere. Probe-only env hooks (`MLX_GQMM_CFG`, `MLX_GQMV_RPS`) exist
  only as uncommitted edits in that clone's Cmlx submodule (see the
  ledger's operational-state section).
- Runtime gotcha (ledger, "macOS/SwiftPM builds JIT the kernels"): device
  init needs a metallib — copy the app bundle's
  `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` next to the
  built binary as `mlx.metallib`. JIT covers the templated kernels.
- Build: `swift build -c release` (Release only for timing).

Nothing here runs in CI; it is lab equipment. Logs from the original
sessions stay in the ledger, not in the repo.
