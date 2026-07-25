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
- `main.swift.c16-conv-backup` — the C16 gate (depthwise conv1d at S == 1
  vs elementwise multiply-adds: f32 accumulation is bitwise-identical in
  all 8192 channels for f16 and bf16, native-dtype accumulation is not)
  plus the qmv latency/fusion sweep behind the 2026-07-25 roofline entry
  (dependent-chain vs independent-call cost, and the control that
  separates rig overhead from kernel time).
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

- `main.swift.c19-softmax-backup` — C19 gate: the router kernel with
  `softmax_single_row` replicated inside it (AccT=float, N_READS=4, masked
  to E/4 threads, MLX's simd_max/simd_sum tree). Bitwise identical;
  rejected on speed. Keep as the reference for reproducing an MLX
  reduction kernel exactly.
- `main.swift.prefill-anchor-backup` — gather_qmm vs dense-qmm anchor at
  production MoE prefill dims, swept over gathered-row counts. Answers
  "is the gather kernel behind the anchor?" (it is only at small shapes).
