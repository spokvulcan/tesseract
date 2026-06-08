---
status: accepted
---

# TriAttention is cut from the app for MVP; the vendor fork's implementation is left inert

TriAttention — the off-by-default sparse-KV-cache attention mode for long-context
text inference on PARO-quantized Qwen3.5 models — is removed from the app for the
MVP. It never worked in practice: the runtime always fell back to dense because no
calibration artifact was ever shipped (the toggle passed
`calibrationArtifactIdentity: nil`, and a missing artifact forces the dense path),
and the MoE sparse path was still under debugging (`fix/triattention-moe-sparse-kv`).
The toggle defaulted to `false`, so removing it changes no working behaviour — the
app already ran dense attention everywhere.

What was removed, **app / superproject only**:

- the `triattentionEnabled` setting, its `ServerConfigurationView` toggle, the
  `DependencyContainer` reload-observation, and the dense-fallback chips in the
  server dashboard / status formatting;
- `TriAttentionRuntimeSelection`, `TriAttentionCalibrationArtifactLoader`,
  `ModelIdentity.isTriAttentionEligible`, and the TriAttention threading through
  `LLMActor` / `AgentEngine` / `InferenceArbiter` / `AgentGenerateParameters`;
- the app's `TriAttentionPartitionIdentity` and the `triAttention` field on
  `CachePartitionKey` / `PartitionMeta`, plus the `\0TA:` segment of
  `partitionDigest`. The dense digest is byte-identical with that branch gone, so
  dense partitions still reattach; `SnapshotManifestSchema.currentVersion` is bumped
  6 → 7 for a clean slate;
- the `scripts/triattention_calibrate*` calibration tooling and all app-side
  TriAttention test coverage.

**What was deliberately *not* touched: the vendor fork** (`Vendor/mlx-swift-lm`, a
submodule of `spokvulcan/mlx-swift-lm`). Its TriAttention implementation stays
present and fully compiled but unused —
`GenerateParameters.{triAttention, triAttentionCalibrationArtifact, triAttentionStablePrefixOffset}`,
`configureTriAttentionCachesForPrefill`, the `TriAttentionSparseKVCache` /
`QuantizedTriAttentionSparseKVCache` classes, the `HybridCacheSnapshot` restore
cases, and the calibration/configuration value types. The app simply stops setting
these fields and relies on their dense defaults (`.v1Disabled`, `nil`);
`configureTriAttentionCachesForPrefill` is a no-op when no sparse caches exist. The
vendor's own `MLXLMTests/TriAttention*` suite is the sole remaining coverage for
that inert code.

This is the surprising part, and the reason this ADR exists: a future reader will
find a complete TriAttention implementation in the fork that nothing in the app
references. **That is intentional. Do not "finish wiring it up", and do not delete it
as dead code, without re-opening this decision.** Revival path: git history for the
app-side wiring, plus the still-present vendor implementation.

## Considered / rejected

- **(A) Full two-repo excision — delete TriAttention from the fork as well.**
  Rejected for the MVP. The TriAttention surface is woven into the fork's two most
  load-bearing files — `Evaluate.swift` (the generate core) and
  `HybridCacheSnapshot.swift` (the serialization the *retained* prefix cache depends
  on) — so excising it means editing the generate and snapshot-restore hot paths and
  carrying a fork divergence on `test/tesseract-integration-v3`, under release
  pressure, for no behavioural gain. Leaving the surface inert costs nothing at
  runtime (it defaults to dense). A later cleanup may still do this; it is a
  deliberate deferral, not an oversight.
- **Disable-only — hide the toggle, keep the app-side machinery dormant.** Rejected.
  Because TriAttention already always fell back to dense, disabling is nearly a
  cosmetic no-op while leaving the exact dead, partly-broken app code (runtime
  selection, calibration loader, the buggy MoE plumbing) we wanted gone.
- **Keep TriAttention and fix it for MVP.** Rejected. Not required for the MVP
  feature set, and the goal is to ship with only paths that are known-working.
- **Preserve on-disk warm-start caches instead of bumping the schema.** Considered.
  The dense digest is stable, so old `v6` dense partitions would reattach without a
  bump; we bump to `v7` anyway because removing a field that
  `partitionDigest` / `PartitionMeta` canonicalize is exactly what the schema-bump
  rule guards, and a one-time cache wipe is free with no shipped users to migrate.
