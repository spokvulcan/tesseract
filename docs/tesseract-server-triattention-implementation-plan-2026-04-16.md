# Tesseract Server + TriAttention Implementation Plan

**Date:** 2026-04-16  
**Status:** Ready for implementation  
**Primary input:** [docs/triattention-qwen35-paro-research-2026-04-16.md](/Users/owl/projects/tesseract/docs/triattention-qwen35-paro-research-2026-04-16.md)  
**Secondary inputs:** [docs/HTTP_SERVER_SPEC.md](/Users/owl/projects/tesseract/docs/HTTP_SERVER_SPEC.md), [docs/prefix-cache-session-2026-04-13-investigation.md](/Users/owl/projects/tesseract/docs/prefix-cache-session-2026-04-13-investigation.md), current code under `tesseract/Features/Server`, `tesseract/Features/Agent`, and `Vendor/mlx-swift-lm`

## Purpose

This plan is for future implementation sessions where the agent will perform the work. It is intentionally server-first:

- the Tesseract server path becomes the canonical inference path
- TriAttention is introduced for Qwen3.5 PARO on that canonical path
- prefix cache correctness is preserved for both dense and TriAttention variants
- comprehensive test coverage is part of the implementation, not a cleanup afterthought
- performance benchmarking stays last and non-blocking

## Working Rules For Future Sessions

1. Implement epics in order.
2. Do not skip the verification gate for an epic before starting the next one.
3. Keep the dense path working until the final cutover gate passes.
4. Do not add extra product scope during implementation.
5. Benchmarking does not block shipping the functional server integration.

## Goals

1. Make the Tesseract server inference pipeline the source of truth for future agent sessions.
2. Add TriAttention support for `qwen3.5-4b-paro`, `qwen3.5-9b-paro`, and `qwen3.5-27b-paro`.
3. Make TriAttention configurable from the UI with a simple on/off control.
4. Keep prefix cache working in both modes:
   - dense attention cache
   - TriAttention sparse/compressed attention cache
5. Preserve current OpenAI-compatible HTTP behavior while unifying internal and external inference semantics.
6. Ship correctness, integration, and end-to-end coverage before the benchmark epic.

## Explicit Non-Goals For V1

- TriAttention for non-PARO models
- TriAttention on the VLM / vision path
- per-request TriAttention overrides over HTTP
- advanced UI tuning for TriAttention budget or calibration artifacts
- reworking the external HTTP API contract
- making the benchmark epic a prerequisite for merging the functional implementation

## Locked Decisions

These decisions answer the implementation-level questions that should be settled before coding starts.

### 1. "Server as source of truth" means server core, not necessarily loopback TCP

The canonical path should be a reusable in-process server inference service extracted from the current HTTP handler flow. The HTTP listener remains one adapter. Internal agent chat and background agents should call the same server-core pipeline directly instead of using a separate inference stack.

Why:

- it gives one inference contract for HTTP clients and internal agent sessions
- it avoids making internal chat depend on `NWListener` lifecycle
- it preserves the existing public server toggle as a transport concern, not a hard dependency for the app itself

### 2. V1 TriAttention scope is text-only PARO

TriAttention is enabled only for:

- `qwen3.5-4b-paro`
- `qwen3.5-9b-paro`
- `qwen3.5-27b-paro`
- text-only container path

When vision mode is enabled, the model continues to run through the current dense path and the UI must communicate that TriAttention is unavailable in that mode.

### 3. TriAttention cache state is non-trimmable

For prefix-cache semantics, TriAttention attention state is treated like `MambaCache`:

- serializable
- restorable
- reusable
- not tail-trimmable

This rule applies in vendor code, server lookup/store logic, normalization handling, and diagnostics.

### 4. Dense and TriAttention prefix caches must never mix

The existing `CachePartitionKey` must be expanded so that dense and TriAttention snapshots live in separate partitions. At minimum the partition key must include:

- model ID
- model fingerprint
- `kvBits`
- `kvGroupSize`
- TriAttention enabled/disabled
- TriAttention budget
- TriAttention calibration fingerprint or artifact version
- TriAttention implementation version

The updated fields must also be reflected in the struct's ordering/equality behavior so deterministic eviction and existing tests do not regress.

### 5. Preserve current quantization timing in V1

Do not redesign quantization policy while adding TriAttention. Preserve the current behavior as closely as possible:

- system snapshots captured during prefill reflect live prefill state
- decode-time active cache and post-generation leaf storage continue to honor the `kvBits` / `kvGroupSize` path

If temporary scaffolding is needed during bring-up, it must stay behind a development-only gate and must not become the final shipped behavior.

### 6. UI scope stays intentionally small

The only required UI control in V1 is a global enable/disable toggle for TriAttention. The default runtime budget is fixed at `12000` and remains internal for V1. Do not expand scope to budget sliders or model-specific controls unless a later review explicitly approves it.

### 7. Calibration artifacts are model-fingerprint keyed

TriAttention requires one calibration artifact per supported PARO checkpoint. V1 should ship precomputed artifacts under a dedicated app resource location such as `Resources/TriAttention/`, load them by model fingerprint, and treat a mismatch as "feature unavailable, fall back to dense mode". The artifact format should be ported from upstream 1:1 rather than redesigned locally. A small developer-only regeneration script is allowed, but runtime behavior should not depend on generating stats inside the app.

### 8. Normalization mismatch rule stays strict

If a stored conversation would require trimming the restored cache to align with a shorter normalized token sequence, TriAttention sessions must skip leaf storage instead of attempting a trim. This follows the same conservative correctness rule already present in [tesseract/Features/Agent/LLMActor.swift](/Users/owl/projects/tesseract/tesseract/Features/Agent/LLMActor.swift).

### 9. V1 keeps the current prefix-cache memory-budget formula

V1 should keep the existing byte-budget policy in [tesseract/Features/Agent/LLMActor.swift](/Users/owl/projects/tesseract/tesseract/Features/Agent/LLMActor.swift) unchanged. TriAttention's initial memory win is used to fit more snapshots into the same budget, not to shrink the budget formula during the rollout. Budget tuning can be revisited after benchmark data exists.

## Implementation Notes Resolved Up Front

### Sparse state encoding

The minimum persisted TriAttention state is:

- retained positions
- retained keys
- retained values
- logical processed offset
- runtime config metadata needed to reject incompatible restores

Do not persist derived scoring intermediates unless the vendor prototype proves they are necessary to continue incremental updates deterministically.

### Pruning schedule

Port upstream TriAttention pruning behavior unchanged in V1. Do not invent a new pruning trigger, retention schedule, or hysteresis policy during the first implementation pass.

### Prefix-cache hit policy

The existing "deepest compatible snapshot wins" rule can remain. The required changes are compatibility checks and non-trimmable handling, not a new hit-selection algorithm.

### Correctness harness strategy

Reuse and extend the existing loaded-model runners instead of building a new correctness framework:

- `HybridCacheCorrectnessRunner`
- `PrefixCacheE2ERunner`
- `PrefillStepBenchmarkRunner`

### Budget policy

Default chat budget is `12000`. This should be hardcoded for V1 and included in partitioning and diagnostics. Lower or per-request budgets are deferred.

## Epic Summary

| Epic | Name | Depends On | Main Output | Exit Gate |
|---|---|---|---|---|
| 0 | Canonical Server Core | none | single server inference service used by HTTP and internal agent adapters | internal and HTTP paths both exercise the same core service |
| 1 | Vendor TriAttention Runtime | 0 | TriAttention cache + Qwen3.5 PARO text path integration in `mlx-swift-lm` | dense-off path unchanged, TriAttention-on path runs on supported PARO models |
| 2 | Snapshot And Prefix Cache Compatibility | 1 | TriAttention snapshot, persistence, partitioning, and non-trim semantics | dense and TriAttention snapshots coexist without contamination |
| 3 | Server Integration And Agent Cutover | 2 | server-core generation path with TriAttention and internal agent adoption | agent/background sessions use server core successfully |
| 4 | UI And Operational Controls | 3 | user-facing toggle, reload behavior, fallback messaging | toggle works and does not require external HTTP listener |
| 5 | Correctness, Integration, And End-To-End Verification | 4 | automated coverage and shipping gate | all required suites pass in dense and TriAttention modes |
| 6 | Benchmark Versus Baseline | 5 | non-blocking performance comparison against current implementation | report produced; regressions understood |

## Epic 0 - Canonical Server Core

**Objective:** make the server pipeline the single inference contract before TriAttention lands.

### Tasks

1. Extract a new `ServerInferenceService` from the logic currently spread across:
   - [tesseract/Features/Server/CompletionHandler.swift](/Users/owl/projects/tesseract/tesseract/Features/Server/CompletionHandler.swift)
   - [tesseract/Features/Server/MessageConverter.swift](/Users/owl/projects/tesseract/tesseract/Features/Server/MessageConverter.swift)
   - [tesseract/Features/Agent/AgentEngine.swift](/Users/owl/projects/tesseract/tesseract/Features/Agent/AgentEngine.swift)
2. Define internal request/response models for server-core generation so the service is not HTTP-transport-specific.
3. Reduce `CompletionHandler` to an HTTP adapter:
   - decode OpenAI request
   - resolve model override
   - invoke `ServerInferenceService`
   - encode streaming or non-streaming OpenAI response
4. Preserve both current server generation branches inside `ServerInferenceService`:
   - prefix-cache-aware completion path
   - non-prefix-cache fallback completion path
   Neither branch is removed during extraction.
5. Add an internal adapter for agent/background sessions and summarization/compaction helpers that call `ServerInferenceService` directly.
6. Update [tesseract/Features/Agent/AgentFactory.swift](/Users/owl/projects/tesseract/tesseract/Features/Agent/AgentFactory.swift) and [tesseract/Features/Agent/BackgroundAgentFactory.swift](/Users/owl/projects/tesseract/tesseract/Features/Agent/BackgroundAgentFactory.swift) to use the server-core adapter behind a temporary rollback flag.
7. Keep the current direct `engine.generate(...)` path available only as a short-lived rollback mechanism until Epic 5 passes.

### Likely Files

- new `tesseract/Features/Server/ServerInferenceService.swift`
- new `tesseract/Features/Server/Models/ServerInferenceTypes.swift`
- modify `tesseract/Features/Server/CompletionHandler.swift`
- modify `tesseract/Features/Agent/AgentFactory.swift`
- modify `tesseract/Features/Agent/BackgroundAgentFactory.swift`
- modify `tesseract/Features/Agent/Context/ContextManager.swift`
- modify `tesseract/App/DependencyContainer.swift`

### Verification Gate

- HTTP completions still work in both streaming and non-streaming modes.
- Internal agent sessions can generate through `ServerInferenceService` with the external HTTP server disabled.
- Dense-path outputs remain unchanged under greedy decoding for the same prompt/tool set.

## Epic 1 - Vendor TriAttention Runtime

**Objective:** add TriAttention to the Qwen3.5 PARO text runtime without changing server semantics yet.

### Tasks

1. Add vendor-side TriAttention configuration types and app-side plumbing for:
   - enabled/disabled
   - budget
   - calibration artifact identity
   - implementation version
2. Add a calibration artifact loader keyed by model fingerprint.
3. Add a new non-trimmable cache type for TriAttention attention layers.
4. Patch the Qwen3.5 PARO text attention path in:
   - [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift](/Users/owl/projects/tesseract/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift)
   so supported PARO attention layers can run dense or TriAttention mode from the same model.
5. Extend the decode-time quantization path so TriAttention honors the same `kvBits` / `kvGroupSize` / `quantizedKVStart` contract used by the current `maybeQuantizeKVCache(...)` flow instead of silently bypassing quantization.
6. Add dense fallback when:
   - model is not PARO
   - vision mode is enabled
   - calibration artifact is missing or mismatched
7. Keep VLM support explicitly deferred; do not partially patch [Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift](/Users/owl/projects/tesseract/Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift) in V1.
8. Port upstream pruning trigger/schedule behavior as-is for the first runtime integration.

### Likely Files

- new vendor files under `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/` for TriAttention config/cache/helpers
- modify `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`
- modify `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift`
- modify `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
- modify `tesseract/Features/Agent/AgentGeneration.swift`
- modify `tesseract/Features/Agent/LLMActor.swift`

### Verification Gate

- Dense mode still produces the same outputs as before for supported models.
- TriAttention mode loads and runs on `qwen3.5-4b-paro` at minimum.
- Unsupported cases fall back to dense mode without request failure.

## Epic 2 - Snapshot And Prefix Cache Compatibility

**Objective:** make TriAttention state compatible with the existing prefix-cache architecture without violating correctness.

### Tasks

1. Extend [Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift](/Users/owl/projects/tesseract/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift) so TriAttention cache state can:
   - capture
   - restore
   - serialize
   - deserialize
2. Extend the SSD persistence stack:
   - [tesseract/Features/Server/SSDSnapshotStore.swift](/Users/owl/projects/tesseract/tesseract/Features/Server/SSDSnapshotStore.swift)
   - [tesseract/Features/Server/SnapshotManifest.swift](/Users/owl/projects/tesseract/tesseract/Features/Server/SnapshotManifest.swift)
   - [tesseract/Features/Server/TieredSnapshotStore.swift](/Users/owl/projects/tesseract/tesseract/Features/Server/TieredSnapshotStore.swift)
3. Expand the existing `CachePartitionKey` with TriAttention config identity so dense and TriAttention snapshots cannot collide, and update its ordering/comparison behavior consistently.
4. Update vendor trim helpers and app-side restore/store logic so any cache containing TriAttention attention state is treated as non-trimmable.
5. Audit [tesseract/Features/Agent/LLMActor.swift](/Users/owl/projects/tesseract/tesseract/Features/Agent/LLMActor.swift) for all places that currently rely on dense-attention assumptions:
   - hit restore path
   - alignment checkpoints
   - transient boundary snapshots
   - canonical leaf capture
   - direct tool leaf capture
   - normalization-trim skip logic
6. Bump persistence versioning as needed so older warm-start manifests are rejected cleanly if they cannot represent the new state.

### Verification Gate

- Dense and TriAttention snapshots can both be captured, restored, and persisted.
- Warm-start restore works for TriAttention sessions.
- Dense and TriAttention leaves never share a partition.
- Any normalization case that would require a trim skips leaf storage safely.

## Epic 3 - Server Integration And Agent Cutover

**Objective:** wire TriAttention into the server-core path and switch the internal agent to that same path.

### Tasks

1. Plumb TriAttention config and the unchanged byte-budget policy through:
   - settings
   - dependency container
   - inference arbiter
   - agent engine
   - LLM actor
   - vendor `GenerateParameters`
2. Ensure TriAttention selection happens in the server-core path, not in a separate agent-only path.
3. Keep request behavior identical between modes:
   - same model routing
   - same tool formatting
   - same response envelope
   - same `cached_tokens` accounting contract
4. Make internal agent/background sessions and summarization helpers call server core instead of direct engine generation.
5. Keep dense fallback automatic and explicit in diagnostics when TriAttention cannot be used.
6. Remove direct internal call sites to `engine.generate(...)` once parity coverage passes.

### Likely Files

- modify `tesseract/Features/Server/ServerInferenceService.swift`
- modify `tesseract/Features/Agent/AgentFactory.swift`
- modify `tesseract/Features/Agent/BackgroundAgentFactory.swift`
- modify `tesseract/Features/Agent/InferenceArbiter.swift`
- modify `tesseract/Features/Agent/AgentEngine.swift`
- modify `tesseract/Features/Agent/LLMActor.swift`

### Verification Gate

- HTTP clients and internal agent sessions both exercise server core.
- A two-turn PARO text conversation gets prefix-cache reuse in dense mode and in TriAttention mode.
- Switching the toggle changes partitioning and cache reuse behavior without corrupting existing dense snapshots.

## Epic 4 - UI And Operational Controls

**Objective:** expose the mode safely without growing product scope.

### Tasks

1. Add `triattentionEnabled` to [tesseract/Features/Settings/SettingsManager.swift](/Users/owl/projects/tesseract/tesseract/Features/Settings/SettingsManager.swift) with default `false`.
2. Add a simple toggle to [tesseract/Features/Settings/ServerSettingsView.swift](/Users/owl/projects/tesseract/tesseract/Features/Settings/ServerSettingsView.swift):
   - label clearly that it is PARO-only
   - mention that vision mode falls back to dense
3. Make toggle changes trigger a model reload via `InferenceArbiter`.
4. Keep TriAttention toggle independent from `isServerEnabled`; internal agent/server-core use must not require the public listener to be enabled.
5. Add minimal diagnostics in UI or logs for:
   - active cache mode
   - supported / unsupported reason
   - dense fallback reason

### Verification Gate

- Setting persists across relaunch.
- Toggling causes a reload and clearly changes active mode.
- If selected model or mode is unsupported, the UI and logs explain the dense fallback.

## Epic 5 - Correctness, Integration, And End-To-End Verification

**Objective:** make correctness the final blocker before any benchmark work.

### Required Automated Coverage

1. **Vendor unit tests**
   - TriAttention cache update and copy
   - quantization compatibility
   - snapshot round-trip
   - non-trimmable behavior
2. **App unit tests**
   - partition-key separation between dense and TriAttention
   - settings persistence
   - fallback selection logic
   - internal server-core adapter behavior
3. **Prefix-cache integration tests**
   - extend [tesseractTests/PrefixCacheIntegrationTests.swift](/Users/owl/projects/tesseract/tesseractTests/PrefixCacheIntegrationTests.swift)
   - extend [tesseractTests/PrefixCacheManagerTests.swift](/Users/owl/projects/tesseract/tesseractTests/PrefixCacheManagerTests.swift)
   - extend [tesseractTests/HybridCacheSnapshotTests.swift](/Users/owl/projects/tesseract/tesseractTests/HybridCacheSnapshotTests.swift)
4. **HTTP/server tests**
   - extend [tesseractTests/CompletionHandlerTests.swift](/Users/owl/projects/tesseract/tesseractTests/CompletionHandlerTests.swift)
   - add `ServerInferenceServiceTests`
5. **Loaded-model correctness runners**
   - extend `HybridCacheCorrectnessRunner` for TriAttention mid-prefill restore
   - extend `PrefixCacheE2ERunner` for dense vs TriAttention cache-hit validation
6. **End-to-end tests**
   - actual local HTTP request against `/v1/chat/completions`
   - actual internal agent session through server core
   - restart / warm-start snapshot reuse

### Required Manual Verification

1. Dense PARO request still works with toggle off.
2. TriAttention PARO request works with toggle on.
3. Same conversation reuses cache on second turn in both modes.
4. Toggle flip does not reuse incompatible snapshots.
5. Vision mode forces dense fallback cleanly.
6. Dense fallback on missing artifact is explicit and safe.

### Verification Gate

All required suites pass in both modes. No dense-path regression is allowed.

## Epic 6 - Benchmark Versus Baseline

**Objective:** measure impact against the current implementation after correctness is already proven.

**This epic is intentionally non-blocking.**

### Tasks

1. Extend the existing harnesses so they can run in:
   - current baseline raw-engine mode
   - dense server-core mode
   - TriAttention server-core mode
2. Treat the raw-engine path as a diagnostic baseline only. The shipping comparison of record must include the server-core path because that is the future source of truth.
3. Compare at minimum:
   - TTFT
   - cached token reuse
   - prompt-time / prefill-time
   - steady-state memory
   - sampled output stability under fixed seeds
4. Use current implementation as the baseline of record.
5. Run at least:
   - `qwen3.5-4b-paro`
   - `qwen3.5-9b-paro`
6. Run `qwen3.5-27b-paro` only when hardware is available and stable.
7. Write a short benchmark report under `docs/` or `benchmarks/results/` with:
   - configuration
   - dense baseline
   - TriAttention result
   - regressions or open concerns

### Expected Benchmark Scope

- long shared-prefix chat turns
- warm-cache continuation turns
- restart / warm-start reuse where feasible
- no new product features added to support the benchmark

## Test Matrix

| Layer | Required Coverage |
|---|---|
| Vendor cache/runtime | TriAttention cache state, quantization, snapshot round-trip, non-trimmable contract |
| Prefix cache | dense vs TriAttention partitioning, lookup/store, eviction, warm-start, SSD hydration |
| Server core | request normalization, fallback behavior, parity between HTTP and internal agent adapters |
| HTTP transport | streaming, non-streaming, usage accounting, model routing, error handling |
| UI/settings | toggle persistence, reload behavior, unsupported-mode messaging |
| Loaded-model correctness | mid-prefill restore correctness and stable cache-hit behavior |
| End-to-end | real local server request, real internal agent session, restart reuse |
| Benchmark | baseline comparison only after all above pass |

## Suggested Verification Commands

Use the existing project/test conventions already referenced elsewhere in the repo.

### Fast automated gate

```bash
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS'
```

During development, narrow the run to the touched suites first, then restore the full suite before closing the epic.

### Loaded-model verification gate

After Release build succeeds, run the existing harness entry points in [tesseract/App/TesseractApp.swift](/Users/owl/projects/tesseract/tesseract/App/TesseractApp.swift):

- `--hybrid-cache-correctness`
- `--prefix-cache-e2e`
- `--prefill-step-benchmark`

Extend those harnesses to accept dense vs TriAttention mode selection before using them as the final gate.

### Benchmark gate

Use the existing [scripts/bench.sh](/Users/owl/projects/tesseract/scripts/bench.sh) flow after Epic 5 is green.

## Rollout Strategy

1. Land server-core extraction first.
2. Land TriAttention runtime behind a disabled-by-default setting.
3. Land prefix-cache compatibility before exposing the toggle.
4. Switch internal agent/background sessions to server core only after dense parity is proven.
5. Keep default `triattentionEnabled = false` until Epic 5 passes.
6. Run the benchmark epic after correctness is already closed.
7. Decide separately whether TriAttention should remain opt-in or become the default for PARO text models.

## Deferred Follow-Ups

- VLM TriAttention support
- UI budget tuning
- public API exposure of TriAttention status
- non-PARO model support
- any optimization that changes the HTTP contract or the current message-format semantics

## Definition Of Done

This effort is done when all of the following are true:

1. The server-core inference path is the canonical path for HTTP and internal agent sessions.
2. TriAttention can be toggled on and off from the UI.
3. Prefix cache works in dense mode and TriAttention mode without cross-contamination.
4. Unsupported cases fall back to dense mode safely.
5. Required automated and manual verification gates are green.
6. Benchmarking, if not yet complete, is clearly isolated as follow-up work and does not block the functional merge.
