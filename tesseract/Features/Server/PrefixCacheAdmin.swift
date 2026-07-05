import Foundation

/// The MainActor **current-cache accessor** for prefix-cache administration
/// (PRD #137, PR C): stats, telemetry snapshots, budget/alpha overrides, and
/// the SSD flush — every admin read/write whose target, the `@MainActor`
/// `PrefixCacheManager`, lives on the MainActor in the first place. Callers
/// used to tunnel these through three isolation domains (MainActor →
/// `LLMActor` → actor-confined module → back to MainActor); this accessor is
/// the direct door.
///
/// It exists because the manager is **rebuilt per model load** (the flop
/// profile comes from the freshly-installed Model Identity) — a bare handle
/// goes stale at every caller on every swap, so "which manager is live"
/// needs one owner. The **Server Completion** module publishes each manager
/// it builds; the reference is weak, so a model unload (which drops the
/// module and with it the manager) reads as "no live cache" with no
/// clear-side wiring to forget.
///
/// All admin entries degrade to `nil`/no-op when no cache is live — the
/// same contract the retired actor forwards had.
@MainActor
final class PrefixCacheAdmin {
    private weak var current: PrefixCacheManager?

    /// Headroom for the Batch Engine's Lane Admission arithmetic — the
    /// admin samples directly rather than through the manager so a budget
    /// exists even between model loads.
    private let headroom = MachMemoryHeadroomSource()

    nonisolated init() {}

    /// The **Lane Admission** inputs (PRD #173, ADR-0022): measured machine
    /// headroom, the cache bytes an eviction drain could free toward a new
    /// lane (`totalSnapshotBytes` — the survival floor is a single leaf,
    /// small enough to ignore here), and the Active-Inference Reserve's
    /// per-lane price. Degrades to zero headroom and the bootstrap reserve
    /// when no cache is live — the policy floors capacity at one lane.
    func batchLaneBudget() -> BatchLaneBudget {
        BatchLaneBudget(
            headroomBytes: headroom.sample()?.headroomBytes ?? 0,
            evictableCacheBytes: current?.totalSnapshotBytes ?? 0,
            perLaneBytes: current?.activeInferenceReserve.perLaneBytes
                ?? ActiveInferenceReserve.bootstrapPerLaneBytes
        )
    }

    /// Install the freshly built manager as the live cache. Called by the
    /// Server Completion module's cache construction, on the MainActor.
    func publish(_ manager: PrefixCacheManager) {
        current = manager
    }

    /// Snapshot of the live prefix-cache state, or `nil` if no cache is
    /// live. Used by the loaded-model E2E runner to verify branch-point
    /// capture and survival.
    var stats: PrefixCacheManager.CacheStats? {
        current?.stats
    }

    /// The live cache's **Eviction Configuration**, or `nil`. Lets tests
    /// assert the cache construction folded the model's `flopProfile` in.
    var evictionConfig: EvictionConfiguration? {
        current?.evictionConfig
    }

    /// Current eviction weighting (`alpha`), or `nil` if no cache is live.
    /// Symmetric with `setEvictionAlpha`, so the E2E runner can save and
    /// restore the weighting around its forced-pressure step.
    var evictionAlpha: Double? {
        current?.evictionConfig.alpha
    }

    func makeTelemetrySnapshot() -> PromptCacheTelemetrySnapshot? {
        current?.makeTelemetrySnapshot()
    }

    /// Override the RAM-tier budget through the manager's band-consistent
    /// mutation. Used by the loaded-model E2E runner to deliberately
    /// trigger eviction pressure.
    func setMemoryBudget(_ bytes: Int) {
        current?.setMemoryBudget(bytes)
    }

    /// Override the eviction weighting (`alpha`). Production code should
    /// not call this; the `AlphaTuner` owns `alpha` after warmup.
    func setEvictionAlpha(_ alpha: Double) {
        current?.setEvictionAlpha(alpha)
    }

    /// Block until pending SSD-tier writes have drained and the manifest
    /// is durably persisted. Callers must invoke this before the model
    /// unload when the on-disk state must survive the teardown. No-op
    /// when SSD is disabled or no cache is live.
    func flushSSDWrites() async {
        await current?.flushSSDWrites()
    }
}
