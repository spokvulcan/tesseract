import Foundation
import MLXLMCommon

// Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
// swiftlint:disable:next type_body_length
nonisolated enum PrefixCacheDiagnostics {
    struct Context: Sendable {
        let requestID: UUID
        let modelID: String
        let kvBits: Int?
        let kvGroupSize: Int

        func render(_ payload: some Payload) -> String {
            PrefixCacheDiagnostics.render(payload, context: self)
        }

        func log(_ payload: some Payload) {
            let line = render(payload)
            Log.agent.info(line)
            PrefixCacheDiagnostics.forwardToSink(line)
            PrefixCacheDiagnostics.forwardTelemetryEvent(
                PrefixCacheDiagnostics.telemetryEvent(payload, context: self)
            )
        }

        func logSkip(
            stage: String,
            reason: String,
            level: Level = .info,
            extraFields: [(String, String)] = []
        ) {
            let line = render(SkipEvent(stage: stage, reason: reason, extraFields: extraFields))
            switch level {
            case .debug:
                Log.agent.debug(line)
            case .info:
                Log.agent.info(line)
            case .warning:
                Log.agent.warning(line)
            case .error:
                Log.agent.error(line)
            }
            PrefixCacheDiagnostics.forwardToSink(line)
            PrefixCacheDiagnostics.forwardTelemetryEvent(
                PrefixCacheDiagnostics.telemetryEvent(
                    SkipEvent(
                        stage: stage,
                        reason: reason,
                        extraFields: extraFields
                    ), context: self)
            )
        }
    }

    enum Level: Sendable, Equatable {
        case debug
        case info
        case warning
        case error
    }

    protocol Payload: Sendable {
        var eventName: String { get }
        var fields: [(String, String)] { get }
    }

    struct LookupEvent: Payload {
        let reason: String
        let promptTokens: Int
        let sharedPrefixLength: Int
        let snapshotOffset: Int?
        let checkpointType: HybridCacheSnapshot.CheckpointType?
        let skippedPrefillTokens: Int
        let newTokensToPrefill: Int
        let lookupMs: TimeInterval
        let restoreMs: TimeInterval
        let plannedCheckpoints: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]
        /// Resolution rewrites a hydrated `.ssdHit`/`.chainPrefixHit` to `.hit`
        /// before this event is logged (`PrefixCacheManager`'s `hydratedHit`), so the
        /// hit *kind* and the Think-Strip Rewind marker must travel as explicit
        /// fields, not in `reason`. The aggregate keys off these in production
        /// (where `reason` is always `"hit"`); the raw reasons still appear on
        /// the no-fingerprint replay path, so the aggregate honors both.
        let hydratedFromSSD: Bool
        let chainPrefixRestore: Bool

        init(
            reason: PrefixCacheManager.LookupReason,
            promptTokens: Int,
            sharedPrefixLength: Int,
            skippedPrefillTokens: Int,
            newTokensToPrefill: Int,
            lookupMs: TimeInterval,
            restoreMs: TimeInterval,
            plannedCheckpoints: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)],
            hydratedFromSSD: Bool = false,
            chainPrefixRestore: Bool = false
        ) {
            switch reason {
            case .hit(let snapshotOffset, _, let type):
                self.reason = "hit"
                self.snapshotOffset = snapshotOffset
                self.checkpointType = type
            case .ssdHit(let ctx):
                self.reason = "ssdHit"
                self.snapshotOffset = ctx.snapshotRef.tokenOffset
                self.checkpointType = ctx.snapshotRef.checkpointType
            case .chainPrefixHit(let ctx):
                self.reason = "chainPrefixHit"
                self.snapshotOffset = ctx.point.boundaryOffset
                self.checkpointType = ctx.point.checkpointType
            case .missNoEntries:
                self.reason = "missNoEntries"
                self.snapshotOffset = nil
                self.checkpointType = nil
            case .missNoSnapshotInPrefix:
                self.reason = "missNoSnapshotInPrefix"
                self.snapshotOffset = nil
                self.checkpointType = nil
            }

            self.promptTokens = promptTokens
            self.sharedPrefixLength = sharedPrefixLength
            self.skippedPrefillTokens = skippedPrefillTokens
            self.newTokensToPrefill = newTokensToPrefill
            self.lookupMs = lookupMs
            self.restoreMs = restoreMs
            self.plannedCheckpoints = plannedCheckpoints
            self.hydratedFromSSD = hydratedFromSSD
            self.chainPrefixRestore = chainPrefixRestore
        }

        let eventName = "lookup"

        var fields: [(String, String)] {
            [
                ("reason", reason),
                ("promptTokens", "\(promptTokens)"),
                ("sharedPrefixLength", "\(sharedPrefixLength)"),
                ("snapshotOffset", PrefixCacheDiagnostics.optionalInt(snapshotOffset)),
                ("checkpointType", PrefixCacheDiagnostics.optionalCheckpointType(checkpointType)),
                ("skippedPrefillTokens", "\(skippedPrefillTokens)"),
                ("newTokensToPrefill", "\(newTokensToPrefill)"),
                ("lookupMs", PrefixCacheDiagnostics.milliseconds(lookupMs)),
                ("restoreMs", PrefixCacheDiagnostics.milliseconds(restoreMs)),
                ("plannedCheckpoints", PrefixCacheDiagnostics.checkpointList(plannedCheckpoints)),
                ("hydratedFromSSD", hydratedFromSSD ? "true" : "false"),
                ("chainPrefixRestore", chainPrefixRestore ? "true" : "false"),
            ]
        }
    }

    struct CaptureEvent: Payload {
        let offset: Int
        let checkpointType: HybridCacheSnapshot.CheckpointType
        let bytes: Int
        let duringPrefill: Bool
        let source: String

        let eventName = "capture"

        var fields: [(String, String)] {
            [
                ("offset", "\(offset)"),
                ("checkpointType", checkpointType.wireString),
                ("bytes", "\(bytes)"),
                ("duringPrefill", "\(duringPrefill)"),
                ("source", source),
            ]
        }
    }

    struct LeafModeEvent: Payload {
        let mode: String
        let continuation: String

        let eventName = "leafMode"

        var fields: [(String, String)] {
            [
                ("mode", mode),
                ("continuation", continuation),
            ]
        }
    }

    /// One admitted **Speculative Canonical Prefill** pass: the background
    /// re-prefill of the **Think-Strip Rewind** span after a stop-finish
    /// answer (ADR-0009). Logged on every admission — full target or the
    /// partial progress of a preempted pass (`preempted=true`); passes that
    /// admit nothing surface as `skip` events with `stage=speculativePrefill`.
    struct SpeculativePrefillEvent: Payload {
        let targetOffset: Int
        let boundaryOffset: Int
        let prefilledTokens: Int
        let prefillSeconds: TimeInterval
        let rewindSpanTokens: Int
        /// `true` when this admission is the partial progress of a preempted
        /// pass (capture-on-preempt) rather than the full planned target.
        let preempted: Bool

        let eventName = "speculativePrefill"

        var fields: [(String, String)] {
            [
                ("targetOffset", "\(targetOffset)"),
                ("boundaryOffset", "\(boundaryOffset)"),
                ("prefilledTokens", "\(prefilledTokens)"),
                ("prefillMs", PrefixCacheDiagnostics.milliseconds(prefillSeconds)),
                ("rewindSpanTokens", "\(rewindSpanTokens)"),
                ("preempted", "\(preempted)"),
            ]
        }
    }

    /// One **Asymmetric-State Restore** pass (issue #134, ADR-0009): the
    /// experimental single-prefill counter to the **Think-Strip Rewind** that
    /// derives a stripped-path snapshot from the think-bearing capture by pure
    /// array surgery, plugged into the **Speculative Canonical Prefill**
    /// scheduling. The outcome surfaces whether synthesis produced a usable
    /// snapshot (`synthesized`), declined preflight (`unavailable`, fell back
    /// to the speculative prefill), or aborted mid-synthesis (`midSynthesis`,
    /// admitted nothing deeper than the canonical leaf). Per ADR-0009's
    /// performance gate, each phase is reported separately so a
    /// capture-dominated outcome is visible rather than hidden behind a fast
    /// synthesis number.
    struct AsymmetricStateRestoreEvent: Payload {
        enum Outcome: String, Sendable {
            case synthesized
            case unavailable
            case midSynthesis
            case disabled
        }

        let outcome: Outcome
        let bearingOffset: Int
        let strippedOffset: Int
        let spanCount: Int
        let excisedTokens: Int
        /// End-to-end pass cost (bearing capture + span scan + surgery +
        /// admission) — the figure the 5 s Stretch-Abandonment window gates on.
        let totalSeconds: TimeInterval
        let captureSeconds: TimeInterval
        let synthesisSeconds: TimeInterval
        /// Preflight decline reason (only meaningful when `outcome == unavailable`).
        let unavailableReason: String?

        let eventName = "asymmetricStateRestore"

        var fields: [(String, String)] {
            var pairs: [(String, String)] = [
                ("outcome", outcome.rawValue),
                ("bearingOffset", "\(bearingOffset)"),
                ("strippedOffset", "\(strippedOffset)"),
                ("spanCount", "\(spanCount)"),
                ("excisedTokens", "\(excisedTokens)"),
                ("totalMs", PrefixCacheDiagnostics.milliseconds(totalSeconds)),
                ("captureMs", PrefixCacheDiagnostics.milliseconds(captureSeconds)),
                ("synthesisMs", PrefixCacheDiagnostics.milliseconds(synthesisSeconds)),
            ]
            if let unavailableReason { pairs.append(("unavailableReason", unavailableReason)) }
            return pairs
        }
    }

    struct EvictionEvent: Payload {
        let strategy: PrefixCacheManager.EvictionEvent.Strategy
        let offset: Int
        let checkpointType: HybridCacheSnapshot.CheckpointType
        let freedBytes: Int
        let budgetBytes: Int
        let snapshotBytesAfter: Int
        let normalizedRecency: Double?
        let normalizedFlopEfficiency: Double?
        let utility: Double?

        init(_ event: PrefixCacheManager.EvictionEvent) {
            self.strategy = event.strategy
            self.offset = event.offset
            self.checkpointType = event.checkpointType
            self.freedBytes = event.freedBytes
            self.budgetBytes = event.budgetBytes
            self.snapshotBytesAfter = event.snapshotBytesAfter
            self.normalizedRecency = event.normalizedRecency
            self.normalizedFlopEfficiency = event.normalizedFlopEfficiency
            self.utility = event.utility
        }

        let eventName = "eviction"

        var fields: [(String, String)] {
            var fields: [(String, String)] = [
                ("strategy", strategy.rawValue),
                ("offset", "\(offset)"),
                ("checkpointType", checkpointType.wireString),
                ("freedBytes", "\(freedBytes)"),
                ("budgetBytes", "\(budgetBytes)"),
                ("snapshotBytesAfter", "\(snapshotBytesAfter)"),
            ]

            if let normalizedRecency {
                fields.append(
                    ("normalizedRecency", PrefixCacheDiagnostics.scalar(normalizedRecency)))
            }
            if let normalizedFlopEfficiency {
                fields.append(
                    (
                        "normalizedFlopEfficiency",
                        PrefixCacheDiagnostics.scalar(normalizedFlopEfficiency)
                    ))
            }
            if let utility {
                fields.append(("utility", PrefixCacheDiagnostics.scalar(utility)))
            }

            return fields
        }
    }

    /// Per-request prompt-latency breakdown. On the cache-aware path the
    /// orchestrator prefills *outside* the MLX generation loop, so the
    /// loop's own prompt time (`residualPromptMs`) covers only the residual
    /// tokens it saw — TTFT is therefore the sum of the stages, not any
    /// single library-reported number.
    struct TTFTEvent: Payload {
        let lookupMs: TimeInterval
        let restoreMs: TimeInterval
        let prefillMs: TimeInterval
        let residualPromptMs: TimeInterval
        let ttftMs: TimeInterval

        init(
            lookupMs: TimeInterval,
            restoreMs: TimeInterval,
            prefillMs: TimeInterval,
            residualPromptMs: TimeInterval
        ) {
            self.lookupMs = lookupMs
            self.restoreMs = restoreMs
            self.prefillMs = prefillMs
            self.residualPromptMs = residualPromptMs
            self.ttftMs = lookupMs + restoreMs + prefillMs + residualPromptMs
        }

        let eventName = "ttft"

        var fields: [(String, String)] {
            [
                ("lookupMs", PrefixCacheDiagnostics.milliseconds(lookupMs)),
                ("restoreMs", PrefixCacheDiagnostics.milliseconds(restoreMs)),
                ("prefillMs", PrefixCacheDiagnostics.milliseconds(prefillMs)),
                ("residualPromptMs", PrefixCacheDiagnostics.milliseconds(residualPromptMs)),
                ("ttftMs", PrefixCacheDiagnostics.milliseconds(ttftMs)),
            ]
        }
    }

    struct MemoryEvent: Payload {
        let snapshotCount: Int
        let totalSnapshotBytes: Int
        let budgetBytes: Int
        let modelWeightBytes: Int64
        let activeMlxBytes: Int64
        let peakMlxBytes: Int64
        let mlxCacheLimitBytes: Int64
        let partitionCount: Int

        init(
            stats: PrefixCacheManager.CacheStats,
            budgetBytes: Int,
            modelWeightBytes: Int64,
            activeMlxBytes: Int64,
            peakMlxBytes: Int64,
            mlxCacheLimitBytes: Int64
        ) {
            self.snapshotCount = stats.snapshotCount
            self.totalSnapshotBytes = stats.totalSnapshotBytes
            self.budgetBytes = budgetBytes
            self.modelWeightBytes = modelWeightBytes
            self.activeMlxBytes = activeMlxBytes
            self.peakMlxBytes = peakMlxBytes
            self.mlxCacheLimitBytes = mlxCacheLimitBytes
            self.partitionCount = stats.partitionCount
        }

        let eventName = "memory"

        var fields: [(String, String)] {
            [
                ("snapshotCount", "\(snapshotCount)"),
                ("totalSnapshotBytes", "\(totalSnapshotBytes)"),
                ("budgetBytes", "\(budgetBytes)"),
                ("modelWeightBytes", "\(modelWeightBytes)"),
                ("activeMlxBytes", "\(activeMlxBytes)"),
                ("peakMlxBytes", "\(peakMlxBytes)"),
                ("mlxCacheLimitBytes", "\(mlxCacheLimitBytes)"),
                ("partitionCount", "\(partitionCount)"),
            ]
        }
    }

    // MARK: - SSD-tier events (Task 4.1.12)

    /// Terminal admission outcome for a single SSD writer item. Each
    /// `tryEnqueue` call eventually fires exactly one `ssdAdmit`
    /// event — synchronously in the front-door reject paths, or
    /// asynchronously from the writer loop after `processPendingItem`
    /// reaches a commit/drop branch.
    enum SSDAdmitOutcome: String, Sendable {
        case accepted
        case droppedByteBudget
        case droppedTooLargeForBudget
        case droppedExceedsBudget
        case droppedSystemProtectionWins
        case droppedDiskFull
        case droppedWriterIOError
        case droppedInvalidCheckpointType
        case droppedUnregisteredPartition
        /// Front door: a **Leaf Extension Admission** found its base
        /// neither resident, queued, nor in flight.
        case droppedExtensionBaseUnavailable
        /// Writer: a pending extension lost its base before the
        /// commit-time fold.
        case droppedExtensionBaseLost
    }

    /// Reason an SSD-resident state-5 hydration failed. Mirrors the
    /// granular branches inside `SSDSnapshotStore.loadSync` so an
    /// operator triaging a miss can tell file-missing apart from
    /// fingerprint mismatch without parsing free-form messages.
    enum SSDMissReason: String, Sendable {
        case partitionNotInManifest
        case fingerprintMismatch
        /// The resident's manifest entry vanished between the lookup
        /// and the hydration (a concurrent eviction).
        case notResident
        case readFailed
        case decodeFailed
    }

    struct SSDAdmitEvent: Payload {
        let id: String
        let bytes: Int
        let outcome: SSDAdmitOutcome

        let eventName = "ssdAdmit"

        var fields: [(String, String)] {
            [
                ("id", id),
                ("bytes", "\(bytes)"),
                ("outcome", outcome.rawValue),
            ]
        }
    }

    struct SSDEvictAtAdmissionEvent: Payload {
        let victimID: String
        let incomingID: String

        let eventName = "ssdEvictAtAdmission"

        var fields: [(String, String)] {
            [
                ("victimID", victimID),
                ("incomingID", incomingID),
            ]
        }
    }

    struct SSDHitEvent: Payload {
        let id: String
        let hydrateMs: TimeInterval

        let eventName = "ssdHit"

        var fields: [(String, String)] {
            [
                ("id", id),
                ("hydrateMs", PrefixCacheDiagnostics.milliseconds(hydrateMs)),
            ]
        }
    }

    struct SSDMissEvent: Payload {
        let id: String
        let reason: SSDMissReason

        let eventName = "ssdMiss"

        var fields: [(String, String)] {
            [
                ("id", id),
                ("reason", reason.rawValue),
            ]
        }
    }

    struct SSDBodyDropEvent: Payload {
        let id: String

        let eventName = "ssdBodyDrop"

        var fields: [(String, String)] {
            [("id", id)]
        }
    }

    struct LeafSupersessionEvent: Payload {
        let offset: Int
        let snapshotRefID: String?
        /// What happened to the superseded leaf's SSD backing:
        /// `transferred` (a **Leaf Extension Admission** took the
        /// chain), `deleted` (a full write replaced it), or
        /// `preserved` (a RAM-only admission kept it for warm start).
        let mode: PrefixCacheManager.LeafSupersession.Mode

        let eventName = "leafSupersession"

        var fields: [(String, String)] {
            [
                ("offset", "\(offset)"),
                ("storageRefID", snapshotRefID ?? "nil"),
                ("mode", mode.rawValue),
            ]
        }
    }

    /// One committed **Leaf Extension Admission**: the suffix landed and
    /// the base's **Segment Chain** transferred. `suffixBytes` is what
    /// actually hit the SSD this turn — the churn metric #78 tracks.
    struct LeafExtensionCommitEvent: Payload {
        let id: String
        let baseID: String
        let suffixBytes: Int
        let chainBytes: Int
        let chainSegments: Int

        let eventName = "leafExtensionCommit"

        var fields: [(String, String)] {
            [
                ("id", id),
                ("baseID", baseID),
                ("suffixBytes", "\(suffixBytes)"),
                ("chainBytes", "\(chainBytes)"),
                ("chainSegments", "\(chainSegments)"),
            ]
        }
    }

    struct SSDRecordHitEvent: Payload {
        let id: String

        let eventName = "ssdRecordHit"

        var fields: [(String, String)] {
            [("id", id)]
        }
    }

    struct SnapshotRefCommitEvent: Payload {
        let id: String

        let eventName = "storageRefCommit"

        var fields: [(String, String)] {
            [("id", id)]
        }
    }

    struct SnapshotRefDropCallbackEvent: Payload {
        let id: String
        let reason: SSDDropReason

        let eventName = "storageRefDropCallback"

        var fields: [(String, String)] {
            [
                ("id", id),
                ("reason", PrefixCacheDiagnostics.ssdDropReasonString(reason)),
            ]
        }
    }

    struct WarmStartCompleteEvent: Payload {
        let partitionCount: Int
        let snapshotCount: Int
        let invalidatedPartitionCount: Int
        let durationSeconds: TimeInterval

        let eventName = "warmStartComplete"

        var fields: [(String, String)] {
            [
                ("partitionCount", "\(partitionCount)"),
                ("snapshotCount", "\(snapshotCount)"),
                ("invalidatedPartitionCount", "\(invalidatedPartitionCount)"),
                ("durationMs", PrefixCacheDiagnostics.milliseconds(durationSeconds)),
            ]
        }
    }

    struct FingerprintMismatchEvent: Payload {
        let partition: String

        let eventName = "fingerprintMismatch"

        var fields: [(String, String)] {
            [("partition", partition)]
        }
    }

    /// Render a payload without per-request context fields. Used by
    /// SSD-tier events that originate outside any request scope —
    /// the SSD writer task, warm-start, the writer's commit/drop
    /// callbacks. Format mirrors the contextful renderer minus the
    /// `requestID` / `modelID` / `kvBits` / `kvGroupSize` prefix.
    nonisolated static func renderSystem(_ payload: some Payload) -> String {
        let fields: [(String, String)] = [("event", payload.eventName)] + payload.fields
        return
            fields
            .map { key, value in "\(key)=\(escape(value))" }
            .joined(separator: " ")
    }

    /// Emit a system-scope event to `Log.agent.info` and forward the
    /// rendered line to the test sink (if installed). Safe to call
    /// from any thread / actor; both `Log.agent` and the sink hop
    /// internally as needed.
    nonisolated static func logSystem(_ payload: some Payload) {
        let line = renderSystem(payload)
        Log.agent.info(line)
        forwardToSink(line)
        forwardTelemetryEvent(telemetrySystemEvent(payload))
    }

    /// Test-only event sink registry. Every emitted diagnostic line
    /// (request-scoped via `Context.log` / `logSkip`, and system-scope
    /// via `logSystem`) is forwarded to every installed sink. Tests
    /// install a sink to assert event ordering deterministically
    /// without scraping the unified-log pipeline. Production never
    /// installs a sink — `forwardToSink` becomes a single uncontended
    /// lock acquisition per event.
    ///
    /// The registry holds an array of `(UUID, handler)` pairs so
    /// multiple parallel tests can each register their own sink
    /// without overwriting one another's slot. Each test filters the
    /// captured lines by its own snapshot IDs to ignore cross-test
    /// leakage; serialization is not required.
    nonisolated final class TestSinkHandle: @unchecked Sendable {
        let id: UUID
        init(id: UUID) { self.id = id }
    }

    nonisolated(unsafe) private static var _sinks: [(UUID, @Sendable (String) -> Void)] = []
    private static let _sinkLock = NSLock()

    @discardableResult
    nonisolated static func addTestSink(
        _ handler: @escaping @Sendable (String) -> Void
    ) -> TestSinkHandle {
        _sinkLock.lock()
        defer { _sinkLock.unlock() }
        let id = UUID()
        _sinks.append((id, handler))
        return TestSinkHandle(id: id)
    }

    nonisolated static func removeTestSink(_ handle: TestSinkHandle) {
        _sinkLock.lock()
        defer { _sinkLock.unlock() }
        _sinks.removeAll { $0.0 == handle.id }
    }

    nonisolated static func forwardToSink(_ line: String) {
        _sinkLock.lock()
        let snapshot = _sinks
        _sinkLock.unlock()
        for (_, handler) in snapshot {
            handler(line)
        }
    }

    /// Structured event sink registry for in-app observability. The string
    /// renderer above remains the stable wire contract for tests and Console;
    /// this side channel carries the same event fields as values so SwiftUI
    /// does not need to parse log lines.
    nonisolated final class TelemetrySinkHandle: @unchecked Sendable {
        let id: UUID
        init(id: UUID) { self.id = id }
    }

    nonisolated(unsafe) private static var _telemetrySinks:
        [(UUID, @Sendable (PromptCacheTelemetryEvent) -> Void)] = []
    private static let _telemetrySinkLock = NSLock()

    @discardableResult
    nonisolated static func addTelemetrySink(
        _ handler: @escaping @Sendable (PromptCacheTelemetryEvent) -> Void
    ) -> TelemetrySinkHandle {
        _telemetrySinkLock.lock()
        defer { _telemetrySinkLock.unlock() }
        let id = UUID()
        _telemetrySinks.append((id, handler))
        return TelemetrySinkHandle(id: id)
    }

    nonisolated static func removeTelemetrySink(_ handle: TelemetrySinkHandle) {
        _telemetrySinkLock.lock()
        defer { _telemetrySinkLock.unlock() }
        _telemetrySinks.removeAll { $0.0 == handle.id }
    }

    nonisolated static func forwardTelemetryEvent(_ event: PromptCacheTelemetryEvent) {
        _telemetrySinkLock.lock()
        let snapshot = _telemetrySinks
        _telemetrySinkLock.unlock()
        for (_, handler) in snapshot {
            handler(event)
        }
    }

    private static func telemetryEvent(
        _ payload: some Payload,
        context: Context
    ) -> PromptCacheTelemetryEvent {
        PromptCacheTelemetryEvent(
            scope: .request,
            eventName: payload.eventName,
            requestID: context.requestID,
            modelID: context.modelID,
            kvBits: context.kvBits,
            kvGroupSize: context.kvGroupSize,
            fields: payload.fields
        )
    }

    private static func telemetrySystemEvent(
        _ payload: some Payload
    ) -> PromptCacheTelemetryEvent {
        PromptCacheTelemetryEvent(
            scope: .system,
            eventName: payload.eventName,
            fields: payload.fields
        )
    }

    /// Wire-format string for an `SSDDropReason`. Pinned in tests so
    /// future enum additions are forced through this switch and the
    /// rendered diagnostics line stays stable.
    nonisolated static func ssdDropReasonString(_ reason: SSDDropReason) -> String {
        switch reason {
        case .backpressureOldest: return "backpressureOldest"
        case .evictedByLRU: return "evictedByLRU"
        case .systemProtectionWins: return "systemProtectionWins"
        case .exceedsBudget: return "exceedsBudget"
        case .diskFull: return "diskFull"
        case .writerIOError: return "writerIOError"
        case .extensionBaseLost: return "extensionBaseLost"
        case .hydrationFailure: return "hydrationFailure"
        }
    }

    private struct SkipEvent: Payload {
        let stage: String
        let reason: String
        let extraFields: [(String, String)]

        let eventName = "skip"

        var fields: [(String, String)] {
            [("stage", stage), ("reason", reason)] + extraFields
        }
    }

    private static func render(_ payload: some Payload, context: Context) -> String {
        let baseFields: [(String, String)] = [
            ("event", payload.eventName),
            ("requestID", context.requestID.uuidString),
            ("modelID", context.modelID),
            ("kvBits", optionalInt(context.kvBits)),
            ("kvGroupSize", "\(context.kvGroupSize)"),
        ]

        return (baseFields + payload.fields)
            .map { key, value in "\(key)=\(escape(value))" }
            .joined(separator: " ")
    }

    private static func optionalInt(_ value: Int?) -> String {
        value.map(String.init) ?? "nil"
    }

    private static func optionalCheckpointType(_ value: HybridCacheSnapshot.CheckpointType?)
        -> String
    {
        value?.wireString ?? "nil"
    }

    private static func checkpointList(
        _ value: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]
    ) -> String {
        guard !value.isEmpty else { return "[]" }
        let parts = value.map { "\($0.offset):\($0.type.wireString)" }
        return "[\(parts.joined(separator: ","))]"
    }

    private static func milliseconds(_ seconds: TimeInterval) -> String {
        String(format: "%.3f", max(0, seconds) * 1000)
    }

    private static func scalar(_ value: Double) -> String {
        String(format: "%.6f", value)
    }

    private static func escape(_ value: String) -> String {
        guard
            value.range(
                of: #"^[A-Za-z0-9._:/,\-\[\]]+$"#,
                options: .regularExpression
            ) != nil
        else {
            let escaped =
                value
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            return "\"\(escaped)\""
        }
        return value
    }
}
