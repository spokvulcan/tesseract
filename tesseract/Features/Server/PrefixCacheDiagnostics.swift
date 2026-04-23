import Foundation
import MLXLMCommon

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
                PrefixCacheDiagnostics.telemetryEvent(SkipEvent(
                    stage: stage,
                    reason: reason,
                    extraFields: extraFields
                ), context: self)
            )
        }
    }

    enum Level: Sendable {
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

        init(
            reason: PrefixCacheManager.LookupReason,
            promptTokens: Int,
            sharedPrefixLength: Int,
            skippedPrefillTokens: Int,
            newTokensToPrefill: Int,
            lookupMs: TimeInterval,
            restoreMs: TimeInterval,
            plannedCheckpoints: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]
        ) {
            switch reason {
            case .hit(let snapshotOffset, _, let type):
                self.reason = "hit"
                self.snapshotOffset = snapshotOffset
                self.checkpointType = type
            case .ssdHit(let ctx):
                self.reason = "ssdHit"
                self.snapshotOffset = ctx.storageRef.tokenOffset
                self.checkpointType = ctx.storageRef.checkpointType
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
                fields.append(("normalizedRecency", PrefixCacheDiagnostics.scalar(normalizedRecency)))
            }
            if let normalizedFlopEfficiency {
                fields.append((
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

    struct TTFTEvent: Payload {
        let lookupMs: TimeInterval
        let restoreMs: TimeInterval
        let prefillMs: TimeInterval
        let firstTokenMs: TimeInterval
        let totalPromptMs: TimeInterval

        init(
            lookupMs: TimeInterval,
            restoreMs: TimeInterval,
            prefillMs: TimeInterval,
            totalPromptMs: TimeInterval
        ) {
            self.lookupMs = lookupMs
            self.restoreMs = restoreMs
            self.prefillMs = prefillMs
            self.totalPromptMs = totalPromptMs
            self.firstTokenMs = max(0, totalPromptMs - prefillMs)
        }

        let eventName = "ttft"

        var fields: [(String, String)] {
            [
                ("lookupMs", PrefixCacheDiagnostics.milliseconds(lookupMs)),
                ("restoreMs", PrefixCacheDiagnostics.milliseconds(restoreMs)),
                ("prefillMs", PrefixCacheDiagnostics.milliseconds(prefillMs)),
                ("firstTokenMs", PrefixCacheDiagnostics.milliseconds(firstTokenMs)),
                ("totalPromptMs", PrefixCacheDiagnostics.milliseconds(totalPromptMs)),
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
    }

    /// Reason an SSD-resident state-5 hydration failed. Mirrors the
    /// granular branches inside `SSDSnapshotStore.loadSync` so an
    /// operator triaging a miss can tell file-missing apart from
    /// fingerprint mismatch without parsing free-form messages.
    enum SSDMissReason: String, Sendable {
        case partitionNotInManifest
        case fingerprintMismatch
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
        let storageRefID: String?

        let eventName = "leafSupersession"

        var fields: [(String, String)] {
            [
                ("offset", "\(offset)"),
                ("storageRefID", storageRefID ?? "nil"),
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

    struct StorageRefCommitEvent: Payload {
        let id: String

        let eventName = "storageRefCommit"

        var fields: [(String, String)] {
            [("id", id)]
        }
    }

    struct StorageRefDropCallbackEvent: Payload {
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
        return fields
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

    private static func optionalCheckpointType(_ value: HybridCacheSnapshot.CheckpointType?) -> String {
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
        guard value.range(
            of: #"^[A-Za-z0-9._:/,\-\[\]]+$"#,
            options: .regularExpression
        ) != nil else {
            let escaped = value
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            return "\"\(escaped)\""
        }
        return value
    }
}
