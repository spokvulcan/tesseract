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
            Log.agent.info(render(payload))
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
