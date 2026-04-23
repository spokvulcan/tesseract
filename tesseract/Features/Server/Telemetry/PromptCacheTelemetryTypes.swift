import Foundation

nonisolated enum PromptCacheStorageState: String, Codable, CaseIterable, Sendable {
    case empty
    case ramOnly
    case pendingWrite
    case pendingWriteBodyDropped
    case ramAndSSD
    case ssdOnly

    var displayName: String {
        switch self {
        case .empty: "Empty"
        case .ramOnly: "RAM"
        case .pendingWrite: "RAM + pending SSD"
        case .pendingWriteBodyDropped: "Pending SSD"
        case .ramAndSSD: "RAM + SSD"
        case .ssdOnly: "SSD only"
        }
    }
}

nonisolated struct PromptCacheTelemetryField: Codable, Hashable, Sendable {
    let key: String
    let value: String

    nonisolated init(key: String, value: String) {
        self.key = key
        self.value = value
    }
}

nonisolated struct PromptCacheTelemetryEvent: Identifiable, Codable, Hashable, Sendable {
    enum Scope: String, Codable, Sendable {
        case request
        case system
    }

    let id: UUID
    let timestamp: Date
    let scope: Scope
    let eventName: String
    let requestID: UUID?
    let modelID: String?
    let kvBits: Int?
    let kvGroupSize: Int?
    let fields: [PromptCacheTelemetryField]

    nonisolated init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        scope: Scope,
        eventName: String,
        requestID: UUID? = nil,
        modelID: String? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int? = nil,
        fields: [(String, String)]
    ) {
        self.id = id
        self.timestamp = timestamp
        self.scope = scope
        self.eventName = eventName
        self.requestID = requestID
        self.modelID = modelID
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.fields = fields.map { PromptCacheTelemetryField(key: $0.0, value: $0.1) }
    }

    func field(_ key: String) -> String? {
        fields.first { $0.key == key }?.value
    }

    func intField(_ key: String) -> Int? {
        field(key).flatMap(Int.init)
    }

    func doubleField(_ key: String) -> Double? {
        field(key).flatMap(Double.init)
    }
}

nonisolated struct PromptCacheTelemetryAggregate: Codable, Equatable, Sendable {
    var lookupCount: Int = 0
    var hitCount: Int = 0
    var missCount: Int = 0
    var ssdHitCount: Int = 0
    var promptTokens: Int = 0
    var cachedTokens: Int = 0
    var captureCount: Int = 0
    var evictionCount: Int = 0
    var admissionCount: Int = 0
    var admissionDropCount: Int = 0
    var averageLookupMs: Double = 0
    var averageRestoreMs: Double = 0
    var averagePrefillMs: Double = 0
    var averageTTFTMs: Double = 0
    var restoredCheckpointCounts: [String: Int] = [:]

    var hitRate: Double {
        guard lookupCount > 0 else { return 0 }
        return Double(hitCount + ssdHitCount) / Double(lookupCount)
    }

    var tokenReuseRate: Double {
        guard promptTokens > 0 else { return 0 }
        return Double(cachedTokens) / Double(promptTokens)
    }

    static func from(events: [PromptCacheTelemetryEvent]) -> Self {
        var aggregate = PromptCacheTelemetryAggregate()
        var lookupMs: [Double] = []
        var restoreMs: [Double] = []
        var prefillMs: [Double] = []
        var ttftMs: [Double] = []

        for event in events {
            switch event.eventName {
            case "lookup":
                aggregate.lookupCount += 1
                let reason = event.field("reason") ?? ""
                if reason == "hit" {
                    aggregate.hitCount += 1
                } else if reason == "ssdHit" {
                    aggregate.ssdHitCount += 1
                } else {
                    aggregate.missCount += 1
                }
                aggregate.promptTokens += event.intField("promptTokens") ?? 0
                aggregate.cachedTokens += event.intField("skippedPrefillTokens") ?? 0
                if let type = event.field("checkpointType"), type != "nil" {
                    aggregate.restoredCheckpointCounts[type, default: 0] += 1
                }
                if let value = event.doubleField("lookupMs") { lookupMs.append(value) }
                if let value = event.doubleField("restoreMs") { restoreMs.append(value) }

            case "capture":
                aggregate.captureCount += 1

            case "eviction", "ssdEvictAtAdmission", "leafSupersession":
                aggregate.evictionCount += 1

            case "ssdAdmit":
                aggregate.admissionCount += 1
                if let outcome = event.field("outcome"), outcome != "accepted" {
                    aggregate.admissionDropCount += 1
                }

            case "ttft":
                if let value = event.doubleField("lookupMs") { lookupMs.append(value) }
                if let value = event.doubleField("restoreMs") { restoreMs.append(value) }
                if let value = event.doubleField("prefillMs") { prefillMs.append(value) }
                if let value = event.doubleField("totalPromptMs") { ttftMs.append(value) }

            default:
                break
            }
        }

        aggregate.averageLookupMs = Self.average(lookupMs)
        aggregate.averageRestoreMs = Self.average(restoreMs)
        aggregate.averagePrefillMs = Self.average(prefillMs)
        aggregate.averageTTFTMs = Self.average(ttftMs)
        return aggregate
    }

    private static func average(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }
}

nonisolated struct PromptCacheMetricSample: Identifiable, Codable, Equatable, Sendable {
    let id: UUID
    let date: Date
    let hitRate: Double
    let tokenReuseRate: Double
    let ramBytes: Int
    let ramBudgetBytes: Int
    let ssdBytes: Int
    let ssdBudgetBytes: Int
    let evictionCount: Int
    let admissionCount: Int
    let averageLookupMs: Double
    let averageRestoreMs: Double
    let averagePrefillMs: Double

    init(
        id: UUID = UUID(),
        date: Date = Date(),
        hitRate: Double,
        tokenReuseRate: Double,
        ramBytes: Int,
        ramBudgetBytes: Int,
        ssdBytes: Int,
        ssdBudgetBytes: Int,
        evictionCount: Int,
        admissionCount: Int,
        averageLookupMs: Double,
        averageRestoreMs: Double,
        averagePrefillMs: Double
    ) {
        self.id = id
        self.date = date
        self.hitRate = hitRate
        self.tokenReuseRate = tokenReuseRate
        self.ramBytes = ramBytes
        self.ramBudgetBytes = ramBudgetBytes
        self.ssdBytes = ssdBytes
        self.ssdBudgetBytes = ssdBudgetBytes
        self.evictionCount = evictionCount
        self.admissionCount = admissionCount
        self.averageLookupMs = averageLookupMs
        self.averageRestoreMs = averageRestoreMs
        self.averagePrefillMs = averagePrefillMs
    }
}

nonisolated struct PromptCacheSSDSnapshot: Codable, Equatable, Sendable {
    let enabled: Bool
    let rootPath: String?
    let budgetBytes: Int
    let currentBytes: Int
    let pendingBytes: Int
    let maxPendingBytes: Int
    let pendingCount: Int
    let snapshotCount: Int
    let partitionCount: Int

    static let disabled = PromptCacheSSDSnapshot(
        enabled: false,
        rootPath: nil,
        budgetBytes: 0,
        currentBytes: 0,
        pendingBytes: 0,
        maxPendingBytes: 0,
        pendingCount: 0,
        snapshotCount: 0,
        partitionCount: 0
    )
}

nonisolated struct PromptCacheTelemetrySnapshot: Codable, Equatable, Sendable {
    let capturedAt: Date
    let memoryBudgetBytes: Int
    let residentSnapshotBytes: Int
    let partitionCount: Int
    let totalNodeCount: Int
    let snapshotCount: Int
    let snapshotsByType: [String: Int]
    let ssd: PromptCacheSSDSnapshot
    let trees: [PromptCacheTreeSnapshot]

    static let empty = PromptCacheTelemetrySnapshot(
        capturedAt: Date(),
        memoryBudgetBytes: 0,
        residentSnapshotBytes: 0,
        partitionCount: 0,
        totalNodeCount: 0,
        snapshotCount: 0,
        snapshotsByType: [:],
        ssd: .disabled,
        trees: []
    )
}

nonisolated struct PromptCacheTreeSnapshot: Identifiable, Codable, Equatable, Sendable {
    let id: String
    let partitionDigest: String
    let partitionSummary: String
    let nodeCount: Int
    let totalSnapshotBytes: Int
    let snapshotCount: Int
    let snapshotsByType: [String: Int]
    let nodes: [PromptCacheTreeNodeSnapshot]
    let edges: [PromptCacheTreeEdgeSnapshot]
}

nonisolated struct PromptCacheTreeNodeSnapshot: Identifiable, Codable, Equatable, Sendable {
    let id: String
    let parentID: String?
    let pathHash: String
    let tokenOffset: Int
    let pathTokenCount: Int
    let edgeTokenCount: Int
    let childCount: Int
    let depth: Int
    let hasSnapshot: Bool
    let checkpointType: String?
    let snapshotBytes: Int
    let storageState: PromptCacheStorageState
    let storageRefID: String?
    let storageBytes: Int
    let lastAccessAgeSeconds: Double
    let normalizedRecency: Double?
    let normalizedFlopEfficiency: Double?
    let utility: Double?
}

nonisolated struct PromptCacheTreeEdgeSnapshot: Identifiable, Codable, Equatable, Sendable {
    let id: String
    let parentID: String
    let childID: String
    let tokenCount: Int
}

nonisolated struct PromptCacheExportPayload: Codable, Equatable, Sendable {
    let exportedAt: Date
    let snapshot: PromptCacheTelemetrySnapshot?
    let aggregate: PromptCacheTelemetryAggregate
    let events: [PromptCacheTelemetryEvent]
}
