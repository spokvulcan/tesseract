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
    /// **Rewind telemetry** (issue #101): lookups whose restore floor sat
    /// below where the request diverged from the cached path — a
    /// **Think-Strip Rewind** that the **Chain-Prefix Restore** floor
    /// served instead of re-prefilling to zero. The token sum is the
    /// re-prefill those rewinds saved versus the strip floor; a rising
    /// count or deepening size is the regression signal.
    var rewindEventCount: Int = 0
    var rewindTokens: Int = 0

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
                // Production rewrites a hydrated `.ssdHit`/`.chainPrefixHit` to
                // `.hit` before this event is logged, so the hit *kind* and the
                // rewind marker arrive as explicit fields. The raw reasons still
                // appear on the no-fingerprint replay path — honor both so the
                // RAM/SSD split and the Rewinds KPI fire on every surface.
                let hydratedFromSSD = event.field("hydratedFromSSD") == "true"
                let chainPrefixRewind = event.field("chainPrefixRestore") == "true"
                let isHit = reason == "hit" || reason == "ssdHit" || reason == "chainPrefixHit"
                if !isHit {
                    aggregate.missCount += 1
                } else if hydratedFromSSD || reason == "ssdHit" || reason == "chainPrefixHit" {
                    // A chain-prefix hit (ADR-0012) is an SSD-tier hit: the body
                    // hydrates from the owning chain's leading segments.
                    aggregate.ssdHitCount += 1
                } else {
                    aggregate.hitCount += 1
                }
                // A chain-prefix restore IS a served Think-Strip Rewind: the
                // restore landed at the floor below the request's divergence
                // (issue #101). `shared > floor` excludes a restore that landed
                // exactly at the divergence (a hit, but not a rewind).
                if chainPrefixRewind || reason == "chainPrefixHit",
                    let shared = event.intField("sharedPrefixLength"),
                    let floor = event.intField("snapshotOffset"),
                    shared > floor
                {
                    aggregate.rewindEventCount += 1
                    aggregate.rewindTokens += shared - floor
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
                if let value = event.doubleField("ttftMs") { ttftMs.append(value) }

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

/// Lifetime counters the `PrefixCacheManager` accumulates across its own
/// life (one model load): what the cache bought (hit tokens, hydrations)
/// and what eviction cost (recovered vs terminal body losses). Lives on
/// the telemetry *snapshot* — unlike the event-derived aggregate, these
/// never lose history to the event-buffer cap.
nonisolated struct PromptCacheCumulativeCounters: Codable, Equatable, Sendable {
    /// Prefill tokens served from cache: restored offsets summed over
    /// RAM hits and successful SSD hydrations.
    var hitTokens: Int = 0
    /// RAM bodies dropped whose node stayed hittable via a surviving
    /// **Snapshot Ref** (SSD-backed drops; **Snapshot Demotion**s).
    var recoveredEvictions: Int = 0
    /// RAM bodies dropped outright — the next hit pays full re-prefill.
    var terminalEvictions: Int = 0
    /// State-5 bodies rebuilt from the SSD tier (`ssdOnly` → `committed`).
    var hydrations: Int = 0
    /// SSD writes skipped by the **Survival Gate** — incoming chains
    /// that would not have survived the eviction their own admission
    /// triggers (a demotion that terminal-dropped instead, a checkpoint
    /// write-through that stayed RAM-only).
    var survivalGateSkips: Int = 0
    /// Prefill wall-clock the cache absorbed: each restored offset's
    /// Marconi FLOPs divided by the measured prefill throughput at hit
    /// time — the same **Recovery Cost** units eviction scores in, so
    /// "seconds saved" and "seconds at risk" are directly comparable.
    var savedPrefillSeconds: Double = 0
    /// SSD writes skipped by **Adaptive Write Eagerness** (ADR-0019,
    /// PRD #150) — redundant copies not taken while RAM comfortably
    /// held the body and the node had not yet proven reuse.
    var eagernessDeferrals: Int = 0
    /// Deferred-class promotion writes issued when a previously
    /// skipped node's hit count crossed the eagerness threshold.
    var eagernessPromotions: Int = 0
}

/// Eviction-tuner state surfaced to the UI: the `AlphaTuner` phase plus the
/// alpha the **Eviction Configuration** is currently scoring with. `alpha`
/// is live even while the tuner is still waiting (0.0 = pure-recency LRU).
nonisolated struct PromptCacheTunerSnapshot: Codable, Equatable, Sendable {
    let phase: String
    let alpha: Double
    let bootstrapProgress: Int
    let bootstrapTarget: Int

    static let unavailable = PromptCacheTunerSnapshot(
        phase: "unavailable",
        alpha: 0,
        bootstrapProgress: 0,
        bootstrapTarget: 0
    )
}

nonisolated struct PromptCacheTelemetrySnapshot: Codable, Equatable, Sendable {
    let capturedAt: Date
    /// The **Pressure-Reactive Budget**'s *current* value — the live
    /// RAM-tier budget the eviction loop enforces.
    let memoryBudgetBytes: Int
    /// The band around `memoryBudgetBytes`: the load-time auto-sized
    /// ceiling and the content-defined **Budget Floor** as computed at
    /// capture time.
    let budgetCeilingBytes: Int
    let budgetFloorBytes: Int
    let residentSnapshotBytes: Int
    let partitionCount: Int
    let totalNodeCount: Int
    let snapshotCount: Int
    let snapshotsByType: [String: Int]
    let ssd: PromptCacheSSDSnapshot
    let tuner: PromptCacheTunerSnapshot
    let counters: PromptCacheCumulativeCounters
    let estimates: MeasuredSecondsEstimates
    let trees: [PromptCacheTreeSnapshot]

    static let empty = PromptCacheTelemetrySnapshot(
        capturedAt: Date(),
        memoryBudgetBytes: 0,
        budgetCeilingBytes: 0,
        budgetFloorBytes: 0,
        residentSnapshotBytes: 0,
        partitionCount: 0,
        totalNodeCount: 0,
        snapshotCount: 0,
        snapshotsByType: [:],
        ssd: .disabled,
        tuner: .unavailable,
        counters: PromptCacheCumulativeCounters(),
        estimates: MeasuredSecondsEstimates(),
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
    /// Mutable so tree filtering can re-parent a survivor to its nearest
    /// visible ancestor when the nodes in between are filtered out.
    var parentID: String?
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
    let snapshotRefID: String?
    let storageBytes: Int
    let lastAccessAgeSeconds: Double
    let normalizedRecency: Double?
    let normalizedFlopEfficiency: Double?
    let utility: Double?

    enum CodingKeys: String, CodingKey {
        case id
        case parentID
        case pathHash
        case tokenOffset
        case pathTokenCount
        case edgeTokenCount
        case childCount
        case depth
        case hasSnapshot
        case checkpointType
        case snapshotBytes
        case storageState
        case snapshotRefID = "storageRefID"
        case storageBytes
        case lastAccessAgeSeconds
        case normalizedRecency
        case normalizedFlopEfficiency
        case utility
    }
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
