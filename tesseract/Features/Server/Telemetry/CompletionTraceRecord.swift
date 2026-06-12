//
//  CompletionTraceRecord.swift
//  tesseract
//
//  The per-completion cache telemetry record (PRD #82, slice #83): one
//  value per cache-aware **Server Completion**, carrying both the
//  *observed outcomes* (TTFT breakdown, restored offset, hit tokens,
//  hydration seconds, eviction tallies) and the *workload inputs* a
//  policy replay needs (prompt length, cumulative prefix block digests,
//  admitted snapshots, the RAM budget in force).
//
//  Records are appended to the on-disk **Completion Trace Log**
//  (`CompletionTraceLog`), which is the replay corpus for the offline
//  trace-replay harness and — later — the rebuilt AlphaTuner's window
//  food. Raw `http-completions/` request JSON is *not* the corpus: it
//  has no token counts without a loaded model (see ADR-0011).
//
//  **Standard-route and Unkeyed Completions never produce a record.**
//  Only the cache-aware path emits one, and `make` returns `nil` for
//  any request that degraded to an **Unkeyed Completion** — those
//  requests never touched the radix tree, so they carry no signal for
//  cache-policy replay.
//

import Foundation

// MARK: - Prefix block digests

/// Content-derived identity for radix-path prefixes, without storing
/// tokens: the **Cache Key Path** is folded through FNV-1a 64 and the
/// cumulative hash is emitted at every `blockSize`-token boundary.
/// Two requests share a prefix of `n` blocks iff their first `n`
/// digests agree, so an offline replay can rebuild prefix-sharing
/// structure (chains, branch points) at block granularity from the
/// digests alone.
nonisolated enum TraceBlockDigest {
    /// Tokens per digest block. Persisted in the trace-log header so a
    /// future change cannot silently mix granularities in one corpus.
    static let blockSize = 256

    /// FNV-1a 64 offset basis. The token fold below is part of the
    /// persisted corpus contract — and the one token-hash algorithm in
    /// the codebase: `TokenRadixTree`'s telemetry path hash folds with
    /// the same primitives, so path identity cannot drift between the
    /// live tree dump and the trace corpus.
    static let fnvOffsetBasis: UInt64 = 0xcbf2_9ce4_8422_2325

    /// Fold one token — 8 little-endian bytes — into a running
    /// FNV-1a 64 hash.
    static func fold(token: Int, into hash: inout UInt64) {
        var value = UInt64(bitPattern: Int64(token))
        for _ in 0..<8 {
            hash ^= value & 0xFF
            hash = hash &* 0x0000_0100_0000_01B3
            value >>= 8
        }
    }

    /// Canonical 16-hex-digit rendering of a folded hash.
    static func hexDigest(_ hash: UInt64) -> String {
        String(format: "%016llx", hash)
    }

    /// Cumulative FNV-1a 64 digests of `tokens`, one per completed
    /// `blockSize` block. The hash chains across blocks (it is never
    /// reset), so digest `i` identifies the whole prefix
    /// `[0, (i+1) * blockSize)`. A trailing partial block emits no
    /// digest.
    static func cumulativeDigests(
        forKeyPath tokens: [Int],
        blockSize: Int = TraceBlockDigest.blockSize
    ) -> [String] {
        guard blockSize > 0 else { return [] }
        var digests: [String] = []
        var hash = fnvOffsetBasis
        var countInBlock = 0
        for token in tokens {
            fold(token: token, into: &hash)
            countInBlock += 1
            if countInBlock == blockSize {
                digests.append(hexDigest(hash))
                countInBlock = 0
            }
        }
        return digests
    }
}

// MARK: - Record

/// One admitted snapshot, as the replay corpus sees it: where it sits
/// on the path, how big its RAM body was, and its checkpoint type
/// (wire string, same vocabulary as `PersistedSnapshotDescriptor`).
nonisolated struct TraceAdmittedSnapshot: Codable, Sendable, Equatable {
    let offset: Int
    let bytes: Int
    let checkpointType: String
}

/// One per-completion record. All durations are seconds (`Double`);
/// the TTFT identity `ttftSeconds == lookup + restore + prefill +
/// residualPrompt` mirrors the live `ttft` diagnostics event.
nonisolated struct CompletionTraceRecord: Codable, Sendable, Equatable {
    /// Bumped when fields change incompatibly. Readers must check the
    /// trace-log header's `schemaVersion` before replaying.
    static let currentSchemaVersion = 1

    // Identity
    /// Seconds since Date's reference date at record time.
    let timestamp: Double
    let requestID: UUID
    let modelID: String
    let partitionDigest: String

    // Workload (replay inputs)
    /// **Cache Key Path** length for this request.
    let promptTokenCount: Int
    /// Cumulative prefix digests of the key path — see `TraceBlockDigest`.
    let prefixBlockDigests: [String]
    /// Mid-prefill snapshots admitted by this completion.
    let admittedCheckpoints: [TraceAdmittedSnapshot]
    /// The leaf admission, when the leaf store completed.
    let admittedLeaf: TraceAdmittedSnapshot?
    /// The RAM-tier budget in force when this completion ended, so a
    /// replay can reproduce the same pressure regime.
    let ramBudgetBytes: Int

    // Outcomes (observed)
    /// Offset of the restored snapshot (`0` on a miss).
    let restoredOffset: Int
    /// True when the restored snapshot was hydrated from the SSD tier.
    let restoredFromSSD: Bool
    /// Prefill tokens skipped thanks to the cache (the planner's base
    /// offset — what TTFT actually avoided paying).
    let hitTokens: Int
    /// Token-level shared prefix between the request and the tree.
    let sharedPrefixLength: Int
    let lookupSeconds: Double
    let restoreSeconds: Double
    /// SSD hydration time (`0` for RAM hits and misses).
    let hydrationSeconds: Double
    let prefillSeconds: Double
    let residualPromptSeconds: Double
    let ttftSeconds: Double
    /// RAM-tier evictions this completion triggered whose body was lost
    /// outright (no surviving SSD ref).
    let terminalEvictionCount: Int
    /// RAM-tier evictions this completion triggered whose node stayed
    /// hittable via a surviving **Snapshot Ref**.
    let recoveredEvictionCount: Int
    /// Rolling measured device estimates at record time, so offline
    /// replay denominates seconds in the same units the live scorer
    /// used (slice #84). `nil` on records written before slice #84.
    let deviceEstimates: MeasuredSecondsEstimates?

    /// Assemble a record for one finished cache-aware completion, or
    /// `nil` for requests that must produce none: any **Unkeyed
    /// Completion** (zero cache participation — no replay signal).
    /// The standard (non-cache-aware) route never reaches this call.
    static func make(
        timestamp: Double,
        requestID: UUID,
        modelID: String,
        partitionDigest: String,
        unkeyedReason: CacheKeySpace.UnkeyedReason?,
        keyPath: [Int],
        admittedCheckpoints: [TraceAdmittedSnapshot],
        admittedLeaf: TraceAdmittedSnapshot?,
        ramBudgetBytes: Int,
        restoredOffset: Int,
        restoredFromSSD: Bool,
        hitTokens: Int,
        sharedPrefixLength: Int,
        lookupSeconds: Double,
        restoreSeconds: Double,
        hydrationSeconds: Double,
        prefillSeconds: Double,
        residualPromptSeconds: Double,
        terminalEvictionCount: Int,
        recoveredEvictionCount: Int,
        deviceEstimates: MeasuredSecondsEstimates?
    ) -> CompletionTraceRecord? {
        guard unkeyedReason == nil else { return nil }
        return CompletionTraceRecord(
            timestamp: timestamp,
            requestID: requestID,
            modelID: modelID,
            partitionDigest: partitionDigest,
            promptTokenCount: keyPath.count,
            prefixBlockDigests: TraceBlockDigest.cumulativeDigests(forKeyPath: keyPath),
            admittedCheckpoints: admittedCheckpoints,
            admittedLeaf: admittedLeaf,
            ramBudgetBytes: ramBudgetBytes,
            restoredOffset: restoredOffset,
            restoredFromSSD: restoredFromSSD,
            hitTokens: hitTokens,
            sharedPrefixLength: sharedPrefixLength,
            lookupSeconds: lookupSeconds,
            restoreSeconds: restoreSeconds,
            hydrationSeconds: hydrationSeconds,
            prefillSeconds: prefillSeconds,
            residualPromptSeconds: residualPromptSeconds,
            ttftSeconds: lookupSeconds + restoreSeconds + prefillSeconds
                + residualPromptSeconds,
            terminalEvictionCount: terminalEvictionCount,
            recoveredEvictionCount: recoveredEvictionCount,
            deviceEstimates: deviceEstimates
        )
    }
}

// MARK: - Log line

/// First line of every trace-log file: the corpus contract a reader
/// validates before replaying.
nonisolated struct CompletionTraceHeader: Codable, Sendable, Equatable {
    let schemaVersion: Int
    let blockSize: Int
    /// Seconds since Date's reference date when the file was opened.
    let createdAt: Double
}

/// One JSONL line of the trace log, discriminated by `kind` so the
/// format can grow new line kinds without breaking old readers.
nonisolated enum CompletionTraceLine: Codable, Sendable, Equatable {
    case header(CompletionTraceHeader)
    case record(CompletionTraceRecord)

    private enum CodingKeys: String, CodingKey {
        case kind
        case header
        case record
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(String.self, forKey: .kind)
        switch kind {
        case "header":
            self = .header(try container.decode(CompletionTraceHeader.self, forKey: .header))
        case "record":
            self = .record(try container.decode(CompletionTraceRecord.self, forKey: .record))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .kind,
                in: container,
                debugDescription: "unknown trace line kind '\(kind)'"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .header(let header):
            try container.encode("header", forKey: .kind)
            try container.encode(header, forKey: .header)
        case .record(let record):
            try container.encode("record", forKey: .kind)
            try container.encode(record, forKey: .record)
        }
    }
}
