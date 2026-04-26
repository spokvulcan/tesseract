//
//  SnapshotManifest.swift
//  tesseract
//
//  Pure value types describing the SSD prefix-cache tier's on-disk
//  layout and its in-memory transport shapes. Intentionally decoupled
//  from the store, writer loop, and radix tree — the composition
//  happens in downstream files that import these shapes.
//
//  Five types live here:
//  - `PersistedSnapshotDescriptor` — one persisted snapshot, Codable,
//    written into the authoritative `manifest.json` and mirrored into
//    the per-file safetensors header so manifest corruption can be
//    recovered from a directory walk.
//  - `PartitionMeta` — per-partition sidecar (`_meta.json`); pins the
//    `modelFingerprint` that warm start validates against the loaded
//    model to prevent cross-weight-swap hits.
//  - `SnapshotManifest` — top-level manifest, keyed by partition
//    digest and snapshot ID; authoritative tree-structure index at
//    warm start.
//  - `SnapshotStorageRef` — in-memory reference attached to a radix
//    node while an SSD write is in flight (`committed == false`) or
//    landed (`committed == true`). Non-Codable: `ContinuousClock.Instant`
//    is the writer's recency input for the LRU cut and has no stable
//    wire format across process restarts.
//  - `SnapshotPayload` — raw bytes extracted from a
//    `HybridCacheSnapshot` inside `container.perform`; the only shape
//    that crosses the LLMActor → `SSDSnapshotStore` writer boundary.
//    Sendable value type, never Codable — consumed immediately by
//    the safetensors write and never serialized through JSON.
//  - `SnapshotPayloadBundle` — a target payload plus an optional
//    DFlash draft companion payload admitted and evicted as one SSD
//    resident.
//

import Foundation
import MLXLMCommon

// MARK: - Schema version

/// Centralized schema version for the persisted data model. Bumped
/// as a unit when `PersistedSnapshotDescriptor`, `PartitionMeta`, or
/// `SnapshotManifest` gain incompatible field changes. A mismatch
/// at warm start triggers a `manifest.v{old}.bak` rename + wipe.
nonisolated enum SnapshotManifestSchema {
    /// Incrementing this value invalidates every on-disk manifest
    /// and discards all previously persisted snapshots on the next
    /// warm start. Bump whenever a field that `partitionDigest` or
    /// `PartitionMeta` canonicalizes changes shape, so partitions
    /// written under the old canonicalization cannot be silently
    /// reattached under a new key and surface stale state.
    ///
    /// **History**
    /// - v4: pre-TriAttention dense-only persistence.
    /// - v5: `PartitionMeta` carries `TriAttentionPartitionIdentity`
    ///   so warm-start reconstructs partition keys (including the
    ///   TriAttention identity) without losing the on-disk digest.
    ///   Older v4 manifests are unable to express TriAttention
    ///   partitions and must be wiped on first boot under v5.
    /// - v6: TriAttention partition canonicalization includes the
    ///   public prefix-protection mode, so persisted TriAttention
    ///   partitions written under v5 must be wiped rather than
    ///   silently reattached under a different runtime policy.
    static let currentVersion: Int = 6
}

// MARK: - Persisted descriptor (Codable, manifest.json + safetensors header)

/// One persisted snapshot. Exactly one file on disk (`.safetensors`),
/// exactly one entry in `SnapshotManifest.snapshots`.
///
/// **What this descriptor deliberately does and does not carry.**
/// The descriptor holds only the inputs the SSD-tier eviction policy
/// actually consumes. The SSD tier runs type-protected LRU, which
/// is equivalent to Marconi scoring at α=0 (LRU within the eligible
/// set). That means the writer needs `checkpointType` for type
/// protection and `lastAccessAt` for recency — nothing else. We
/// deliberately do NOT store `parentTokenOffset` or `childCount`
/// here because those inputs are only meaningful against live
/// radix-tree state. Promoting the SSD tier to the full Marconi
/// utility formula would require either a cross-actor hop per
/// scoring call or a descriptor schema extension, and is deferred
/// until production traces show the alpha tuner raising α above 0.
///
/// `lastAccessAt` is the single mutable field: `SSDSnapshotStore`
/// bumps it to "now" under its front-door lock on every RAM hit
/// that lands on a committed ref and on every SSD hydration, so
/// the writer's LRU cut sees fresh recency data without needing
/// live tree inputs.
nonisolated struct PersistedSnapshotDescriptor: Codable, Sendable, Equatable {
    /// Stable, UUID-shaped identifier. Survives restarts; used as
    /// both the on-disk filename (`{snapshotID}.safetensors`) and
    /// the lookup key across `SnapshotManifest.snapshots`, the
    /// store's in-memory book-keeping, and the MainActor
    /// `pendingRefsByID` map.
    let snapshotID: String

    /// 8-hex FNV digest of the owning `CachePartitionKey` (including
    /// `modelFingerprint`). Matches the directory layout
    /// `partitions/{partitionDigest}/snapshots/.../{id}.safetensors`.
    let partitionDigest: String

    /// Radix-tree path from the root to this snapshot's node, as a
    /// sequence of token IDs. Enables manifest rebuild: a directory
    /// walk + safetensors header parse is enough to reconstruct the
    /// tree structure even when `manifest.json` is missing or corrupt.
    let pathFromRoot: [Int]

    /// Absolute token offset into the prompt at which this snapshot
    /// was captured. Mirrors `HybridCacheSnapshot.tokenOffset`.
    let tokenOffset: Int

    /// String wire form of `HybridCacheSnapshot.CheckpointType`. Stored
    /// as a String rather than the enum so the descriptor stays
    /// trivially `Codable` and survives enum-case additions without
    /// forcing a schema bump. Use
    /// `HybridCacheSnapshot.CheckpointType(wireString:)` to recover
    /// the enum; an unrecognized value is treated as "drop this
    /// descriptor" by the warm-start path.
    let checkpointType: String

    /// On-disk file size in bytes. Matches
    /// `SnapshotStorageRef.bytesOnDisk` and is the sole byte-budget
    /// input for the writer's admission-time LRU cut. For descriptors
    /// with a DFlash draft companion this is target bytes + draft bytes.
    let bytes: Int

    /// Optional on-disk size of the DFlash draft-cache companion.
    /// Missing for old target-only descriptors and for snapshots
    /// captured while DFlash was unavailable.
    let dflashDraftBytes: Int?

    /// Seconds since Date's reference date (2001-01-01). Stable across
    /// restarts (unlike `ContinuousClock.Instant`) and suitable for
    /// the Codable round-trip. Produced via
    /// `Date().timeIntervalSinceReferenceDate`.
    let createdAt: Double

    /// Mutable recency input, seconds since reference date. Bumped
    /// to "now" by `SSDSnapshotStore.recordHit(id:)` on every hit
    /// (RAM state-4 lookup OR SSD state-5 hydration). Sole eviction
    /// input under the α=0 LRU rule.
    var lastAccessAt: Double

    /// Relative path from `SSDPrefixCacheConfig.rootURL` to the file,
    /// in the form `partitions/{digest}/snapshots/{shardByte}/{id}.safetensors`.
    /// Persisted alongside the ID so recovery does not depend on the
    /// store's URL-construction helper.
    let fileRelativePath: String

    /// Optional relative path to the DFlash draft-cache companion.
    /// Stored separately so target-only snapshots remain readable when
    /// the companion is absent or corrupt.
    let dflashDraftFileRelativePath: String?

    /// Schema version stamped at write time. Warm start rejects
    /// descriptors whose version differs from
    /// `SnapshotManifestSchema.currentVersion`.
    let schemaVersion: Int

    init(
        snapshotID: String,
        partitionDigest: String,
        pathFromRoot: [Int],
        tokenOffset: Int,
        checkpointType: String,
        bytes: Int,
        dflashDraftBytes: Int? = nil,
        createdAt: Double,
        lastAccessAt: Double,
        fileRelativePath: String,
        dflashDraftFileRelativePath: String? = nil,
        schemaVersion: Int
    ) {
        self.snapshotID = snapshotID
        self.partitionDigest = partitionDigest
        self.pathFromRoot = pathFromRoot
        self.tokenOffset = tokenOffset
        self.checkpointType = checkpointType
        self.bytes = bytes
        self.dflashDraftBytes = dflashDraftBytes
        self.createdAt = createdAt
        self.lastAccessAt = lastAccessAt
        self.fileRelativePath = fileRelativePath
        self.dflashDraftFileRelativePath = dflashDraftFileRelativePath
        self.schemaVersion = schemaVersion
    }

    var hasDFlashDraftCompanion: Bool {
        dflashDraftBytes != nil && dflashDraftFileRelativePath != nil
    }

    func withoutDFlashDraftCompanion() -> PersistedSnapshotDescriptor {
        let draftBytes = dflashDraftBytes ?? 0
        return PersistedSnapshotDescriptor(
            snapshotID: snapshotID,
            partitionDigest: partitionDigest,
            pathFromRoot: pathFromRoot,
            tokenOffset: tokenOffset,
            checkpointType: checkpointType,
            bytes: max(0, bytes - draftBytes),
            createdAt: createdAt,
            lastAccessAt: lastAccessAt,
            fileRelativePath: fileRelativePath,
            schemaVersion: schemaVersion
        )
    }

    /// Single source of truth for the on-disk path layout
    /// (`partitions/{digest}/snapshots/{shardByte}/{id}.safetensors`,
    /// where `shardByte` is the first character of `snapshotID`).
    /// Called by the writer (`SSDSnapshotStore.fileURL`), the
    /// descriptor factory (`PrefixCacheManager.makePersistedDescriptor`),
    /// and the warm-start path. Any change to this layout invalidates
    /// every previously persisted snapshot — see the stability
    /// contract on `CachePartitionKey.partitionDigest`.
    static func relativeFilePath(
        snapshotID: String,
        partitionDigest: String
    ) -> String {
        let shardByte = String(snapshotID.prefix(1))
        return "partitions/\(partitionDigest)/snapshots/\(shardByte)/\(snapshotID).safetensors"
    }

    static func relativeDFlashDraftFilePath(
        snapshotID: String,
        partitionDigest: String
    ) -> String {
        let shardByte = String(snapshotID.prefix(1))
        return "partitions/\(partitionDigest)/snapshots/\(shardByte)/\(snapshotID).dflash.safetensors"
    }
}

// MARK: - Partition metadata (Codable, _meta.json)

/// Per-partition sidecar. One file per `partitions/{digest}/_meta.json`,
/// pinned to the `modelFingerprint` snapshotted at model load time.
/// The warm-start path validates this fingerprint against the currently
/// loaded model's fingerprint: on mismatch, every descriptor under the
/// partition is dropped and the partition directory is scheduled for
/// async cleanup.
nonisolated struct PartitionMeta: Codable, Sendable, Equatable {
    /// Human-readable model identifier (e.g. `"mlx-community/Qwen3-4B-4bit"`).
    /// Redundant with the digest — carried here so a human reading a
    /// `_meta.json` can tell which partition they are looking at.
    let modelID: String

    /// Hex SHA-256 over `config.json` + `tokenizer.json` + sorted
    /// `(filename, size, mtime)` tuples for every `*.safetensors` in
    /// the model directory. Computed by `ModelFingerprint.compute...`.
    /// A mismatch between this value and the model's current
    /// fingerprint invalidates the entire partition at warm start.
    let modelFingerprint: String

    /// Quantization bits for the KV cache (`nil` for unquantized
    /// pure-attention models, otherwise typically 4 or 8). Mirrors
    /// `CachePartitionKey.kvBits`.
    let kvBits: Int?

    /// Quantization group size for the KV cache (e.g. 32, 64, 128).
    /// Mirrors `CachePartitionKey.kvGroupSize`.
    let kvGroupSize: Int

    /// Seconds since Date's reference date. Not used for any eviction
    /// decision; recorded for diagnostics only.
    let createdAt: Double

    /// Schema version for this `PartitionMeta`. Tracks the same
    /// `SnapshotManifestSchema.currentVersion` lifecycle as the
    /// top-level manifest.
    let schemaVersion: Int

    /// TriAttention identity carried so warm-start can reconstruct
    /// the partition's full `CachePartitionKey` — including the
    /// TriAttention discriminator — and verify the on-disk digest
    /// matches without collapsing TriAttention partitions back to
    /// `.dense`. Defaults to `.dense` so v5 reads of files that
    /// somehow omit the field still produce a valid dense partition
    /// rather than throwing a decode error. Added in schema v5.
    let triAttention: TriAttentionPartitionIdentity

    init(
        modelID: String,
        modelFingerprint: String,
        kvBits: Int?,
        kvGroupSize: Int,
        createdAt: Double,
        schemaVersion: Int,
        triAttention: TriAttentionPartitionIdentity = .dense
    ) {
        self.modelID = modelID
        self.modelFingerprint = modelFingerprint
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.createdAt = createdAt
        self.schemaVersion = schemaVersion
        self.triAttention = triAttention
    }

    private enum CodingKeys: String, CodingKey {
        case modelID
        case modelFingerprint
        case kvBits
        case kvGroupSize
        case createdAt
        case schemaVersion
        case triAttention
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelID = try container.decode(String.self, forKey: .modelID)
        self.modelFingerprint = try container.decode(String.self, forKey: .modelFingerprint)
        self.kvBits = try container.decodeIfPresent(Int.self, forKey: .kvBits)
        self.kvGroupSize = try container.decode(Int.self, forKey: .kvGroupSize)
        self.createdAt = try container.decode(Double.self, forKey: .createdAt)
        self.schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        self.triAttention =
            try container.decodeIfPresent(
                TriAttentionPartitionIdentity.self, forKey: .triAttention
            ) ?? .dense
    }
}

// MARK: - Top-level manifest (Codable, manifest.json)

/// Authoritative tree-structure index on disk. The warm-start path
/// reads this file first; if it is missing or corrupt, the per-file
/// safetensors headers (which duplicate every descriptor's critical
/// fields) provide the recovery path.
///
/// Both dictionaries are `var` rather than `let` so the store can
/// incrementally mutate them under its front-door lock and schedule a
/// debounced persist without reconstructing the whole value.
nonisolated struct SnapshotManifest: Codable, Sendable, Equatable {
    /// Schema version stamped on the whole manifest. A mismatch at
    /// warm start renames `manifest.json` → `manifest.v{old}.bak`
    /// and starts fresh.
    var schemaVersion: Int

    /// Keyed by `partitionDigest`. Empty on a freshly created manifest.
    var partitions: [String: PartitionMeta]

    /// Keyed by `PersistedSnapshotDescriptor.snapshotID`. Every entry
    /// must reference a partition digest that is present in
    /// `partitions` — the warm-start path drops dangling descriptors
    /// as a defensive repair.
    var snapshots: [String: PersistedSnapshotDescriptor]

    /// Empty-manifest constructor used by the store on first start and
    /// by the warm-start path on any `schemaVersion` mismatch.
    static func empty() -> SnapshotManifest {
        SnapshotManifest(
            schemaVersion: SnapshotManifestSchema.currentVersion,
            partitions: [:],
            snapshots: [:]
        )
    }
}

// MARK: - In-memory storage ref (non-Codable)

/// In-memory reference attached to a `RadixTreeNode` while an SSD
/// write is pending or after it has committed. Lives alongside
/// `node.snapshot` (the RAM body), not instead of it — the
/// combination drives the five-state lifecycle:
///
/// | State | Body   | Ref     | Committed | Semantics                |
/// |-------|--------|---------|-----------|--------------------------|
/// | 1     | yes    | none    | —         | RAM-only                 |
/// | 2     | yes    | present | false     | Pending write            |
/// | 3     | no     | present | false     | Pending, body dropped    |
/// | 4     | yes    | present | true      | Committed + RAM          |
/// | 5     | no     | present | true      | SSD-only (hydratable)    |
///
/// Not `Codable`: `lastAccessTime` uses `ContinuousClock.Instant`,
/// which has no stable wire format across process restarts. The
/// persisted analogue is `PersistedSnapshotDescriptor`, which stores
/// `lastAccessAt` as seconds since the reference date.
nonisolated struct SnapshotStorageRef: Sendable, Equatable {
    /// Mirrors `PersistedSnapshotDescriptor.snapshotID`. Stable for
    /// the lifetime of the file on disk.
    let snapshotID: String

    /// Mirrors `PersistedSnapshotDescriptor.partitionDigest`. Used by
    /// `SSDSnapshotStore.loadSync` (via `fileURL(for:)`) to resolve
    /// the on-disk path from the store's immutable `rootURL`.
    let partitionDigest: String

    /// Absolute token offset at which the snapshot was captured.
    /// Mirrors `PersistedSnapshotDescriptor.tokenOffset`.
    let tokenOffset: Int

    /// The enum form of the checkpoint type. Unlike the descriptor's
    /// String wire form, this stays as the vendor enum because this
    /// value never crosses a persistence boundary — it is read by
    /// `SSDSnapshotStore`'s type-protection filter and the eviction
    /// path to tell `.system` residents apart from the rest.
    let checkpointType: HybridCacheSnapshot.CheckpointType

    /// Mirrors `PersistedSnapshotDescriptor.bytes`. Present so the
    /// eviction path can reach the SSD byte count via
    /// `node.storageRef` alone, without maintaining a separate
    /// `snapshotID → descriptor` map just to look up a size. The RAM
    /// lookup path does not touch this value.
    let bytesOnDisk: Int

    /// In-process recency input. Updated on every hit so the writer's
    /// LRU cut can rank SSD-resident entries without touching the
    /// descriptor map. Not persisted — on warm start the rebuilt refs
    /// start at `.now` by construction.
    var lastAccessTime: ContinuousClock.Instant

    /// Write commit gate. `false` while the write is enqueued or
    /// in-flight; `true` after `SSDSnapshotStore.writerLoop` has
    /// fsync'd and atomically renamed the file. Lookups landing on
    /// a ref with `committed == false` treat the node as a miss so
    /// no race can surface a half-written file.
    var committed: Bool
}

// MARK: - In-memory transport payload (Sendable, not Codable)

/// Raw bytes extracted from a `HybridCacheSnapshot` inside
/// `container.perform`, plus the metadata needed to reconstruct the
/// snapshot on hydration. The only shape that crosses the
/// LLMActor → `SSDSnapshotStore` boundary — pure `Sendable` value
/// type with no MLX references.
///
/// Not `Codable`. The payload is the input to the safetensors
/// writer, which consumes it byte-for-byte via
/// `HybridCacheSnapshot.serialize(to:metadata:)` inside the writer
/// task and never round-trips it through JSON. Making this
/// `Codable` would encourage callers to persist the payload without
/// going through the safetensors path, which is exactly what the
/// Metal-affinity rules forbid.
nonisolated struct SnapshotPayload: Sendable {

    /// One MLX state array, pre-extracted from Metal-resident memory
    /// into a plain `Data` blob.
    struct ArrayPayload: Sendable {
        /// Raw bytes from `MLXArray.asData()`. Already owned by the
        /// CPU after `eval(cache)`; the Apple Silicon unified-memory
        /// footing means there is no staging copy.
        let data: Data

        /// MLX dtype name as reported by the vendor (e.g. `"bfloat16"`,
        /// `"int8"`, `"float16"`). Stored as a String so the payload
        /// type does not have to import the MLX enum.
        let dtype: String

        /// Array shape. Preserved for the safetensors header so the
        /// reader can reconstruct the array before MLX gets involved.
        let shape: [Int]
    }

    /// One `HybridCacheSnapshot.LayerState`, mirrored as a flat value
    /// type. The vendor type is `@unchecked Sendable` because it
    /// holds `[MLXArray]`; this mirror is genuinely `Sendable`.
    struct LayerPayload: Sendable {
        /// Cache class name matching the `savePromptCache` convention
        /// (e.g. `"KVCache"`, `"QuantizedKVCache"`, `"RotatingKVCache"`,
        /// `"ChunkedKVCache"`, `"MambaCache"`, `"ArraysCache"`).
        let className: String

        /// Per-array extracted bytes. One entry per
        /// `HybridCacheSnapshot.LayerState.state` element, in the
        /// same order.
        let state: [ArrayPayload]

        /// `HybridCacheSnapshot.LayerState.metaState`, verbatim.
        /// Stable `String` values that the vendor reconstructs from.
        let metaState: [String]

        /// Absolute token offset captured from
        /// `HybridCacheSnapshot.LayerState.offset`. Mirrors the
        /// vendor's explicit-offset contract.
        let offset: Int
    }

    /// The snapshot's absolute prompt token offset. Mirrors
    /// `HybridCacheSnapshot.tokenOffset`.
    let tokenOffset: Int

    /// Mirrors `HybridCacheSnapshot.checkpointType`. Kept as the enum
    /// (not the wire String) because the payload is in-memory only.
    let checkpointType: HybridCacheSnapshot.CheckpointType

    /// Per-layer extracted payloads, in the same order as the vendor
    /// snapshot's `layers` array.
    let layers: [LayerPayload]

    /// Sum of every `state` array's byte count across every layer.
    /// The front-door `tryEnqueue` uses this to enforce
    /// `SSDPrefixCacheConfig.maxPendingBytes`, so the value has to be
    /// cheap — it walks the `[LayerPayload]` once, no allocation.
    var totalBytes: Int {
        var total = 0
        for layer in layers {
            for array in layer.state {
                total += array.data.count
            }
        }
        return total
    }
}

nonisolated struct SnapshotPayloadBundle: Sendable {
    let target: SnapshotPayload
    let dflashDraft: SnapshotPayload?

    init(target: SnapshotPayload, dflashDraft: SnapshotPayload? = nil) {
        self.target = target
        self.dflashDraft = dflashDraft
    }

    var totalBytes: Int {
        target.totalBytes + (dflashDraft?.totalBytes ?? 0)
    }
}

// MARK: - Partition digest (wire contract for on-disk directory layout)

extension CachePartitionKey {
    /// 8-hex-character FNV-1a 32-bit digest over the canonical
    /// string form of the key fields. Used by the SSD prefix-cache
    /// tier as both the partition directory name
    /// (`partitions/{digest}/...`) and the join key between
    /// in-memory `SnapshotStorageRef`s and their on-disk partitions.
    ///
    /// **Load-bearing stability contract.** Once a snapshot has
    /// been written to disk, the digest of its partition key must
    /// stay identical across restarts, process upgrades, and
    /// refactors. Any change to the canonicalization rule, the
    /// separator, or the hash function makes every previously
    /// persisted snapshot silently unreachable. The writer and
    /// warm-start paths both call this helper so the two cannot
    /// disagree — and `SnapshotManifestTests` pins a known
    /// input→output pair as a regression trap.
    ///
    /// **Canonicalization.** Concatenates the four base fields in fixed
    /// order (`modelID`, `kvBits`, `kvGroupSize`, `modelFingerprint`)
    /// separated by a single null byte (`\0`). Nullable fields
    /// (`kvBits`, `modelFingerprint`) use a presence tag: `"N"` for `nil`,
    /// `"S"` followed by the value for `Some`. The tag is load-
    /// bearing — a bare sentinel string like `"none"` would
    /// collide with a real value of `"none"` and silently merge
    /// two structurally distinct partitions on disk. The presence
    /// tag prevents the collision because `nil` always encodes as
    /// `"N"` and a `Some` value always encodes with a leading `"S"`,
    /// and no value starting with `"S"` can ever equal the bare
    /// `"N"` sentinel.
    ///
    /// When `triAttention == .dense` the canonical string stops here,
    /// keeping the pre-TriAttention digest stable so dense partitions
    /// persisted before TriAttention landed remain reachable under the
    /// exact same on-disk digest. `.triAttention(...)` appends a
    /// `\0TA:` tagged segment so TriAttention-enabled partitions can
    /// never collide with their dense counterparts (nor with each
    /// other across budget / calibration / impl-version changes).
    ///
    /// The null-byte separator cannot appear inside any field:
    /// `modelID` is a HuggingFace ID, `kvBits`/`kvGroupSize` are
    /// decimal integers, `modelFingerprint` is hex SHA-256,
    /// `budgetTokens` is a decimal integer,
    /// `calibrationArtifactIdentity` is hex SHA-256, and
    /// `implementationVersion` is a restricted identifier string
    /// (`"v1"` today), and `prefixProtectionMode` is a restricted
    /// identifier string (`"protectNone"`, `"protectStablePrefixOnly"`,
    /// or `"protectAllPrefill"`).
    nonisolated var partitionDigest: String {
        let kvBitsField = kvBits.map { "S\($0)" } ?? "N"
        let fingerprintField = modelFingerprint.map { "S" + $0 } ?? "N"
        var canonical =
            "\(modelID)\0\(kvBitsField)\0\(kvGroupSize)\0\(fingerprintField)"
        if case let .triAttention(budget, artifact, impl, mode) = triAttention {
            let artifactField = artifact.map { "S" + $0.rawValue } ?? "N"
            canonical += "\0TA:S\(budget)\0\(artifactField)\0\(impl.rawValue)\0\(mode.rawValue)"
        }

        // FNV-1a 32-bit: offset_basis = 0x811c9dc5, prime = 0x01000193.
        var hash: UInt32 = 0x811c_9dc5
        for byte in canonical.utf8 {
            hash ^= UInt32(byte)
            hash &*= 0x0100_0193
        }
        return String(format: "%08x", hash)
    }
}

// MARK: - Schema-version compatibility

extension SnapshotManifest {
    /// True when this manifest's `schemaVersion` matches
    /// `SnapshotManifestSchema.currentVersion`. Warm start uses
    /// this as the gate for the wipe/rebuild decision: compatible
    /// → load descriptors normally; incompatible → rename
    /// `manifest.json` → `manifest.v{old}.bak` and start fresh.
    ///
    /// Codable decoding does **not** throw on a version mismatch.
    /// An older- or newer-schema manifest deserializes cleanly as
    /// long as its structural shape matches; the responsibility
    /// for acting on the mismatch lives in the warm-start path.
    /// This helper is the signal warm start branches on.
    nonisolated var isSchemaCompatible: Bool {
        schemaVersion == SnapshotManifestSchema.currentVersion
    }
}
