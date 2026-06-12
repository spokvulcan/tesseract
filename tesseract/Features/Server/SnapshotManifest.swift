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
//  - `SnapshotRef` — immutable in-memory identity of an SSD-resident
//    snapshot, carried as the payload of the ref-bearing `SnapshotState`
//    cases. Models *what and where on disk* only — never the write
//    phase (encoded by the `SnapshotState` case) and never recency.
//    Non-Codable: the persisted analogue is `PersistedSnapshotDescriptor`.
//  - `SnapshotPayload` — raw bytes extracted from a
//    `HybridCacheSnapshot` inside `container.perform`; the only shape
//    that crosses the LLMActor → `SSDSnapshotStore` writer boundary.
//    Sendable value type, never Codable — consumed immediately by
//    the safetensors write and never serialized through JSON.
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
    /// - v7: TriAttention removed from the app; `PartitionMeta` no longer
    ///   carries a TriAttention identity and `partitionDigest` drops the
    ///   `\0TA:` segment. Dense digests are unchanged, but v6 manifests are
    ///   wiped on first boot under v7 so no stale TriAttention partition
    ///   lingers (see docs/adr/0005).
    /// - v8: **Segment Chain** support for **Leaf Extension Admission**
    ///   (docs/adr/0010): descriptors gain `segmentBaseOffset` (the own
    ///   file's slice start) and `inheritedSegments` (the chain transferred
    ///   from superseded ancestor leaves), and per-layer container headers
    ///   gain `suffix_base_offset`. v7 manifests are wiped on first boot
    ///   under v8.
    static let currentVersion: Int = 8
}

// MARK: - Segment chain (Codable, manifest.json + safetensors header)

/// One inherited **Snapshot Segment** of a **Segment Chain**: a file an
/// earlier leaf wrote, whose ownership a later **Leaf Extension
/// Admission** transferred to the chain's head descriptor. The head's
/// own file is described by the descriptor's `fileRelativePath` /
/// `segmentBaseOffset` / `tokenOffset` / `bytes` fields, never
/// duplicated here. Ordered shallow→deep; contiguity
/// (`next.baseOffset == previous.tokenOffset`, first at 0) is the
/// chain invariant hydration validates per layer.
nonisolated struct SnapshotSegment: Codable, Sendable, Equatable {
    /// Token offset this segment's sliceable layers start at.
    /// 0 for a full segment (always true for the chain's first).
    let baseOffset: Int

    /// Token offset this segment covers up to — the offset the segment's
    /// originating leaf was captured at.
    let tokenOffset: Int

    /// Relative path from the SSD root, same form as
    /// `PersistedSnapshotDescriptor.fileRelativePath`.
    let fileRelativePath: String

    /// On-disk file size in bytes, counted into the chain total.
    let bytes: Int
}

// MARK: - Leaf extension intent (in-memory, not Codable)

/// The base a **Leaf Extension Admission** slices against: the deepest
/// SSD-backed ancestor leaf on the new leaf's path. Resolved tree-side
/// by `PrefixCacheManager.extensionBase`, carried on the
/// `SnapshotPayload`, validated by the SSD front door, and consumed by
/// the writer's commit-time chain fold.
nonisolated struct SnapshotExtension: Sendable, Equatable {
    /// `snapshotID` of the base leaf's manifest entry / pending write.
    let baseSnapshotID: String

    /// The base leaf's capture offset — where the suffix slice starts.
    let baseOffset: Int
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

    /// On-disk size of the descriptor's **own** file in bytes. For a
    /// chain head this is the newest segment only; budget accounting
    /// and `SnapshotRef.bytesOnDisk` use `totalBytes` (own +
    /// inherited), the value the writer's admission-time LRU cut frees.
    let bytes: Int

    /// Token offset the own file's sliceable layers start at. 0 for a
    /// full snapshot (every non-leaf descriptor); the base leaf's
    /// capture offset for a **Leaf Extension Admission**.
    let segmentBaseOffset: Int

    /// The **Segment Chain** below the own file, ordered shallow→deep.
    /// Empty for a single-file full snapshot. Populated by the writer's
    /// commit-time fold (the base entry's chain plus the base's own
    /// file), never at descriptor minting — the chain is only stable
    /// once the base has settled.
    let inheritedSegments: [SnapshotSegment]

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
        segmentBaseOffset: Int = 0,
        inheritedSegments: [SnapshotSegment] = [],
        createdAt: Double,
        lastAccessAt: Double,
        fileRelativePath: String,
        schemaVersion: Int
    ) {
        self.snapshotID = snapshotID
        self.partitionDigest = partitionDigest
        self.pathFromRoot = pathFromRoot
        self.tokenOffset = tokenOffset
        self.checkpointType = checkpointType
        self.bytes = bytes
        self.segmentBaseOffset = segmentBaseOffset
        self.inheritedSegments = inheritedSegments
        self.createdAt = createdAt
        self.lastAccessAt = lastAccessAt
        self.fileRelativePath = fileRelativePath
        self.schemaVersion = schemaVersion
    }

    private enum CodingKeys: String, CodingKey {
        case snapshotID, partitionDigest, pathFromRoot, tokenOffset
        case checkpointType, bytes, segmentBaseOffset, inheritedSegments
        case createdAt, lastAccessAt, fileRelativePath, schemaVersion
    }

    /// Decode with chain-field defaults so a pre-v8 manifest still
    /// *decodes* cleanly and reaches the schema-version gate (backup +
    /// fresh start) instead of detouring through the corrupt-manifest
    /// rebuild.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.snapshotID = try container.decode(String.self, forKey: .snapshotID)
        self.partitionDigest = try container.decode(String.self, forKey: .partitionDigest)
        self.pathFromRoot = try container.decode([Int].self, forKey: .pathFromRoot)
        self.tokenOffset = try container.decode(Int.self, forKey: .tokenOffset)
        self.checkpointType = try container.decode(String.self, forKey: .checkpointType)
        self.bytes = try container.decode(Int.self, forKey: .bytes)
        self.segmentBaseOffset =
            try container.decodeIfPresent(Int.self, forKey: .segmentBaseOffset) ?? 0
        self.inheritedSegments =
            try container.decodeIfPresent(
                [SnapshotSegment].self, forKey: .inheritedSegments
            ) ?? []
        self.createdAt = try container.decode(Double.self, forKey: .createdAt)
        self.lastAccessAt = try container.decode(Double.self, forKey: .lastAccessAt)
        self.fileRelativePath = try container.decode(String.self, forKey: .fileRelativePath)
        self.schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
    }

    /// Chain-total byte count: the own file plus every inherited
    /// segment. The single byte-budget input — `currentSSDBytes`, the
    /// LRU cut, and `SnapshotRef.bytesOnDisk` all use this, never the
    /// bare `bytes`.
    var totalBytes: Int {
        inheritedSegments.reduce(bytes) { $0 + $1.bytes }
    }

    /// Every file in the chain, shallow→deep, own file last. The unit
    /// the store deletes when this descriptor leaves the manifest and
    /// the read order `loadSync` composes in.
    var chainFileRelativePaths: [String] {
        inheritedSegments.map(\.fileRelativePath) + [fileRelativePath]
    }

    /// The own file's segment record, as a later extension would
    /// inherit it.
    var ownSegment: SnapshotSegment {
        SnapshotSegment(
            baseOffset: segmentBaseOffset,
            tokenOffset: tokenOffset,
            fileRelativePath: fileRelativePath,
            bytes: bytes
        )
    }

    /// Single source of truth for the on-disk path layout
    /// (`partitions/{digest}/snapshots/{shardByte}/{id}.safetensors`,
    /// where `shardByte` is the first character of `snapshotID`).
    /// Called by the writer (`SSDSnapshotStore.fileURL`), the
    /// descriptor factory (`SnapshotLedger.makeDescriptor`),
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

    init(
        modelID: String,
        modelFingerprint: String,
        kvBits: Int?,
        kvGroupSize: Int,
        createdAt: Double,
        schemaVersion: Int
    ) {
        self.modelID = modelID
        self.modelFingerprint = modelFingerprint
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.createdAt = createdAt
        self.schemaVersion = schemaVersion
    }

    private enum CodingKeys: String, CodingKey {
        case modelID
        case modelFingerprint
        case kvBits
        case kvGroupSize
        case createdAt
        case schemaVersion
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelID = try container.decode(String.self, forKey: .modelID)
        self.modelFingerprint = try container.decode(String.self, forKey: .modelFingerprint)
        self.kvBits = try container.decodeIfPresent(Int.self, forKey: .kvBits)
        self.kvGroupSize = try container.decode(Int.self, forKey: .kvGroupSize)
        self.createdAt = try container.decode(Double.self, forKey: .createdAt)
        self.schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
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

// MARK: - In-memory snapshot ref (non-Codable identity value)

/// Immutable on-disk identity of an SSD-resident snapshot, carried as
/// the payload of the ref-bearing `SnapshotState` cases (`pendingWrite`,
/// `pendingDropped`, `committed`, `ssdOnly`). Models *what and where on
/// disk* only.
///
/// **Deliberately phase-free and recency-free.** The write phase
/// (pending vs committed) is encoded by the owning `SnapshotState`
/// case, not by a `committed` flag on the ref — that is exactly the
/// drift this consolidation removes (a committed-without-an-ID combo is
/// now unrepresentable). Recency is *not* carried here either: the
/// writer's LRU cut ranks by `PersistedSnapshotDescriptor.lastAccessAt`
/// and the in-RAM fallback ranks by `RadixTreeNode.lastAccessTime`, so
/// the old `lastAccessTime` field on the ref had zero readers and is
/// dropped.
///
/// Not `Codable`: the persisted analogue is
/// `PersistedSnapshotDescriptor`, which stores `lastAccessAt` as
/// seconds since the reference date.
nonisolated struct SnapshotRef: Sendable, Equatable {
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
    /// eviction path can reach the SSD byte count via the node's
    /// `SnapshotState` alone, without maintaining a separate
    /// `snapshotID → descriptor` map just to look up a size. The RAM
    /// lookup path does not touch this value.
    let bytesOnDisk: Int

    /// Copy with a refreshed on-disk byte count. The writer's commit
    /// is the byte authority: an extension fold makes the committed
    /// entry own its whole **Segment Chain**, so the live ref must
    /// report the chain total (`PersistedSnapshotDescriptor.totalBytes`)
    /// — the same value a warm start would restore.
    func settingBytesOnDisk(_ bytes: Int) -> SnapshotRef {
        SnapshotRef(
            snapshotID: snapshotID,
            partitionDigest: partitionDigest,
            tokenOffset: tokenOffset,
            checkpointType: checkpointType,
            bytesOnDisk: bytes
        )
    }
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

        /// Non-nil when this layer's arrays hold only the suffix token
        /// range `(suffixBaseOffset..offset]` along the token axis — a
        /// **Leaf Extension Admission** sliced a sliceable class
        /// (`KVCacheSimple` / `QuantizedKVCache`). `nil` means the
        /// arrays are the layer's whole state (always for recurrent /
        /// rotating / chunked classes, and for every non-extension
        /// payload).
        let suffixBaseOffset: Int?

        init(
            className: String,
            state: [ArrayPayload],
            metaState: [String],
            offset: Int,
            suffixBaseOffset: Int? = nil
        ) {
            self.className = className
            self.state = state
            self.metaState = metaState
            self.offset = offset
            self.suffixBaseOffset = suffixBaseOffset
        }
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

    /// Non-nil when this payload is a **Leaf Extension Admission**: the
    /// sliceable layers carry only the suffix past `extending.baseOffset`
    /// and the SSD front door must validate-and-transfer the base's
    /// **Segment Chain**. `nil` for every full payload.
    let extending: SnapshotExtension?

    init(
        tokenOffset: Int,
        checkpointType: HybridCacheSnapshot.CheckpointType,
        layers: [LayerPayload],
        extending: SnapshotExtension? = nil
    ) {
        self.tokenOffset = tokenOffset
        self.checkpointType = checkpointType
        self.layers = layers
        self.extending = extending
    }

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

// MARK: - Partition digest (wire contract for on-disk directory layout)

extension CachePartitionKey {
    /// 8-hex-character FNV-1a 32-bit digest over the canonical
    /// string form of the key fields. Used by the SSD prefix-cache
    /// tier as both the partition directory name
    /// (`partitions/{digest}/...`) and the join key between
    /// in-memory `SnapshotRef`s and their on-disk partitions.
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
    /// The null-byte separator cannot appear inside any field:
    /// `modelID` is a HuggingFace ID, `kvBits`/`kvGroupSize` are
    /// decimal integers, and `modelFingerprint` is hex SHA-256.
    nonisolated var partitionDigest: String {
        let kvBitsField = kvBits.map { "S\($0)" } ?? "N"
        let fingerprintField = modelFingerprint.map { "S" + $0 } ?? "N"
        let canonical =
            "\(modelID)\0\(kvBitsField)\0\(kvGroupSize)\0\(fingerprintField)"

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
