import Foundation
import MLX
import MLXLMCommon

/// Full snapshot of all per-layer cache state at a specific token offset.
/// Mirrors the savePromptCache/loadPromptCache serialization contract.
/// Immutable after creation.
///
/// App-owned since the migration off the mlx-swift-lm fork (ADR-0006): built
/// entirely on upstream's public `KVCache` surface — `state` / `metaState`
/// get+set, the `BaseKVCache.offset` setter, and `loadPromptCache` — so the
/// inference package needs no patches for the prefix cache to work.
nonisolated struct HybridCacheSnapshot: @unchecked Sendable {
    let tokenOffset: Int

    /// Per-layer cache state. Mirrors savePromptCache's serialization format.
    struct LayerState: @unchecked Sendable {
        /// Cache class name matching savePromptCache convention.
        /// "KVCache" (not "KVCacheSimple") for Python compat.
        let className: String
        /// Deep-copied cache.state arrays.
        let state: [MLXArray]
        /// cache.metaState strings.
        let metaState: [String]
        /// Absolute token offset. Stored explicitly because ChunkedKVCache's
        /// state setter (inherited from KVCacheSimple) only sets offset = keys.dim(2),
        /// and its metaState setter restores chunkSize/startPosition but not offset.
        let offset: Int

        init(
            className: String,
            state: [MLXArray],
            metaState: [String],
            offset: Int
        ) {
            self.className = className
            self.state = state
            self.metaState = metaState
            self.offset = offset
        }
    }

    let layers: [LayerState]
    let checkpointType: CheckpointType
    /// Pre-computed sum of all state array nbytes, for eviction decisions.
    let memoryBytes: Int
    let createdAt: ContinuousClock.Instant

    /// Memberwise initializer. Task 4.1.9 lazy hydration reads raw payload
    /// bytes and rebuilds a snapshot in `SSDSnapshotStore.loadSync`.
    init(
        tokenOffset: Int,
        layers: [LayerState],
        checkpointType: CheckpointType,
        memoryBytes: Int,
        createdAt: ContinuousClock.Instant
    ) {
        self.tokenOffset = tokenOffset
        self.layers = layers
        self.checkpointType = checkpointType
        self.memoryBytes = memoryBytes
        self.createdAt = createdAt
    }

    enum CheckpointType: Comparable, Sendable {
        case system               // stable-prefix reuse (system + tools)
        case leaf                 // standard conversation-prefix reuse
        case branchPoint          // Phase 2: speculative Marconi checkpoint
    }

    /// Capture from live cache during prefill. Deep-copies all state arrays.
    /// Returns nil if the cache contains unsupported layer types (e.g. CacheList
    /// from FalconH1/BaichuanM1) — callers should fall back to a no-cache path.
    static func capture(
        cache: [any KVCache],
        offset: Int,
        type: CheckpointType
    ) -> HybridCacheSnapshot? {
        var totalBytes = 0
        var layers: [LayerState] = []
        layers.reserveCapacity(cache.count)

        for layer in cache {
            guard let className = classNameForCache(layer) else {
                return nil
            }
            let state = layer.state.map { array -> MLXArray in
                let copy = array[.ellipsis]
                totalBytes += array.nbytes
                return copy
            }
            layers.append(LayerState(
                className: className,
                state: state,
                metaState: layer.metaState,
                offset: layer.offset
            ))
        }

        return HybridCacheSnapshot(
            tokenOffset: offset,
            layers: layers,
            checkpointType: type,
            memoryBytes: totalBytes,
            createdAt: .now
        )
    }

    /// Restore into a live cache array. Creates correct class per layer.
    /// Mirrors the upstream loadPromptCache() reconstruction logic.
    func restore(
        kvBitsHint: Int? = nil,
        kvGroupSizeHint: Int? = nil
    ) -> [any KVCache] {
        let classBreakdown = Dictionary(grouping: layers, by: { $0.className })
            .mapValues { $0.count }
            .map { "\($0.key):\($0.value)" }
            .sorted()
            .joined(separator: ",")
        let maxOffset = layers.map { $0.offset }.max() ?? 0
        Log.server.info(
            "snapshot restore-begin tokenOffset=\(tokenOffset) "
            + "layers=\(layers.count) maxLayerOffset=\(maxOffset) "
            + "classBreakdown=\(classBreakdown)"
        )
        return layers.map { layerState -> any KVCache in
            // ArraysCache (and its MambaCache subclass) reject direct metaState
            // assignment upstream — slot reconstruction goes through its own path.
            if layerState.className == "MambaCache" || layerState.className == "ArraysCache" {
                return Self.makeArraysCache(
                    mamba: layerState.className == "MambaCache",
                    state: layerState.state,
                    metaState: layerState.metaState,
                    offset: layerState.offset
                )
            }

            // `var` because `KVCache.state` has no class constraint upstream, so
            // assigning through the existential needs a mutable binding (every
            // conformer is in fact a class).
            var cache: any KVCache = switch layerState.className {
            case "KVCache", "KVCacheSimple":
                KVCacheSimple()

            case "QuantizedKVCache":
                Self.makeQuantizedCache(
                    metaState: layerState.metaState,
                    kvGroupSizeHint: kvGroupSizeHint,
                    kvBitsHint: kvBitsHint
                )

            case "RotatingKVCache":
                Self.makeRotatingCache(metaState: layerState.metaState)

            case "ChunkedKVCache":
                ChunkedKVCache()

            default:
                fatalError("HybridCacheSnapshot: unsupported cache class '\(layerState.className)'")
            }

            if !layerState.state.isEmpty {
                cache.state = layerState.state.map { $0[.ellipsis] }
            }
            cache.metaState = layerState.metaState

            // Explicitly restore offset. Required for ChunkedKVCache where the
            // state setter sets offset = keys.dim(2) but the correct absolute
            // offset is startPosition + used tokens. Safe for all types since
            // they all inherit from BaseKVCache.
            if let baseCache = cache as? BaseKVCache {
                baseCache.offset = layerState.offset
            }

            return cache
        }
    }

    // MARK: - Chunked Prefill

    /// Runs a checkpoint-aware prefill loop: main chunking + tail drain.
    /// The `processChunk` closure must run the forward pass for `chunkSize` tokens,
    /// call `eval(cache)`, and advance the caller's token cursor past the chunk.
    /// Returns (tokensConsumed, snapshots).
    static func chunkedPrefill(
        totalTokens: Int,
        prefillStepSize: Int,
        checkpoints: [Int: CheckpointType],
        checkpointBaseOffset: Int,
        initialOffset: Int = 0,
        cache: [KVCache],
        processChunk: (_ chunkSize: Int) throws -> Void
    ) rethrows -> (consumed: Int, snapshots: [HybridCacheSnapshot]) {
        let relativeCheckpoints = checkpoints.keys
            .map { $0 - checkpointBaseOffset }
            .filter { $0 > 0 }
            .sorted()

        var currentOffset = initialOffset
        var remaining = totalTokens
        var snapshots: [HybridCacheSnapshot] = []

        func captureAt(_ relativeOffset: Int) {
            let absoluteOffset = checkpointBaseOffset + relativeOffset
            // Invariant: relativeCheckpoints derives from checkpoints.keys, so the
            // map always has an entry for any offset we capture at.
            let type = checkpoints[absoluteOffset]!
            if let snap = capture(cache: cache, offset: absoluteOffset, type: type) {
                snapshots.append(snap)
            }
        }

        // Capture at initialOffset if it's a checkpoint (e.g. after vision prefix)
        if relativeCheckpoints.contains(currentOffset) {
            captureAt(currentOffset)
        }

        // Main loop — processes chunks up to prefillStepSize, adjusting to land on checkpoints
        while remaining > prefillStepSize {
            var chunkSize = prefillStepSize
            if let next = relativeCheckpoints.first(where: {
                $0 > currentOffset && $0 < currentOffset + chunkSize
            }) {
                chunkSize = next - currentOffset
            }

            try processChunk(chunkSize)
            currentOffset += chunkSize
            remaining -= chunkSize

            if relativeCheckpoints.contains(currentOffset) {
                captureAt(currentOffset)
            }
        }

        // Tail drain — captures checkpoints in the final remainder
        while let nextCP = relativeCheckpoints.first(where: {
            $0 > currentOffset && $0 < currentOffset + remaining
        }) {
            let chunkSize = nextCP - currentOffset
            guard chunkSize > 0 else { break }

            try processChunk(chunkSize)
            currentOffset += chunkSize
            remaining -= chunkSize

            captureAt(currentOffset)
        }

        return (totalTokens - remaining, snapshots)
    }

    // MARK: - Private

    /// Determine className via type check. Subclass before superclass order
    /// matching upstream savePromptCache(). Returns nil for unsupported
    /// types (CacheList).
    private static func classNameForCache(_ cache: any KVCache) -> String? {
        switch cache {
        case is ChunkedKVCache:
            return "ChunkedKVCache"
        case is KVCacheSimple:
            return "KVCache"
        case is RotatingKVCache:
            return "RotatingKVCache"
        case is QuantizedKVCache:
            return "QuantizedKVCache"
        case is MambaCache:
            return "MambaCache"
        case is ArraysCache:
            return "ArraysCache"
        case is CacheList:
            return nil
        default:
            return nil
        }
    }

    /// Parse groupSize/bits from metaState, falling back to hints then defaults.
    /// Stricter than loadPromptCache() which routes through the type-specific
    /// metaState setter.
    private static func makeQuantizedCache(
        metaState: [String],
        kvGroupSizeHint: Int?,
        kvBitsHint: Int?
    ) -> QuantizedKVCache {
        let groupSize: Int
        let bits: Int
        if metaState.count >= 4,
           let parsedGroupSize = Int(metaState[2]),
           let parsedBits = Int(metaState[3])
        {
            groupSize = parsedGroupSize
            bits = parsedBits
        } else {
            groupSize = kvGroupSizeHint ?? 64
            bits = kvBitsHint ?? 8
        }
        return QuantizedKVCache(groupSize: groupSize, bits: bits)
    }

    /// Parse maxSize from metaState for RotatingKVCache constructor.
    private static func makeRotatingCache(metaState: [String]) -> RotatingKVCache {
        guard metaState.count >= 5 else {
            fatalError("Invalid RotatingKVCache metaState — expected 5 values, got \(metaState.count)")
        }
        guard metaState[1] != "None", let maxSize = Int(metaState[1]) else {
            fatalError("Failed to parse RotatingKVCache maxSize from: \(metaState[1])")
        }
        return RotatingKVCache(maxSize: maxSize)
    }

    /// Rebuild an `ArraysCache`/`MambaCache` from its slot-aware metaState.
    ///
    /// Upstream's `ArraysCache.metaState` setter intentionally asserts —
    /// in-package restoration goes through the internal
    /// `restoreFromMetaState(state:savedMetaState:)`. This mirrors that method
    /// on the public surface: `init(size:leftPadding:)` plus the public slot
    /// subscript. metaState format:
    /// `[slotCount, presentSlots (comma-separated), leftPadding (comma-separated, optional)]`;
    /// the legacy format (`[""]`) restores compacted state via the `state` setter.
    private static func makeArraysCache(
        mamba: Bool, state: [MLXArray], metaState: [String], offset: Int
    ) -> ArraysCache {
        let copied = state.map { $0[.ellipsis] }
        let cache: ArraysCache
        if metaState.count >= 2, let slotCount = Int(metaState[0]) {
            let presentSlots = metaState[1].isEmpty
                ? [] : metaState[1].split(separator: ",").compactMap { Int($0) }
            let leftPadding: [Int]? = metaState.count >= 3
                ? metaState[2].split(separator: ",").compactMap { Int($0) } : nil
            cache = mamba
                ? MambaCache(leftPadding: leftPadding)
                : ArraysCache(size: slotCount, leftPadding: leftPadding)
            for (arrayIdx, slotIdx) in presentSlots.enumerated()
            where slotIdx < slotCount && arrayIdx < copied.count {
                cache[slotIdx] = copied[arrayIdx]
            }
        } else {
            // Legacy [""]-style metaState: state arrays are compacted.
            cache = mamba ? MambaCache() : ArraysCache(size: copied.count)
            if !copied.isEmpty {
                cache.state = copied
            }
        }
        cache.offset = offset
        return cache
    }
}

// MARK: - Checkpoint-type wire format

nonisolated extension HybridCacheSnapshot.CheckpointType {
    /// Stable wire-format string used in both the safetensors-header
    /// `tesse.hybrid_cache.checkpoint_type` field written by
    /// ``HybridCacheSnapshot/serialize(to:metadata:)`` and in
    /// `PersistedSnapshotDescriptor.checkpointType` in the downstream
    /// SSD persistence tier. Case names match the enum verbatim so
    /// `.wireString` round-trips through ``init(wireString:)`` without
    /// a lookup table.
    var wireString: String {
        switch self {
        case .system: return "system"
        case .leaf: return "leaf"
        case .branchPoint: return "branchPoint"
        }
    }

    /// Inverse of ``wireString``. Returns `nil` for any unrecognized
    /// input so warm-start paths can treat it as "drop this descriptor"
    /// without crashing on unknown data (e.g. a file written by a
    /// future build with a new case).
    init?(wireString: String) {
        switch wireString {
        case "system": self = .system
        case "leaf": self = .leaf
        case "branchPoint": self = .branchPoint
        default: return nil
        }
    }
}

// MARK: - Safetensors persistence

nonisolated extension HybridCacheSnapshot {

    /// Reserved metadata keys written by ``serialize(to:metadata:)`` and
    /// consumed by ``deserialize(from:expectedFingerprint:)``. Caller-supplied
    /// metadata values under the `tesse.hybrid_cache.` namespace are
    /// overwritten by the serializer; everything else is persisted verbatim.
    enum MetadataKey {
        /// Model fingerprint (see `ModelFingerprint.computeFingerprint(modelDir:)`).
        /// The caller must include this key in the metadata passed to
        /// `serialize`, so that `deserialize` can reject stale files after a
        /// weight swap under a stable `modelID`.
        static let fingerprint = "tesse.hybrid_cache.fingerprint"

        /// Wire-format `HybridCacheSnapshot.CheckpointType`. Overwritten by
        /// the serializer.
        static let checkpointType = "tesse.hybrid_cache.checkpoint_type"

        /// Absolute `HybridCacheSnapshot.tokenOffset`. Overwritten by the
        /// serializer.
        static let tokenOffset = "tesse.hybrid_cache.token_offset"

        /// Prefix for per-layer absolute offsets (`{prefix}{layerIndex}`).
        /// Required to survive round-trips for `ChunkedKVCache`, whose
        /// `state` setter derives `offset = keys.dim(2)` — which is the
        /// truncated chunk count, not the caller's absolute prompt
        /// position. Overwritten by the serializer.
        static let layerOffsetPrefix = "tesse.hybrid_cache.layer_offset."

        /// Prefix for per-layer `metaState` entries
        /// (`{prefix}{layerIndex}.count` plus `{prefix}{layerIndex}.{j}`
        /// for each string). Keeps the snapshot's metaState authoritative
        /// across serialize/deserialize even where a cache type's
        /// `metaState` setter is lossy. Overwritten by the serializer.
        static let layerMetaPrefix = "tesse.hybrid_cache.layer_meta."

        /// Caller-supplied integer schema version for the wire format.
        /// Optional on write; when present, downstream
        /// ``deserialize(from:expectedFingerprint:expectedSchemaVersion:)``
        /// rejects files whose stored value disagrees with the caller's
        /// `expectedSchemaVersion` before reconstructing the snapshot.
        static let schemaVersion = "tesse.hybrid_cache.schema_version"
    }

    /// Errors thrown by ``serialize(to:metadata:)`` /
    /// ``deserialize(from:expectedFingerprint:expectedSchemaVersion:)``.
    enum SerializationError: LocalizedError {
        case missingFingerprint
        case fingerprintMismatch(expected: String, actual: String)
        case missingCheckpointType
        case unknownCheckpointType(String)
        case missingTokenOffset
        case invalidTokenOffset(String)
        case unsupportedCacheClass(String)
        case missingSchemaVersion
        case invalidSchemaVersion(String)
        case schemaVersionMismatch(expected: Int, actual: Int)

        var errorDescription: String? {
            switch self {
            case .missingFingerprint:
                return "Prompt cache file has no '\(MetadataKey.fingerprint)' metadata."
            case .fingerprintMismatch(let expected, let actual):
                return "Model fingerprint mismatch: expected \(expected), got \(actual)."
            case .missingCheckpointType:
                return "Prompt cache file has no '\(MetadataKey.checkpointType)' metadata."
            case .unknownCheckpointType(let value):
                return "Unknown HybridCacheSnapshot.CheckpointType wire value: '\(value)'."
            case .missingTokenOffset:
                return "Prompt cache file has no '\(MetadataKey.tokenOffset)' metadata."
            case .invalidTokenOffset(let value):
                return "Invalid HybridCacheSnapshot.tokenOffset wire value: '\(value)'."
            case .unsupportedCacheClass(let name):
                return "HybridCacheSnapshot cannot represent cache class '\(name)'."
            case .missingSchemaVersion:
                return "Prompt cache file has no '\(MetadataKey.schemaVersion)' metadata, but the caller required one."
            case .invalidSchemaVersion(let value):
                return "Invalid HybridCacheSnapshot schema-version wire value: '\(value)'."
            case .schemaVersionMismatch(let expected, let actual):
                return "HybridCacheSnapshot schema-version mismatch: expected \(expected), got \(actual)."
            }
        }
    }

    /// Serialize this snapshot to a safetensors file at `url`.
    ///
    /// **Thread-affinity contract: must be called from inside `container.perform`
    /// on `LLMActor` (or another Metal-affine context).** The safetensors writer
    /// reads MLX-array bytes, which forces evaluation of any pending lazy
    /// Metal command-queue work. Calling this outside a Metal-affine context
    /// risks silent state corruption or a crash. Swift has no runtime
    /// Metal-context detection, so the contract is enforced by convention
    /// plus a debug-build smoke check that evaluates the first layer's first
    /// state array at function entry.
    ///
    /// Writes the snapshot's tensors directly from `self.layers` using
    /// `save(arrays:metadata:url:)` and the same flattened-safetensors wire
    /// format as `savePromptCache` (`"i.j"` tensors, `"0.i.j"` metaState,
    /// `"1.{key}"` user metadata, `"2.i"` class names). This avoids both
    /// constructing throwaway `[KVCache]` instances via `restore()` and
    /// paying a second copy of the state arrays on the hot path. Output
    /// files remain fully compatible with `loadPromptCache` as well as
    /// ``deserialize(from:expectedFingerprint:)``.
    ///
    /// The snapshot's `tokenOffset`, `checkpointType`, per-layer absolute
    /// offsets, and (authoritatively) per-layer `metaState` are written
    /// into the safetensors metadata under the reserved keys defined by
    /// ``MetadataKey``. The caller must also place the model fingerprint
    /// under `MetadataKey.fingerprint` so that
    /// ``deserialize(from:expectedFingerprint:)`` can reject stale files
    /// after a weight swap under a stable `modelID`.
    ///
    /// - Parameters:
    ///   - url: destination `.safetensors` file. Atomic rename, parent
    ///     directory creation, and replacement of existing files are the
    ///     caller's responsibility (the SSD writer handles these).
    ///   - metadata: caller-supplied string metadata. Keys under the
    ///     `tesse.hybrid_cache.` namespace are reserved and overwritten by
    ///     this function; all other keys are persisted verbatim.
    func serialize(to url: URL, metadata: [String: String] = [:]) throws {
        #if DEBUG
        Self.debugSmokeCheckMetalAffine(firstArray: layers.first?.state.first)
        #endif

        // Build the "user metadata" dictionary — the caller's keys plus
        // our reserved `tesse.hybrid_cache.*` keys. These become
        // `1.{key}` in the flattened safetensors metadata below.
        var userMetadata = metadata
        userMetadata[MetadataKey.checkpointType] = checkpointType.wireString
        userMetadata[MetadataKey.tokenOffset] = String(tokenOffset)
        for (layerIndex, layer) in layers.enumerated() {
            userMetadata[Self.layerOffsetKey(layerIndex)] = String(layer.offset)
            userMetadata[Self.layerMetaCountKey(layerIndex)] = String(layer.metaState.count)
            for (j, value) in layer.metaState.enumerated() {
                userMetadata[Self.layerMetaKey(layerIndex, j)] = value
            }
        }

        // Flatten directly from `self.layers`, mirroring the wire format
        // produced by `savePromptCache` (upstream KVCache.swift). Produces
        // byte-for-byte equivalent files so any existing reader — our
        // `deserialize`, the upstream `loadPromptCache`, etc. — can ingest
        // them without discrimination.
        var flattenedArrays: [String: MLXArray] = [:]
        var flattenedMetadata: [String: String] = [:]
        for (i, layer) in layers.enumerated() {
            for (j, array) in layer.state.enumerated() {
                flattenedArrays["\(i).\(j)"] = array
            }
            for (j, info) in layer.metaState.enumerated() {
                flattenedMetadata["0.\(i).\(j)"] = info
            }
            flattenedMetadata["2.\(i)"] = layer.className
        }
        for (key, value) in userMetadata {
            flattenedMetadata["1.\(key)"] = value
        }

        try save(arrays: flattenedArrays, metadata: flattenedMetadata, url: url)
    }

    /// Deserialize a snapshot previously written by
    /// ``serialize(to:metadata:)``.
    ///
    /// **Thread-affinity contract: must be called from inside `container.perform`
    /// on `LLMActor` (or another Metal-affine context).** `loadPromptCache`
    /// creates MLX arrays backed by the safetensors payload; touching those
    /// arrays outside a Metal-affine context is undefined. The contract is
    /// enforced by convention plus a debug-build smoke check that evaluates
    /// the first loaded layer's first state array immediately after load.
    ///
    /// - Parameters:
    ///   - url: source `.safetensors` file previously produced by
    ///     ``serialize(to:metadata:)``.
    ///   - expectedFingerprint: fingerprint of the loading model (see
    ///     `ModelFingerprint.computeFingerprint(modelDir:)`). The file's
    ///     persisted fingerprint under `MetadataKey.fingerprint` must match;
    ///     on mismatch this function throws
    ///     ``SerializationError/fingerprintMismatch(expected:actual:)``
    ///     without returning a snapshot.
    /// - Returns: a fully reconstructed `HybridCacheSnapshot` with a fresh
    ///   `createdAt` wall-clock timestamp. The captured moment is not
    ///   persisted because `ContinuousClock.Instant` has no stable wire
    ///   format across process restarts.
    static func deserialize(
        from url: URL,
        expectedFingerprint: String,
        expectedSchemaVersion: Int? = nil
    ) throws -> HybridCacheSnapshot {
        let (caches, metadata) = try loadPromptCache(url: url)

        #if DEBUG
        debugSmokeCheckMetalAffine(firstArray: caches.first?.state.first)
        #endif

        guard let storedFingerprint = metadata[MetadataKey.fingerprint] else {
            throw SerializationError.missingFingerprint
        }
        guard storedFingerprint == expectedFingerprint else {
            throw SerializationError.fingerprintMismatch(
                expected: expectedFingerprint,
                actual: storedFingerprint
            )
        }

        // Optional schema-version gate: a v(N) file cannot be safely
        // reattached after the persistence schema bumps to v(N+1)
        // because per-layer metaState may mean something different.
        // Skip the metadata lookup entirely when the caller didn't pin
        // a version — keeps the dictionary access off the legacy path.
        if let expectedSchemaVersion {
            guard let storedRaw = metadata[MetadataKey.schemaVersion] else {
                throw SerializationError.missingSchemaVersion
            }
            guard let storedVersion = Int(storedRaw) else {
                throw SerializationError.invalidSchemaVersion(storedRaw)
            }
            guard storedVersion == expectedSchemaVersion else {
                throw SerializationError.schemaVersionMismatch(
                    expected: expectedSchemaVersion,
                    actual: storedVersion
                )
            }
        }

        guard let checkpointWire = metadata[MetadataKey.checkpointType] else {
            throw SerializationError.missingCheckpointType
        }
        guard let checkpointType = CheckpointType(wireString: checkpointWire) else {
            throw SerializationError.unknownCheckpointType(checkpointWire)
        }

        guard let tokenOffsetRaw = metadata[MetadataKey.tokenOffset] else {
            throw SerializationError.missingTokenOffset
        }
        guard let tokenOffset = Int(tokenOffsetRaw) else {
            throw SerializationError.invalidTokenOffset(tokenOffsetRaw)
        }

        var totalBytes = 0
        var layers: [LayerState] = []
        layers.reserveCapacity(caches.count)
        for (layerIndex, cache) in caches.enumerated() {
            guard let className = classNameForCache(cache) else {
                throw SerializationError.unsupportedCacheClass(
                    String(describing: type(of: cache))
                )
            }
            let state = cache.state
            for array in state {
                totalBytes += array.nbytes
            }

            // Restore absolute per-layer offset. Required for ChunkedKVCache,
            // whose state setter in loadPromptCache resets offset to
            // `keys.dim(2)` (the truncated chunk count) rather than the
            // caller's absolute prompt position. For other cache types the
            // stored value equals `keys.dim(2)` anyway, so the override is
            // a no-op.
            let layerOffset: Int
            if let raw = metadata[Self.layerOffsetKey(layerIndex)],
               let parsed = Int(raw)
            {
                layerOffset = parsed
            } else {
                layerOffset = cache.offset
            }

            // Restore metaState from the reserved-key mirror instead of
            // the live cache's `metaState` getter, so the snapshot's
            // metaState stays authoritative across serialize/deserialize
            // even where a type-specific `metaState =` setter is lossy.
            // Legacy files (absent mirror) fall through to the live
            // cache — the behavior the previous version shipped.
            let metaState = Self.layerMetaState(from: metadata, layerIndex: layerIndex)
                ?? cache.metaState

            layers.append(LayerState(
                className: className,
                state: state,
                metaState: metaState,
                offset: layerOffset
            ))
        }

        return HybridCacheSnapshot(
            tokenOffset: tokenOffset,
            layers: layers,
            checkpointType: checkpointType,
            memoryBytes: totalBytes,
            createdAt: .now
        )
    }

    // MARK: - Private helpers

    /// Build the reserved metadata key for the per-layer absolute
    /// offset at `layerIndex`. Single construction point so serialize
    /// and deserialize cannot drift.
    private static func layerOffsetKey(_ layerIndex: Int) -> String {
        "\(MetadataKey.layerOffsetPrefix)\(layerIndex)"
    }

    /// Build the reserved metadata key for the metaState element at
    /// `(layerIndex, metaIndex)`. Paired with ``layerMetaCountKey(_:)``.
    private static func layerMetaKey(_ layerIndex: Int, _ metaIndex: Int) -> String {
        "\(MetadataKey.layerMetaPrefix)\(layerIndex).\(metaIndex)"
    }

    /// Build the reserved metadata key that holds the metaState element
    /// count for `layerIndex`. Needed because each metaState element is
    /// stored under its own key (see ``layerMetaKey(_:_:)``), so the
    /// reader needs an authoritative count to stop at.
    private static func layerMetaCountKey(_ layerIndex: Int) -> String {
        "\(MetadataKey.layerMetaPrefix)\(layerIndex).count"
    }

    /// Recover the persisted metaState for `layerIndex` from the
    /// reserved-key mirror written by ``serialize(to:metadata:)``.
    /// Returns `nil` if the mirror is absent (legacy file) or
    /// structurally invalid — callers fall through to the live cache's
    /// `metaState` in both cases.
    private static func layerMetaState(
        from metadata: [String: String],
        layerIndex: Int
    ) -> [String]? {
        guard let countRaw = metadata[layerMetaCountKey(layerIndex)],
              let count = Int(countRaw),
              count >= 0
        else {
            return nil
        }
        var result: [String] = []
        result.reserveCapacity(count)
        for metaIndex in 0 ..< count {
            guard let value = metadata[layerMetaKey(layerIndex, metaIndex)] else {
                return nil
            }
            result.append(value)
        }
        return result
    }

    #if DEBUG
    /// Smoke check for the thread-affinity contract. Evaluates a single
    /// MLX array to force any pending lazy Metal work to drain; if the
    /// caller is outside a Metal-affine context, this will trap or crash
    /// loudly, which is the desired failure mode per the contract in the
    /// doc comments on ``serialize(to:metadata:)`` and
    /// ``deserialize(from:expectedFingerprint:)``.
    private static func debugSmokeCheckMetalAffine(firstArray: MLXArray?) {
        guard let firstArray else { return }
        eval(firstArray)
    }
    #endif
}
