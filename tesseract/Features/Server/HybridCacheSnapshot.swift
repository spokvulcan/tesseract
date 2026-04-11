import Foundation
import MLX
import MLXLMCommon

/// Full snapshot of all per-layer cache state at a specific token offset.
/// Mirrors the savePromptCache/loadPromptCache serialization contract.
/// Immutable after creation.
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
    }

    let layers: [LayerState]
    let checkpointType: CheckpointType
    /// Pre-computed sum of all state array nbytes, for eviction decisions.
    let memoryBytes: Int
    let createdAt: ContinuousClock.Instant

    enum CheckpointType: Comparable, Sendable {
        case system         // stable-prefix reuse; highest priority in Phase 1
        case leaf           // standard conversation-prefix reuse
        case branchPoint    // Phase 2: speculative Marconi checkpoint
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
    /// Mirrors loadPromptCache() reconstruction logic from KVCache.swift:1340-1378.
    func restore(kvBitsHint: Int? = nil, kvGroupSizeHint: Int? = nil) -> [any KVCache] {
        layers.map { layerState -> any KVCache in
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

            case "MambaCache":
                MambaCache()

            case "ArraysCache":
                ArraysCache(size: 0)

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

    // MARK: - Private

    /// Determine className via type check. Subclass before superclass order
    /// matching savePromptCache() at KVCache.swift:1252-1271.
    /// Returns nil for unsupported types (CacheList).
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
    /// Stricter than loadPromptCache() which ignores stored groupSize/bits.
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
}
