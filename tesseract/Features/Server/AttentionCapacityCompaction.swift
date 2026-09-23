import Foundation
import MLX
import MLXLMCommon

/// Compaction of retained attention capacity (#534). A cancelled long
/// generation leaves the rows it grew into behind after **Leaf Rewind**, and
/// an ordinary check-in keeps whatever the growth granule rounded up; the
/// leaf inherits both and the next check-out carries them on. When the
/// retained capacity exceeds the threshold, the full-attention layers are
/// rebuilt at the offset's rows plus one growth step — private arrays, the
/// logical rows copied in — so the next append does not immediately pay a
/// whole-body concatenation. Whole-state and quantized layers are untouched.
///
/// Threshold: the smaller of a quarter of the logical body and
/// `maximumThresholdBytes`, measured by the
/// [2026-09-21 profile](../../../benchmarks/allocation-profile/2026-09-21/README.md):
/// cancelled generations retained 50–117 MB, ordinary check-ins 5–17 MB.
nonisolated enum AttentionCapacityCompaction {
    /// 64 MB, from the #501 profile's decision rule (2026-09-21).
    static let maximumThresholdBytes = 64 * 1_048_576

    static func threshold(bodyBytes: Int) -> Int {
        min(bodyBytes / 4, maximumThresholdBytes)
    }

    struct Outcome: Equatable, Sendable {
        var retainedBytes = 0
        var bodyBytes = 0
        var freedBytes = 0
        var compactedLayers = 0
        var thresholdBytes = 0
    }

    /// The plain full-attention layers `RequestMemoryTelemetry.cacheFacts`
    /// measures, with their row extents.
    private static func measurable(_ cache: [any KVCache]) -> [(KVCacheSimple, Int, Int)] {
        cache.compactMap { entry in
            guard type(of: entry) == KVCacheSimple.self, let layer = entry as? KVCacheSimple else {
                return nil
            }
            let arrays = layer.innerState()
            guard arrays.count == 2,
                arrays.allSatisfy({ $0.ndim == 4 && $0.dim(2) > 0 && layer.offset <= $0.dim(2) })
            else { return nil }
            let rowBytes = arrays.reduce(0) { $0 + $1.nbytes / $1.dim(2) }
            return (layer, arrays[0].dim(2), rowBytes)
        }
    }

    static func measure(_ cache: [any KVCache]) -> Outcome {
        var outcome = Outcome()
        for (layer, capacity, rowBytes) in measurable(cache) {
            outcome.bodyBytes += rowBytes * layer.offset
            outcome.retainedBytes += rowBytes * (capacity - layer.offset)
        }
        outcome.thresholdBytes = threshold(bodyBytes: outcome.bodyBytes)
        return outcome
    }

    /// Compact in place when the retained capacity exceeds the threshold.
    /// Runs inside the Model Session on a request-private cache. One layer
    /// at a time (#554): each layer's replacement is evaluated, and its old
    /// arrays released, before the next layer allocates, so the transient is
    /// one layer's share of the body rather than the whole of it.
    @discardableResult
    static func compactIfNeeded(_ cache: [any KVCache]) -> Outcome {
        var outcome = measure(cache)
        guard outcome.retainedBytes > outcome.thresholdBytes else { return outcome }
        for (layer, capacity, rowBytes) in measurable(cache) {
            let step = layer.step
            let target = layer.offset + step
            guard capacity > target else { continue }
            let offset = layer.offset
            let fresh = layer.innerState().map { array -> MLXArray in
                var shape = array.shape
                shape[2] = target
                let replacement = MLXArray.zeros(shape, dtype: array.dtype)
                if offset > 0 {
                    replacement[.ellipsis, 0..<offset, 0...] = array[.ellipsis, 0..<offset, 0...]
                }
                return replacement
            }
            layer.state = fresh
            layer.trim(step)
            precondition(layer.offset == offset)
            eval(fresh)
            outcome.freedBytes += rowBytes * (capacity - target)
            outcome.compactedLayers += 1
        }
        return outcome
    }
}
