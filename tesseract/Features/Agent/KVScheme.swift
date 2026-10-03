import Foundation
import MLXLMCommon

/// **KV Scheme** (CONTEXT.md): how a request holds its full-attention KV
/// once the prompt is prefilled. Both schemes are TurboQuant (#603): 4-bit
/// values in a rotated codebook, with 8-bit affine or full-precision keys.
/// On Qwen3.8-27B they shrink the cache 2.53x or 1.59x against bf16 and
/// decode at bf16's speed, and DFlash2 verifies over them. The scheme is
/// part of the request's cache partition identity, so a snapshot never
/// restores into a request of another scheme.
nonisolated enum KVScheme: String, Sendable, Equatable, CaseIterable, Codable {
    /// 8-bit affine keys, 4-bit values: 25,856 bytes per token on
    /// Qwen3.8-27B (bf16: 65,536).
    case turbo8v4
    /// bf16 keys, 4-bit values: 41,216 bytes per token on Qwen3.8-27B.
    case turbo0v4

    /// Key bits as the vendor counts them: 8 is affine, 0 is unquantized.
    var keyBits: Int {
        switch self {
        case .turbo8v4: 8
        case .turbo0v4: 0
        }
    }

    var valueBits: Int { 4 }

    /// Full-precision attention bytes over this scheme's, at Qwen3.8-27B's
    /// shape: what a turn's live cache weighs while its prompt still
    /// prefills at full precision, relative to the leaf it stores.
    var fullPrecisionRatio: Double {
        switch self {
        case .turbo8v4: 65_536.0 / 25_856.0
        case .turbo0v4: 65_536.0 / 41_216.0
        }
    }

    /// An empty cache of this scheme, for a layer `capture` stores in it.
    func makeCache() -> TurboQuantKVCache {
        TurboQuantKVCache(
            bits: max(keyBits, valueBits), keyBits: keyBits, valueBits: valueBits)
    }

    /// The models the schemes were measured on (#603), Qwen3.8-27B and its
    /// PARO pack. Elsewhere a request keeps full precision.
    static func supports(modelID: String) -> Bool {
        modelID.hasPrefix("qwen3.8-27b") && !modelID.hasSuffix("-draft")
    }
}

/// The **KV Cache Compression** setting: the scheme requests use when the
/// loaded model supports one.
nonisolated enum KVCacheCompression: String, Sendable, CaseIterable, Identifiable {
    case off
    case turbo8v4
    case turbo0v4

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .off: "Off"
        case .turbo8v4: "Smallest (8-bit keys)"
        case .turbo0v4: "Balanced (full-precision keys)"
        }
    }

    /// The scheme a request for `modelID` uses.
    func scheme(forModelID modelID: String) -> KVScheme? {
        guard let scheme = KVScheme(rawValue: rawValue), KVScheme.supports(modelID: modelID)
        else { return nil }
        return scheme
    }
}
