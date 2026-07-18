import Foundation

/// The frozen (digest, index) → pseudo-token expansion behind the **Cache Key
/// Path** (ADR-0007).
///
/// **FROZEN — do not change.** Expansions persist inside SSD admission paths
/// across restarts; changing any constant here invalidates every image-bearing
/// snapshot on disk. The golden-value tests pin this function forever.
///
/// Values are always negative, so they can never collide with vocabulary
/// token ids (non-negative) and key-path divergence between different images
/// of the same pixel size is guaranteed at the first expanded position with
/// overwhelming probability (63-bit values).
nonisolated enum ImagePseudoToken {

    /// Pseudo-token for position `index` of the placeholder run keyed by
    /// `digest`. Seed = first 8 digest bytes (little-endian), mixed with the
    /// index by one splitmix64 round, folded into [-2^63, -1].
    static func value(digest: ImageDigest, index: Int) -> Int {
        MediaPseudoTokenCore.value(
            seed: MediaPseudoTokenCore.seed(fromRawDigestBytes: digest.rawBytes), index: index
        )
    }

    /// The full length-`runLength` expansion for one image's placeholder run.
    static func expansion(digest: ImageDigest, runLength: Int) -> [Int] {
        MediaPseudoTokenCore.expansion(rawDigestBytes: digest.rawBytes, runLength: runLength)
    }
}

/// The audio face of the same frozen expansion. Domain separation from images
/// lives in `AudioDigest` (its hash is prefixed), so the shared core stays
/// byte-identical to the pinned image scheme.
///
/// **FROZEN — do not change** (same contract as `ImagePseudoToken`).
nonisolated enum AudioPseudoToken {

    /// The full length-`runLength` expansion for one clip's placeholder run.
    static func expansion(digest: AudioDigest, runLength: Int) -> [Int] {
        MediaPseudoTokenCore.expansion(rawDigestBytes: digest.rawBytes, runLength: runLength)
    }
}

/// The shared splitmix64 core both pseudo-token faces call. Extracted, not
/// changed: `ImagePseudoToken`'s golden-value tests pin these exact numbers.
private nonisolated enum MediaPseudoTokenCore {

    static func expansion(rawDigestBytes: Data, runLength: Int) -> [Int] {
        let seed = seed(fromRawDigestBytes: rawDigestBytes)
        return (0..<runLength).map { value(seed: seed, index: $0) }
    }

    static func seed(fromRawDigestBytes rawBytes: Data) -> UInt64 {
        var seed: UInt64 = 0
        for (offset, byte) in rawBytes.prefix(8).enumerated() {
            seed |= UInt64(byte) << (8 * offset)
        }
        return seed
    }

    static func value(seed: UInt64, index: Int) -> Int {
        var z = seed &+ (UInt64(bitPattern: Int64(index)) &* 0x9E37_79B9_7F4A_7C15)
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return -1 - Int(bitPattern: UInt(z >> 1))
    }
}
