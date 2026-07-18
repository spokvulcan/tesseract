import CryptoKit
import Foundation

/// The **Audio Digest**: the content identity of one audio clip for
/// prefix-cache keying — SHA-256 over a domain-separation prefix plus the raw
/// encoded bytes exactly as received. Identity is exact-byte, mirroring
/// `ImageDigest` (a re-encoded variant of the same recording is a different
/// clip — always a miss, never a wrong hit).
///
/// The `"audio\0"` prefix keeps the audio digest space disjoint from the
/// image digest space: the same byte payload attached as an image and as an
/// audio clip must never collide into one pseudo-token expansion.
nonisolated struct AudioDigest: Hashable, Sendable {
    /// The 32 SHA-256 bytes. `AudioPseudoToken` seeds its frozen expansion
    /// from the leading bytes, so the byte layout is part of the on-disk
    /// compatibility surface.
    let rawBytes: Data

    init(audioBytes: Data) {
        var hasher = SHA256()
        hasher.update(data: Data("audio\0".utf8))
        hasher.update(data: audioBytes)
        self.rawBytes = Data(hasher.finalize())
    }

    /// Test seam: adopt an already-computed 32-byte digest.
    init?(rawDigest: Data) {
        guard rawDigest.count == SHA256Digest.byteCount else { return nil }
        self.rawBytes = rawDigest
    }

    var hexString: String {
        rawBytes.map { String(format: "%02x", $0) }.joined()
    }
}
