import CryptoKit
import Foundation

/// The **Image Digest** (CONTEXT.md → Image-aware prefix caching): the content
/// identity of one image for prefix-cache keying — SHA-256 over the raw
/// encoded bytes exactly as received, per image. Identity is exact-byte: a
/// re-encoded or resized variant of the same picture is a *different* image —
/// always a miss, never a wrong hit (ADR-0007).
nonisolated struct ImageDigest: Hashable, Sendable {
    /// The 32 SHA-256 bytes. `ImagePseudoToken` seeds its frozen expansion
    /// from the leading bytes, so the byte layout is part of the on-disk
    /// compatibility surface.
    let rawBytes: Data

    init(imageBytes: Data) {
        self.rawBytes = Data(SHA256.hash(data: imageBytes))
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
