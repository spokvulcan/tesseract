//
//  RenderTokenCache+Keys.swift
//  tesseract
//
//  The **Render+Token Cache**'s keying and byte-comparison layer, split out of
//  `RenderTokenCache.swift` so the resolve/verify logic and the pure
//  value-identity machinery read separately. Everything here is `static` and
//  side-effect free.
//
//  Two groups:
//  - **Digests** — the cumulative per-message SHA-256 chain and the
//    deterministic canonical serialization it hashes. A cheap pre-filter, NOT
//    an exactness arbiter: the render-level byte/trim/junction checks in
//    `RenderTokenCache.swift` are what make a hit exact.
//  - **Byte helpers** — the prefix/suffix primitives every resolve uses.
//    These exist because Swift `String`'s `==`/`hasPrefix`/`hasSuffix` compare
//    under Unicode canonical equivalence, which would let an NFC render match
//    an NFD one and hand back tokens for the wrong bytes. See the header of
//    `RenderTokenCache.swift`.
//

import CryptoKit
import Foundation
import MLXLMCommon

// `nonisolated` explicitly: the build sets `SWIFT_DEFAULT_ACTOR_ISOLATION`
// to `MainActor`, and an extension does NOT inherit the `nonisolated` on the
// type it extends — without this every helper here would be main-actor-isolated
// and unreachable from the cache's synchronous, off-MainActor resolves.
nonisolated extension RenderTokenCache {

    // MARK: - Digest chain

    /// Internal (not private) so `--agent-cpu-bench` can time the digest chain
    /// directly (its per-turn cost inside the request-keying resolve).
    static func digestChain(_ messages: [[String: any Sendable]]) -> [String] {
        var chain: [String] = []
        chain.reserveCapacity(messages.count)
        var previous = "rtc1"
        for message in messages {
            previous = sha256Hex(previous + "|" + canonicalForm(message))
            chain.append(previous)
        }
        return chain
    }

    /// C29: the digest chain for `messages`, reusing `entry`'s stored head
    /// when it is short enough — the chain is cumulative, so a conversation
    /// extending the stored one has the same head values and only the tail
    /// messages need hashing. A head that does NOT match (edited history)
    /// vacuously passes the caller's head-match guard; the render arbiters
    /// downstream (byte-prefix + trim + junction/cut verification) reject
    /// the candidate instead, so exactness is unchanged and the miss simply
    /// lands on a different reason (`.renderNotExtended`, not
    /// `.digestMismatch`).
    static func digestChain(
        _ messages: [[String: any Sendable]], reusingHeadOf entry: Entry?
    ) -> [String] {
        guard let entry, entry.messageDigests.count <= messages.count else {
            return digestChain(messages)
        }
        var chain = entry.messageDigests
        var previous = chain.last ?? "rtc1"
        for message in messages[chain.count...] {
            previous = sha256Hex(previous + "|" + canonicalForm(message))
            chain.append(previous)
        }
        return chain
    }

    static func sha256Hex(_ string: String) -> String {
        SHA256.hash(data: Data(string.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Canonical serialization

    /// Optional-aware entry point: `nil` serializes distinctly from an empty
    /// container (`nil` tools and `[]` tools can render differently).
    static func canonicalForm(optional value: Any?) -> String {
        guard let value else { return "null" }
        return canonicalForm(value)
    }

    /// Deterministic canonical serialization for the JSON-shaped values that
    /// appear in message dicts, tool specs, and additional context. Type-tagged
    /// and length-prefixed so distinct values never serialize alike; unknown
    /// leaf types fall back to `String(describing:)` rather than failing.
    ///
    /// The `NSNumber` case comes FIRST and dispatches on the CoreFoundation
    /// type id, because JSON-decoded numbers arrive as `NSNumber` and
    /// `NSNumber(value: 1) as? Bool` SUCCEEDS — a plain `case let bool as Bool`
    /// first would serialize the integer `1` and the boolean `true`
    /// identically. A native Swift `Bool` bridges to `__NSCFBoolean` and is
    /// caught by the same case, so both spellings land on `b:`.
    static func canonicalForm(_ value: Any) -> String {
        switch value {
        case is NSNull:
            return "null"
        case let number as NSNumber where CFGetTypeID(number) == CFBooleanGetTypeID():
            return "b:\(number.boolValue)"
        case let bool as Bool where !(value is NSNumber):
            return "b:\(bool)"
        case let string as String:
            return "s:\(string.utf8.count):\(string)"
        case let int as Int:
            return "i:\(int)"
        case let int as Int8:
            return "i:\(int)"
        case let int as Int16:
            return "i:\(int)"
        case let int as Int32:
            return "i:\(int)"
        case let int as Int64:
            return "i:\(int)"
        case let uint as UInt:
            return "u:\(uint)"
        case let double as Double:
            return "d:\(double.bitPattern)"
        case let float as Float:
            return "f:\(float.bitPattern)"
        case let array as [Any]:
            return "[\(array.count)]" + array.map { canonicalForm($0) }.joined(separator: ",")
        case let dictionary as [String: Any]:
            return "{\(dictionary.count)}"
                + dictionary.keys.sorted().map { key in
                    "s:\(key.utf8.count):\(key)=\(canonicalForm(dictionary[key] ?? NSNull()))"
                }.joined(separator: ",")
        default:
            return "?:\(String(describing: value))"
        }
    }

    // MARK: - Byte helpers

    /// One token's decoded UTF-8 bytes. `skipSpecialTokens: false` so template
    /// scaffolding decodes to its literal text, exactly as the render emitted
    /// it.
    static func utf8(ofToken token: Int, tokenizer: any Tokenizer) -> [UInt8] {
        Array(tokenizer.decode(tokenIds: [token], skipSpecialTokens: false).utf8)
    }

    /// Whether `bytes[..<end]` ends with `needle`. An empty `needle` is NOT a
    /// match: a token that decodes to nothing cannot be trimmed off the text
    /// side, and treating it as a no-op match would advance the token cursor
    /// without advancing the byte cursor.
    static func bytes(_ bytes: [UInt8], endingAt end: Int, equal needle: [UInt8]) -> Bool {
        guard !needle.isEmpty, end >= needle.count, end <= bytes.count else { return false }
        return bytes[(end - needle.count)..<end].elementsEqual(needle)
    }

    /// Whether `bytes` begins with the first `count` bytes of `other`.
    static func bytes(_ bytes: [UInt8], startsWith other: [UInt8], count: Int) -> Bool {
        guard count >= 0, count <= bytes.count, count <= other.count else { return false }
        return bytes[0..<count].elementsEqual(other[0..<count])
    }

    /// `bytes[start...]` as a `String`, or `nil` when that slice is empty or
    /// is not exactly the UTF-8 of the returned text.
    ///
    /// Both guards exist to catch a cut landing mid-scalar. The failable
    /// initializer is deliberate over `String(decoding:as:)`, which would
    /// silently substitute U+FFFD and hand back text that encodes to tokens the
    /// render never contained; the round-trip check then holds independently of
    /// how strict Foundation's decoder is. A cut produced by the trim walk
    /// always lands on a scalar boundary — the trimmed bytes are whole decoded
    /// tokens — so this only fires on a tokenizer whose decode is not
    /// byte-faithful, where falling back to the full encode is exactly right.
    static func suffixString(of bytes: [UInt8], from start: Int) -> String? {
        guard start >= 0, start < bytes.count else { return nil }
        let slice = bytes[start...]
        guard let text = String(bytes: slice, encoding: .utf8),
            text.utf8.elementsEqual(slice)
        else { return nil }
        return text
    }
}
