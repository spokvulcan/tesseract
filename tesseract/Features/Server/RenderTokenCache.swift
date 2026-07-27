//
//  RenderTokenCache.swift
//  tesseract
//
//  Experiment C25 — the **Render+Token Cache**: a single-entry last-request
//  cache over the fused render+encode `applyChatTemplate` call every
//  server/agent request pays. On prefix-stable request sequences (agent
//  multi-turn, growing conversations) the bulk of the render is byte-identical
//  to the previous request's, so the cache re-renders (cheap — the Jinja
//  render is ~0.03% of the fused call), verifies the old render is a byte
//  prefix of the new one, and tokenizes only the suffix — reproducing the
//  EXACT token list `applyChatTemplate` would return.
//
//  Exactness contract (the binding gate):
//  - The MISS path is `renderChatTemplate` + `encode(rendered)` — byte-exact
//    with `applyChatTemplate` by construction (Layers 1+2 split the fused
//    call without changing it; Gate 1 asserts the equality on a battery).
//  - The HIT path is empirical: a junction-window verification re-encodes a
//    window bracketing the cached/suffix seam and requires the exact token
//    slice back (BPE merges spanning the cut are detected, never assumed
//    away). Failure trims back one token (k = 1...4) and retries; all
//    failures degrade to the miss path. Correctness never hinges on the
//    cache.
//  - An identical repeat render returns the cached tokens outright:
//    `encode` is deterministic, so byte-identical render ⇒ identical tokens.
//
//  The cache keys on (modelFingerprint, templateHash, tools digest,
//  additional-context digest, per-message digest chain). `templateHash` is
//  the SHA-256 of a probe render (the template boilerplate); note the load
//  path's `ModelFingerprint.computeFingerprint` already hashes
//  `tokenizer_config.json` / `chat_template.*`, so template identity is
//  covered twice. Images and vision-family models never reach here — both
//  integration seams bypass upstream of this type.
//

import CryptoKit
import Foundation
import MLXLMCommon

/// `@unchecked Sendable`: all mutable state is NSLock-guarded.
nonisolated final class RenderTokenCache: @unchecked Sendable {

    nonisolated static let shared = RenderTokenCache()

    // MARK: - Public surface

    /// How one `resolve` produced its token list.
    enum Path: Equatable, Sendable {
        /// Cache hit: cached prefix tokens (trimmed by k tokens) + fresh
        /// suffix tokens, junction-verified.
        case hit(trimmedBy: Int)
        /// Byte-identical repeat render — cached tokens returned outright.
        case hitRepeat
        /// Cache miss: full render+encode (today's behavior, split).
        case miss(MissReason)
    }

    enum MissReason: Equatable, Sendable {
        /// No entry stored yet.
        case cold
        /// Different model fingerprint.
        case modelMismatch
        /// Same model, different template probe hash.
        case templateMismatch
        /// Tools or additional-context digests differ.
        case toolsOrContextMismatch
        /// Per-message digest chain is not a head-match (edited history).
        case digestMismatch
        /// Cached render is not a byte prefix of the new render.
        case renderNotExtended
        /// Every trim-back attempt failed junction verification.
        case junctionUnverified
    }

    struct Resolution: Sendable {
        let tokens: [Int]
        let path: Path
    }

    struct Stats: Equatable, Sendable {
        var hits = 0
        var repeats = 0
        var misses = 0
        /// k (tokens trimmed) -> number of hits verified at that trim.
        var trimHistogram: [Int: Int] = [:]
        /// Trim attempts rejected by the junction-window verification.
        var junctionFailures = 0
        /// Times the junction window had to grow past 256 characters.
        var windowEnlargements = 0
    }

    /// Tokenize one request through the cache. Returns `nil` when the
    /// tokenizer cannot render (`ChatTemplateRendering` unimplemented) — the
    /// caller falls back to its processor's `prepare`. Throws only when the
    /// render/encode itself throws; callers fall back to `prepare` too, which
    /// reproduces today's error handling (e.g. the missing-template plain-text
    /// fallback stays in the processor).
    func resolve(
        tokenizer: any Tokenizer,
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?,
        modelFingerprint: String
    ) throws -> Resolution? {
        guard let rendering = tokenizer as? any ChatTemplateRendering else {
            return nil
        }

        // 1. Always render — the Jinja pass is ~0.03% of the fused call.
        let rendered = try rendering.renderChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)

        // 2. Identical repeat render: encode is deterministic, so the cached
        //    tokens are provably exact — as long as the tokenizer is the same
        //    (the fingerprint covers tokenizer.json).
        if let repeatEntry = lock.withLock({
            entry.flatMap {
                $0.renderedText == rendered && $0.modelFingerprint == modelFingerprint ? $0 : nil
            }
        }) {
            lock.withLock { stats.repeats += 1 }
            return Resolution(tokens: repeatEntry.tokens, path: .hitRepeat)
        }

        // 3. Digest chain + template probe hash.
        let toolsDigest = Self.sha256Hex(Self.canonicalForm(optional: tools))
        let contextDigest = Self.sha256Hex(Self.canonicalForm(optional: additionalContext))
        let chain = Self.digestChain(messages)
        let templateHash = try templateHash(for: modelFingerprint, rendering: rendering)

        // 4. Candidate lookup: fingerprint, template, tools/context, and the
        //    per-message digest head-match select the entry. The byte-prefix
        //    requirement is NOT checked here against the full stored render:
        //    this template's generation-prompt tail (`<|im_start|>assistant
        //    \n<think>\n`) diverges from the next request's render right at
        //    the seam, so a hittable entry routinely fails a full-text
        //    hasPrefix — the trim loop below re-discovers the true shared
        //    prefix in token space and verifies each attempt's text.
        let missReason: MissReason
        var candidate: Entry?
        if let entry = lock.withLock({ entry }) {
            if entry.modelFingerprint != modelFingerprint {
                missReason = .modelMismatch
            } else if entry.templateHash != templateHash {
                missReason = .templateMismatch
            } else if entry.toolsDigest != toolsDigest || entry.contextDigest != contextDigest {
                missReason = .toolsOrContextMismatch
            } else if entry.messageDigests.count > chain.count
                || !zip(entry.messageDigests, chain).allSatisfy(==)
            {
                missReason = .digestMismatch
            } else {
                missReason = .junctionUnverified
                candidate = entry
            }
        } else {
            missReason = .cold
        }

        // 5. HIT candidate: cut at k=0 trimmed tokens, then trim back k=1...4.
        var finalReason = missReason
        if let candidate {
            switch resolveHit(
                candidate: candidate,
                rendered: rendered,
                chain: chain,
                toolsDigest: toolsDigest,
                contextDigest: contextDigest,
                tokenizer: tokenizer
            ) {
            case .resolved(let resolution):
                return resolution
            case .failed(let reason):
                finalReason = reason
            }
        }

        // 6. MISS path: render+encode(full) — byte-exact with today's
        //    `applyChatTemplate`, via the split (one code path).
        let fullTokens = tokenizer.encode(text: rendered, addSpecialTokens: false)
        store(
            Entry(
                modelFingerprint: modelFingerprint,
                templateHash: templateHash,
                toolsDigest: toolsDigest,
                contextDigest: contextDigest,
                messageDigests: chain,
                renderedText: rendered,
                tokens: fullTokens
            ))
        lock.withLock { stats.misses += 1 }
        return Resolution(tokens: fullTokens, path: .miss(finalReason))
    }

    func statsSnapshot() -> Stats {
        lock.withLock { stats }
    }

    /// Test/bench hook: drop the cached entry, template hashes, and stats.
    func reset() {
        lock.withLock {
            entry = nil
            templateHashes = [:]
            stats = Stats()
        }
    }

    // MARK: - Entry

    struct Entry: Sendable {
        let modelFingerprint: String
        let templateHash: String
        let toolsDigest: String
        let contextDigest: String
        /// Cumulative per-message SHA-256 chain over the canonical message
        /// dicts as passed to `resolve` (`chain[i]` covers messages `0...i`).
        let messageDigests: [String]
        let renderedText: String
        let tokens: [Int]
    }

    private var entry: Entry?
    /// Probe-render template hash, memoized per model fingerprint.
    private var templateHashes: [String: String] = [:]
    private var stats = Stats()
    private let lock = NSLock()

    init() {}

    // MARK: - Hit path

    /// Largest trim-back count tried after the exact cut (k = 0...4 total).
    private static let maxTrimBack = 4

    private enum HitOutcome {
        case resolved(Resolution)
        case failed(MissReason)
    }

    private func resolveHit(
        candidate: Entry,
        rendered: String,
        chain: [String],
        toolsDigest: String,
        contextDigest: String,
        tokenizer: any Tokenizer
    ) -> HitOutcome {
        /// Whether any trim attempt produced a textual prefix at all — the
        /// discriminator between "the new render does not extend the old one"
        /// and "the seam is there but the junction never verified".
        var sawTextualPrefix = false
        for k in 0...Self.maxTrimBack {
            guard candidate.tokens.count - k > 0 else { break }
            let prefixTokens = Array(candidate.tokens.dropLast(k))
            // Byte-level BPE decode is byte-exact in practice; the hasPrefix
            // check per attempt is what makes it safe to rely on.
            let prefixText = tokenizer.decode(tokenIds: prefixTokens, skipSpecialTokens: false)
            guard rendered.hasPrefix(prefixText) else { continue }
            sawTextualPrefix = true
            let suffix = String(rendered.dropFirst(prefixText.count))
            // An empty suffix at k>0 would mean the trimmed tokens decoded to
            // nothing — not trustworthy as `encode(rendered)`; keep trimming.
            guard !suffix.isEmpty else { continue }
            let suffixTokens = tokenizer.encode(text: suffix, addSpecialTokens: false)
            if verifyJunction(
                prefixTokens: prefixTokens, suffixTokens: suffixTokens, tokenizer: tokenizer)
            {
                let fullTokens = prefixTokens + suffixTokens
                store(
                    Entry(
                        modelFingerprint: candidate.modelFingerprint,
                        templateHash: candidate.templateHash,
                        toolsDigest: toolsDigest,
                        contextDigest: contextDigest,
                        messageDigests: chain,
                        renderedText: rendered,
                        tokens: fullTokens
                    ))
                lock.withLock {
                    stats.hits += 1
                    stats.trimHistogram[k, default: 0] += 1
                }
                return .resolved(Resolution(tokens: fullTokens, path: .hit(trimmedBy: k)))
            }
            lock.withLock { stats.junctionFailures += 1 }
        }
        return .failed(sawTextualPrefix ? .junctionUnverified : .renderNotExtended)
    }

    /// The exactness arbiter. Re-encodes a window bracketing the
    /// cached/suffix seam — the last ≥C characters of the cached prefix plus
    /// the first ≥C characters of the suffix, C = 256 enlarged ×4 up to
    /// 16,384 on failure — and requires the identical token slice back. Any
    /// BPE merge spanning the cut changes the window's tokenization and is
    /// detected here; the check is an exact token-array equality, never a
    /// heuristic.
    private func verifyJunction(
        prefixTokens: [Int],
        suffixTokens: [Int],
        tokenizer: any Tokenizer
    ) -> Bool {
        var target = 256
        while true {
            let a = trailingTokenCount(
                of: prefixTokens, coveringCharacters: target, tokenizer: tokenizer)
            let b = leadingTokenCount(
                of: suffixTokens, coveringCharacters: target, tokenizer: tokenizer)
            if a >= 1, b >= 1 {
                let span = Array(prefixTokens.suffix(a)) + Array(suffixTokens.prefix(b))
                // Decoding the span re-splits at the seam exactly: the prefix
                // side ends at a character boundary (the `hasPrefix` check
                // upstream guarantees a valid seam) and the suffix side
                // starts there, so `decode(span)` is the window text.
                let windowText = tokenizer.decode(tokenIds: span, skipSpecialTokens: false)
                let windowTokens = tokenizer.encode(text: windowText, addSpecialTokens: false)
                if windowTokens == span { return true }
            }
            let canGrow = a < prefixTokens.count || b < suffixTokens.count
            guard canGrow, target < 16_384 else { return false }
            target = min(target * 4, 16_384)
            lock.withLock { stats.windowEnlargements += 1 }
        }
    }

    /// Smallest token count from the end whose decode covers at least
    /// `target` UTF-8 bytes — doubling probes so the search costs O(log)
    /// decodes, never a full re-decode per token.
    private func trailingTokenCount(
        of tokens: [Int],
        coveringCharacters target: Int,
        tokenizer: any Tokenizer
    ) -> Int {
        var count = 1
        while count < tokens.count {
            let text = tokenizer.decode(
                tokenIds: Array(tokens.suffix(count)), skipSpecialTokens: false)
            if text.utf8.count >= target { return count }
            count = min(count * 2, tokens.count)
        }
        return tokens.count
    }

    private func leadingTokenCount(
        of tokens: [Int],
        coveringCharacters target: Int,
        tokenizer: any Tokenizer
    ) -> Int {
        var count = 1
        while count < tokens.count {
            let text = tokenizer.decode(
                tokenIds: Array(tokens.prefix(count)), skipSpecialTokens: false)
            if text.utf8.count >= target { return count }
            count = min(count * 2, tokens.count)
        }
        return tokens.count
    }

    // MARK: - Keys and digests

    private func store(_ newEntry: Entry) {
        lock.withLock { entry = newEntry }
    }

    /// SHA-256 of a probe render — a fixed one-message conversation rendered
    /// through the template. A template change produces a different probe
    /// render, hence a different key. (An empty message list would be the
    /// purer boilerplate probe, but templates may reject it — the loaded
    /// Qwen3.5 template raises `No messages provided`.) A throwing probe
    /// hashes the empty render; the model fingerprint still covers the
    /// template files on disk (`ModelFingerprint.computeFingerprint` hashes
    /// `tokenizer_config.json` / `chat_template.*`), so template identity
    /// remains keyed.
    private func templateHash(
        for modelFingerprint: String,
        rendering: any ChatTemplateRendering
    ) throws -> String {
        if let cached = lock.withLock({ templateHashes[modelFingerprint] }) {
            return cached
        }
        let probeMessages: [[String: any Sendable]] = [
            ["role": "user", "content": "render-token-cache probe"]
        ]
        let probe =
            (try? rendering.renderChatTemplate(
                messages: probeMessages, tools: nil, additionalContext: nil)) ?? ""
        let hash = Self.sha256Hex("probe:" + probe)
        lock.withLock { templateHashes[modelFingerprint] = hash }
        return hash
    }

    private static func digestChain(_ messages: [[String: any Sendable]]) -> [String] {
        var chain: [String] = []
        chain.reserveCapacity(messages.count)
        var previous = "rtc1"
        for message in messages {
            previous = sha256Hex(previous + "|" + canonicalForm(message))
            chain.append(previous)
        }
        return chain
    }

    static func sha256Hex(_ string: String) -> String {
        SHA256.hash(data: Data(string.utf8)).map { String(format: "%02x", $0) }.joined()
    }

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
    static func canonicalForm(_ value: Any) -> String {
        switch value {
        case is NSNull:
            return "null"
        case let bool as Bool:
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
}
