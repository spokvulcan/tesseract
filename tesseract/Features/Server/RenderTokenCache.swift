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
//  Experiment C27 — the **truncated resolve**: `PrefillPlanner` re-renders
//  every cache-aware request truncated at the last user message (with
//  `add_generation_prompt: false` merged into the context) to find the
//  last-user boundary — a second full-size encode right after the Request
//  Keying phase cached the FULL render+tokens of the same conversation.
//  `resolveTruncated` recovers that token list as a TRIM of the stored one:
//  when the truncated render is a byte prefix of the stored render (the
//  common case — the conversation's last message is its last user message —
//  and verified empirically per call), the truncated tokens are the stored
//  tokens minus the k tail tokens covering the tail text (the generation
//  prompt). The exactness arbiter is the **cut verification**: a standalone
//  re-encode of the truncated render's trailing ≥256 characters (enlarged ×4
//  up to 16,384 on failure) must reproduce `candidateTokens.suffix(a)` for
//  the `a` covering the same text — this catches right-context effects at
//  the cut (the stored encode saw the tail following; the standalone encode
//  sees end-of-text, e.g. the `\s+(?!\S)` pretoken alternative). Left of the
//  window the tokenizations coincide by construction: byte-level BPE splits
//  text into pretokens by regex (bounded, context-free) and merges only
//  within a pretoken, so with the same merge table and identical text the
//  only place the two encodes can disagree is a pretoken touching the cut —
//  and every such pretoken lies inside the verified window. Any failure
//  degrades to the caller's `applyChatTemplate`; correctness never hinges on
//  the cache.
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
        /// C27: truncated resolves served as a verified trim of the entry.
        var truncatedHits = 0
        /// C27: truncated resolves that fell back to `applyChatTemplate`.
        var truncatedFallbacks = 0
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

    /// C27 truncated resolve: recover the token list of a render TRUNCATED at
    /// the conversation's last user message as a verified trim of the stored
    /// full-conversation entry. Returns exactly the tokens
    /// `applyChatTemplate(messages, tools, mergedAdditionalContext)` would
    /// produce, or `nil` — the caller falls back to that call.
    ///
    /// Chain of exactness (each step's failure is a `nil`, never a guess):
    /// 1. Candidate: the stored entry's fingerprint, template hash, tools
    ///    digest, and context digest all match — the context digest compared
    ///    on the UNMERGED `baseAdditionalContext`, because the entry was
    ///    rendered with it while the truncated render uses the merged one —
    ///    and the entry's per-message digest chain head-matches the truncated
    ///    messages' chain.
    /// 2. Empirical arbiter: the truncated render (made here with the merged
    ///    context) is a byte prefix of the stored render. The tail text must
    ///    be short — a generation prompt, not dropped content
    ///    (`maxTruncatedTailBytes`); a long tail means the conversation's last
    ///    message is not its last user message, the case this path does not
    ///    exist for.
    /// 3. The trim k: tokens stripped from the tail of the stored list (one
    ///    at a time, each decode checked against the remaining tail text)
    ///    must consume the tail exactly; a token whose decoded text is not a
    ///    suffix of the remainder spans the cut — not trimmable.
    /// 4. Cut verification (`verifyCut`): the right-context arbiter; see the
    ///    file header.
    ///
    /// The stored entry is left untouched: the truncated list must never
    /// poison the full-conversation entry the next request resolves against.
    func resolveTruncated(
        tokenizer: any Tokenizer,
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?,
        baseAdditionalContext: [String: any Sendable]?,
        mergedAdditionalContext: [String: any Sendable]?,
        modelFingerprint: String
    ) throws -> [Int]? {
        guard let rendering = tokenizer as? any ChatTemplateRendering else {
            return nil
        }

        func fallback() -> [Int]? {
            lock.withLock { stats.truncatedFallbacks += 1 }
            return nil
        }

        // 1. Keyed candidate selection (cheap digests before any render).
        guard let entry = lock.withLock({ entry }),
            entry.modelFingerprint == modelFingerprint
        else {
            return fallback()
        }
        let toolsDigest = Self.sha256Hex(Self.canonicalForm(optional: tools))
        let baseContextDigest = Self.sha256Hex(Self.canonicalForm(optional: baseAdditionalContext))
        guard entry.toolsDigest == toolsDigest, entry.contextDigest == baseContextDigest
        else {
            return fallback()
        }
        let chain = Self.digestChain(messages)
        guard entry.messageDigests.count >= chain.count,
            zip(entry.messageDigests, chain).allSatisfy(==)
        else {
            return fallback()
        }
        let templateHash = try templateHash(for: modelFingerprint, rendering: rendering)
        guard entry.templateHash == templateHash else {
            return fallback()
        }

        // 2. The truncated render must be a byte prefix of the stored render
        //    — the empirical arbiter that the two contexts render the same
        //    head regardless of what the digests claim.
        let truncatedRender = try rendering.renderChatTemplate(
            messages: messages, tools: tools, additionalContext: mergedAdditionalContext)
        guard entry.renderedText.hasPrefix(truncatedRender) else {
            return fallback()
        }
        let tail = String(entry.renderedText.dropFirst(truncatedRender.count))
        guard !tail.isEmpty, tail.utf8.count <= Self.maxTruncatedTailBytes else {
            return fallback()
        }

        // 3. Find the trim k: strip each tail token's decoded text off the
        //    tail, one token at a time (decode is per-token concatenation,
        //    so stripping reproduces exactly `decode(tailTokens)` — a
        //    doubling probe can jump over the exact k when token lengths
        //    vary). A token whose text is not a suffix of the remaining
        //    tail spans the cut — not trimmable. Tails are capped at
        //    `maxTruncatedTailBytes`, so the walk is a handful of tokens.
        var remaining = tail
        var k = 0
        while !remaining.isEmpty {
            guard k < entry.tokens.count else {
                return fallback()
            }
            let token = entry.tokens[entry.tokens.count - 1 - k]
            let decoded = tokenizer.decode(tokenIds: [token], skipSpecialTokens: false)
            guard !decoded.isEmpty, remaining.hasSuffix(decoded) else {
                return fallback()
            }
            remaining = String(remaining.dropLast(decoded.count))
            k += 1
        }
        guard k < entry.tokens.count else {
            return fallback()
        }
        let candidateTokens = Array(entry.tokens.dropLast(k))

        // 4. Cut verification — the right-context arbiter.
        guard
            verifyCut(
                candidateTokens: candidateTokens, truncatedRender: truncatedRender,
                tokenizer: tokenizer
            )
        else {
            return fallback()
        }
        lock.withLock { stats.truncatedHits += 1 }
        return candidateTokens
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

    /// C27: the longest tail text a truncated resolve will trim — a
    /// generation prompt (`<|im_start|>assistant\n<think>\n` is 28 bytes),
    /// never dropped conversation content. A longer tail means the
    /// conversation's last message is not its last user message; that case
    /// falls back to `applyChatTemplate`.
    private static let maxTruncatedTailBytes = 128

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
        // C26: `prefixText` is derived from the entry's stored render by
        // stripping each trimmed token's decoded text from the tail — decode
        // is per-token concatenation, so stripping reproduces exactly
        // `decode(prefixTokens)` — instead of decoding the whole prefix per
        // attempt (that decode dominated the hit path at long prefixes). The
        // `hasSuffix` guard keeps any decode/strip inconsistency honest, and
        // the per-attempt `hasPrefix` check is unchanged.
        var prefixText = candidate.renderedText
        for k in 0...Self.maxTrimBack {
            guard candidate.tokens.count - k > 0 else { break }
            if k > 0 {
                let dropped = candidate.tokens[candidate.tokens.count - k]
                let droppedText = tokenizer.decode(tokenIds: [dropped], skipSpecialTokens: false)
                guard prefixText.hasSuffix(droppedText) else {
                    return .failed(.renderNotExtended)
                }
                prefixText = String(prefixText.dropLast(droppedText.count))
            }
            let prefixTokens = Array(candidate.tokens.dropLast(k))
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

    /// C27: the truncated resolve's exactness arbiter — the right-context
    /// counterpart to `verifyJunction`. The candidate tokens were encoded
    /// with the tail text following them; the standalone encode this result
    /// is substituted for sees end-of-text at the cut. Re-encodes the
    /// truncated render's trailing ≥C characters (C = 256 enlarged ×4 up to
    /// 16,384 on failure) and requires the exact token slice back: the span
    /// `candidateTokens.suffix(a)` covering the window must equal the
    /// standalone encode of its own decoded text. Any right-context effect
    /// at the cut (a pretoken alternative like `\s+(?!\S)` matching
    /// differently at end-of-text) changes the window's tokenization and is
    /// detected here. Left of the window the two encodes coincide by
    /// construction (see the file header for the pretoken argument).
    private func verifyCut(
        candidateTokens: [Int],
        truncatedRender: String,
        tokenizer: any Tokenizer
    ) -> Bool {
        var target = 256
        while true {
            let a = trailingTokenCount(
                of: candidateTokens, coveringCharacters: target, tokenizer: tokenizer)
            if a >= 1 {
                let span = Array(candidateTokens.suffix(a))
                let windowText = tokenizer.decode(tokenIds: span, skipSpecialTokens: false)
                // The span's text must be the render's own tail: anything
                // else is a decode/trim inconsistency no enlargement fixes.
                guard truncatedRender.hasSuffix(windowText) else { return false }
                let windowTokens = tokenizer.encode(text: windowText, addSpecialTokens: false)
                if windowTokens == span { return true }
            }
            let canGrow = a < candidateTokens.count
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
