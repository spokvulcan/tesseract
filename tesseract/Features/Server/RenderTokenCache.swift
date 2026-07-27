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
//  **Everything here compares UTF-8 bytes, never `String`s.** Swift `String`
//  equality, `hasPrefix` and `hasSuffix` compare under Unicode *canonical
//  equivalence*, so an NFC render and an NFD render of the same text are `==`
//  while their bytes — and therefore their token lists — differ. The junction
//  and cut arbiters below cannot catch that class of mismatch: they decode the
//  cached tokens and re-encode that decode, which is self-consistent by
//  construction and never re-examines the new render's actual bytes. Entries
//  therefore store `renderedBytes: [UInt8]`, and every prefix/suffix/equality
//  test goes through the byte helpers in `RenderTokenCache+Keys.swift`. A
//  normalization-shifting client now misses (and re-encodes correctly) instead
//  of silently receiving tokens for the other byte string.
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
//    This is the ONLY resolve with no empirical arbiter behind it — its
//    exactness rests entirely on (byte equality, model fingerprint), which is
//    why an unknown fingerprint bypasses the cache upstream rather than
//    sharing a synthetic key (`RenderTokenSource`).
//
//  The cache keys on (modelFingerprint, templateHash, tools digest,
//  additional-context digest, per-message digest chain). `templateHash` is
//  the SHA-256 of a probe render (the template boilerplate); note the load
//  path's `ModelFingerprint.computeFingerprint` already hashes
//  `tokenizer_config.json` / `chat_template.*`, so template identity is
//  covered twice. Images and vision-family models never reach here — every
//  integration seam bypasses upstream of this type via `RenderTokenSource`.
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
//  tokens minus the k tail tokens covering the tail bytes (the generation
//  prompt). The exactness arbiter is the **cut verification**: a standalone
//  re-encode of the truncated render's trailing ≥256 bytes (enlarged ×4 up to
//  16,384 on failure) must reproduce `candidateTokens.suffix(a)` for the `a`
//  covering the same text — this catches right-context effects at the cut (the
//  stored encode saw the tail following; the standalone encode sees
//  end-of-text, e.g. the `\s+(?!\S)` pretoken alternative). Left of the
//  window the tokenizations coincide by construction: byte-level BPE splits
//  text into pretokens by regex (bounded, context-free) and merges only
//  within a pretoken, so with the same merge table and identical bytes the
//  only place the two encodes can disagree is a pretoken touching the cut —
//  and every such pretoken lies inside the verified window. Any failure
//  degrades to the caller's `applyChatTemplate`; correctness never hinges on
//  the cache.
//
//  Experiment C28 — the **tail-replacement resolve**: post-generation, the
//  Leaf Store phase re-tokenizes the stored conversation (the request plus
//  the appended assistant turn, `add_generation_prompt: false`) and the Leaf
//  Admission Builder renders that conversation twice more (alone and with a
//  synthetic continuation appended) — up to three more full-size encodes per
//  turn, serialized against the next request. Each target render shares the
//  stored entry's head and diverges exactly where the entry's generation-
//  prompt tail sits, so `resolveReplacingTail` COMPOSES the two verified
//  primitives: trim the tail (C27 machinery — strip one tail token at a
//  time, each decode checked against the remaining prefix bytes; a token
//  whose bytes are not a suffix spans the cut and falls back), then suffix-
//  encode the extension and junction-verify (C25 machinery — the window
//  re-encode bracketing the cut must reproduce the exact token slice). The
//  two verification windows make the composition exact by the same
//  arguments: the trim walk detects any token spanning the cut from the
//  entry's side, and the junction window arbitrates the right-context
//  effects the extension introduces at the cut (a pretoken like `\s+(?!\S)`
//  matching differently with the extension following than it did with the
//  generation prompt following) — the trim-back retries absorb those into
//  the suffix until the window verifies. `verifyCut` is NOT needed here:
//  that arbiter exists for the end-of-text right context of a pure trim;
//  the tail-replacement's cut is always followed by the freshly encoded
//  suffix, which is precisely the case the junction window covers. The
//  stored entry is left untouched — the next request's C25 resolve still
//  finds this request's full render cached, and the replaced lists must
//  never poison it. Any failure degrades to the caller's
//  `applyChatTemplate`; correctness never hinges on the cache.
//
//  Experiment C31 — compute-once-and-plumb on the truncated resolve:
//  `PrefillPlanner`'s truncated conversation is a message prefix of the
//  conversation the Request Keying phase resolved for the same request, and
//  the digest chain is cumulative, so the truncated chain is the entry's
//  stored head — the same values, reused under the caller's
//  `messagesAreEntryPrefix` assertion instead of recomputed per request.
//  The head-match guard still runs; exactness still rests on the
//  byte-prefix/trim/cut-verify arbiters.
//
//  Observability: every resolve counts into `Stats` with a typed reason, a
//  summary lands in `Log.server` every `summaryInterval` resolves, and a
//  throwing render is logged before it propagates. The seams all call through
//  `try?`, so without this the subsystem's own safety mechanism — silent
//  degradation to the full encode — would be invisible in production.
//

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

    enum MissReason: String, Equatable, Sendable {
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

    /// Why a C27/C28 resolve handed the caller back to `applyChatTemplate`.
    /// Distinct from `MissReason`: those resolves never store an entry, so
    /// there is no "miss" to attribute — only a fallback.
    enum FallbackReason: String, Equatable, Sendable {
        /// No entry stored yet.
        case cold
        /// Different model fingerprint.
        case modelMismatch
        /// Tools or base additional-context digests differ.
        case toolsOrContextMismatch
        /// Per-message digest chain is not a head-match in the required
        /// direction (C27 wants a prefix, C28 a strict extension).
        case digestMismatch
        /// Same model, different template probe hash.
        case templateMismatch
        /// The target render is not a byte prefix of / extension of the entry.
        case renderNotPrefix
        /// The trim tail is empty — the renders are byte-identical, which is
        /// the repeat case `resolve` owns.
        case tailEmpty
        /// The trim tail is longer than a generation prompt: the entry's last
        /// message is not the truncation point this path exists for.
        case tailTooLong
        /// A tail token's bytes cross the cut — not trimmable.
        case trimSpansCut
        /// The cut-window re-encode did not reproduce the token slice.
        case cutUnverified
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
        /// `MissReason.rawValue` -> count.
        var missReasons: [String: Int] = [:]
        /// C25 hit-path trim attempts rejected by the junction window.
        var junctionFailures = 0
        /// C28 trim attempts rejected by the junction window. Kept separate
        /// from `junctionFailures`: the two paths cross different seams and
        /// conflating them hides which one regressed.
        var replacedJunctionFailures = 0
        /// Times `verifyJunction` had to grow past `initialWindowBytes`.
        var junctionWindowEnlargements = 0
        /// Times `verifyCut` had to grow past `initialWindowBytes`.
        var cutWindowEnlargements = 0
        /// C27: truncated resolves served as a verified trim of the entry.
        var truncatedHits = 0
        /// C27: truncated resolves that fell back to `applyChatTemplate`.
        var truncatedFallbacks = 0
        /// C27: `FallbackReason.rawValue` -> count.
        var truncatedFallbackReasons: [String: Int] = [:]
        /// C28: tail-replacement resolves served as a verified trim+extension
        /// of the entry.
        var replacedHits = 0
        /// C28: tail-replacement resolves that fell back to
        /// `applyChatTemplate`.
        var replacedFallbacks = 0
        /// C28: `FallbackReason.rawValue` -> count.
        var replacedFallbackReasons: [String: Int] = [:]
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
        let rendered = try render(
            rendering, messages: messages, tools: tools, additionalContext: additionalContext,
            label: "resolve")
        let renderedBytes = Array(rendered.utf8)

        // 2. ONE entry read for the whole resolve. Re-reading would let the
        //    entry change between the chain build and the candidate select,
        //    which would store a chain whose head does not derive from these
        //    messages — permanently vacuating the head-match guard for the
        //    singleton (exactness would still hold on the render arbiters,
        //    but a documented pre-filter would be silently dead).
        let snapshot = lock.withLock { entry }

        // 3. Identical repeat render: encode is deterministic, so the cached
        //    tokens are provably exact — as long as the tokenizer is the same
        //    (the fingerprint covers tokenizer.json). Byte equality, not
        //    `String ==`: see the file header.
        if let snapshot, snapshot.modelFingerprint == modelFingerprint,
            snapshot.renderedBytes == renderedBytes
        {
            lock.withLock { stats.repeats += 1 }
            logSummaryIfDue()
            return Resolution(tokens: snapshot.tokens, path: .hitRepeat)
        }

        // 4. Digest chain + template probe hash. C29: the chain reuses the
        //    entry's stored head when it is short enough — cumulative hashing
        //    makes the extending chain's head the same values, so only the
        //    tail messages are hashed. A head that does NOT match (edited
        //    history) then vacuously passes the head-match guard below — and
        //    the render arbiters (byte-prefix + trim + junction verification)
        //    reject the candidate instead, so the miss lands on
        //    `.renderNotExtended` rather than `.digestMismatch`, never on a
        //    wrong token list.
        let toolsDigest = Self.sha256Hex(Self.canonicalForm(optional: tools))
        let contextDigest = Self.sha256Hex(Self.canonicalForm(optional: additionalContext))
        let chain = Self.digestChain(messages, reusingHeadOf: snapshot)
        let templateHash = try templateHash(for: modelFingerprint, rendering: rendering)

        // 5. Candidate lookup: fingerprint, template, tools/context, and the
        //    per-message digest head-match select the entry. The byte-prefix
        //    requirement is NOT checked here against the full stored render:
        //    this template's generation-prompt tail (`<|im_start|>assistant
        //    \n<think>\n`) diverges from the next request's render right at
        //    the seam, so a hittable entry routinely fails a whole-buffer
        //    prefix test — the trim loop below re-discovers the true shared
        //    prefix in token space and verifies each attempt's bytes.
        let missReason: MissReason
        var candidate: Entry?
        if let snapshot {
            if snapshot.modelFingerprint != modelFingerprint {
                missReason = .modelMismatch
            } else if snapshot.templateHash != templateHash {
                missReason = .templateMismatch
            } else if snapshot.toolsDigest != toolsDigest
                || snapshot.contextDigest != contextDigest
            {
                missReason = .toolsOrContextMismatch
            } else if snapshot.messageDigests.count > chain.count
                || !zip(snapshot.messageDigests, chain).allSatisfy(==)
            {
                missReason = .digestMismatch
            } else {
                missReason = .junctionUnverified
                candidate = snapshot
            }
        } else {
            missReason = .cold
        }

        // 6. HIT candidate: cut at k=0 trimmed tokens, then trim back k=1...4.
        var finalReason = missReason
        if let candidate {
            switch resolveHit(
                candidate: candidate,
                renderedBytes: renderedBytes,
                chain: chain,
                toolsDigest: toolsDigest,
                contextDigest: contextDigest,
                tokenizer: tokenizer
            ) {
            case .resolved(let resolution):
                logSummaryIfDue()
                return resolution
            case .failed(let reason):
                finalReason = reason
            }
        }

        // 7. MISS path: render+encode(full) — byte-exact with today's
        //    `applyChatTemplate`, via the split (one code path).
        let fullTokens = tokenizer.encode(text: rendered, addSpecialTokens: false)
        store(
            Entry(
                modelFingerprint: modelFingerprint,
                templateHash: templateHash,
                toolsDigest: toolsDigest,
                contextDigest: contextDigest,
                messageDigests: chain,
                renderedBytes: renderedBytes,
                tokens: fullTokens
            ))
        lock.withLock {
            stats.misses += 1
            stats.missReasons[finalReason.rawValue, default: 0] += 1
        }
        logSummaryIfDue()
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
    ///    context) is a BYTE prefix of the stored render. The tail must be
    ///    short — a generation prompt, not dropped content
    ///    (`maxTruncatedTailBytes`); a long tail means the conversation's last
    ///    message is not its last user message, the case this path does not
    ///    exist for.
    /// 3. The trim k: tokens stripped from the tail of the stored list (one
    ///    at a time, each decode's bytes checked against the remaining tail)
    ///    must consume the tail exactly; a token whose bytes reach left of the
    ///    cut spans it — not trimmable.
    /// 4. Cut verification (`verifyCut`): the right-context arbiter; see the
    ///    file header.
    ///
    /// The stored entry is left untouched: the truncated list must never
    /// poison the full-conversation entry the next request resolves against.
    ///
    /// C31: `messagesAreEntryPrefix` asserts `messages` are a prefix of the
    /// conversation whose FULL digest chain the stored entry already holds.
    /// That holds for `PrefillPlanner`'s last-user truncation *when the
    /// Request Keying phase resolved this request through the cache* (the same
    /// conversation value flows to both, and `promptMessages` maps
    /// per-message, so a message-level truncation is a prompt-message prefix).
    /// It does NOT hold when Request Keying bypassed or fell back, leaving an
    /// older entry in place — which is why the assertion is a cost hint, not a
    /// correctness input. The chain is cumulative (`chain[i]` covers messages
    /// `0...i`), so under the assertion the truncated chain IS the entry's
    /// stored head — the same values, reused instead of recomputed. The
    /// head-match guard below still runs (against the entry's own head when
    /// this is set); exactness continues to rest on the empirical arbiters
    /// (byte-prefix render, trim walk, cut verification), so a false assertion
    /// costs a render before those arbiters reject the candidate — never a
    /// wrong token list.
    func resolveTruncated(
        tokenizer: any Tokenizer,
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?,
        baseAdditionalContext: [String: any Sendable]?,
        mergedAdditionalContext: [String: any Sendable]?,
        modelFingerprint: String,
        messagesAreEntryPrefix: Bool = false
    ) throws -> [Int]? {
        guard let rendering = tokenizer as? any ChatTemplateRendering else {
            return nil
        }

        func fallback(_ reason: FallbackReason) -> [Int]? {
            lock.withLock {
                stats.truncatedFallbacks += 1
                stats.truncatedFallbackReasons[reason.rawValue, default: 0] += 1
            }
            logSummaryIfDue()
            return nil
        }

        // 1. Keyed candidate selection (cheap digests before any render).
        guard let entry = lock.withLock({ entry }) else {
            return fallback(.cold)
        }
        guard entry.modelFingerprint == modelFingerprint else {
            return fallback(.modelMismatch)
        }
        let toolsDigest = Self.sha256Hex(Self.canonicalForm(optional: tools))
        let baseContextDigest = Self.sha256Hex(Self.canonicalForm(optional: baseAdditionalContext))
        guard entry.toolsDigest == toolsDigest, entry.contextDigest == baseContextDigest
        else {
            return fallback(.toolsOrContextMismatch)
        }
        // C31: under the caller's entry-prefix assertion the truncated chain
        // is the entry's stored head (cumulative hashing — the same values,
        // computed once by the Request Keying resolve); the head-match guard
        // below then checks the entry against its own head. Without the
        // assertion the chain is computed and the guard does its historical
        // job (edited-history detection before paying for the render).
        let chain: [String]
        if messagesAreEntryPrefix, messages.count <= entry.messageDigests.count {
            chain = Array(entry.messageDigests.prefix(messages.count))
        } else {
            chain = Self.digestChain(messages)
        }
        guard entry.messageDigests.count >= chain.count,
            zip(entry.messageDigests, chain).allSatisfy(==)
        else {
            return fallback(.digestMismatch)
        }
        let templateHash = try templateHash(for: modelFingerprint, rendering: rendering)
        guard entry.templateHash == templateHash else {
            return fallback(.templateMismatch)
        }

        // 2. The truncated render must be a BYTE prefix of the stored render
        //    — the empirical arbiter that the two contexts render the same
        //    head regardless of what the digests claim.
        let truncatedRender = try render(
            rendering, messages: messages, tools: tools,
            additionalContext: mergedAdditionalContext, label: "resolveTruncated")
        let truncatedBytes = Array(truncatedRender.utf8)
        guard entry.renderedBytes.starts(with: truncatedBytes) else {
            return fallback(.renderNotPrefix)
        }
        let tailByteCount = entry.renderedBytes.count - truncatedBytes.count
        guard tailByteCount > 0 else { return fallback(.tailEmpty) }
        guard tailByteCount <= Self.maxTruncatedTailBytes else { return fallback(.tailTooLong) }

        // 3. Find the trim k: strip each tail token's decoded bytes off the
        //    tail, one token at a time (decode is per-token concatenation,
        //    so stripping reproduces exactly `decode(tailTokens)` — a
        //    doubling probe can jump over the exact k when token lengths
        //    vary). A token whose bytes reach left of the cut spans it — not
        //    trimmable. Tails are capped at `maxTruncatedTailBytes`, so the
        //    walk is a handful of tokens.
        var end = entry.renderedBytes.count
        var k = 0
        while end > truncatedBytes.count {
            guard k < entry.tokens.count else {
                return fallback(.trimSpansCut)
            }
            let token = entry.tokens[entry.tokens.count - 1 - k]
            let decoded = Self.utf8(ofToken: token, tokenizer: tokenizer)
            guard !decoded.isEmpty,
                end - decoded.count >= truncatedBytes.count,
                Self.bytes(entry.renderedBytes, endingAt: end, equal: decoded)
            else {
                return fallback(.trimSpansCut)
            }
            end -= decoded.count
            k += 1
        }
        guard k < entry.tokens.count else {
            return fallback(.trimSpansCut)
        }
        let candidateTokens = Array(entry.tokens.dropLast(k))

        // 4. Cut verification — the right-context arbiter.
        guard
            verifyCut(
                candidateTokens: candidateTokens, truncatedBytes: truncatedBytes,
                tokenizer: tokenizer
            )
        else {
            return fallback(.cutUnverified)
        }
        lock.withLock { stats.truncatedHits += 1 }
        logSummaryIfDue()
        return candidateTokens
    }

    /// C28 tail-replacement resolve: recover the token list of a render that
    /// EXTENDS the stored entry's conversation past its generation-prompt
    /// tail — the post-generation shapes (the stored conversation with the
    /// assistant turn appended, and that conversation plus a synthetic
    /// continuation). Returns exactly the tokens
    /// `applyChatTemplate(messages, tools, mergedAdditionalContext)` would
    /// produce, or `nil` — the caller falls back to that call.
    ///
    /// Chain of exactness (each step's failure is a `nil`, never a guess):
    /// 1. Candidate: same keyed selection as `resolveTruncated` — fingerprint,
    ///    tools digest, context digest compared on the UNMERGED
    ///    `baseAdditionalContext`, template probe hash — except the digest
    ///    chain must head-match with the entry's chain STRICTLY shorter: the
    ///    target conversation extends the entry's (an equal-length chain is
    ///    the trim case `resolveTruncated` owns).
    /// 2. Empirical arbiter: the target render (made here with the merged
    ///    context) must extend the entry's render past its tail — found by
    ///    the trim-back walk (3), never assumed.
    /// 3. Trim-back walk: strip one tail token at a time (decode is per-token
    ///    concatenation, so stripping reproduces `decode(prefixTokens)`
    ///    exactly), until the remaining prefix bytes are a byte prefix of the
    ///    target render. A token whose bytes are not a suffix of the
    ///    remaining prefix spans the cut — not trimmable. The k budget is
    ///    `maxTrimBack`, the same junction the C25 hit path crosses on every
    ///    growing turn.
    /// 4. Junction verification (`verifyJunction`): the freshly encoded
    ///    suffix's seam against the trimmed prefix — the right-context
    ///    arbiter; see the file header for why `verifyCut` does not apply.
    ///
    /// The stored entry is left untouched, exactly as `resolveTruncated`
    /// leaves it: the next request's C25 resolve must still find this
    /// request's full render cached.
    func resolveReplacingTail(
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

        func fallback(_ reason: FallbackReason) -> [Int]? {
            lock.withLock {
                stats.replacedFallbacks += 1
                stats.replacedFallbackReasons[reason.rawValue, default: 0] += 1
            }
            logSummaryIfDue()
            return nil
        }

        // 1. Keyed candidate selection (cheap digests before any render).
        guard let entry = lock.withLock({ entry }) else {
            return fallback(.cold)
        }
        guard entry.modelFingerprint == modelFingerprint else {
            return fallback(.modelMismatch)
        }
        let toolsDigest = Self.sha256Hex(Self.canonicalForm(optional: tools))
        let baseContextDigest = Self.sha256Hex(Self.canonicalForm(optional: baseAdditionalContext))
        guard entry.toolsDigest == toolsDigest, entry.contextDigest == baseContextDigest
        else {
            return fallback(.toolsOrContextMismatch)
        }
        // C29: reuse the entry's stored chain head (same values for an
        // extending conversation) — only the new tail messages are hashed;
        // an edited base vacuously passes the head match and is rejected by
        // the render arbiters below instead.
        let chain = Self.digestChain(messages, reusingHeadOf: entry)
        guard entry.messageDigests.count < chain.count,
            zip(entry.messageDigests, chain).allSatisfy(==)
        else {
            return fallback(.digestMismatch)
        }
        let templateHash = try templateHash(for: modelFingerprint, rendering: rendering)
        guard entry.templateHash == templateHash else {
            return fallback(.templateMismatch)
        }

        // 2–4. Render the target (the Jinja pass is ~0.03% of the fused
        // call), then the trim-back walk: k = 0 is the direct extension (the
        // non-thinking template, whose generation prompt is a pure prefix);
        // deeper k cross the generation-prompt tail and any merges spanning
        // the cut. Each k's candidate is junction-verified; the first
        // verified k wins, anything else falls back.
        let rendered = try render(
            rendering, messages: messages, tools: tools,
            additionalContext: mergedAdditionalContext, label: "resolveReplacingTail")
        let renderedBytes = Array(rendered.utf8)
        var prefixLength = entry.renderedBytes.count
        var sawBytePrefix = false
        for k in 0...Self.maxTrimBack {
            guard entry.tokens.count - k > 0 else { break }
            if k > 0 {
                let dropped = entry.tokens[entry.tokens.count - k]
                let droppedBytes = Self.utf8(ofToken: dropped, tokenizer: tokenizer)
                guard !droppedBytes.isEmpty,
                    Self.bytes(entry.renderedBytes, endingAt: prefixLength, equal: droppedBytes)
                else {
                    return fallback(.trimSpansCut)
                }
                prefixLength -= droppedBytes.count
            }
            let prefixTokens = Array(entry.tokens.dropLast(k))
            guard Self.bytes(renderedBytes, startsWith: entry.renderedBytes, count: prefixLength)
            else { continue }
            sawBytePrefix = true
            // An empty suffix at k>0 would mean the trimmed tokens decoded to
            // nothing — not trustworthy as `encode(rendered)`; keep trimming.
            // At k=0 it means the target does not extend the entry at all.
            guard let suffix = Self.suffixString(of: renderedBytes, from: prefixLength) else {
                continue
            }
            let suffixTokens = tokenizer.encode(text: suffix, addSpecialTokens: false)
            // Cheap seam pre-check: if the tokens adjacent to the cut merge
            // across it, the full window can only fail — skip to the next
            // trim WITHOUT paying the junction ladder (a futile 256→16384
            // enlargement costs a full encode's time; e.g. the empty think
            // scaffold a thinking template gives the latest turn makes the
            // suffix start with `\n`, merging the prefix's trailing `\n`
            // run). A skipped k just trims deeper; `verifyJunction` remains
            // the authority whenever the pre-check passes, so exactness is
            // unchanged.
            if let lastPrefix = prefixTokens.last, let firstSuffix = suffixTokens.first {
                let seamText =
                    tokenizer.decode(tokenIds: [lastPrefix], skipSpecialTokens: false)
                    + tokenizer.decode(tokenIds: [firstSuffix], skipSpecialTokens: false)
                if tokenizer.encode(text: seamText, addSpecialTokens: false)
                    != [lastPrefix, firstSuffix]
                {
                    continue
                }
            }
            if verifyJunction(
                prefixTokens: prefixTokens, suffixTokens: suffixTokens, tokenizer: tokenizer)
            {
                lock.withLock { stats.replacedHits += 1 }
                logSummaryIfDue()
                return prefixTokens + suffixTokens
            }
            lock.withLock { stats.replacedJunctionFailures += 1 }
        }
        return fallback(sawBytePrefix ? .junctionUnverified : .renderNotPrefix)
    }

    func statsSnapshot() -> Stats {
        lock.withLock { stats }
    }

    /// Drop the cached entry, template hashes, and stats. Called on model
    /// unload (the entry holds a whole render's bytes plus its token list —
    /// megabytes at long context — and none of it is valid for the next
    /// model), and by the test/bench harnesses between cases.
    func reset() {
        lock.withLock {
            entry = nil
            templateHashes = [:]
            stats = Stats()
            resolutionsSinceSummary = 0
        }
    }

    /// Emit the current counters at `info`, unconditionally. The unload path
    /// calls this before `reset()` so a session's hit rate is recoverable from
    /// the log even when the periodic summary did not land on a boundary.
    func logSummary(context: String) {
        let snapshot = lock.withLock { stats }
        Log.server.info("render-token-cache [\(context)] \(Self.summary(of: snapshot))")
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
        /// The render's UTF-8 bytes — NOT a `String`. Every prefix/suffix test
        /// in this file is a byte comparison; see the file header for why
        /// `String`'s canonical-equivalence semantics are unusable here.
        let renderedBytes: [UInt8]
        let tokens: [Int]
    }

    private var entry: Entry?
    /// Probe-render template hash, memoized per model fingerprint. Safe as a
    /// memo key because an unknown fingerprint never reaches this type — the
    /// seams bypass the cache instead of synthesizing a shared key, so two
    /// models can never share a memo slot.
    private var templateHashes: [String: String] = [:]
    private var stats = Stats()
    private var resolutionsSinceSummary = 0
    private let lock = NSLock()

    init() {}

    // MARK: - Tuning

    /// Largest trim-back count tried after the exact cut (k = 0...4 total).
    private static let maxTrimBack = 4

    /// C27: the longest tail a truncated resolve will trim — a generation
    /// prompt (`<|im_start|>assistant\n<think>\n` is 28 bytes), never dropped
    /// conversation content. A longer tail means the conversation's last
    /// message is not its last user message; that case falls back to
    /// `applyChatTemplate`.
    private static let maxTruncatedTailBytes = 128

    /// Verification-window width, in UTF-8 bytes: the first attempt, and the
    /// ceiling the ×4 ladder stops at.
    private static let initialWindowBytes = 256
    private static let maxWindowBytes = 16_384

    /// How many resolves between `Log.server` summaries.
    private static let summaryInterval = 256

    // MARK: - Hit path

    private enum HitOutcome {
        case resolved(Resolution)
        case failed(MissReason)
    }

    private func resolveHit(
        candidate: Entry,
        renderedBytes: [UInt8],
        chain: [String],
        toolsDigest: String,
        contextDigest: String,
        tokenizer: any Tokenizer
    ) -> HitOutcome {
        /// Whether any trim attempt produced a byte prefix at all — the
        /// discriminator between "the new render does not extend the old one"
        /// and "the seam is there but the junction never verified".
        var sawBytePrefix = false
        // C26: `prefixLength` walks back through the entry's stored bytes by
        // stripping each trimmed token's decoded bytes from the tail — decode
        // is per-token concatenation, so stripping reproduces exactly
        // `decode(prefixTokens)` — instead of decoding the whole prefix per
        // attempt (that decode dominated the hit path at long prefixes). The
        // trailing-bytes guard keeps any decode/strip inconsistency honest,
        // and the per-attempt prefix check is unchanged.
        var prefixLength = candidate.renderedBytes.count
        for k in 0...Self.maxTrimBack {
            guard candidate.tokens.count - k > 0 else { break }
            if k > 0 {
                let dropped = candidate.tokens[candidate.tokens.count - k]
                let droppedBytes = Self.utf8(ofToken: dropped, tokenizer: tokenizer)
                guard !droppedBytes.isEmpty,
                    Self.bytes(candidate.renderedBytes, endingAt: prefixLength, equal: droppedBytes)
                else {
                    return .failed(.renderNotExtended)
                }
                prefixLength -= droppedBytes.count
            }
            let prefixTokens = Array(candidate.tokens.dropLast(k))
            guard
                Self.bytes(
                    renderedBytes, startsWith: candidate.renderedBytes, count: prefixLength)
            else { continue }
            sawBytePrefix = true
            // An empty suffix at k>0 would mean the trimmed tokens decoded to
            // nothing — not trustworthy as `encode(rendered)`; keep trimming.
            guard let suffix = Self.suffixString(of: renderedBytes, from: prefixLength) else {
                continue
            }
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
                        renderedBytes: renderedBytes,
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
        return .failed(sawBytePrefix ? .junctionUnverified : .renderNotExtended)
    }

    /// The exactness arbiter. Re-encodes a window bracketing the
    /// cached/suffix seam — the last ≥C bytes of the cached prefix plus the
    /// first ≥C bytes of the suffix, C = `initialWindowBytes` enlarged ×4 up
    /// to `maxWindowBytes` on failure — and requires the identical token slice
    /// back. Any BPE merge spanning the cut changes the window's tokenization
    /// and is detected here; the check is an exact token-array equality, never
    /// a heuristic.
    private func verifyJunction(
        prefixTokens: [Int],
        suffixTokens: [Int],
        tokenizer: any Tokenizer
    ) -> Bool {
        var target = Self.initialWindowBytes
        while true {
            let a = trailingTokenCount(
                of: prefixTokens, coveringBytes: target, tokenizer: tokenizer)
            let b = leadingTokenCount(
                of: suffixTokens, coveringBytes: target, tokenizer: tokenizer)
            if a >= 1, b >= 1 {
                let span = Array(prefixTokens.suffix(a)) + Array(suffixTokens.prefix(b))
                // Decoding the span re-splits at the seam exactly: the prefix
                // side ends at a scalar boundary (the byte-prefix check
                // upstream guarantees a valid seam) and the suffix side
                // starts there, so `decode(span)` is the window text.
                let windowText = tokenizer.decode(tokenIds: span, skipSpecialTokens: false)
                let windowTokens = tokenizer.encode(text: windowText, addSpecialTokens: false)
                if windowTokens == span { return true }
            }
            let canGrow = a < prefixTokens.count || b < suffixTokens.count
            guard canGrow, target < Self.maxWindowBytes else { return false }
            target = min(target * 4, Self.maxWindowBytes)
            lock.withLock { stats.junctionWindowEnlargements += 1 }
        }
    }

    /// C27: the truncated resolve's exactness arbiter — the right-context
    /// counterpart to `verifyJunction`. The candidate tokens were encoded
    /// with the tail following them; the standalone encode this result is
    /// substituted for sees end-of-text at the cut. Re-encodes the truncated
    /// render's trailing ≥C bytes (C = `initialWindowBytes` enlarged ×4 up to
    /// `maxWindowBytes` on failure) and requires the exact token slice back:
    /// the span `candidateTokens.suffix(a)` covering the window must equal the
    /// standalone encode of its own decoded text. Any right-context effect at
    /// the cut (a pretoken alternative like `\s+(?!\S)` matching differently
    /// at end-of-text) changes the window's tokenization and is detected here.
    /// Left of the window the two encodes coincide by construction (see the
    /// file header for the pretoken argument).
    private func verifyCut(
        candidateTokens: [Int],
        truncatedBytes: [UInt8],
        tokenizer: any Tokenizer
    ) -> Bool {
        var target = Self.initialWindowBytes
        while true {
            let a = trailingTokenCount(
                of: candidateTokens, coveringBytes: target, tokenizer: tokenizer)
            if a >= 1 {
                let span = Array(candidateTokens.suffix(a))
                let windowText = tokenizer.decode(tokenIds: span, skipSpecialTokens: false)
                let windowBytes = Array(windowText.utf8)
                // The span's bytes must be the render's own tail: anything
                // else is a decode/trim inconsistency no enlargement fixes.
                guard
                    Self.bytes(truncatedBytes, endingAt: truncatedBytes.count, equal: windowBytes)
                else { return false }
                let windowTokens = tokenizer.encode(text: windowText, addSpecialTokens: false)
                if windowTokens == span { return true }
            }
            let canGrow = a < candidateTokens.count
            guard canGrow, target < Self.maxWindowBytes else { return false }
            target = min(target * 4, Self.maxWindowBytes)
            lock.withLock { stats.cutWindowEnlargements += 1 }
        }
    }

    /// A token count from the end of `tokens` whose decode covers at least
    /// `target` UTF-8 bytes, found by doubling — so the search costs O(log n)
    /// decodes rather than one per token. The result OVERSHOOTS the minimum
    /// such count (the doubling can step past it); that is deliberate, since
    /// the verifiers need a window at least this wide, not exactly this wide.
    /// Returns `tokens.count` when the whole list does not reach `target`.
    private func trailingTokenCount(
        of tokens: [Int],
        coveringBytes target: Int,
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

    /// `trailingTokenCount`'s mirror, counting from the front.
    private func leadingTokenCount(
        of tokens: [Int],
        coveringBytes target: Int,
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

    // MARK: - Store and template identity

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
    ///
    /// Memoized per model fingerprint. That is only sound because an unknown
    /// fingerprint never reaches this type: `RenderTokenSource` bypasses the
    /// cache rather than passing a synthetic key, so two different models can
    /// never collide on one memo slot (they would then share a probe hash and
    /// the template-mismatch check would pass vacuously).
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

    // MARK: - Logging

    /// Render through the layer-1 split, logging (and rethrowing) a throw.
    /// Every seam calls the resolves through `try?`, so a systematic render
    /// failure would otherwise be completely silent.
    private func render(
        _ rendering: any ChatTemplateRendering,
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?,
        label: String
    ) throws -> String {
        do {
            return try rendering.renderChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch {
            Log.server.warning(
                "render-token-cache \(label): render threw, caller falls back to "
                    + "applyChatTemplate/prepare — \(error.localizedDescription)")
            throw error
        }
    }

    private func logSummaryIfDue() {
        let due: Stats? = lock.withLock {
            resolutionsSinceSummary += 1
            guard resolutionsSinceSummary >= Self.summaryInterval else { return nil }
            resolutionsSinceSummary = 0
            return stats
        }
        guard let due else { return }
        Log.server.debug("render-token-cache \(Self.summary(of: due))")
    }

    private static func summary(of stats: Stats) -> String {
        let resolves = stats.hits + stats.repeats + stats.misses
        let hitRate = resolves > 0 ? Double(stats.hits + stats.repeats) / Double(resolves) : 0
        return "resolves=\(resolves) hitRate=\(String(format: "%.3f", hitRate)) "
            + "hits=\(stats.hits) repeats=\(stats.repeats) misses=\(stats.misses) "
            + "missReasons=\(histogram(stats.missReasons)) "
            + "trims=\(histogram(stats.trimHistogram)) "
            + "junctionFailures=\(stats.junctionFailures) "
            + "junctionWindowEnlargements=\(stats.junctionWindowEnlargements) "
            + "cutWindowEnlargements=\(stats.cutWindowEnlargements) "
            + "truncated=\(stats.truncatedHits)/\(stats.truncatedFallbacks) "
            + "truncatedFallbackReasons=\(histogram(stats.truncatedFallbackReasons)) "
            + "replaced=\(stats.replacedHits)/\(stats.replacedFallbacks) "
            + "replacedJunctionFailures=\(stats.replacedJunctionFailures) "
            + "replacedFallbackReasons=\(histogram(stats.replacedFallbackReasons))"
    }

    /// Deterministic rendering of a counter histogram (sorted keys) so log
    /// lines diff across runs.
    private static func histogram<Key: Comparable & CustomStringConvertible>(
        _ counts: [Key: Int]
    ) -> String {
        "["
            + counts.sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: " ") + "]"
    }
}
