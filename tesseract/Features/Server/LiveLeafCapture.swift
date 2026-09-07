//
//  LiveLeafCapture.swift
//  tesseract
//
//  The **Live Leaf Capture** eligibility: whether the finished turn's leaf
//  is taken straight from the live decode cache, at the cache's own offset,
//  under the **Emitted Path** — the **Leaf Store** fast path of ADR-0063
//  (decisions 10 to 13) — or whether the turn keeps the boundary path
//  (restore a boundary snapshot, re-prefill the canonical residual).
//
//  Under ADR-0062 this decision proved, per turn and per token, that the
//  fed path equalled the template's canonical re-render before trusting the
//  live cache. The Emitted Path Index inverts the reference: for a turn the
//  server generated, the ids the model fed are the truth and the next
//  request resolves to them, so there is nothing to compare — the leaf is
//  keyed on the fed path by construction. What survives of the old decision
//  is its structural eligibility: an intervened turn (the registered final
//  cache is the cancelled phase's), a non-identity key space (image
//  placeholders make key space and model input differ), no fed ids, and a
//  cache offset outside the live path (the loop and the cache disagree).
//  Each keeps its log reason. One render rule joins them: a think-stripping
//  template at a new-user-message boundary will not render the turn
//  verbatim for the next request, so ADR-0009's boundary path and
//  speculative seed remain its answer (decision 11).
//
//  Pure and GPU-free: plain values in, a decision out, so the rules are
//  unit-tested without a model (`LeafStoreFastPathTests`).
//

import Foundation

nonisolated enum LiveLeafCapture {

    /// Where the finished turn's leaf comes from.
    enum Decision: Equatable, Sendable {
        /// Capture the live final cache at `offset` and admit it under the
        /// live path's first `offset` ids (`livePath`). No prefill.
        case live(offset: Int)
        /// Restore the boundary snapshot and re-prefill the canonical
        /// residual — the pre-ADR-0063 path — for the given reason.
        case boundary(FallbackReason)
    }

    /// Why the live cache is not the leaf. Every case carries the numbers
    /// its skip record prints (`LeafStorePhase.liveFallbackLog`) and names
    /// the boundary reason the `leafStore` event reports.
    enum FallbackReason: Equatable, Sendable {
        /// A thinking-safeguard continuation swapped the raw generation; the
        /// registered final cache is the cancelled phase's, not the turn's.
        /// Kept on the boundary path and counted (decision 11).
        case intervened
        /// Image placeholders make key-space and render-space differ; the
        /// fast path is defined over identity key spaces only.
        case nonIdentityKeySpace
        /// The loop recorded no fed ids (an empty turn, or an iterator that
        /// bypassed the recorder).
        case noGeneratedTokens
        /// The cache's reported offset does not sit inside the live path
        /// (`promptCount < offset <= liveCount`) — the loop's accounting and
        /// the cache disagree, so nothing can be trusted.
        case cacheOffsetOutsideLivePath(cacheOffset: Int, promptCount: Int, liveCount: Int)
        /// A stop-finish turn under a template that strips the thinking of
        /// earlier turns once a new user message arrives: the next request
        /// renders this turn differently from what the model fed, so the
        /// canonical-user leaf is synthesized from the boundary as before
        /// and the ADR-0009 seed extends it.
        case thinkStrippingUserBoundary
    }

    /// Decide for one finished turn.
    ///
    /// - `mode`: the selected leaf-store mode (`selectHTTPLeafStoreMode`).
    /// - `preservesThinking`: whether the request rendered under the
    ///   **Preserve-Thinking Render**, which keeps every turn verbatim.
    /// - `promptKeyPath`: the request's **Cache Key Path** (the prompt as
    ///   prefilled, identity key space).
    /// - `generatedTokens`: every id the decode loop fed past the prompt, in
    ///   order, stop token included (`GeneratedTokenRecorder`).
    /// - `cacheOffset`: the live final cache's reported token count. It may
    ///   trail `promptKeyPath.count + generatedTokens.count` by the
    ///   iterator's unfed tail (DFlash2's bonus token has no cache entry
    ///   yet); the leaf is captured at the cache's own offset, never past it.
    ///
    /// The structural guards are checked first, in the order ADR-0062
    /// logged them, so an intervened or image-bearing turn names that
    /// reason whatever the render; the render rule comes last.
    static func decide(
        mode: HTTPLeafStoreMode,
        preservesThinking: Bool,
        promptKeyPath: [Int],
        generatedTokens: [Int],
        cacheOffset: Int,
        intervened: Bool,
        keySpaceIsIdentity: Bool
    ) -> Decision {
        if intervened { return .boundary(.intervened) }
        guard keySpaceIsIdentity else { return .boundary(.nonIdentityKeySpace) }
        guard !generatedTokens.isEmpty else { return .boundary(.noGeneratedTokens) }

        let promptCount = promptKeyPath.count
        let liveCount = promptCount + generatedTokens.count
        guard cacheOffset > promptCount, cacheOffset <= liveCount else {
            return .boundary(
                .cacheOffsetOutsideLivePath(
                    cacheOffset: cacheOffset, promptCount: promptCount, liveCount: liveCount))
        }
        // Every tool-stretch turn renders verbatim under the Qwen-family
        // templates, and every turn does under the preserve-thinking render;
        // only a stop-finish turn under a think-stripping template is
        // re-rendered by the next user message.
        if mode == .canonicalUserLeaf, !preservesThinking {
            return .boundary(.thinkStrippingUserBoundary)
        }
        return .live(offset: cacheOffset)
    }

    /// The path a live leaf is admitted under: the prompt key path plus the
    /// fed ids, cut at the cache offset (an unfed bonus token past it has no
    /// cache entry and is left for the next request to prefill).
    static func livePath(promptKeyPath: [Int], generatedTokens: [Int], offset: Int) -> [Int] {
        var path = promptKeyPath
        path.reserveCapacity(offset)
        path.append(contentsOf: generatedTokens.prefix(max(0, offset - promptKeyPath.count)))
        return Array(path.prefix(offset))
    }
}
