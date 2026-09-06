//
//  LiveLeafCapture.swift
//  tesseract
//
//  The **Live Leaf Capture** decision: whether the finished turn's leaf can
//  be taken straight from the live decode cache instead of restored-and-
//  re-prefilled from a boundary snapshot.
//
//  Why the boundary path exists at all: a leaf must key on the template's
//  re-render of the turn — the token path the client's next request will
//  walk — and that re-render is not guaranteed to equal the ids the model
//  emitted (tool calls are re-rendered from parsed arguments, content is
//  trimmed, BPE can emit non-canonical splits, strip-by-default templates
//  drop prior thinking). The live KV cache holds the *emitted* path, and the
//  recurrent (GDN/Mamba) state cannot be rewound to an arbitrary offset, so
//  the safe general answer was to restore a boundary and recompute the
//  canonical residual — at prefill speed, for every generated token.
//
//  On an append-stable render (Qwen3.8 with preserved thinking, every tool
//  stretch under Qwen3-family templates) the re-render *is* the emitted path
//  plus a couple of glue tokens, so that recomputation buys nothing: measured
//  7.3 ms per generated token of post-EOS stall, 47 s on a 6.4k-token turn.
//  This decision proves the equality per turn (prompt key path + fed ids is
//  a prefix of the canonical stored path) and, when it holds, hands the drive
//  a zero-prefill capture of the live cache at its fed offset. Any mismatch
//  falls back to the boundary path unchanged — correctness never rests on
//  the flag, only on the comparison.
//
//  Pure and GPU-free: plain token arrays in, a decision out, so the rules are
//  unit-tested without a model (`LiveLeafCaptureTests`).
//

import Foundation

nonisolated enum LiveLeafCapture {

    /// Where the finished turn's leaf comes from.
    enum Decision: Equatable, Sendable {
        /// Capture the live final cache at `offset` and admit it under the
        /// stored path's first `offset` tokens. No prefill.
        case live(offset: Int)
        /// Restore the boundary snapshot and re-prefill the canonical
        /// residual — today's path — for the given reason.
        case boundary(FallbackReason)
    }

    /// Why the live cache could not be used. Every case carries the numbers
    /// its skip record prints (`LeafStorePhase.liveFallbackLog`), so a
    /// divergence on a render believed append-stable is diagnosable from the
    /// log alone.
    enum FallbackReason: Equatable, Sendable {
        /// A thinking-safeguard continuation swapped the raw generation; the
        /// registered final cache is the cancelled phase's, not the turn's.
        case intervened
        /// Image placeholders make key-space and render-space differ; the
        /// live comparison is defined over identity key spaces only.
        case nonIdentityKeySpace
        /// The loop recorded no fed ids (an empty turn, or an iterator that
        /// bypassed the recorder).
        case noGeneratedTokens
        /// The cache's reported offset does not sit inside the live path
        /// (`promptCount < offset <= liveCount`) — the loop's accounting and
        /// the cache disagree, so nothing can be trusted.
        case cacheOffsetOutsideLivePath(cacheOffset: Int, promptCount: Int, liveCount: Int)
        /// The fed path is longer than the canonical stored path — the render
        /// dropped emitted tokens (whitespace normalization, a stripped
        /// span); the live state past the stored end has no key.
        case liveLongerThanStored(cacheOffset: Int, storedLen: Int)
        /// First position where the fed id and the re-rendered id differ,
        /// with a few ids of context on each side so the *kind* of
        /// divergence (a BPE re-split of identical text, a trimmed
        /// newline, a re-rendered tool call) is readable from the log.
        case divergence(
            offset: Int, liveToken: Int, storedToken: Int,
            liveContext: [Int], storedContext: [Int])

        /// Ids kept on each side of a divergence in the log fields.
        static let contextRadius = 4
    }

    /// Decide for one finished turn.
    ///
    /// - `promptKeyPath`: the request's **Cache Key Path** (the prompt as
    ///   prefilled, identity key space).
    /// - `generatedTokens`: every id the decode loop fed past the prompt, in
    ///   order, stop token included (`GeneratedTokenRecorder`).
    /// - `cacheOffset`: the live final cache's reported token count. It may
    ///   trail `promptKeyPath.count + generatedTokens.count` by the
    ///   iterator's unfed tail (DFlash2's bonus token has no cache entry
    ///   yet); the leaf is captured at the cache's own offset, never past it.
    /// - `storedTokens`: the canonical stored path the boundary plan would
    ///   admit under (the **Leaf Admission Builder**'s probe result).
    static func decide(
        promptKeyPath: [Int],
        generatedTokens: [Int],
        cacheOffset: Int,
        storedTokens: [Int],
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
        guard cacheOffset <= storedTokens.count else {
            return .boundary(
                .liveLongerThanStored(cacheOffset: cacheOffset, storedLen: storedTokens.count))
        }

        // The prompt prefix is the same render on both sides by construction,
        // but the comparison is the proof — check every fed position.
        func liveToken(_ index: Int) -> Int {
            index < promptCount ? promptKeyPath[index] : generatedTokens[index - promptCount]
        }
        for index in 0..<cacheOffset {
            let live = liveToken(index)
            let stored = storedTokens[index]
            if live != stored {
                let radius = FallbackReason.contextRadius
                let window = max(0, index - radius)..<min(liveCount, index + radius + 1)
                return .boundary(
                    .divergence(
                        offset: index, liveToken: live, storedToken: stored,
                        liveContext: window.map(liveToken),
                        storedContext: Array(
                            storedTokens[window.clamped(to: 0..<storedTokens.count)])
                    ))
            }
        }
        return .live(offset: cacheOffset)
    }
}
