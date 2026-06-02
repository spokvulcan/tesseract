import Foundation
import MLXLMCommon

/// The GPU-free decision the **Leaf Admission Builder** emits for one leaf
/// **Snapshot Admission**: which token path to capture, and from where. A
/// *total* value — every `HTTPLeafStoreMode` maps to exactly one case — so the
/// actor's post-generation `switch` is exhaustive and a future leaf mode
/// surfaces as a compile error rather than a silently missed branch.
nonisolated enum LeafCapturePlan: Sendable {
    /// *directLeaf*: snapshot the live final KV cache at `storedTokens.count`.
    case liveCache(storedTokens: [Int])
    /// *directTool* / *canonical*: restore `boundary`, reprefill the residual
    /// `storedTokens[boundary.tokenOffset...]`, capture a leaf at `storedTokens.count`.
    case fromBoundary(boundary: HybridCacheSnapshot, storedTokens: [Int])
    /// A decidable skip — the routing ruled out a capture before any Metal work.
    case skip(reason: LeafSkipReason)
}

/// The typed vocabulary of **decidable** leaf-store skips: the failures the
/// GPU-free builder rules on from the tokenizer and the resolved boundary
/// alone. Each case carries the diagnostic payload the actor reproduces in
/// today's `logSkip` fields, so the decidable-skip vocabulary lives in one
/// typed place instead of string literals scattered across the capture
/// helpers. The live-`finalCache` skips (`no-final-cache`, `no-reusable-cache-state`,
/// `normalization-trim`, `unsupported-cache-type`, `invalid-path`,
/// `capturedThenEvicted`) and the `intervention` guard stay actor-side and are
/// not represented here.
nonisolated enum LeafSkipReason: Sendable, Equatable {
    /// The reusable-prefix probe's chat-template render threw.
    case tokenizationFailed(error: String)
    /// The probe found no usable common prefix — no distinct continuation boundary.
    case probeDivergence
    /// *directTool*: no request-local transient boundary snapshot to restore from.
    case noTransientBoundary
    /// *canonical*: neither the transient boundary nor **Snapshot Resolution**
    /// yielded a usable restore boundary.
    case noResolvedBoundary(canonicalLen: Int)
    /// *directTool*: the residual is empty — the stored path ends at or before
    /// the boundary.
    case storedAtOrBeforeBoundary(storedLen: Int, boundaryOffset: Int)
    /// *canonical*: the canonical path ends at or before the boundary. Defensive:
    /// boundary selection already guarantees `boundaryOffset < canonicalLen`.
    case canonicalAtOrBeforeBoundary(canonicalLen: Int, boundaryOffset: Int)
    /// *canonical*: the canonical prefix is longer than the stored path — the
    /// render disagreement the canonical leaf exists to sidestep.
    case canonicalLongerThanStored(canonicalLen: Int, storedLen: Int)
}

/// The GPU-free routing decisions for capturing a **leaf** Snapshot Admission:
/// which token path a future continuation can hydrate. Tokenizer-affine, no
/// live KV cache — the actor executes the Metal capture/`admit` from the
/// decision this produces.
///
/// Today this owns the **reusable-prefix probes** shared by the canonical-user
/// and tool-continuation leaf modes. Leaf-mode selection still lives on
/// `LLMActor.selectHTTPLeafStoreMode` (already a tested pure function), and the
/// Metal capture stays in the actor.
nonisolated enum LeafAdmissionBuilder {

    /// The synthetic continuation a reusable-prefix probe appends to discover
    /// the shared token path. A stop turn re-renders as a user continuation; a
    /// tool-call turn re-renders as a tool-result continuation.
    enum Continuation: Sendable {
        case userTurn
        case toolResult

        var probeMessage: [String: any Sendable] {
            switch self {
            case .userTurn:
                ["role": Chat.Message.Role.user.rawValue, "content": "Aqkz_strip_probe"]
            case .toolResult:
                ["role": Chat.Message.Role.tool.rawValue, "content": "Aqkz_tool_probe"]
            }
        }
    }

    /// The reusable token prefix for the stored assistant turn: the shared
    /// prefix between the turn's isolated render (no generation prompt) and the
    /// render that appends a single synthetic continuation. That shared prefix
    /// is the exact token path the immediate continuation can hydrate.
    ///
    /// Returns `nil` when the probe diverges — no common prefix, or the prefix
    /// is not strictly shorter than the continuation (the appended turn added
    /// no tokens, so there is no distinct boundary to align to).
    static func reusablePrefix(
        continuation: Continuation,
        storedConversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        tokenizer: any Tokenizer
    ) throws -> [Int]? {
        let baseMessages = storedConversation.promptMessages
        let storedTokens = try tokenizer.applyChatTemplate(
            messages: baseMessages,
            tools: toolSpecs,
            additionalContext: ["add_generation_prompt": false]
        )
        let continuationTokens = try tokenizer.applyChatTemplate(
            messages: baseMessages + [continuation.probeMessage],
            tools: toolSpecs,
            additionalContext: ["add_generation_prompt": false]
        )

        let common = zip(storedTokens, continuationTokens).prefix { $0 == $1 }.count
        guard common > 0, common <= storedTokens.count, common < continuationTokens.count else {
            return nil
        }
        return Array(continuationTokens[0..<common])
    }

    /// The whole GPU-free leaf-capture routing decision for one stored turn.
    ///
    /// Takes the already-selected `mode` (mode selection stays the separately
    /// tested `selectHTTPLeafStoreMode`), the re-tokenized stored token path, the
    /// mode-relevant transient boundary snapshot, a tokenizer, and a
    /// `resolveBoundary` closure-**peer** for the canonical fallback. Runs the
    /// reusable-prefix probe, chooses the boundary source, applies the
    /// offset-guard arithmetic, and emits a `LeafCapturePlan`. No live KV cache,
    /// no Metal — the actor executes the capture/`admit` from the decision.
    static func plan(
        mode: HTTPLeafStoreMode,
        storedConversation: HTTPPrefixCacheConversation,
        storedTokens: [Int],
        toolSpecs: [ToolSpec]?,
        transientBoundary: HybridCacheSnapshot?,
        tokenizer: any Tokenizer,
        resolveBoundary: @Sendable ([Int]) async -> HybridCacheSnapshot?
    ) async -> LeafCapturePlan {
        switch mode {
        case .directLeaf:
            return .liveCache(storedTokens: storedTokens)

        case .directToolLeaf:
            // Tool-call turns are reused by the immediate tool-result
            // continuation: restore the request-local boundary and reprefill
            // the tool-continuation render.
            guard let transientBoundary else {
                return .skip(reason: .noTransientBoundary)
            }
            let toolTokens: [Int]?
            do {
                toolTokens = try reusablePrefix(
                    continuation: .toolResult,
                    storedConversation: storedConversation,
                    toolSpecs: toolSpecs,
                    tokenizer: tokenizer
                )
            } catch {
                return .skip(reason: .tokenizationFailed(error: error.localizedDescription))
            }
            guard let toolTokens else {
                return .skip(reason: .probeDivergence)
            }
            guard toolTokens.count > transientBoundary.tokenOffset else {
                return .skip(reason: .storedAtOrBeforeBoundary(
                    storedLen: toolTokens.count,
                    boundaryOffset: transientBoundary.tokenOffset
                ))
            }
            return .fromBoundary(boundary: transientBoundary, storedTokens: toolTokens)

        case .canonicalUserLeaf:
            // Thinking templates may re-render non-latest assistant turns
            // differently from the just-generated form: capture the canonical
            // render under the token path a future non-latest render will see.
            let canonicalTokens: [Int]?
            do {
                canonicalTokens = try reusablePrefix(
                    continuation: .userTurn,
                    storedConversation: storedConversation,
                    toolSpecs: toolSpecs,
                    tokenizer: tokenizer
                )
            } catch {
                return .skip(reason: .tokenizationFailed(error: error.localizedDescription))
            }
            guard let canonicalTokens else {
                return .skip(reason: .probeDivergence)
            }

            // Prefer the request-local transient boundary when it sits strictly
            // before the canonical path; otherwise fall back through the
            // injected **Snapshot Resolution** closure-peer.
            let boundary: HybridCacheSnapshot
            if let transientBoundary, transientBoundary.tokenOffset < canonicalTokens.count {
                boundary = transientBoundary
            } else if let resolved = await resolveBoundary(canonicalTokens),
                      resolved.tokenOffset > 0,
                      resolved.tokenOffset < canonicalTokens.count {
                boundary = resolved
            } else {
                return .skip(reason: .noResolvedBoundary(canonicalLen: canonicalTokens.count))
            }

            // Defensive: boundary selection above already guarantees
            // `boundary.tokenOffset < canonicalTokens.count`, so this never fires —
            // kept to mirror today's guard and document the invariant.
            guard canonicalTokens.count > boundary.tokenOffset else {
                return .skip(reason: .canonicalAtOrBeforeBoundary(
                    canonicalLen: canonicalTokens.count,
                    boundaryOffset: boundary.tokenOffset
                ))
            }
            guard canonicalTokens.count <= storedTokens.count else {
                return .skip(reason: .canonicalLongerThanStored(
                    canonicalLen: canonicalTokens.count,
                    storedLen: storedTokens.count
                ))
            }
            return .fromBoundary(boundary: boundary, storedTokens: canonicalTokens)
        }
    }
}
