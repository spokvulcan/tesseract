import Foundation
import MLXLMCommon

/// The offline **Canonical-Echo Fidelity** harness (PRD #94): proves that the
/// boundary-leaf token paths the server derives when a turn completes — the
/// canonical-user / tool-continuation reusable prefixes, and the
/// **Speculative Canonical Prefill** target path — are token-identical
/// prefixes of what the client actually sends next. A leaf admitted on an
/// infidel render sits off every future request's path and is dead weight; a
/// speculated spine built from one is worse, because it spends idle GPU on a
/// branch nobody will ever walk.
///
/// The harness replays a recorded session (the request JSONs
/// `HTTPRequestLogger` writes) through the same normalization, reasoning
/// repair, and probe machinery the live server runs, then diffs each derived
/// leaf path against the *next* request's actual render. Pure and
/// tokenizer-affine — no GPU, no cache state. The corpus stays outside the
/// repo (it contains user project content); see `docs/testing.md` for
/// invocation.
nonisolated enum CanonicalEchoFidelity {

    /// The completed-turn boundary one adjacent request pair (N, N+1) of a
    /// session implies: request N finished a turn, request N+1 echoes it.
    struct Boundary: Sendable {
        enum Kind: String, Sendable {
            /// Stop answer under a thinking template — the canonical-user
            /// leaf and the speculation seed both derive from it.
            case canonicalUser
            /// Tool-call turn whose next request continued the stretch
            /// with tool results — the directTool leaf derives from it.
            case toolContinuation
            /// Tool-call turn whose next request instead carries a later
            /// *real user message* (the post-interrupt steering message,
            /// or a multi-turn catch-up): the **Think-Strip Rewind**
            /// future. The directTool leaf legitimately serves a future
            /// that never arrived, so the gate checks the rewind geometry
            /// instead — the strip-floor path (`.userTurn` probe, ending
            /// at the stretch base — the **Chain-Prefix Restore** floor)
            /// and the `futureSharedPrefix` spine (the **Stretch
            /// Abandonment** speculation target) must both be faithful
            /// prefixes of the rewound render.
            case interruptRewind
        }

        let kind: Kind
        /// Index of the *generating* request within the session walk.
        let requestIndex: Int
        /// Whether the client's echo carried reasoning. When it does not and
        /// no repair recovered it, the reconstruction of the server's stored
        /// conversation is incomplete and the verdict is advisory.
        let echoHasReasoning: Bool
    }

    /// One derived path checked against the next request's real render.
    enum Verdict: Sendable {
        /// The derived path is a token-identical prefix of the next render.
        case faithful(pathLength: Int)
        /// The derived path forks off the next render before its end — the
        /// admitted leaf would sit past a divergence, unusable for exactly
        /// the future it was built to serve. Carries decoded windows around
        /// the fork for diagnosis.
        case mismatch(pathLength: Int, matched: Int, derivedTail: String, nextTail: String)
        /// The probe declined (divergence/translation/tokenization) — the
        /// server would have skipped this leaf, so there is nothing to hold
        /// faithful.
        case noPath(reason: String)

        var isMismatch: Bool {
            if case .mismatch = self { return true }
            return false
        }
    }

    struct BoundaryReport: Sendable {
        let boundary: Boundary
        /// The boundary-leaf path (canonical-user or tool-continuation).
        let leaf: Verdict
        /// Stop answers only: the `futureSharedPrefix` speculation target
        /// path — the gate for abandonment-seeded speculation.
        let speculation: Verdict?

        var hasMismatch: Bool { leaf.isMismatch || (speculation?.isMismatch ?? false) }
    }

    /// Tokens of decoded context shown on each side of a fork.
    private static let mismatchContextTokens = 48

    /// Check the boundary one adjacent request pair implies.
    ///
    /// `previous`/`next` are the server-normalized conversations of requests
    /// N and N+1; `echo` is N+1's echo of the turn N generated (the message
    /// at `previous.messages.count`). The reconstruction uses the echo as
    /// the stored turn — exact whenever the client echoes faithfully, which
    /// is itself part of what the prefix diff verifies.
    static func check(
        previous: HTTPPrefixCacheConversation,
        echo: HTTPPrefixCacheMessage,
        next: HTTPPrefixCacheConversation,
        probeToolSpecs: [ToolSpec]?,
        nextToolSpecs: [ToolSpec]?,
        requestIndex: Int,
        tokenizer: any Tokenizer
    ) -> BoundaryReport {
        let stored = previous.appendingAssistant(echo)
        // The kind is decided by the *actual* future: a tool-call turn whose
        // appended tail carries a later real user message renders the whole
        // stretch think-stripped (`last_query_index` moved), so the tool
        // leaf serves a future that never arrived — check the rewind paths.
        let appendedTail = next.messages.dropFirst(previous.messages.count + 1)
        let kind: Boundary.Kind
        if echo.toolCalls.isEmpty {
            kind = .canonicalUser
        } else if appendedTail.contains(where: { $0.role == .user }) {
            kind = .interruptRewind
        } else {
            kind = .toolContinuation
        }
        let boundary = Boundary(
            kind: kind,
            requestIndex: requestIndex,
            echoHasReasoning: echo.reasoning != nil
        )

        let nextRender: [Int]
        do {
            nextRender = try tokenizer.applyChatTemplate(
                messages: next.promptMessages,
                tools: nextToolSpecs,
                additionalContext: ["add_generation_prompt": false]
            )
        } catch {
            let verdict = Verdict.noPath(reason: "next-render-failed: \(error)")
            return BoundaryReport(boundary: boundary, leaf: verdict, speculation: nil)
        }

        // The boundary-leaf path. For `interruptRewind` this is the
        // strip-floor path: LCP(render(stored), render(stored + user
        // probe)) ends right past the stretch base's assistant header —
        // the deepest offset the rewound future shares with the stored
        // spine, i.e. the **Chain-Prefix Restore** floor (ADR-0012).
        let leafVerdict = verdict(
            of: {
                try LeafAdmissionBuilder.reusablePrefix(
                    continuation: kind == .toolContinuation ? .toolResult : .userTurn,
                    storedConversation: stored,
                    toolSpecs: probeToolSpecs,
                    tokenizer: tokenizer,
                    keySpace: .identity()
                )
            },
            against: nextRender,
            tokenizer: tokenizer
        )

        // The speculation target path predicts a *user* turn directly
        // after the echo; it has an exact counterpart to diff against only
        // when the next request appended one — a stop answer's next user
        // turn, or a steering message sent before any tool result landed,
        // where it covers the whole think-stripped stretch (the **Stretch
        // Abandonment** speculation spine). When the appended tail begins
        // with tool results (a pure continuation, or an interrupt that
        // kept the already-run results before the steering message), the
        // spine's trailing user-header tokens legitimately walk a
        // different branch — observed forks sit 4 tokens from the spine
        // end, at the `<|im_start|>user` BPE seam — so there is nothing
        // to hold faithful past the floor.
        var speculationVerdict: Verdict?
        if kind != .toolContinuation, appendedTail.first?.role == .user {
            speculationVerdict = verdict(
                of: {
                    try LeafAdmissionBuilder.futureSharedPrefix(
                        storedConversation: stored,
                        toolSpecs: probeToolSpecs,
                        tokenizer: tokenizer,
                        keySpace: .identity()
                    )
                },
                against: nextRender,
                tokenizer: tokenizer
            )
        }

        return BoundaryReport(
            boundary: boundary,
            leaf: leafVerdict,
            speculation: speculationVerdict
        )
    }

    private static func verdict(
        of probe: () throws -> Result<[Int], CacheKeySpace.TranslationFailure>?,
        against nextRender: [Int],
        tokenizer: any Tokenizer
    ) -> Verdict {
        let path: [Int]
        do {
            switch try probe() {
            case .none:
                return .noPath(reason: "probe-divergence")
            case .failure(let failure):
                return .noPath(reason: "render-translation-failed: \(failure)")
            case .success(let tokens):
                path = tokens
            }
        } catch {
            return .noPath(reason: "tokenization-failed: \(error)")
        }

        let matched = zip(path, nextRender).prefix { $0 == $1 }.count
        if matched >= path.count {
            return .faithful(pathLength: path.count)
        }
        let window = mismatchContextTokens
        let derivedTail = tokenizer.decode(
            tokenIds: Array(path[max(0, matched - window)..<min(path.count, matched + window)]),
            skipSpecialTokens: false
        )
        let nextTail = tokenizer.decode(
            tokenIds: Array(
                nextRender[max(0, matched - window)..<min(nextRender.count, matched + window)]),
            skipSpecialTokens: false
        )
        return .mismatch(
            pathLength: path.count,
            matched: matched,
            derivedTail: derivedTail,
            nextTail: nextTail
        )
    }
}

// MARK: - Session walk

extension CanonicalEchoFidelity {

    /// One recorded request, already decoded from an `HTTPRequestLogger`
    /// recording: the raw OpenAI messages plus tool definitions.
    struct RecordedRequest: Sendable {
        let messages: [OpenAI.ChatMessage]
        let tools: [OpenAI.ToolDefinition]?

        init(messages: [OpenAI.ChatMessage], tools: [OpenAI.ToolDefinition]?) {
            self.messages = messages
            self.tools = tools
        }
    }

    struct SessionReport: Sendable {
        let sessionAffinity: String?
        let boundaries: [BoundaryReport]
        /// Pairs that could not be reduced to a boundary (history edit,
        /// ineligible conversation, no clean echo) — reported, never failed.
        let skipped: [(requestIndex: Int, reason: String)]

        var mismatchCount: Int { boundaries.count(where: \.hasMismatch) }
    }

    /// Walk one session's requests in order, mirroring the live pipeline:
    /// reasoning repair → normalization → boundary reduction → probe diff.
    /// The repair store is threaded exactly as `CompletionHandler` does it,
    /// so turns whose reasoning the client drops are reconstructed from the
    /// session replay record rather than reported as spurious mismatches.
    @MainActor
    static func walkSession(
        requests: [RecordedRequest],
        sessionAffinity: String?,
        modelID: String,
        tokenizer: any Tokenizer
    ) async -> SessionReport {
        let repairStore = HTTPPrefixCacheSessionReplayStore()
        var boundaries: [BoundaryReport] = []
        var skipped: [(requestIndex: Int, reason: String)] = []

        var previous: (conversation: HTTPPrefixCacheConversation, toolSpecs: [ToolSpec]?)?

        for (index, request) in requests.enumerated() {
            let repaired = await repairStore.repair(
                messages: request.messages,
                sessionAffinity: sessionAffinity,
                modelID: modelID,
                visionMode: false
            )
            let normalized = MessageConverter.normalizeRequest(
                repaired.messages,
                tools: request.tools
            )
            guard case .eligible(let conversation) = normalized.prefixCacheEligibility else {
                skipped.append((index, "ineligible: \(normalized.prefixCacheEligibility)"))
                previous = nil
                continue
            }
            let toolSpecs = LLMActor.canonicalizeToolSpecs(
                MessageConverter.convertToolDefinitions(request.tools)
            )

            if let (previousConversation, previousToolSpecs) = previous {
                let echoIndex = previousConversation.messages.count
                if previousConversation.isPrefix(of: conversation),
                    echoIndex < conversation.messages.count,
                    conversation.messages[echoIndex].role == .assistant
                {
                    let echo = conversation.messages[echoIndex]
                    boundaries.append(
                        check(
                            previous: previousConversation,
                            echo: echo,
                            next: conversation,
                            probeToolSpecs: previousToolSpecs,
                            nextToolSpecs: toolSpecs,
                            requestIndex: index - 1,
                            tokenizer: tokenizer
                        ))
                    // Mirror the live server: the completed turn enters the
                    // replay record so later requests that drop its
                    // reasoning get repaired, not misdiagnosed.
                    await repairStore.record(
                        sessionAffinity: sessionAffinity,
                        modelID: modelID,
                        visionMode: false,
                        assistantMessage: echo
                    )
                } else {
                    skipped.append((index, "no-clean-echo-extension"))
                }
            }

            previous = (conversation, toolSpecs)
        }

        return SessionReport(
            sessionAffinity: sessionAffinity,
            boundaries: boundaries,
            skipped: skipped
        )
    }

    /// Human-readable summary, one line per unfaithful or skipped boundary.
    static func renderText(_ report: SessionReport) -> String {
        var lines: [String] = []
        let faithfulLeaves = report.boundaries.count { !$0.leaf.isMismatch }
        lines.append(
            "session=\(report.sessionAffinity ?? "-") boundaries=\(report.boundaries.count) "
                + "faithful=\(faithfulLeaves) mismatches=\(report.mismatchCount) "
                + "skipped=\(report.skipped.count)"
        )
        for boundaryReport in report.boundaries where boundaryReport.hasMismatch {
            let boundary = boundaryReport.boundary
            lines.append("MISMATCH request#\(boundary.requestIndex) kind=\(boundary.kind.rawValue)")
            for (label, verdict) in [
                ("leaf", boundaryReport.leaf),
                ("speculation", boundaryReport.speculation ?? .faithful(pathLength: 0)),
            ] {
                if case .mismatch(let length, let matched, let derived, let next) = verdict {
                    lines.append("  \(label): fork at \(matched)/\(length)")
                    lines.append("  derived: …\(derived)")
                    lines.append("  next:    …\(next)")
                }
            }
        }
        for (index, reason) in report.skipped {
            lines.append("skipped request#\(index): \(reason)")
        }
        return lines.joined(separator: "\n")
    }
}
