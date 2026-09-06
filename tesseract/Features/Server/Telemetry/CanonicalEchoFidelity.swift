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
        /// The Emitted Path Index's account of this boundary when the walk
        /// learns one (ADR-0063).
        var emittedPath: EmittedPathVerdict?

        var hasMismatch: Bool { leaf.isMismatch || (speculation?.isMismatch ?? false) }
    }

    // MARK: - Emitted Path learning (ADR-0063, ticket #475)

    /// The private index an offline walk learns and resolves against: each
    /// echoed turn registers its path exactly as the Leaf Store would (the
    /// emitted ids simulated as the canonical encode of the stored render
    /// past request N's prompt), and request N+1's render resolves against
    /// it through the same Conversation Render verbs the server runs.
    struct EmittedPathLearning: Sendable {
        let index: EmittedPathIndex
        let fingerprint: String
        let toolCallFormat: ToolCallFormat
        /// Whether the template's generation prompt opens a `<think>` block
        /// (the model identity's `promptStartsThinking`).
        let promptStartsThinking: Bool

        init(
            fingerprint: String, toolCallFormat: ToolCallFormat, promptStartsThinking: Bool,
            byteBudget: Int = EmittedPathIndex.defaultByteBudget
        ) {
            self.index = EmittedPathIndex(byteBudget: byteBudget)
            self.fingerprint = fingerprint
            self.toolCallFormat = toolCallFormat
            self.promptStartsThinking = promptStartsThinking
        }
    }

    struct EmittedPathVerdict: Sendable {
        /// `registered`, a `EmittedPathRegistration.SkipReason` raw value,
        /// or `promptNotTokenPrefix` when request N's prompt is not a token
        /// prefix of the stored render — the junction merge the Live Leaf
        /// Capture would have refused as a divergence, so no live-stored
        /// turn is simulated for it.
        let registration: String
        let pathLength: Int?
        /// Request N+1's resolve (the harness renders it as the leaf-store
        /// continuation spelling — same bytes through the last marker as the
        /// request edge, so the same hit): the indexed prefix on a hit.
        let nextIndexedPrefix: Int?
        let nextMissReason: String?
        /// Shadow differences over every resolve this boundary ran.
        let shadowDifferences: Int

        var registered: Bool { registration == "registered" }
    }

    struct EmittedPathSessionSummary: Sendable, Equatable {
        var boundaries = 0
        var registered = 0
        /// Registration outcome (other than `registered`) → count.
        var registrationSkips: [String: Int] = [:]
        /// Registered boundaries whose next request resolved with a
        /// non-zero indexed prefix.
        var nextResolved = 0
        var nextMisses: [String: Int] = [:]
        var shadowDifferences = 0
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
        tokenizer: any Tokenizer,
        learning: EmittedPathLearning? = nil
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

        // Replay renders are always uncached — no live entry to resolve
        // against, so every render runs in full through the same verbs the
        // server uses. A learning walk hands the same renders a private
        // Emitted Path Index: the stored render registers the turn (and is
        // carried as the probe's base render, as the Leaf Store carries it),
        // and request N+1's render resolves against it — no render is run
        // twice for the index's sake.
        let telemetry = learning.map { _ in EmittedPathRequestTelemetry(diagnostics: nil) }
        var probeRender = ConversationRender.uncached(
            tokenizer: tokenizer, toolSpecs: probeToolSpecs,
            emittedPathIndex: learning?.index, emittedPathFingerprint: learning?.fingerprint,
            emittedPathTelemetry: telemetry
        )
        var learned: (registration: String, pathLength: Int?)?
        if let learning {
            let registered = registerEmittedPath(
                learning, previous: previous, echo: echo, stored: stored,
                probeToolSpecs: probeToolSpecs, render: probeRender, tokenizer: tokenizer)
            learned = (registered.registration, registered.pathLength)
            if let storedTokens = registered.storedTokens {
                probeRender = probeRender.carryingBaseRender(storedTokens)
            }
        }
        let nextRender: [Int]
        do {
            nextRender =
                try ConversationRender
                .uncached(
                    tokenizer: tokenizer, toolSpecs: nextToolSpecs,
                    emittedPathIndex: learning?.index,
                    emittedPathFingerprint: learning?.fingerprint,
                    emittedPathTelemetry: telemetry
                )
                .continuationRender(messages: next.promptMessages)
        } catch {
            let verdict = Verdict.noPath(reason: "next-render-failed: \(error)")
            return BoundaryReport(boundary: boundary, leaf: verdict, speculation: nil)
        }
        let emittedPath = learned.map { learned in
            let summary = telemetry?.summary ?? .init()
            return EmittedPathVerdict(
                registration: learned.registration, pathLength: learned.pathLength,
                nextIndexedPrefix: summary.lastIndexedPrefix,
                nextMissReason: summary.lastMissReason,
                shadowDifferences: summary.shadowDifferences)
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
                    keySpace: .identity(),
                    render: probeRender
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
                        keySpace: .identity(),
                        render: probeRender
                    )
                },
                against: nextRender,
                tokenizer: tokenizer
            )
        }

        return BoundaryReport(
            boundary: boundary,
            leaf: leafVerdict,
            speculation: speculationVerdict,
            emittedPath: emittedPath
        )
    }

    /// The Leaf Store's registration, simulated on the recorded pair:
    /// request N's prompt is its full render (generation prompt included);
    /// the emitted ids are the canonical encode of the stored render past
    /// that prompt, through the last end-of-turn marker — what a live-stored
    /// turn's path equals once the Live Leaf Capture has proved the fed path
    /// canonical. Returns the stored render's tokens, for the probe render
    /// to carry as its base render (the harness renders `stored` once).
    private static func registerEmittedPath(
        _ learning: EmittedPathLearning,
        previous: HTTPPrefixCacheConversation,
        echo: HTTPPrefixCacheMessage,
        stored: HTTPPrefixCacheConversation,
        probeToolSpecs: [ToolSpec]?,
        render: ConversationRender,
        tokenizer: any Tokenizer
    ) -> (registration: String, pathLength: Int?, storedTokens: [Int]?) {
        guard
            case .eligible(let index, let fingerprint, let marker) = render.emittedPathEligibility()
        else {
            return (EmittedPathRegistration.SkipReason.noEndOfTurnMarker.rawValue, nil, nil)
        }
        do {
            let rendered = try render.storedRender(messages: stored.promptMessages)
            guard let bytes = rendered.bytes,
                let markerIndex = rendered.tokens.lastIndex(of: marker.tokenID)
            else {
                return (
                    EmittedPathRegistration.SkipReason.noEndOfTurnMarker.rawValue, nil,
                    rendered.tokens
                )
            }
            let prompt = try ConversationRender.stablePrefixProbeRender(
                tokenizer: tokenizer, messages: previous.promptMessages, tools: probeToolSpecs,
                additionalContext: TemplateRenderContext.canonical.additionalContext())
            let path = Array(rendered.tokens[...markerIndex])
            guard path.starts(with: prompt) else {
                return ("promptNotTokenPrefix", nil, rendered.tokens)
            }
            let outcome = EmittedPathRegistration.register(
                EmittedPathRegistration.Inputs(
                    index: index, fingerprint: fingerprint, marker: marker, tokenizer: tokenizer,
                    storedRenderBytes: bytes, storedMessage: echo, promptKeyPath: prompt,
                    generatedTokens: Array(path[prompt.count...]), stoppedOn: marker.tokenID,
                    toolCallFormat: learning.toolCallFormat, tools: probeToolSpecs,
                    startsInsideThinkBlock: TemplateRenderContext.canonical
                        .startsInsideThinkBlock(promptStartsThinking: learning.promptStartsThinking)
                ))
            switch outcome {
            case .registered(let registered):
                return ("registered", registered.pathLength, rendered.tokens)
            case .skipped(let skip):
                return (skip.reason.rawValue, nil, rendered.tokens)
            }
        } catch {
            return (EmittedPathRegistration.SkipReason.renderUnavailable.rawValue, nil, nil)
        }
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
    }

    struct SessionReport: Sendable {
        let sessionAffinity: String?
        let boundaries: [BoundaryReport]
        /// Pairs that could not be reduced to a boundary (history edit,
        /// ineligible conversation, no clean echo) — reported, never failed.
        let skipped: [(requestIndex: Int, reason: String)]

        var mismatchCount: Int { boundaries.count(where: \.hasMismatch) }

        /// The Emitted Path account over the walk, when it learned an index.
        var emittedPathSummary: EmittedPathSessionSummary? {
            let verdicts = boundaries.compactMap(\.emittedPath)
            guard !verdicts.isEmpty else { return nil }
            var summary = EmittedPathSessionSummary()
            summary.boundaries = verdicts.count
            for verdict in verdicts {
                summary.shadowDifferences += verdict.shadowDifferences
                guard verdict.registered else {
                    summary.registrationSkips[verdict.registration, default: 0] += 1
                    continue
                }
                summary.registered += 1
                if let prefix = verdict.nextIndexedPrefix, prefix > 0 {
                    summary.nextResolved += 1
                } else {
                    summary.nextMisses[verdict.nextMissReason ?? "noHit", default: 0] += 1
                }
            }
            return summary
        }
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
        tokenizer: any Tokenizer,
        learning: EmittedPathLearning? = nil
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
                            tokenizer: tokenizer,
                            learning: learning
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
        if let summary = report.emittedPathSummary {
            lines.append(
                "emitted-path boundaries=\(summary.boundaries) registered=\(summary.registered) "
                    + "skips=\(summary.registrationSkips) nextResolved=\(summary.nextResolved) "
                    + "nextMisses=\(summary.nextMisses) shadowDifferences=\(summary.shadowDifferences)"
            )
            for boundaryReport in report.boundaries {
                guard let verdict = boundaryReport.emittedPath,
                    !verdict.registered || (verdict.nextIndexedPrefix ?? 0) == 0
                        || verdict.shadowDifferences > 0
                else { continue }
                lines.append(
                    "emitted-path request#\(boundaryReport.boundary.requestIndex) "
                        + "kind=\(boundaryReport.boundary.kind.rawValue) "
                        + "registration=\(verdict.registration) "
                        + "nextPrefix=\(verdict.nextIndexedPrefix ?? 0) "
                        + "nextMiss=\(verdict.nextMissReason ?? "-") "
                        + "shadow=\(verdict.shadowDifferences)")
            }
        }
        return lines.joined(separator: "\n")
    }
}
