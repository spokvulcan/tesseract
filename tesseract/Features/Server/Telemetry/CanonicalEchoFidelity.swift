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
        /// Generation-prompt token counts per render context, measured once.
        let generationPromptProbe = GenerationPromptProbe()

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

    /// The tokens a render context's generation prompt adds after a
    /// conversation's last end-of-turn marker: the difference between the
    /// probe conversation rendered with and without the prompt. Special
    /// tokens bound the prompt on both sides, so the count is independent
    /// of the conversation it follows. Memoized per context digest — one
    /// pair of tiny renders per distinct context over a walk.
    nonisolated final class GenerationPromptProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var counts: [String: Int] = [:]

        func tokenCount(tokenizer: any Tokenizer, renderContext: TemplateRenderContext) -> Int? {
            let key = renderContext.digest
            if let known = lock.withLock({ counts[key] }) { return known }
            let probe: [[String: any Sendable]] = [["role": "user", "content": "probe"]]
            guard
                let withPrompt = try? ConversationRender.stablePrefixProbeRender(
                    tokenizer: tokenizer, messages: probe, tools: nil,
                    additionalContext: renderContext.additionalContext()),
                let withoutPrompt = try? ConversationRender.stablePrefixProbeRender(
                    tokenizer: tokenizer, messages: probe, tools: nil,
                    additionalContext: renderContext.additionalContext(
                        merging: ["add_generation_prompt": false]))
            else { return nil }
            let count = withPrompt.count - withoutPrompt.count
            lock.withLock { counts[key] = count }
            return count
        }
    }

    /// The Emitted Path account of one boundary: what the Leaf Store would
    /// have done with the turn (decided exactly as the live fast path
    /// decides), what the index learned, and what the next request would
    /// have prefilled. The replay gate (`EmittedPathReplayGate`) judges
    /// these per turn.
    struct EmittedPathVerdict: Sendable {
        /// `registered`, a `EmittedPathRegistration.SkipReason` raw value,
        /// or `promptNotTokenPrefix` when request N's prompt is not a token
        /// prefix of the stored render — the junction merge the Live Leaf
        /// Capture would have refused as a divergence, so no live-stored
        /// turn is simulated for it.
        let registration: String
        let pathLength: Int?
        /// The leaf-store mode the turn selects (`HTTPLeafStoreMode`).
        let mode: String
        /// The leaf source the Live Leaf Capture decides for the turn:
        /// `live` (the fed ids, registered) or `boundary` (synthesized from
        /// the boundary, nothing registered) with its reason.
        let source: String
        let boundaryReason: String?
        /// The key was already registered under another path (decision 6):
        /// last writer wins, counted by the index as an overwrite.
        let overwrote: Bool
        /// Request N+1's resolve (the harness renders it as the leaf-store
        /// continuation spelling — same bytes through the last marker as the
        /// request edge, so the same hit): the indexed prefix on a hit.
        let nextIndexedPrefix: Int?
        /// The tokens request N+1 encoded past the hit — its new messages
        /// plus the glue after the marker: what it would prefill beyond the
        /// stored leaf.
        let nextSuffixTokens: Int?
        let nextMissReason: String?
        /// Request N+1's request-edge prefill beyond what the index served:
        /// its full render (generation prompt included) past the indexed
        /// prefix — zero indexed prefix on a miss, so the whole request.
        let nextPrefilled: Int?
        /// The tokens request N+1's new messages add to the stored render.
        let nextNewMessageTokens: Int?
        /// The fast path's CPU tail simulated on the recording: the stored
        /// render to bytes plus the registration (fidelity replay, hashing,
        /// insertion) — the post-EOS work the live turn pays before its
        /// capture. Only measured for a live-decided turn.
        let tailSeconds: Double?
        /// The tail's two halves: the template render, and the registration.
        let renderSeconds: Double?
        let registerSeconds: Double?

        var registered: Bool { registration == "registered" }
        var fidelityRejected: Bool {
            registration == EmittedPathRegistration.SkipReason.fidelityRejected.rawValue
        }
        /// The prefilled tokens that are neither the new messages nor the
        /// stored turn: the generation prompt plus the newline closing the
        /// stored turn's marker line — more when the hit was shallow.
        var glueTokens: Int? {
            guard let nextPrefilled, let nextNewMessageTokens else { return nil }
            return nextPrefilled - nextNewMessageTokens
        }
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
        /// Suffix tokens summed over the resolved next requests.
        var nextSuffixTokens = 0
        /// Leaf source (`live`/`boundary`) → count.
        var sources: [String: Int] = [:]
        var overwrites = 0
        var fidelityRejections = 0
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
        learning: EmittedPathLearning? = nil,
        previousRenderContext: TemplateRenderContext = .canonical,
        nextRenderContext: TemplateRenderContext = .canonical
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
            tokenizer: tokenizer, toolSpecs: probeToolSpecs, renderContext: previousRenderContext,
            emittedPathIndex: learning?.index, emittedPathFingerprint: learning?.fingerprint,
            emittedPathTelemetry: telemetry
        )
        var learned: SimulatedRegistration?
        if let learning {
            let registered = registerEmittedPath(
                learning, previous: previous, echo: echo, stored: stored,
                probeToolSpecs: probeToolSpecs, render: probeRender, tokenizer: tokenizer,
                renderContext: previousRenderContext)
            learned = registered
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
                    renderContext: nextRenderContext,
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
            // The request edge renders the same bytes plus the generation
            // prompt, which special tokens bound on both sides: its token
            // count is the continuation render's plus the prompt's.
            let generationPrompt = learning?.generationPromptProbe.tokenCount(
                tokenizer: tokenizer, renderContext: nextRenderContext)
            let nextPrefilled = generationPrompt.map {
                nextRender.count + $0 - (summary.lastIndexedPrefix ?? 0)
            }
            return EmittedPathVerdict(
                registration: learned.registration, pathLength: learned.pathLength,
                mode: learned.mode, source: learned.source,
                boundaryReason: learned.boundaryReason, overwrote: learned.overwrote,
                nextIndexedPrefix: summary.lastIndexedPrefix,
                nextSuffixTokens: summary.lastSuffixTokens,
                nextMissReason: summary.lastMissReason,
                nextPrefilled: nextPrefilled,
                nextNewMessageTokens: learned.storedTokens.map { nextRender.count - $0.count },
                tailSeconds: learned.tailSeconds,
                renderSeconds: learned.renderSeconds,
                registerSeconds: learned.registerSeconds)
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

    /// What `registerEmittedPath` simulated for one turn.
    private struct SimulatedRegistration {
        var registration: String
        var pathLength: Int?
        /// The stored render's tokens, for the probe render to carry as its
        /// base render (the harness renders `stored` once).
        var storedTokens: [Int]?
        var mode: String
        var source: String
        var boundaryReason: String?
        var overwrote = false
        var tailSeconds: Double?
        var renderSeconds: Double?
        var registerSeconds: Double?

        init(
            registration: String, pathLength: Int? = nil, storedTokens: [Int]? = nil,
            mode: HTTPLeafStoreMode, decision: LiveLeafCapture.Decision? = nil
        ) {
            self.registration = registration
            self.pathLength = pathLength
            self.storedTokens = storedTokens
            self.mode = mode.rawValue
            switch decision {
            case .live, .none:
                self.source = LeafStorePhase.Report.Source.live.rawValue
            case .boundary(let reason):
                self.source = LeafStorePhase.Report.Source.boundary.rawValue
                self.boundaryReason = EmittedPathRegistration.skipReason(for: reason).rawValue
            }
        }
    }

    /// The Leaf Store's registration, simulated on the recorded pair:
    /// request N's prompt is its full render (generation prompt included);
    /// the emitted ids are the canonical encode of the stored render past
    /// that prompt, through the last end-of-turn marker — what a live-stored
    /// turn's path equals when the model's split was the canonical one (a
    /// recording keeps no fed ids; the live fast path registers the fed ids
    /// themselves). The turn's leaf source is decided exactly as the live
    /// fast path decides it (`LiveLeafCapture.decide`, with the cache offset
    /// at the end of the fed ids and no intervention — the recording keeps
    /// neither): a boundary-decided turn registers nothing, as live.
    private static func registerEmittedPath(
        _ learning: EmittedPathLearning,
        previous: HTTPPrefixCacheConversation,
        echo: HTTPPrefixCacheMessage,
        stored: HTTPPrefixCacheConversation,
        probeToolSpecs: [ToolSpec]?,
        render: ConversationRender,
        tokenizer: any Tokenizer,
        renderContext: TemplateRenderContext
    ) -> SimulatedRegistration {
        let mode = LeafStorePhase.selectHTTPLeafStoreMode(
            promptStartsThinking: learning.promptStartsThinking,
            emittedToolCalls: !echo.toolCalls.isEmpty)
        let index: EmittedPathIndex
        let fingerprint: String
        let marker: EndOfTurnMarker
        switch render.emittedPathEligibility() {
        case .ineligible(let reason, _):
            return SimulatedRegistration(registration: reason.rawValue, mode: mode)
        case .eligible(let engaged, let scoped, let derived):
            index = engaged
            fingerprint = scoped
            marker = derived
        }
        do {
            let rendered = try render.storedRender(messages: stored.promptMessages)
            guard let markerIndex = rendered.tokens.lastIndex(of: marker.tokenID) else {
                return SimulatedRegistration(
                    registration: EmittedPathRegistration.SkipReason.noEndOfTurnMarker.rawValue,
                    storedTokens: rendered.tokens, mode: mode)
            }
            let prompt = try ConversationRender.stablePrefixProbeRender(
                tokenizer: tokenizer, messages: previous.promptMessages, tools: probeToolSpecs,
                additionalContext: renderContext.additionalContext())
            let path = Array(rendered.tokens[...markerIndex])
            guard path.starts(with: prompt) else {
                return SimulatedRegistration(
                    registration: EmittedPathReplayGate.promptNotTokenPrefix,
                    storedTokens: rendered.tokens,
                    mode: mode)
            }
            let generatedTokens = Array(path[prompt.count...])
            let decision = LiveLeafCapture.decide(
                mode: mode, preservesThinking: renderContext.preservesThinking,
                promptKeyPath: prompt, generatedTokens: generatedTokens,
                cacheOffset: path.count, intervened: false, keySpaceIsIdentity: true)
            if case .boundary(let reason) = decision {
                return SimulatedRegistration(
                    registration: EmittedPathRegistration.skipReason(for: reason).rawValue,
                    storedTokens: rendered.tokens, mode: mode, decision: decision)
            }
            var simulated = SimulatedRegistration(
                registration: "registered", storedTokens: rendered.tokens, mode: mode,
                decision: decision)

            // The fast path's own CPU work, timed as the live turn pays it:
            // the stored render to bytes (no tokenization), then the
            // registration over those bytes.
            let tailStart = monotonicSeconds()
            guard let bytes = try render.storedRenderBytes(messages: stored.promptMessages)
            else {
                simulated.registration =
                    EmittedPathRegistration.SkipReason.renderUnavailable.rawValue
                return simulated
            }
            let registerStart = monotonicSeconds()
            simulated.renderSeconds = registerStart - tailStart
            let outcome = EmittedPathRegistration.register(
                EmittedPathRegistration.Inputs(
                    index: index, fingerprint: fingerprint, marker: marker, tokenizer: tokenizer,
                    storedRenderBytes: bytes, storedMessage: echo, promptKeyPath: prompt,
                    generatedTokens: generatedTokens, stoppedOn: marker.tokenID,
                    toolCallFormat: learning.toolCallFormat, tools: probeToolSpecs,
                    startsInsideThinkBlock: renderContext.startsInsideThinkBlock(
                        promptStartsThinking: learning.promptStartsThinking)
                ))
            let registered = monotonicSeconds()
            simulated.registerSeconds = registered - registerStart
            simulated.tailSeconds = registered - tailStart
            switch outcome {
            case .registered(let registered):
                simulated.pathLength = registered.pathLength
                simulated.overwrote = registered.previousPathLength != nil
            case .skipped(let skip):
                simulated.registration = skip.reason.rawValue
            }
            return simulated
        } catch {
            return SimulatedRegistration(
                registration: EmittedPathRegistration.SkipReason.renderUnavailable.rawValue,
                mode: mode)
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

/// The simulated tail's clock: monotonic, so a wall-clock adjustment never
/// counts against a turn.
nonisolated private func monotonicSeconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
}

extension CanonicalEchoFidelity {

    /// One recorded request, already decoded from an `HTTPRequestLogger`
    /// recording: the raw OpenAI messages plus tool definitions.
    struct RecordedRequest: Sendable {
        let messages: [OpenAI.ChatMessage]
        let tools: [OpenAI.ToolDefinition]?
        /// The render context the server resolved for the request
        /// (`TemplateRenderContext.resolve`): its kwargs shape the render
        /// bytes and its digest the conversation's partition — a change
        /// between adjacent requests is a history edit to the walk, as it
        /// is to the server's prefix check.
        var renderContext: TemplateRenderContext = .canonical
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
                summary.sources[verdict.source, default: 0] += 1
                if verdict.overwrote { summary.overwrites += 1 }
                if verdict.fidelityRejected { summary.fidelityRejections += 1 }
                guard verdict.registered else {
                    summary.registrationSkips[verdict.registration, default: 0] += 1
                    continue
                }
                summary.registered += 1
                if let prefix = verdict.nextIndexedPrefix, prefix > 0 {
                    summary.nextResolved += 1
                    summary.nextSuffixTokens += verdict.nextSuffixTokens ?? 0
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

        var previous:
            (
                conversation: HTTPPrefixCacheConversation, toolSpecs: [ToolSpec]?,
                renderContext: TemplateRenderContext
            )?

        for (index, request) in requests.enumerated() {
            let repaired = await repairStore.repair(
                messages: request.messages,
                sessionAffinity: sessionAffinity,
                modelID: modelID,
                visionMode: false
            )
            let normalized = MessageConverter.normalizeRequest(
                repaired.messages,
                tools: request.tools,
                templateContextDigest: request.renderContext.digest
            )
            guard case .eligible(let conversation) = normalized.prefixCacheEligibility else {
                skipped.append((index, "ineligible: \(normalized.prefixCacheEligibility)"))
                previous = nil
                continue
            }
            let toolSpecs = LLMActor.canonicalizeToolSpecs(
                MessageConverter.convertToolDefinitions(request.tools)
            )

            if let (previousConversation, previousToolSpecs, previousRenderContext) = previous {
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
                            learning: learning,
                            previousRenderContext: previousRenderContext,
                            nextRenderContext: request.renderContext
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

            previous = (conversation, toolSpecs, request.renderContext)
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
                    + "nextMisses=\(summary.nextMisses) nextSuffixTokens=\(summary.nextSuffixTokens) "
                    + "sources=\(summary.sources) overwrites=\(summary.overwrites) "
                    + "fidelityRejections=\(summary.fidelityRejections)"
            )
            for boundaryReport in report.boundaries {
                guard let verdict = boundaryReport.emittedPath,
                    !verdict.registered || (verdict.nextIndexedPrefix ?? 0) == 0
                else { continue }
                lines.append(
                    "emitted-path request#\(boundaryReport.boundary.requestIndex) "
                        + "kind=\(boundaryReport.boundary.kind.rawValue) "
                        + "source=\(verdict.source) "
                        + "boundary=\(verdict.boundaryReason ?? "-") "
                        + "registration=\(verdict.registration) "
                        + "nextPrefix=\(verdict.nextIndexedPrefix ?? 0) "
                        + "nextMiss=\(verdict.nextMissReason ?? "-")")
            }
        }
        return lines.joined(separator: "\n")
    }
}
