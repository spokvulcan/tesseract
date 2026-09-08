//
//  LeafStorePhase.swift
//  tesseract
//
//  The **Leaf Store** phase of a cache-aware **Server Completion**: after the
//  stream drive finishes, decide how (or whether) the finished turn's KV
//  state is admitted as a leaf, and dispatch to the model-affine executors
//  (`LeafStorePhase+Executors.swift`).
//
//  Two routes (ADR-0063, decisions 10 to 13):
//  - the fast path, for every turn the template renders verbatim for the
//    next request and whose structural guards hold (`LiveLeafCapture`):
//    register the turn's **Emitted Path** (one Jinja render to bytes, no
//    tokenization), capture the leaf from the live final cache at the
//    cache's own offset, admit it under the live path. No canonical
//    render-and-tokenize, no continuation probe, no boundary lookup, no
//    comparison against a re-render — the next request resolves to the
//    same ids through the **Emitted Path Resolve**.
//  - the boundary path, for a think-stripping template at a new-user-message
//    boundary and for guard failures: the stored-conversation
//    re-tokenization and key-space translation, the reusable-prefix probe,
//    the boundary plan, the **Speculative Canonical Prefill** seeding, and
//    the restore-and-re-prefill executor (or, under a non-thinking
//    template, the render-trusting direct executor) — the pre-ADR-0063
//    path, unchanged.
//
//  Isolation matches the drive that calls it (nonisolated, off-actor); every
//  model-affine step hops through the **Model Session** (ADR-0016) and cache
//  admissions hop to the MainActor-confined Prefix Cache, exactly as before.
//

import Foundation
import MLX
import MLXLMCommon

nonisolated enum LeafStorePhase {

    /// What the phase concluded: the alpha tuner's leaf-store record if one
    /// landed, the Speculative Canonical Prefill seed if this turn armed
    /// one (handed to the post-finish hook by the drive), and the report of
    /// which path ran and where its time went.
    struct Result: Sendable {
        var leafStore: AlphaTuner.LeafStore?
        var speculativeSeed: SpeculativeCanonicalPrefill.Seed?
        /// The admission's store diagnostics, for the per-request trace
        /// (nil when no admission was attempted).
        var admission: PrefixCacheManager.StoreDiagnostics?
        var report = Report()
    }

    /// The request facts every step of the phase reads: the generation the
    /// drive ran, the session provider, and the admission context.
    struct Inputs: Sendable {
        let mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>
        let sessions: any ModelSessionProviding
        let requestID: UUID
        let prefixCache: PrefixCacheManager
        let diagnosticsContext: PrefixCacheDiagnostics.Context
        let containsImages: Bool
        var mlxStart: HTTPPrefixCacheGeneration { mlxStartBox.value }
    }

    /// The turn facts shared by both routes: the stored conversation (prompt
    /// plus the generated assistant turn), the request's render, the
    /// selected mode and the appended message the fidelity check compares
    /// against.
    struct Turn: Sendable {
        let storedConversation: HTTPPrefixCacheConversation
        let render: ConversationRender
        let mode: HTTPLeafStoreMode
        /// The assistant message appended to the stored conversation.
        let storedMessage: HTTPPrefixCacheMessage
        /// Every id the decode loop fed past the prompt, in order, stop id
        /// included (`GeneratedTokenRecorder`).
        let generatedTokens: [Int]
        /// Whether the generation began inside a `<think>` block.
        let startsInsideThinkBlock: Bool
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); the phase keeps the drive's
    // lenient structural limits — splitting further is deferred. The wide
    // parameter list is the phase's honest input set (the drive's request
    // context); `Inputs` carries its request-constant subset to the helpers.
    // swiftlint:disable function_parameter_count
    static func run(
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        conversation: HTTPPrefixCacheConversation,
        sessions: any ModelSessionProviding,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        promptStartsThinking: Bool,
        assistantText: String,
        assistantReasoning: String?,
        toolCalls: [HTTPPrefixCacheToolCall],
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        trace: inout CompletionTraceAccumulator
    ) async -> Result {
        // swiftlint:enable function_parameter_count
        let inputs = Inputs(
            mlxStartBox: mlxStartBox, sessions: sessions, requestID: requestID,
            prefixCache: prefixCache, diagnosticsContext: diagnosticsContext,
            containsImages: conversation.messages.contains { !$0.images.isEmpty })
        let mlxStart = inputs.mlxStart
        var result = Result()

        // An Unkeyed Completion never touches the radix tree — construction
        // failed, so no token path of this request can be trusted as a key.
        if let unkeyedReason = mlxStart.unkeyedReason {
            result.report.recordSkip(
                LeafSkipLog(
                    stage: "leafStore", reason: "unkeyed-completion", level: .info,
                    extraFields: [("unkeyedReason", unkeyedReason.rawValue)]),
                in: diagnosticsContext)
            result.report.recordEmittedPathSkip(
                .ineligibleRender, fields: [("cause", "unkeyed")], in: diagnosticsContext)
            return result
        }
        guard let render = mlxStart.render else {
            // The render is nil exactly for an Unkeyed Completion, which the
            // guard above already returned on.
            return result
        }

        // 1. Build the stored conversation (prompt + generated assistant turn).
        let storedMessage = HTTPPrefixCacheMessage.assistant(
            content: assistantText,
            reasoning: assistantReasoning ?? "",
            toolCalls: toolCalls
        )
        let storedConversation = conversation.appendingAssistant(storedMessage)

        let leafStoreMode = Self.selectHTTPLeafStoreMode(
            promptStartsThinking: promptStartsThinking,
            emittedToolCalls: !toolCalls.isEmpty
        )
        result.report.mode = leafStoreMode.rawValue
        diagnosticsContext.log(
            PrefixCacheDiagnostics.LeafModeEvent(
                mode: leafStoreMode.rawValue,
                continuation: toolCalls.isEmpty
                    ? HTTPLeafContinuationKind.userTurn.rawValue
                    : HTTPLeafContinuationKind.toolResult.rawValue
            ))
        let turn = Turn(
            storedConversation: storedConversation,
            render: render,
            mode: leafStoreMode,
            storedMessage: storedMessage,
            generatedTokens: mlxStart.generatedTokens.snapshot,
            startsInsideThinkBlock: promptStartsThinking
        )

        // 2. The fast-path eligibility (GPU-free, no render): the live final
        // cache is the leaf unless a structural guard or the render rule
        // sends the turn to the boundary path.
        let preservesThinking = render.renderContext.preservesThinking
        let decision = LiveLeafCapture.decide(
            mode: leafStoreMode,
            preservesThinking: preservesThinking,
            promptKeyPath: mlxStart.keySpace.keyPath,
            generatedTokens: turn.generatedTokens,
            cacheOffset: httpPrefixCacheReportedTokenCount(mlxStart.finalCache),
            keySpaceIsIdentity: mlxStart.keySpace.isIdentity
        )
        switch decision {
        case .live(let offset):
            await storeLive(offset: offset, turn: turn, inputs: inputs, result: &result)
        case .boundary(let reason):
            let record = liveFallbackLog(
                for: reason, mode: leafStoreMode, preservesThinking: preservesThinking)
            record.emit(in: diagnosticsContext)
            result.report.boundaryReason = record.reason
            LeafStoreCounters.shared.noteBoundaryTurn(reason: record.reason)
            result.report.recordEmittedPathSkip(
                EmittedPathRegistration.skipReason(for: reason), in: diagnosticsContext)
            await storeFromBoundary(turn: turn, inputs: inputs, result: &result)
        }

        if let admission = result.admission {
            trace.ingest(evictions: admission.evictions, diagnostics: diagnosticsContext)
            trace.logSupersessions(admission.supersededLeaves, diagnostics: diagnosticsContext)
        }
        return result
    }

    // MARK: - The fast path

    /// Register the turn's Emitted Path, capture the live final cache at
    /// `offset` and admit it under the live path (decision 10). A fidelity
    /// rejection registers nothing and still stores the leaf: the state
    /// matches the fed ids by construction, and the next request renders
    /// canonically and re-prefills from the first differing token, visibly.
    private static func storeLive(
        offset: Int,
        turn: Turn,
        inputs: Inputs,
        result: inout Result
    ) async {
        let mlxStart = inputs.mlxStart

        // 1. Register the turn's Emitted Path.
        registerEmittedPath(turn: turn, inputs: inputs, report: &result.report)

        // 2/3. Capture at the cache's offset, admit under the live path.
        let livePath = LiveLeafCapture.livePath(
            promptKeyPath: mlxStart.keySpace.keyPath,
            generatedTokens: turn.generatedTokens,
            offset: offset)
        // A tool-call boundary under a think-stripping template still arms
        // **Stretch Abandonment** (ADR-0009): the next real user message
        // re-renders the stretch, and the pass pre-prefills that render
        // while the GPU is idle. No canonical leaf exists here for the
        // immediate seed to extend — the live path is the future path.
        let pendingSeed = speculativeSeed(for: turn, inputs: inputs, canonicalLeafOffset: nil)
        let stages = leafStages(for: turn.mode)
        let capture = await captureLiveLeaf(
            sessions: inputs.sessions, mlxStartBox: inputs.mlxStartBox,
            context: LeafAdmissionContext(storedTokens: livePath, inputs: inputs, stages: stages))
        result.report.absorb(capture, path: .live)
        result.conclude(capture, pendingSeed: pendingSeed)
    }

    // MARK: - The boundary path

    /// The pre-ADR-0063 route: measure the stored conversation (render and
    /// tokenize, translate into key space), then either the boundary
    /// executor behind the reusable-prefix probe and the boundary plan, or
    /// the render-trusting direct executor under a non-thinking template.
    private static func storeFromBoundary(
        turn: Turn,
        inputs: Inputs,
        result: inout Result
    ) async {
        let mlxStart = inputs.mlxStart
        let diagnostics = inputs.diagnosticsContext

        // The stored conversation's render-space tokens, translated into key
        // space (identity for text-only). The translated path is what every
        // capture offset and admission below keys on — length-equal to the
        // prepared sequence, so key index == KV offset holds. Raw prompt
        // messages, so assistant `reasoning_content` and `tool_calls`
        // survive template rendering.
        let renderStart = Date.timeIntervalSinceReferenceDate
        let storedRenderTokens: [Int]
        do {
            storedRenderTokens = try turn.render.storedRender(
                messages: turn.storedConversation.promptMessages
            ).tokens
        } catch {
            Log.agent.warning(
                "Stored token sequence measurement failed — error=\(error.localizedDescription)"
            )
            result.report.recordSkip(
                LeafSkipLog(
                    stage: "leafStore", reason: "tokenization-failed", level: .warning,
                    extraFields: []),
                in: diagnostics)
            return
        }
        let storedTokens: [Int]
        switch mlxStart.keySpace.translate(renderTokens: storedRenderTokens) {
        case .success(let translated):
            storedTokens = translated
        case .failure(let failure):
            result.report.recordSkip(
                LeafSkipLog(
                    stage: "leafStore", reason: "render-translation-failed", level: .warning,
                    extraFields: [("failure", "\(failure)")]),
                in: diagnostics)
            return
        }
        result.report.renderSeconds = secondsSince(renderStart)

        guard let boundaryMode = turn.mode.boundaryMode else {
            // Non-thinking templates: the render-trusting direct executor
            // snapshots the live final cache under the stored path.
            let capture = await captureDirectLeaf(
                sessions: inputs.sessions, mlxStartBox: inputs.mlxStartBox,
                context: LeafAdmissionContext(
                    storedTokens: storedTokens, inputs: inputs, stages: .direct))
            result.report.absorb(capture, path: .direct)
            result.conclude(capture, pendingSeed: nil)
            return
        }

        let planStart = Date.timeIntervalSinceReferenceDate
        guard
            let route = await routeBoundaryMode(
                boundaryMode,
                inputs: inputs,
                storedConversation: turn.storedConversation,
                storedTokens: storedTokens,
                // C31: the stored render just computed above is the identical
                // computation the builder's base probe would re-run — carry
                // it so the base render runs once per request.
                probeRender: turn.render.carryingBaseRender(storedRenderTokens),
                report: &result.report
            )
        else { return }
        result.report.planSeconds = secondsSince(planStart)

        // Seed the **Speculative Canonical Prefill** before the GPU-side
        // store: the seed spawns the future-path probe immediately, so its
        // CPU render+tokenize overlaps the store (#76's earlier start). Kept
        // only if the leaf store below succeeds. The worth-it floor differs
        // by trigger: a canonical leaf IS the strip floor; a tool stretch
        // measures its rewind span from the last-user boundary.
        let pendingSeed = speculativeSeed(
            for: turn, inputs: inputs, canonicalLeafOffset: route.leafOffset)
        let capture = await captureStructuredLeafFromBoundary(
            sessions: inputs.sessions,
            boundarySnapshot: route.boundary,
            positionAnchorRopeDelta: route.positionAnchorRopeDelta,
            prefillStepSize: mlxStart.prefillStepSize,
            tokenNDim: mlxStart.tokenNDim,
            context: LeafAdmissionContext(
                storedTokens: route.storedTokens, inputs: inputs,
                stages: leafStages(for: boundaryMode))
        )
        result.report.absorb(capture, path: .boundary)
        result.conclude(capture, pendingSeed: pendingSeed)
    }

    /// The restore boundary a boundary mode's leaf is synthesized from, once
    /// the reusable-prefix probe and the boundary plan have run.
    struct BoundaryRoute {
        let boundary: HybridCacheSnapshot
        /// The position-anchor delta the residual re-prefill is seeded with
        /// on the vision container.
        let positionAnchorRopeDelta: Int?
        /// The canonical path the leaf is admitted under: restore `boundary`,
        /// re-prefill `storedTokens[boundary.tokenOffset...]`, capture at
        /// `storedTokens.count`.
        let storedTokens: [Int]

        /// The leaf's offset — what the speculative seed measures from.
        var leafOffset: Int { storedTokens.count }
    }

    /// Route one boundary mode: probe the shared token path and choose a
    /// restore boundary for it (**Snapshot Resolution** may hydrate from
    /// SSD). Returns `nil` after recording the decidable skip in `report`.
    static func routeBoundaryMode(
        _ mode: BoundaryLeafMode,
        inputs: Inputs,
        storedConversation: HTTPPrefixCacheConversation,
        storedTokens: [Int],
        probeRender: ConversationRender,
        report: inout Report
    ) async -> BoundaryRoute? {
        let mlxStart = inputs.mlxStart
        let diagnostics = inputs.diagnosticsContext

        let probedTokens: [Int]
        switch LeafAdmissionBuilder.probe(
            mode: mode,
            storedConversation: storedConversation,
            storedTokens: storedTokens,
            keySpace: mlxStart.keySpace,
            render: probeRender
        ) {
        case .tokens(let tokens):
            probedTokens = tokens
        case .skip(let reason):
            report.recordSkip(leafSkipLog(for: reason, mode: mode), in: diagnostics)
            return nil
        }

        let transientBoundary: HybridCacheSnapshot? =
            switch mode {
            case .directTool: mlxStart.transientLastMessageBoundarySnapshot
            case .canonical: mlxStart.transientLastUserBoundarySnapshot
            }
        let plan = await LeafAdmissionBuilder.plan(
            mode: mode,
            probedTokens: probedTokens,
            transientBoundary: transientBoundary,
            keySpace: mlxStart.keySpace,
            resolveBoundary: { tokens in
                // Drive Snapshot Resolution inside the Model Session so the
                // SSD `loadSync` stays off-MainActor (ADR-0001). Session
                // entry cannot fail with a non-throwing body; the
                // hypothetical failure degrades to "no boundary snapshot".
                let resolved = try? await inputs.sessions.withSession { _ in
                    await inputs.prefixCache.resolve(
                        tokens: tokens,
                        promptTokenCount: tokens.count,
                        partitionKey: mlxStart.partitionKey,
                        modelFingerprint: mlxStart.partitionKey.modelFingerprint,
                        diagnostics: diagnostics,
                        pinningRestorePathFor: diagnostics.requestID
                    ).lookup.snapshot
                }
                return resolved.flatMap { $0 }
            }
        )
        switch plan {
        case .skip(let reason):
            report.recordSkip(leafSkipLog(for: reason, mode: mode), in: diagnostics)
            return nil
        case .fromBoundary(let boundary, let tokens):
            // The boundary sits past the image prefix (builder guard), so the
            // residual is real tokens in both spaces and the anchor delta is
            // always defined; on the vision container the residual reprefill
            // must resume with it seeded.
            var positionAnchorRopeDelta: Int?
            if mlxStart.seedsPositionAnchor {
                guard
                    let delta = mlxStart.keySpace.positionAnchorDelta(upTo: boundary.tokenOffset)
                else {
                    report.recordSkip(
                        LeafSkipLog(
                            stage: leafStages(for: mode).store,
                            reason: "boundary-splits-image-run", level: .warning,
                            extraFields: [("offset", "\(boundary.tokenOffset)")]),
                        in: diagnostics)
                    return nil
                }
                positionAnchorRopeDelta = delta
            }
            return BoundaryRoute(
                boundary: boundary, positionAnchorRopeDelta: positionAnchorRopeDelta,
                storedTokens: tokens)
        }
    }

    // MARK: - Emitted Path registration (ADR-0063)

    /// Register the finished turn's Emitted Path: the fed prompt ids and
    /// generated ids, the stop id, and the stored conversation's render to
    /// bytes — the fast path's one Jinja render, no tokenization — hashed
    /// through its last end-of-turn marker. Every outcome lands in the
    /// diagnostics net and the report; an ineligible render never renders.
    static func registerEmittedPath(turn: Turn, inputs: Inputs, report: inout Report) {
        let diagnostics = inputs.diagnosticsContext
        let mlxStart = inputs.mlxStart
        let render = turn.render
        let index: EmittedPathIndex
        let fingerprint: String
        let marker: EndOfTurnMarker
        switch render.emittedPathEligibility() {
        case .ineligible(let reason, let cause):
            report.recordEmittedPathSkip(
                reason, fields: cause.map { [("cause", $0)] } ?? [], in: diagnostics)
            return
        case .eligible(let engaged, let scoped, let derived):
            index = engaged
            fingerprint = scoped
            marker = derived
        }

        let renderStart = Date.timeIntervalSinceReferenceDate
        let storedRenderBytes: [UInt8]?
        do {
            defer { report.renderSeconds = secondsSince(renderStart) }
            storedRenderBytes = try render.storedRenderBytes(
                messages: turn.storedConversation.promptMessages)
        } catch {
            Log.agent.warning("Stored render failed — error=\(error.localizedDescription)")
            report.recordEmittedPathSkip(
                .renderUnavailable, fields: [("cause", "renderThrew")], in: diagnostics)
            return
        }
        guard let storedRenderBytes else {
            // A tokenizer that cannot render to bytes.
            report.recordEmittedPathSkip(.renderUnavailable, in: diagnostics)
            return
        }

        let start = Date.timeIntervalSinceReferenceDate
        let outcome = EmittedPathRegistration.register(
            EmittedPathRegistration.Inputs(
                index: index,
                fingerprint: fingerprint,
                marker: marker,
                tokenizer: render.tokenizer,
                storedRenderBytes: storedRenderBytes,
                storedMessage: turn.storedMessage,
                promptKeyPath: mlxStart.keySpace.keyPath,
                generatedTokens: turn.generatedTokens,
                stoppedOn: mlxStart.generatedTokens.stopToken,
                toolCallFormat: mlxStart.toolCallFormat,
                tools: render.toolSpecs,
                startsInsideThinkBlock: turn.startsInsideThinkBlock
            ))
        let seconds = secondsSince(start)
        EmittedPathRegistration.emit(outcome, registerSeconds: seconds, in: diagnostics)
        report.absorbEmittedPath(outcome, registerSeconds: seconds)
    }

    // MARK: - Mode selection

    static func selectHTTPLeafStoreMode(
        promptStartsThinking: Bool,
        emittedToolCalls: Bool
    ) -> HTTPLeafStoreMode {
        if emittedToolCalls {
            return .directToolLeaf
        }
        if promptStartsThinking {
            return .canonicalUserLeaf
        }
        return .directLeaf
    }

    /// The diagnostics stage labels of a boundary leaf mode — the exact
    /// strings the dissolved `captureDirectToolLeaf` /
    /// `captureCanonicalTemplateLeaf` helpers passed to the shared executor.
    static func leafStages(for mode: BoundaryLeafMode) -> LeafStages {
        switch mode {
        case .directTool:
            LeafStages(
                store: "directToolLeafStore", capture: "directToolLeafCapture",
                admission: "directToolLeafAdmission", source: "directToolLeaf")
        case .canonical:
            LeafStages(
                store: "canonicalLeafStore", capture: "canonicalLeafCapture",
                admission: "canonicalLeafAdmission", source: "canonicalLeaf")
        }
    }

    /// The stage labels of a leaf-store mode: the boundary modes' own, and
    /// the pre-existing direct labels for the non-thinking mode.
    static func leafStages(for mode: HTTPLeafStoreMode) -> LeafStages {
        mode.boundaryMode.map { leafStages(for: $0) } ?? .direct
    }

    // MARK: - Speculative seeding

    /// The seed this turn arms, if its mode and render call for one (the
    /// trigger table below); `nil` under the preserve-thinking render and
    /// for the direct mode. `canonicalLeafOffset` is where the boundary
    /// path's canonical leaf ends — what the immediate seed extends — and
    /// `nil` on the fast path, whose leaf is the live path: only Stretch
    /// Abandonment can arm there, whatever the mode.
    private static func speculativeSeed(
        for turn: Turn,
        inputs: Inputs,
        canonicalLeafOffset: Int?
    ) -> SpeculativeCanonicalPrefill.Seed? {
        guard let boundaryMode = turn.mode.boundaryMode,
            let plan = speculativeSeedPlan(
                boundaryMode: boundaryMode, renderContext: turn.render.renderContext)
        else { return nil }
        let mlxStart = inputs.mlxStart
        let leafOffset: Int
        switch boundaryMode {
        case .canonical:
            guard let canonicalLeafOffset else { return nil }
            leafOffset = canonicalLeafOffset
        case .directTool:
            leafOffset = mlxStart.transientLastUserBoundarySnapshot?.tokenOffset ?? 0
        }
        return SpeculativeCanonicalPrefill.makeSeed(
            storedConversation: turn.storedConversation,
            render: turn.render,
            keySpace: mlxStart.keySpace,
            partitionKey: mlxStart.partitionKey,
            prefillStepSize: mlxStart.prefillStepSize,
            ssdEnabled: mlxStart.ssdEnabled,
            seedsPositionAnchor: mlxStart.seedsPositionAnchor,
            canonicalLeafOffset: leafOffset,
            idleDelay: plan.idleDelay,
            ramOnlySpine: plan.ramOnlySpine,
            diagnostics: inputs.diagnosticsContext
        )
    }

    /// How a finished turn seeds the **Speculative Canonical Prefill** —
    /// the trigger table (issues #76, #100):
    /// - A canonical-user boundary (stop-finish answer) seeds immediately,
    ///   durable — the original #76 trigger.
    /// - A tool-call boundary arms **Stretch Abandonment**'s timer: the
    ///   pass starts only if no follow-up request lands inside the idle
    ///   window, and its spine admits RAM-only so a false alarm (the tool
    ///   result arrives) costs zero SSD writes (ADR-0009).
    /// - Under the **Preserve-Thinking Render** (issue #98) nothing seeds:
    ///   the render is append-stable, so the canonical future path equals
    ///   the live path and there is no Think-Strip Rewind span to
    ///   pre-prefill.
    struct SpeculativeSeedPlan: Equatable {
        let idleDelay: Duration
        let ramOnlySpine: Bool
    }

    static func speculativeSeedPlan(
        boundaryMode: BoundaryLeafMode,
        renderContext: TemplateRenderContext
    ) -> SpeculativeSeedPlan? {
        guard !renderContext.preservesThinking else { return nil }
        switch boundaryMode {
        case .canonical:
            return SpeculativeSeedPlan(idleDelay: .zero, ramOnlySpine: false)
        case .directTool:
            return SpeculativeSeedPlan(
                idleDelay: SpeculativeCanonicalPrefill.stretchAbandonmentIdleWindow,
                ramOnlySpine: true
            )
        }
    }

    // MARK: - Skip wire format

    /// The exact `logSkip` record a decidable skip reproduces — the
    /// stage/reason/level/fields the dissolved capture helpers logged.
    struct LeafSkipLog: Sendable {
        let stage: String
        let reason: String
        let level: PrefixCacheDiagnostics.Level
        let extraFields: [(String, String)]

        /// Emit through the request's diagnostics net.
        func emit(in diagnostics: PrefixCacheDiagnostics.Context) {
            diagnostics.logSkip(
                stage: stage, reason: reason, level: level, extraFields: extraFields)
        }
    }

    /// Map a decidable skip to its wire record. The reason carries the payload
    /// (offsets, lengths); the stage prefix follows the boundary mode, exactly as
    /// the dissolved `captureDirectToolLeaf` / `captureCanonicalTemplateLeaf`
    /// helpers did. `.info` is the `logSkip` default those untyped helpers relied
    /// on, made explicit so the level is pinned too. A pure value (no `Context`,
    /// no side effect) so `ServerCompletionLeafSkipLogTests` pins the
    /// byte-for-byte wire format — mirroring `ssdDropReasonString` — and any
    /// future drift (a renamed stage, a flipped level) fails a test rather than
    /// silently shifting dashboards and the diagnostics net.
    static func leafSkipLog(
        for reason: LeafSkipReason,
        mode: BoundaryLeafMode
    ) -> LeafSkipLog {
        let stage = leafStages(for: mode).store
        switch reason {
        case .tokenizationFailed(let error):
            // The probe's chat-template render threw — today's helpers catch this
            // in the same `do/catch` as the prefill, logged as `prefill-threw`.
            return LeafSkipLog(
                stage: stage, reason: "prefill-threw", level: .warning,
                extraFields: [("error", error)]
            )
        case .probeDivergence:
            return LeafSkipLog(
                stage: stage, reason: "probe-divergence-failed", level: .info, extraFields: []
            )
        case .noTransientBoundary:
            return LeafSkipLog(
                stage: stage, reason: "no-transient-boundary-snapshot", level: .info,
                extraFields: []
            )
        case .noResolvedBoundary(let canonicalLen):
            return LeafSkipLog(
                stage: stage, reason: "no-canonical-restore-boundary", level: .info,
                extraFields: [("canonicalLen", "\(canonicalLen)")]
            )
        case .storedAtOrBeforeBoundary(let storedLen, let boundaryOffset):
            return LeafSkipLog(
                stage: stage, reason: "stored-at-or-before-boundary", level: .info,
                extraFields: [
                    ("storedLen", "\(storedLen)"), ("boundaryOffset", "\(boundaryOffset)"),
                ]
            )
        case .canonicalLongerThanStored(let canonicalLen, let storedLen):
            return LeafSkipLog(
                stage: stage, reason: "canonical-longer-than-stored", level: .warning,
                extraFields: [("canonicalLen", "\(canonicalLen)"), ("storedLen", "\(storedLen)")]
            )
        case .renderTranslationFailed(let failure):
            // The probe render's image-placeholder arithmetic disagreed with
            // the request's image table — feature-level skip, request unharmed.
            return LeafSkipLog(
                stage: stage, reason: "render-translation-failed", level: .warning,
                extraFields: [("failure", "\(failure)")]
            )
        case .boundaryInsideImagePrefix(let boundaryOffset, let minimumWarmOffset):
            // The residual would contain an image run, which cannot be
            // reprefilled — expected on image-add turns, hence `.info`.
            return LeafSkipLog(
                stage: stage, reason: "boundary-inside-image-prefix", level: .info,
                extraFields: [
                    ("boundaryOffset", "\(boundaryOffset)"),
                    ("minimumWarmOffset", "\(minimumWarmOffset)"),
                ]
            )
        }
    }

    /// The wire record of a turn the fast path did not take (stage
    /// `liveLeafCapture`): why the turn took the boundary path. The
    /// structural guards keep the reasons and levels ADR-0062 gave them —
    /// an image key space and no fed ids are `.info`
    /// like every other expected skip; a cache offset outside the live path
    /// is the loop and the cache disagreeing about what was fed, always
    /// `.warning`. The render rule (a think-stripping template at a user
    /// boundary) is the expected shape of that template, `.info`. Pure, so
    /// `ServerCompletionLeafSkipLogTests` pins it beside `leafSkipLog`.
    static func liveFallbackLog(
        for reason: LiveLeafCapture.FallbackReason,
        mode: HTTPLeafStoreMode,
        preservesThinking: Bool
    ) -> LeafSkipLog {
        func record(
            _ token: String, _ level: PrefixCacheDiagnostics.Level,
            _ fields: [(String, String)] = []
        ) -> LeafSkipLog {
            LeafSkipLog(
                stage: "liveLeafCapture", reason: token, level: level,
                extraFields: fields + [
                    ("mode", leafStages(for: mode).source),
                    ("preservesThinking", "\(preservesThinking)"),
                ])
        }
        switch reason {
        case .nonIdentityKeySpace:
            return record("non-identity-key-space", .info)
        case .noGeneratedTokens:
            return record("no-generated-tokens", .info)
        case .cacheOffsetOutsideLivePath(let cacheOffset, let promptCount, let liveCount):
            return record(
                "cache-offset-outside-live-path", .warning,
                [
                    ("cacheOffset", "\(cacheOffset)"), ("promptCount", "\(promptCount)"),
                    ("liveCount", "\(liveCount)"),
                ])
        case .thinkStrippingUserBoundary:
            return record("think-stripping-user-boundary", .info)
        }
    }
}

// MARK: - Result assembly

nonisolated extension LeafStorePhase.Result {
    /// Fold what an executor produced: the tuner record when the leaf
    /// survived, and the seed only then — a stored canonical leaf still ends
    /// at the think-strip divergence, and everything past it would re-prefill
    /// interactively on the next user message, so the seed goes to the
    /// post-finish hook to extend the leaf while the GPU is idle (#76).
    fileprivate mutating func conclude(
        _ capture: LeafStorePhase.LeafCapture,
        pendingSeed: SpeculativeCanonicalPrefill.Seed?
    ) {
        leafStore = capture.leafStore
        admission = capture.admission
        if capture.leafStore != nil {
            speculativeSeed = pendingSeed
        } else {
            pendingSeed?.discard()
        }
    }
}

// MARK: - Mode vocabulary

nonisolated extension HTTPLeafStoreMode {
    /// The boundary route's narrower vocabulary; `nil` for the
    /// render-trusting direct mode, which never enters the builder. The one
    /// place that knows `directLeaf` is that mode, so a future
    /// `HTTPLeafStoreMode` surfaces as a compile error here rather than a
    /// silently missed branch.
    var boundaryMode: BoundaryLeafMode? {
        switch self {
        case .directToolLeaf: .directTool
        case .canonicalUserLeaf: .canonical
        case .directLeaf: nil
        }
    }
}
