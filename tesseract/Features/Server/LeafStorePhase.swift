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
        /// The request as Request Keying keyed it: its key space, render and
        /// facts, whole (ADR-0070).
        let request: KeyedRequest
        /// The request's **Cache Claim**, owned by the drive: the boundary
        /// route resolves for it.
        let claim: CacheClaim
        let sessions: any ModelSessionProviding
        let requestID: UUID
        let prefixCache: PrefixCacheManager
        let diagnosticsContext: PrefixCacheDiagnostics.Context
        let containsImages: Bool
        let memory: RequestMemoryTelemetry?
        var mlxStart: HTTPPrefixCacheGeneration { mlxStartBox.value }
    }

    /// The turn facts shared by both routes: the stored conversation (prompt
    /// plus the generated assistant turn), the request's render, the
    /// selected mode and the appended message the fidelity check compares
    /// against. The request's own facts ride on `Inputs.request`.
    struct Turn: Sendable {
        let storedConversation: HTTPPrefixCacheConversation
        let render: ConversationRender
        let mode: HTTPLeafStoreMode
        /// The assistant message appended to the stored conversation.
        let storedMessage: HTTPPrefixCacheMessage
        /// Every id the decode loop fed past the prompt, in order, stop id
        /// included (`GeneratedTokenRecorder`).
        let generatedTokens: [Int]
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); the phase keeps the drive's
    // lenient structural limits — splitting further is deferred. The wide
    // parameter list is the phase's honest input set (the drive's request
    // context); `Inputs` carries its request-constant subset to the helpers.
    // swiftlint:disable function_parameter_count
    static func run(
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        claim: CacheClaim,
        conversation: HTTPPrefixCacheConversation,
        sessions: any ModelSessionProviding,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        assistantText: String,
        assistantReasoning: String?,
        toolCalls: [HTTPPrefixCacheToolCall],
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        trace: inout CompletionTraceAccumulator,
        memory: RequestMemoryTelemetry? = nil
    ) async -> Result {
        // swiftlint:enable function_parameter_count
        let mlxStart = mlxStartBox.value
        var result = Result()
        result.report.generationPrompt = mlxStart.facts.generationPrompt.traceValue

        // An Unkeyed Completion never touches the radix tree — construction
        // failed, so no token path of this request can be trusted as a key.
        let request: KeyedRequest
        switch mlxStart.keying {
        case .unkeyed(let unkeyed):
            result.report.recordSkip(
                LeafSkipLog(
                    stage: "leafStore", reason: "unkeyed-completion", level: .info,
                    extraFields: [("unkeyedReason", unkeyed.reason.rawValue)]),
                in: diagnosticsContext)
            result.report.recordEmittedPathSkip(
                .ineligibleRender, fields: [("cause", "unkeyed")], in: diagnosticsContext)
            return result
        case .keyed(let keyed):
            request = keyed
        }
        let inputs = Inputs(
            mlxStartBox: mlxStartBox, request: request, claim: claim, sessions: sessions,
            requestID: requestID, prefixCache: prefixCache, diagnosticsContext: diagnosticsContext,
            containsImages: conversation.messages.contains { !$0.images.isEmpty }, memory: memory)
        let render = request.render

        // 1. Build the stored conversation (prompt + generated assistant turn).
        let storedMessage = HTTPPrefixCacheMessage.assistant(
            content: assistantText,
            reasoning: assistantReasoning ?? "",
            toolCalls: toolCalls
        )
        let storedConversation = conversation.appendingAssistant(storedMessage)

        let leafStoreMode = request.facts.leafStoreMode(emittedToolCalls: !toolCalls.isEmpty)
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
            generatedTokens: mlxStart.generatedTokens.snapshot
        )

        // 2. The fast-path eligibility (GPU-free, no render): the live final
        // cache is the leaf unless a structural guard or the render rule
        // sends the turn to the boundary path.
        let preservesThinking = render.renderContext.preservesThinking
        let decision = LiveLeafCapture.decide(
            mode: leafStoreMode,
            preservesThinking: preservesThinking,
            promptKeyPath: inputs.request.keySpace.keyPath,
            generatedTokens: turn.generatedTokens,
            cacheOffset: httpPrefixCacheReportedTokenCount(mlxStart.finalCache)
        )
        // The live leaf a think-stripping boundary turn checks in for its
        // transient views; released once the canonical leaf backs them.
        var boundaryBackingLeafPath: [Int]?
        switch decision {
        case .live(let offset):
            await storeLive(offset: offset, turn: turn, inputs: inputs, result: &result)
        case .boundary(let reason):
            // The render rule rejects canonical reuse of the generated tail,
            // not the already-validated fed path. Check in its full leaf at
            // this quiescent point before consuming request-local views; the
            // canonical leaf takes over as their backer and the live leaf is
            // released below (ADR-0068 amendment).
            if reason == .thinkStrippingUserBoundary,
                [
                    mlxStart.transientLastUserBoundarySnapshot,
                    mlxStart.transientLastMessageBoundarySnapshot,
                ].contains(where: { $0?.isPrefixView == true })
            {
                let offset = httpPrefixCacheReportedTokenCount(mlxStart.finalCache)
                let path = LiveLeafCapture.livePath(
                    promptKeyPath: inputs.request.keySpace.keyPath,
                    generatedTokens: turn.generatedTokens, offset: offset)
                let backer = await captureLiveLeaf(
                    sessions: sessions, mlxStartBox: mlxStartBox,
                    context: LeafAdmissionContext(
                        storedTokens: path, inputs: inputs,
                        stages: LeafStages(
                            store: "boundaryBackingLeafStore",
                            capture: "boundaryBackingLeafCapture",
                            admission: "boundaryBackingLeafAdmission",
                            source: "boundaryBackingLeaf"), ssdEnabled: false), path: .live)
                if let admission = backer.admission {
                    trace.ingest(evictions: admission.evictions, diagnostics: diagnosticsContext)
                    trace.logSupersessions(
                        admission.supersededLeaves, diagnostics: diagnosticsContext)
                }
                if backer.leafStore != nil { boundaryBackingLeafPath = path }
            }
            // A boundary or intervened turn must return the original leaf
            // before running the existing restore-and-re-prefill strategy:
            // the claim's explicit rewind step.
            let checkedOutOffset = await sessions.withSession { session in
                await inputs.claim.rewind(in: session)
            }
            let record = liveFallbackLog(
                for: reason, mode: leafStoreMode, preservesThinking: preservesThinking)
            record.emit(in: diagnosticsContext)
            result.report.boundaryReason = record.reason
            LeafStoreCounters.shared.noteBoundaryTurn(reason: record.reason)
            result.report.recordEmittedPathSkip(
                EmittedPathRegistration.skipReason(for: reason), in: diagnosticsContext)
            if let checkedOutOffset, turn.mode.boundaryMode == nil {
                // A structural guard rejected the direct live path. Its
                // rendered tokens cannot prove the cache's current state;
                // preserve the original leaf instead of capturing after rewind.
                result.report.path = .rewind
                result.report.leafOffset = checkedOutOffset
            } else {
                await storeFromBoundary(turn: turn, inputs: inputs, result: &result)
            }
        }

        if let admission = result.admission {
            trace.ingest(evictions: admission.evictions, diagnostics: diagnosticsContext)
            trace.logSupersessions(admission.supersededLeaves, diagnostics: diagnosticsContext)
        }
        if let boundaryBackingLeafPath, let canonical = result.leafStore?.storedTokens {
            if let released = await inputs.prefixCache.releaseBoundaryBackingLeaf(
                path: boundaryBackingLeafPath, sparing: canonical,
                partitionKey: inputs.request.facts.partitionKey)
            {
                trace.logSupersessions([released], diagnostics: diagnosticsContext)
            } else {
                result.report.recordSkip(
                    LeafSkipLog(
                        stage: "boundaryBackingLeafRelease", reason: "not-releasable",
                        level: .info,
                        extraFields: [("offset", "\(boundaryBackingLeafPath.count)")]),
                    in: diagnosticsContext)
            }
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
        // 1. Register the turn's Emitted Path.
        registerEmittedPath(turn: turn, inputs: inputs, report: &result.report)

        // 2/3. Capture at the cache's offset, admit under the live path.
        let livePath = LiveLeafCapture.livePath(
            promptKeyPath: inputs.request.keySpace.keyPath,
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
            context: LeafAdmissionContext(storedTokens: livePath, inputs: inputs, stages: stages),
            path: .live)
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
        switch inputs.request.keySpace.translate(renderTokens: storedRenderTokens) {
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
            backingLeaf: route.backingLeaf,
            positionAnchorRopeDelta: route.positionAnchorRopeDelta,
            prefillStepSize: inputs.request.facts.prefillStepSize,
            tokenNDim: inputs.request.facts.tokenNDim,
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
        let backingLeaf: HybridCacheSnapshot?
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
            keySpace: inputs.request.keySpace,
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
        // Keep resolution's Backing Leaf alongside a view until materialization.
        // Every call runs inside the Model Session so SSD reads stay off MainActor.
        let resolveBoundary:
            @Sendable ([Int], HybridCacheSnapshot?) async -> PrefixCacheManager.LookupResult? = {
                tokens, view in
                let matchingView = view.flatMap { boundary in
                    guard inputs.request.keySpace.keyPath.count >= boundary.tokenOffset,
                        tokens.starts(
                            with: inputs.request.keySpace.keyPath.prefix(boundary.tokenOffset))
                    else { return Optional<HybridCacheSnapshot>.none }
                    return boundary
                }
                return await inputs.sessions.withSession { _ in
                    await inputs.prefixCache.resolve(
                        tokens: tokens,
                        promptTokenCount: tokens.count,
                        partitionKey: inputs.request.facts.partitionKey,
                        modelFingerprint: inputs.request.facts.partitionKey.modelFingerprint,
                        diagnostics: diagnostics,
                        transientBoundary: matchingView,
                        for: inputs.claim
                    ).lookup
                }
            }
        let plan = await LeafAdmissionBuilder.plan(
            mode: mode,
            probedTokens: probedTokens,
            transientBoundary: transientBoundary,
            keySpace: inputs.request.keySpace,
            resolveBoundary: { await resolveBoundary($0, nil)?.snapshot }
        )
        switch plan {
        case .skip(let reason):
            report.recordSkip(leafSkipLog(for: reason, mode: mode), in: diagnostics)
            return nil
        case .fromBoundary(let plannedBoundary, let tokens):
            var boundary = plannedBoundary
            var backingLeaf: HybridCacheSnapshot?
            if boundary.isPrefixView {
                // Request-local boundaries never enter the tree. Resolve the
                // view against its current Backing Leaf, or use the existing
                // shallower boundary re-prefill when that leaf is unavailable.
                guard
                    let resolved = await resolveBoundary(
                        Array(tokens.prefix(boundary.tokenOffset)), boundary),
                    let snapshot = resolved.snapshot,
                    snapshot.tokenOffset > 0, snapshot.tokenOffset < tokens.count,
                    snapshot.tokenOffset >= inputs.request.keySpace.minimumWarmOffset
                else {
                    report.recordSkip(
                        leafSkipLog(
                            for: .noResolvedBoundary(canonicalLen: tokens.count), mode: mode),
                        in: diagnostics)
                    return nil
                }
                boundary = snapshot
                backingLeaf = resolved.backingLeaf
            }
            // The boundary sits past the image prefix (builder guard), so the
            // residual is real tokens in both spaces and the anchor delta is
            // always defined; on the vision container the residual reprefill
            // must resume with it seeded.
            var positionAnchorRopeDelta: Int?
            if inputs.request.seedsPositionAnchor {
                guard
                    let delta = inputs.request.keySpace.positionAnchorDelta(
                        upTo: boundary.tokenOffset)
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
                boundary: boundary, backingLeaf: backingLeaf,
                positionAnchorRopeDelta: positionAnchorRopeDelta,
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
                promptPath: inputs.request.keySpace.renderSpacePath,
                generatedTokens: turn.generatedTokens,
                stoppedOn: mlxStart.generatedTokens.stopToken,
                toolCallFormat: inputs.request.facts.toolCallFormat,
                tools: render.toolSpecs,
                generationPrompt: inputs.request.facts.generationPrompt
            ))
        let seconds = secondsSince(start)
        EmittedPathRegistration.emit(outcome, registerSeconds: seconds, in: diagnostics)
        report.absorbEmittedPath(outcome, registerSeconds: seconds)
    }

    // MARK: - Mode selection

    /// The one leaf-store mode rule (ADR-0070), read by the Leaf Store after
    /// the turn, by MTP engagement before it (with defined tools counting as
    /// called), and by the replay harness. A tool-call turn renders verbatim
    /// for its result, so it takes the direct tool leaf. A stop turn takes
    /// the canonical user leaf whenever its prompt carries a think block,
    /// open or closed, under a think-stripping render: the template drops
    /// either from history once a new user message arrives, so the leaf has
    /// to be the re-rendered form. Under the Preserve-Thinking Render a
    /// closed block stays verbatim and takes the direct leaf; an open one
    /// keeps the canonical user leaf, which the fast path captures live. A
    /// prompt with no think block takes the direct leaf, and an unknown one
    /// the canonical user leaf, which is correct for any template.
    static func selectHTTPLeafStoreMode(
        generationPrompt: GenerationPrompt,
        renderContext: TemplateRenderContext,
        emittedToolCalls: Bool
    ) -> HTTPLeafStoreMode {
        if emittedToolCalls {
            return .directToolLeaf
        }
        switch generationPrompt.thinkBlock {
        case .opens, .unknown:
            return .canonicalUserLeaf
        case .closed:
            return renderContext.preservesThinking ? .directLeaf : .canonicalUserLeaf
        case .none:
            return .directLeaf
        }
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
            request: inputs.request,
            canonicalLeafOffset: leafOffset,
            transientBoundary: mlxStart.transientLastUserBoundarySnapshot,
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
