import CoreImage
import Foundation
import Metal
import MLX
import MLXLMCommon
import Tokenizers
import os

// Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
// swiftlint:disable file_length

/// Workaround for a region-based isolation checker limitation: capturing
/// `HTTPPrefixCacheGeneration` (an `@unchecked Sendable` struct) directly in
/// the driving `Task` fails to compile with "pattern that the region-based
/// isolation checker does not understand how to check" — boxing it in a
/// `Sendable` class routes it past the checker. Do not "simplify" this away
/// without building first.
nonisolated final class UnsafeSendableBox<T>: @unchecked Sendable {
    let value: T

    init(_ value: T) {
        self.value = value
    }
}

/// Output of `ServerCompletion.makeHTTPPrefixCacheGeneration`. Bundles the
/// lower-level MLX generation handles together so the module can drive the
/// stream and capture the final KV cache after generation completes.
///
/// The **Server Completion** module's private cross-step value — it never
/// crosses the module's interface (ADR-0015). Internal (not `private`) only
/// so the sequencing suite can drive a converted arm with a toy-backed
/// **Model Session** and read the resulting handles (ADR-0016).
nonisolated struct HTTPPrefixCacheGeneration: @unchecked Sendable {
    let stream: AsyncStream<RawGeneration>
    let completion: Task<Void, Never>
    /// The app-owned KV cache array the generation runs on. The module
    /// quantizes it *before* building the `TokenIterator` (so the iterator's
    /// decode-time quantization pass cannot swap elements behind our back),
    /// which makes this array the live final cache once `completion`
    /// finishes — the post-generation leaf capture reads it directly.
    /// Replaces the fork's `FinalizedKVCacheHandle` (ADR-0006).
    let finalCache: [any KVCache]
    let diagnosticsContext: PrefixCacheDiagnostics.Context
    let lookupMs: TimeInterval
    let restoreMs: TimeInterval
    let prefillMs: TimeInterval
    /// Seconds `loadSync` spent materializing an SSD-resident body for
    /// this request (`0` for RAM hits and misses). Feeds the
    /// per-completion trace record.
    let hydrationSeconds: TimeInterval
    /// True when the restored snapshot was hydrated from the SSD tier.
    let restoredFromSSD: Bool
    /// Total prompt tokens (full conversation, ignoring slicing).
    let promptTokenCount: Int
    /// Number of leading tokens skipped because the cache already covered them.
    let skippedPrefillTokens: Int
    /// Lookup outcome classification, surfaced for in-app observability.
    let lookupReason: PrefixCacheManager.LookupReason
    /// Shared-prefix length in tokens between the request and the best cache entry.
    let sharedPrefixLength: Int

    // -- Post-generation store context (radix tree flow) --

    /// Flat token sequence for the full prompt (1D extraction from potentially
    /// 2D VLM tensor). REAL prepared tokens — safe to re-forward through the
    /// model (the thinking-safeguard continuation does). Radix-tree paths use
    /// `keySpace.keyPath` instead, which replaces image runs with
    /// digest-derived pseudo-tokens that must never reach an embedding lookup.
    let fullTokens: [Int]
    /// The request's **Cache Key Space** — identity for text-only requests.
    /// Owns the key path the radix tree was driven with and translates the
    /// post-generation stored render the same way.
    let keySpace: CacheKeySpace
    /// Non-nil when this is an **Unkeyed Completion**: no valid Cache Key Path
    /// could be built, so the request was served with zero cache
    /// participation — the post-generation store flow must stay away from the
    /// radix tree entirely.
    let unkeyedReason: CacheKeySpace.UnkeyedReason?
    /// Whether warm forwards must seed the **Position Anchor** (the loaded
    /// family is the recognized vision container). False for text models,
    /// where restored prefills keep their nil-state behavior.
    let seedsPositionAnchor: Bool

    /// The wire string for `Diagnostics.cacheReason`: the lookup outcome, or
    /// the unkeyed degradation when the request never reached the lookup.
    var cacheReasonDescription: String {
        unkeyedReason.map { "unkeyed(\($0.rawValue))" } ?? String(describing: lookupReason)
    }
    /// Validated mid-prefill checkpoint admission, if any checkpoints survived
    /// extraction-edge path validation.
    let snapshotAdmission: SnapshotAdmission?
    /// SSD persistence tier gate, sampled once on `LLMActor`'s own
    /// isolation at `makeHTTPPrefixCacheGeneration` entry. Downstream
    /// post-generation sites (unstripped leaf + stripped leaf) read
    /// this through the captured `mlxStart` instead of re-sampling
    /// the module's `ssdConfig`, which they cannot do without crossing
    /// the Metal-affine scope boundary.
    let ssdEnabled: Bool
    /// Partition key used for cache routing.
    let partitionKey: CachePartitionKey
    /// Request-local helper snapshot captured at the end of the last history
    /// message. Never stored or persisted; used to synthesize the direct
    /// tool-continuation leaf for tool-call turns.
    let transientLastMessageBoundarySnapshot: HybridCacheSnapshot?
    /// Request-local helper snapshot captured at the end of the last real
    /// user message. Never stored or persisted; used to synthesize the
    /// canonical user-continuation leaf for templates that rewrite the
    /// assistant/tool suffix after the last user.
    let transientLastUserBoundarySnapshot: HybridCacheSnapshot?
    /// Chunked prefill step size from the request's `GenerateParameters`,
    /// plumbed out so the post-generation canonical-leaf path can use the
    /// same chunk size when re-prefilling the canonical assistant residual
    /// on top of the restored last-message-boundary snapshot.
    let prefillStepSize: Int
    /// Rank of the processor-prepared token tensor (1 for pure LLMs, 2
    /// for conditional-generation models like Qwen3.5). Post-generation
    /// leaf-capture rebuilds residual inputs from a raw `[Int]` via
    /// `MLXArray(...)` (which is 1D), but the MLXVLM `prepare` indexes
    /// the tensor with two axes — passing 1D there crashes in
    /// `getRopeIndex` on `inputIds.dim(1)`.
    let tokenNDim: Int
}

extension GenerationStreamLoop.RawGenerationHandle {
    /// The server's rich prefill handle collapses to `{ stream, cancel, wait }`;
    /// the prefill/cache metadata stays with the **Server Completion** module
    /// and never crosses the seam.
    fileprivate nonisolated init(_ generation: HTTPPrefixCacheGeneration) {
        self.init(stream: generation.stream, completion: generation.completion)
    }
}

nonisolated enum HTTPLeafContinuationKind: String, Sendable {
    case toolResult
    case userTurn
}

nonisolated enum HTTPLeafStoreMode: String, Sendable {
    case directToolLeaf
    case canonicalUserLeaf
    case directLeaf
}

nonisolated enum VisionPrefixMemoryGuard {
    struct Rejection: Equatable, Sendable {
        let prefixTokens: Int
        let estimatedBytes: UInt64
        let maxBufferBytes: UInt64

        var message: String {
            "vision prefill is too large: \(prefixTokens) image-prefix tokens would allocate "
                + "\(VisionPrefixMemoryGuard.formatBytes(estimatedBytes)) for one "
                + "Qwen3.5/Qwen3.6 full-attention score matrix, above this Mac's Metal "
                + "buffer limit of \(VisionPrefixMemoryGuard.formatBytes(maxBufferBytes))"
        }
    }

    /// The vision-tower analogue of `Rejection`: the request's *combined* image
    /// patches would allocate one `[vision_heads, ΣP, ΣP]` global-attention
    /// score matrix above the Metal single-buffer limit. The per-image cap
    /// bounds one image, but the global ViT attends over every image's patches
    /// jointly, so a many-image turn still re-crosses the cliff — this turns
    /// that corner into a typed, actionable rejection instead of an OOM abort
    /// (ADR-0014).
    struct VisionRejection: Equatable, Sendable {
        let totalPatches: Int
        let estimatedBytes: UInt64
        let maxBufferBytes: UInt64

        var message: String {
            // Lead with the problem and the action so neither is clipped if the
            // banner truncates; the byte/limit detail trails in parentheses.
            "This image set is too large to process. Reduce the number or size of "
                + "the attached images. (\(totalPatches) combined image patches would "
                + "allocate \(VisionPrefixMemoryGuard.formatBytes(estimatedBytes)) for the "
                + "vision tower's attention, above this Mac's Metal buffer limit of "
                + "\(VisionPrefixMemoryGuard.formatBytes(maxBufferBytes)).)"
        }
    }

    static func formatBytes(_ bytes: UInt64) -> String {
        if bytes == UInt64.max { return "more than \(UInt64.max) bytes" }
        let gib = Double(bytes) / 1_073_741_824.0
        return String(format: "%.2f GiB", gib)
    }

    static func rejection(
        prefixTokens: Int,
        profile: ModelIdentity.FullAttentionScratchProfile?,
        maxBufferBytes: UInt64
    ) -> Rejection? {
        guard let profile else { return nil }
        let estimatedBytes = profile.scoreMatrixBytes(sequenceLength: prefixTokens) ?? UInt64.max
        guard estimatedBytes > maxBufferBytes else { return nil }
        return Rejection(
            prefixTokens: prefixTokens,
            estimatedBytes: estimatedBytes,
            maxBufferBytes: maxBufferBytes
        )
    }

    /// The windowed-continuation backstop (ADR-0007 phase 2). The chunked
    /// forward's peak full-attention scratch is `[heads, windowSize,
    /// contextTokens]` — one query window over the whole `[0, contextTokens)`
    /// span — not the single-shot `[heads, L, L]`. So a long image span that
    /// would have tripped the single-shot guard now passes: this fires only when
    /// even one bounded window cannot fit, which is effectively unreachable for
    /// real inputs (hence "rarely-fired backstop").
    static func chunkedRejection(
        windowSize: Int,
        contextTokens: Int,
        profile: ModelIdentity.FullAttentionScratchProfile?,
        maxBufferBytes: UInt64
    ) -> Rejection? {
        guard let profile else { return nil }
        let query = min(max(1, windowSize), max(1, contextTokens))
        let estimatedBytes =
            profile.scoreMatrixBytes(queryLength: query, contextLength: contextTokens)
            ?? UInt64.max
        guard estimatedBytes > maxBufferBytes else { return nil }
        return Rejection(
            prefixTokens: contextTokens,
            estimatedBytes: estimatedBytes,
            maxBufferBytes: maxBufferBytes
        )
    }

    /// Price the vision tower's global-attention score matrix for a forward over
    /// `totalPatches` patches: `[vision_heads, totalPatches, totalPatches]` in
    /// the profile's element size. Rejects when that single buffer would exceed
    /// the Metal limit, *before* the tower runs. `totalPatches` is the patch
    /// count of the images actually fed to this forward (the whole request on a
    /// cold/unkeyed turn; only the newly-added images on a warm continuation,
    /// since earlier images are already in the restored cache and not re-fed).
    /// Inert (`nil`) for an unknown profile, mirroring `rejection` (ADR-0014).
    static func visionRejection(
        totalPatches: Int,
        profile: ModelIdentity.FullAttentionScratchProfile?,
        maxBufferBytes: UInt64
    ) -> VisionRejection? {
        guard let profile else { return nil }
        let estimatedBytes = profile.scoreMatrixBytes(sequenceLength: totalPatches) ?? UInt64.max
        guard estimatedBytes > maxBufferBytes else { return nil }
        return VisionRejection(
            totalPatches: totalPatches,
            estimatedBytes: estimatedBytes,
            maxBufferBytes: maxBufferBytes
        )
    }
}

// Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
// swiftlint:disable type_body_length
/// **Server Completion** — the deep module owning one cache-aware HTTP
/// completion on `LLMActor`'s isolation (CONTEXT.md → Server completion,
/// ADR-0015).
///
/// Non-`Sendable` and actor-confined: `LLMActor` stores it, installs the
/// load-time facts (prefix cache budget, SSD config snapshot, model identity)
/// at model load, and clears it at unload. Every state-touching entry takes an
/// `isolated LLMActor` parameter so module state and every model-affine step
/// stay on the actor's executor — this is a module split, not an isolation
/// split (a second actor was rejected; ADR-0015). The GPU lease, held across
/// the whole HTTP request by `CompletionHandler`, remains the primary guard
/// against unload/reload interleaving.
///
/// The module owns: **Snapshot Resolution** → restore → suffix prefill from
/// the **Prefill Plan** → the **Generation Stream Loop** drive (with the
/// server's sink: accumulator fold + tool-call projection) → **Snapshot
/// Admission** at the MLX edge → **Leaf Capture Plan** execution, plus the
/// prefix-cache admin entries and the snapshot-payload extraction statics.
/// It composes the actor's thinking-continuation primitives for the
/// safeguard's continuation swap.
nonisolated final class ServerCompletion {

    /// The MainActor **current-cache accessor** this module publishes each
    /// freshly built `PrefixCacheManager` into, so cache admin (stats,
    /// telemetry, budget/alpha, flush) reaches the live manager without
    /// tunnelling through the inference actor.
    let cacheAdmin: PrefixCacheAdmin

    init(cacheAdmin: PrefixCacheAdmin) {
        self.cacheAdmin = cacheAdmin
    }

    // MARK: - Load-time facts

    /// Load-time, directory-derived facts about the current model — tool-call
    /// format, Qwen3.5 family/MoE, prompt-starts-thinking, and flop profile.
    /// Installed by the actor's load path; `nil` before load and after unload
    /// (the actor drops the whole module at unload).
    private(set) var modelIdentity: ModelIdentity?

    /// Stable SHA-256 of the loaded model's weight files. Folded into every
    /// `CachePartitionKey` so a weight swap under the same `modelID`
    /// cannot surface stale persisted snapshots.
    private(set) var modelFingerprint: String?

    /// Snapshot of the SSD prefix-cache config captured at load time.
    /// Synchronously readable from inside `container.perform`, which cannot
    /// await MainActor.
    private(set) var ssdConfig: SSDPrefixCacheConfig?

    /// User RAM-budget cap snapshot captured at load time (ADR-0018:
    /// caps, never floors). `nil` = "Automatic (recommended)".
    private(set) var ramBudgetCapBytes: Int?

    /// Headroom source for dynamic ceiling measurement (ADR-0018).
    /// Injected at install time: production (`LLMActor.loadModel`) passes
    /// `MachMemoryHeadroomSource`; the default `nil` disables measurement
    /// so test fixtures keep their exact static budgets — measuring the
    /// live machine inside a unit test collapses the ceiling whenever the
    /// host is short on free RAM (mirrors `SSDPrefixCacheConfig.measuresFreeDisk`).
    private(set) var headroomSource: (any MemoryHeadroomSource)?

    /// Whether the loaded model's template starts generation inside a
    /// `<think>` block. Installed after the container verify.
    private var promptStartsThinking = false
    private var modelWeightBytes: Int64 = 0
    private var defaultPrefixCacheMemoryBudgetBytes =
        LLMActor.Defaults.fallbackPrefixCacheMemoryBudgetBytes

    private var _prefixCache: PrefixCacheManager?

    /// Active-completion registry: the most recent cache-aware start, keyed
    /// by its request ID so the natural-finish clear and the drain can tell
    /// handles apart. The GPU lease — held across the whole HTTP request by
    /// `CompletionHandler` — is the primary guard against unload/reload
    /// interleaving; this handle is the in-actor backstop
    /// `LLMActor.unloadModel` drains (cancel-and-await) before the container
    /// is released, replacing the engine's old fire-and-forget cancel.
    /// Replaced on each start; cleared by the drain and by the driving
    /// task's natural-finish hook (`clearFinishedCompletion`), so an idle
    /// server doesn't retain the finished handle — and through it the
    /// final-cache handle's KV tensors — until the next request.
    private var activeCompletion: (id: UUID, handle: HTTPServerGenerationStart)?

    /// `start` calls still in their heavy restore/prefill phase, which runs
    /// *before* the handle lands in `activeCompletion` — the drain cannot see
    /// those starts through the registry alone, so it parks on this count.
    private var inflightStartCount = 0

    /// Continuations parked by `drainActiveCompletion` until every in-flight
    /// start settles (aborts or registers).
    private var inflightStartWaiters: [CheckedContinuation<Void, Never>] = []

    /// Bumped by every drain. A `start` that observes a bump across its heavy
    /// awaits aborts instead of handing a live generation into model state
    /// that is being torn down.
    private var drainGeneration = 0

    /// Durable per-completion trace log (PRD #82, slice #83): every
    /// finished cache-aware completion appends one
    /// `CompletionTraceRecord` — the replay corpus for the offline
    /// harness and, later, the rebuilt tuner's window food. Owned here
    /// (not per cache) so the corpus spans model loads.
    private let completionTraceLog = CompletionTraceLog()

    /// The background **Speculative Canonical Prefill** task — at most one.
    /// Scheduled only when the module is quiescent (no active completion, no
    /// in-flight start), cancelled-and-awaited by every new generation entry,
    /// and drained the same way on unload. The task observes cancellation
    /// between prefill chunks and then settles — admitting partial progress
    /// past the capture threshold as a RAM-only leaf the preempting request
    /// restores instead of re-prefilling — so a preempting generation
    /// acquires the container actor within ~one chunk plus at most one
    /// capture, and never races past an admission it should have hit.
    private var speculativePrefill: (id: UUID, task: Task<Void, Never>)?

    // MARK: - Install / lifecycle

    /// Single install site for per-load snapshot state. Called from the
    /// actor's `loadModel` before the container load is attempted so the
    /// state is visible even on failed loads; this lets the unit suite
    /// exercise the full config-resolution chain via a fake directory that
    /// trips the container load. The actor's unload path drops the module.
    func installLoadTimeState(
        modelIdentity: ModelIdentity,
        fingerprint: String,
        ssdConfig: SSDPrefixCacheConfig?,
        ramBudgetCapBytes: Int? = nil,
        headroomSource: (any MemoryHeadroomSource)? = nil
    ) {
        self.modelIdentity = modelIdentity
        self.modelFingerprint = fingerprint
        self.ssdConfig = ssdConfig
        self.ramBudgetCapBytes = ramBudgetCapBytes
        self.headroomSource = headroomSource
    }

    /// Container-derived facts, installed by the actor's `verifyAndStore`
    /// after a successful load. Drops any pre-load prefix cache so the next
    /// use rebuilds it with the real FLOP profile and auto-sized budget.
    func installLoadedModelFacts(
        promptStartsThinking: Bool,
        modelWeightBytes: Int64,
        prefixCacheBudgetBytes: Int
    ) {
        self.promptStartsThinking = promptStartsThinking
        self.modelWeightBytes = modelWeightBytes
        self.defaultPrefixCacheMemoryBudgetBytes = prefixCacheBudgetBytes
        self._prefixCache = nil
    }

    /// Cancel-and-await every cache-aware completion the module knows about:
    /// the registered handle plus any `start` still in its pre-registration
    /// restore/prefill phase. Called by `LLMActor.unloadModel` (and by the
    /// engine's unload task ahead of the SSD flush) before the model
    /// container is released, so no in-flight server completion can touch
    /// model state during teardown.
    ///
    /// Reentrancy-safe: the slot is cleared only *after* the awaited handle
    /// has fully finished, and both conditions re-check until the module is
    /// quiescent — a concurrent second drain or a start interleaved during
    /// an await cannot slip past the teardown.
    func drainActiveCompletion(on actor: isolated LLMActor) async {
        drainGeneration += 1
        while inflightStartCount > 0 || activeCompletion != nil || speculativePrefill != nil {
            if let active = activeCompletion {
                active.handle.cancel()
                await active.handle.waitForCompletion()
                if activeCompletion?.id == active.id {
                    activeCompletion = nil
                }
            } else if speculativePrefill != nil {
                await preemptSpeculativePrefill(on: actor)
            } else {
                await withCheckedContinuation { continuation in
                    inflightStartWaiters.append(continuation)
                }
            }
        }
    }

    /// Natural-finish hook from the driving task: drop the registry slot for
    /// `requestID` once its stream has fully completed. Keyed by request ID
    /// so a newer registered start is never dropped by a stale finisher.
    func clearFinishedCompletion(_ requestID: UUID, on actor: isolated LLMActor) {
        if activeCompletion?.id == requestID {
            activeCompletion = nil
        }
    }

    // MARK: - Speculative Canonical Prefill lifecycle

    /// Schedule the background **Speculative Canonical Prefill** for the turn
    /// that just finished (issue #76, ADR-0009). Skips unless the module is
    /// quiescent and no drain ran since the originating `start` — a newer
    /// generation or a teardown always wins; this pass is strictly droppable.
    func scheduleSpeculativePrefill(
        seed: SpeculativeCanonicalPrefill.Seed,
        container: ModelContainer,
        entryDrainGeneration: Int,
        on actor: isolated LLMActor
    ) async {
        // Settle any previous occupant before the quiescence check — the
        // await-everywhere preemption invariant makes a live occupant
        // unreachable here, but the settle suspends, so the guard must run
        // after it to observe any start or drain that interleaved.
        await preemptSpeculativePrefill(on: actor)
        guard drainGeneration == entryDrainGeneration,
            inflightStartCount == 0,
            activeCompletion == nil,
            let prefixCache = _prefixCache
        else {
            seed.discard()
            seed.diagnostics.logSkip(stage: "speculativePrefill", reason: "not-idle")
            return
        }
        let id = UUID()
        let actorRef = actor
        let task = Task {
            // **Stretch Abandonment**'s idle window (issue #100): a timer-
            // triggered seed sleeps before touching the GPU. A follow-up
            // request preempts (cancel-and-await) the sleeping task, so a
            // tool result landing inside the window costs nothing — the
            // pass never starts and the seed's probe is discarded.
            if seed.idleDelay > .zero {
                try? await Task.sleep(for: seed.idleDelay)
                guard !Task.isCancelled else {
                    seed.discard()
                    seed.diagnostics.logSkip(
                        stage: "speculativePrefill",
                        reason: "follow-up-within-idle-window"
                    )
                    await actorRef.clearFinishedSpeculativeServerPrefill(id)
                    return
                }
            }
            await prefixCache.storageActivityGate.withPrefillMarked {
                await SpeculativeCanonicalPrefill.run(
                    seed: seed,
                    container: container,
                    prefixCache: prefixCache
                )
            }
            await actorRef.clearFinishedSpeculativeServerPrefill(id)
        }
        speculativePrefill = (id: id, task: task)
    }

    /// Cancel-and-await any background speculative prefill. `start` calls
    /// this at entry so its lookup sees the settled pass's partial-leaf
    /// admission; generation entries that bypass `start` (the standard
    /// non-cache-aware path) call it through the actor so they never queue
    /// behind background chunks. The wait is bounded by ~one chunk plus at
    /// most one RAM-only capture; the slot is cleared only after the awaited
    /// task has fully finished (same reentrancy contract as the drain).
    func preemptSpeculativePrefill(on actor: isolated LLMActor) async {
        guard let speculative = speculativePrefill else { return }
        speculative.task.cancel()
        await speculative.task.value
        if speculativePrefill?.id == speculative.id {
            speculativePrefill = nil
        }
    }

    /// Natural-finish hook from the speculative task: drop the slot once the
    /// pass has fully finished. Keyed by ID so a newer scheduled pass is
    /// never dropped by a stale finisher.
    func clearFinishedSpeculativePrefill(_ id: UUID, on actor: isolated LLMActor) {
        if speculativePrefill?.id == id {
            speculativePrefill = nil
        }
    }

    // MARK: - Cache-aware completion

    /// Start the HTTP text-based prefix-cache path for `/v1/chat/completions`.
    ///
    /// The request shape is the dispatcher's problem: the **Completion Route**
    /// has already decided this conversation is servable (non-empty, not
    /// assistant-last), so there is no bypass here.
    func start(
        on actor: isolated LLMActor,
        sessions: any ModelSessionProviding,
        modelID: String,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters,
        renderContext: TemplateRenderContext = .canonical,
        progressHandler: ServerInferenceProgressHandler? = nil
    ) async throws -> HTTPServerGenerationStart {
        Memory.cacheLimit = LLMActor.Defaults.cacheLimitMB * 1024 * 1024

        // A new generation always preempts the background speculative pass —
        // interactive work owns the GPU. Cancel-and-await: the pass settles
        // (admitting partial progress as a RAM-only leaf) before this request
        // proceeds, so the lookup below sees that admission instead of racing
        // past it and re-prefilling the same span. Bounded by ~one chunk plus
        // at most one capture; reentrancy keeps the actor free meanwhile.
        await preemptSpeculativePrefill(on: actor)

        // The heavy restore/prefill phase below suspends before the handle is
        // registered. Track the start so a concurrent drain (the unload
        // backstop) can both abort it — via the generation bump checked after
        // the last await — and park until it has stopped touching the
        // container before teardown proceeds.
        let entryDrainGeneration = drainGeneration
        inflightStartCount += 1
        defer {
            inflightStartCount -= 1
            if inflightStartCount == 0 {
                let waiters = inflightStartWaiters
                inflightStartWaiters = []
                for waiter in waiters {
                    waiter.resume()
                }
            }
        }

        let prefixCache = await ensurePrefixCache(on: actor)
        let requestID = UUID()
        let genParams = LLMActor.makeGenerateParameters(from: parameters)
        // Canonicalize tools once so the leaf re-tokenization uses the same dict
        // iteration order as the prefill path inside makeHTTPPrefixCacheGeneration.
        let canonicalTools = LLMActor.canonicalizeToolSpecs(toolSpecs)
        let mlxStart = try await makeHTTPPrefixCacheGeneration(
            on: actor,
            sessions: sessions,
            conversation: conversation,
            requestID: requestID,
            modelID: modelID,
            parameters: genParams,
            toolSpecs: canonicalTools,
            prefixCache: prefixCache,
            renderContext: renderContext,
            progressHandler: progressHandler
        )

        // A drain ran while restore/prefill was suspended: the model is
        // tearing down, so stop the freshly started generation and bail
        // before wiring up a handle nothing would ever drain.
        if drainGeneration != entryDrainGeneration {
            mlxStart.completion.cancel()
            await mlxStart.completion.value
            Memory.clearCache()
            throw CancellationError()
        }

        let (stream, continuation) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
        let loadedModelWeightBytes = modelWeightBytes

        let driver = ManagedGenerationDriver(
            parameters: parameters,
            startsInsideThinkBlock: promptStartsThinking,
            logContext: "request_id=\(requestID.uuidString)"
        )
        let fullTokensForContinuation = mlxStart.fullTokens
        let tokenNDimForContinuation = mlxStart.tokenNDim
        let continuationInjection = driver.safeguard.continuationHandOff
        let continuationToolSpecs = canonicalTools
        let actorRef = actor
        let continuationStarter: @Sendable (String) async throws -> HTTPServerRawGenerationStart = {
            safePrefix in
            try await actorRef.startThinkingContinuationFromTokens(
                originalTokens: fullTokensForContinuation,
                tokenNDim: tokenNDimForContinuation,
                safeThinkingPrefix: safePrefix,
                injection: continuationInjection,
                toolSpecs: continuationToolSpecs,
                parameters: parameters
            )
        }

        // The loop owns the cross-swap cancel invariant (which raw handle is live
        // after an intervention swap). Its `cancelCurrent` must be wired into
        // `start.cancel` synchronously, but the loop isn't built until the task
        // starts — bridge through a late-bound cancel the task fills.
        let loopCancel = LateBoundCancel()

        // Hoisted before the Task so the closure's `mlxStart` capture is its
        // last use — the region-isolation checker rejects a capture-then-use
        // of the @unchecked Sendable struct.
        let cachedTokenCount = mlxStart.skippedPrefillTokens
        let completionDiagnostics = HTTPServerGenerationStart.Diagnostics.fromSeconds(
            lookup: mlxStart.lookupMs,
            restore: mlxStart.restoreMs,
            prefill: mlxStart.prefillMs,
            cacheReason: mlxStart.cacheReasonDescription,
            sharedPrefixLength: mlxStart.sharedPrefixLength,
            promptTokenCount: mlxStart.promptTokenCount
        )

        // The driving work lives in a nonisolated static helper, so it runs
        // off the actor's executor — exactly like the pre-carve driving task,
        // whose capture list omitted `self` and therefore never inherited the
        // actor's isolation. Keeping it off-actor means per-token sink work
        // never serializes against unrelated actor calls; every model-affine
        // step inside hops through `container.perform`, and module state
        // crosses only as the immutable copies captured above. After the
        // drive finishes (naturally or cancelled), the task hops back to the
        // actor to release this request's registry slot.
        // Region-isolation workaround, part 2: the checker rejects sending
        // this argument bundle into the nonisolated callee directly ("pattern
        // that the region-based isolation checker does not understand").
        // Every captured value is an immutable copy or a Sendable handle, so
        // boxing the whole drive closure is safe for the same reason
        // `mlxStartBox` is.
        let mlxStartBox = UnsafeSendableBox(mlxStart)
        let traceLog = completionTraceLog
        let driveBox = UnsafeSendableBox<() async -> Void>({
            await Self.driveCompletion(
                mlxStartBox: mlxStartBox,
                conversation: conversation,
                sessions: sessions,
                canonicalTools: canonicalTools,
                requestID: requestID,
                loadedModelWeightBytes: loadedModelWeightBytes,
                prefixCache: prefixCache,
                renderContext: renderContext,
                traceLog: traceLog,
                driver: driver,
                loopCancel: loopCancel,
                continuationStarter: continuationStarter,
                continuation: continuation,
                finishHook: { await actorRef.clearFinishedServerCompletion(requestID) },
                scheduleSpeculative: { seed in
                    await actorRef.scheduleServerSpeculativePrefill(
                        seed: seed,
                        entryDrainGeneration: entryDrainGeneration
                    )
                }
            )
        })
        let task = Task {
            await driveBox.value()
        }

        let completionStart = ManagedGenerationDriver.makeStart(
            stream: stream,
            continuation: continuation,
            cachedTokenCount: cachedTokenCount,
            diagnostics: completionDiagnostics,
            cancelBridge: loopCancel,
            task: task
        )
        activeCompletion = (id: requestID, handle: completionStart)
        return completionStart
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_body_length cyclomatic_complexity function_parameter_count
    /// Drive one cache-aware completion to its end: the stream-loop run with
    /// the server's sink, snapshot admissions, leaf capture, and the
    /// request-end tuner record. Deliberately nonisolated — see the comment
    /// at the call site's `Task`. `finishHook` runs after every exit path
    /// (natural finish, cancellation, error) and releases this request's
    /// registry slot back on the actor.
    private static func driveCompletion(
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        conversation: HTTPPrefixCacheConversation,
        sessions: any ModelSessionProviding,
        canonicalTools: [ToolSpec]?,
        requestID: UUID,
        loadedModelWeightBytes: Int64,
        prefixCache: PrefixCacheManager,
        renderContext: TemplateRenderContext,
        traceLog: CompletionTraceLog,
        driver: ManagedGenerationDriver,
        loopCancel: LateBoundCancel,
        continuationStarter:
            @escaping @Sendable (String) async throws -> HTTPServerRawGenerationStart,
        continuation: AsyncThrowingStream<AgentGeneration, Error>.Continuation,
        finishHook: @escaping @Sendable () async -> Void,
        scheduleSpeculative: @escaping @Sendable (SpeculativeCanonicalPrefill.Seed) async -> Void
    ) async {
        // swiftlint:enable function_body_length cyclomatic_complexity function_parameter_count
        let mlxStart = mlxStartBox.value
        let diagnosticsContext = mlxStart.diagnosticsContext
        // The accumulator fold and the `.toolCall → HTTPPrefixCacheToolCall`
        // projection are the server's per-event side effects (its sink); the
        // streaming spine itself lives in `GenerationStreamLoop`.
        var accumulator = GenerationAccumulator()
        var toolCalls: [HTTPPrefixCacheToolCall] = []
        // Set only when a canonical leaf landed: the **Speculative Canonical
        // Prefill** seed handed to the post-finish hook (issue #76).
        var speculativeSeed: SpeculativeCanonicalPrefill.Seed?
        // The one home for the trace record's derivation rules: eviction
        // tallies (with their correlated diagnostics events), the
        // restored-offset rule, and the admitted-snapshot projections.
        var trace = CompletionTraceAccumulator()

        drive: do {
            func handle(_ event: AgentGeneration) {
                // Fold shared accumulation (text/thinking/safeguard prefix)
                // in one place. The leaf-store tool-call projection
                // (raw `ToolCall` → `HTTPPrefixCacheToolCall`) stays here, as
                // does the continuation yield that drives downstream
                // consumers (the Requests-log UI).
                accumulator.ingest(event)
                if case .toolCall(let call) = event {
                    toolCalls.append(
                        HTTPPrefixCacheToolCall(
                            name: call.function.name,
                            arguments: call.function.arguments
                        ))
                }
                continuation.yield(event)
            }

            // Restore-state snapshot: what cache state does this generation
            // begin from? Pair this with the silent-close warning to
            // correlate model misbehavior with cache hits (e.g. the Qwen3.6
            // hybrid-linear-attention stale-state bug, jundot/omlx#825).
            Log.agent.info(
                "Generation starting — "
                    + "request_id=\(requestID.uuidString) "
                    + "cached=\(mlxStart.skippedPrefillTokens)/"
                    + "\(mlxStart.promptTokenCount) "
                    + "sharedPrefix=\(mlxStart.sharedPrefixLength) "
                    + "lookup=\(mlxStart.lookupReason) "
                    + "restoreMs=\(String(format: "%.1f", mlxStart.restoreMs * 1000)) "
                    + "prefillMs=\(String(format: "%.1f", mlxStart.prefillMs * 1000))"
            )

            // Drive the shared spine through the Managed Generation Driver.
            // `handle` is the sink (fold + project + yield); the driver's
            // shared tail re-yields the terminal `.info` through it — so
            // CompletionHandler's non-streaming and SSE paths still read
            // final completion metrics from the stream — and emits the
            // completion log and unparsed-tool-call warning. The server
            // always supplies a continuation starter.
            let outcome = try await driver.run(
                initial: .init(mlxStart),
                cancelBridge: loopCancel,
                continuationStarter: { safePrefix in
                    GenerationStreamLoop.RawGenerationHandle(
                        try await continuationStarter(safePrefix)
                    )
                },
                sink: handle
            )

            if outcome.cancelled {
                Memory.clearCache()
                continuation.finish()
                break drive
            }

            if let completionInfo = outcome.completionInfo {
                // Server-local extras: the cache-correlated TTFT event and
                // the raw-chunk debug dump.
                diagnosticsContext.log(
                    PrefixCacheDiagnostics.TTFTEvent(
                        lookupMs: mlxStart.lookupMs,
                        restoreMs: mlxStart.restoreMs,
                        prefillMs: mlxStart.prefillMs,
                        residualPromptMs: completionInfo.promptTime
                    ))
                Log.agent.debug(
                    "Raw library chunks (after ToolCallProcessor):\n\(outcome.diagnostics.rawChunksJoined)"
                )
            } else {
                // Stream closed without an `.info` event from MLX — the case we
                // were previously blind to (jundot/omlx#825: Qwen3.6 hybrid
                // linear attention losing tool-calling after prefix-cache hit).
                // The loop-owned diagnostics plus server-local cache context
                // give the operator one correlatable log cluster.
                let rawChunks = outcome.diagnostics.rawChunksJoined
                let parserState = outcome.diagnostics.finalizeState
                Log.agent.warning(
                    "Generation stream closed without .info event — "
                        + "request_id=\(requestID.uuidString) "
                        + "rawLen=\(rawChunks.count) "
                        + "libraryParsedToolCalls=\(outcome.diagnostics.libraryParsedToolCalls) "
                        + "cachedTokens=\(mlxStart.skippedPrefillTokens)/"
                        + "\(mlxStart.promptTokenCount) "
                        + "lookupReason=\(mlxStart.lookupReason) "
                        + "parserInsideThink=\(parserState.insideThinkBlock) "
                        + "parserThinkClosed=\(parserState.thinkBlockClosed) "
                        + "parserBufferLen=\(parserState.bufferLen) "
                        + "rawTail=\(String(rawChunks.suffix(200)).debugDescription)"
                )
            }

            // -- Post-generation: store snapshots in radix tree --

            // Store mid-prefill snapshots (e.g. stable-prefix boundary) unconditionally.
            // These are captured during prefill and independent of the leaf path — if
            // final-cache recovery or leaf capture fails, the stable-prefix checkpoint
            // still saves future requests from a full re-prefill.
            var storedSnapshotsForTuner: [HybridCacheSnapshot] = []
            if !Task.isCancelled, let admission = mlxStart.snapshotAdmission {
                let diagnostics = await MainActor.run {
                    prefixCache.admit(admission)
                }
                trace.ingest(evictions: diagnostics.evictions, diagnostics: diagnosticsContext)
                storedSnapshotsForTuner = admission.snapshots
            }

            if Task.isCancelled {
                Memory.clearCache()
                continuation.finish()
                break drive
            }

            // Leaf store, wrapped so any skip path falls through to
            // the request-end recordRequest call below — the alpha
            // tuner needs to see every request, not just the ones
            // whose leaf store completed.
            //
            // Canonical leaf policy:
            // - thinking templates store one template-canonical leaf
            //   synthesized from the transient boundary snapshot
            // - non-thinking templates store the direct post-response
            //   leaf captured from the final cache
            var leafStoreForTuner: AlphaTuner.LeafStore?
            leafBlock: do {
                // Skip leaf-store when a thinking-safeguard intervention
                // fired: the continuation ran through the raw path, so the
                // on-device KV cache no longer matches the radix-tree
                // logical snapshot we'd compute from
                // `textContent + thinkingContent + toolCalls`. Storing
                // anything here would corrupt future prefix-cache hits
                // for requests sharing this prefix. The stable-prefix
                // snapshot captured pre-generation is still stored
                // unconditionally earlier in this task, so future requests
                // still benefit from partial cache reuse; only the leaf
                // is lost for this one turn.
                if outcome.intervened {
                    diagnosticsContext.logSkip(
                        stage: "leafStore",
                        reason: "thinking-safeguard-intervention"
                    )
                    break leafBlock
                }

                // An Unkeyed Completion never touches the radix tree —
                // construction failed, so no token path of this request can
                // be trusted as a key.
                if let unkeyedReason = mlxStart.unkeyedReason {
                    diagnosticsContext.logSkip(
                        stage: "leafStore",
                        reason: "unkeyed-completion",
                        extraFields: [("unkeyedReason", unkeyedReason.rawValue)]
                    )
                    break leafBlock
                }

                // 1. Build stored conversation (prompt + generated assistant turn).
                let storedConversation = conversation.appendingAssistant(
                    .assistant(
                        content: accumulator.text,
                        reasoning: accumulator.thinking ?? "",
                        toolCalls: toolCalls
                    ))

                // 2. Re-tokenize stored conversation → flat render sequence,
                // then translate into key space (identity for text-only). The
                // translated path is what every capture offset and admission
                // below keys on — length-equal to the prepared sequence, so
                // key index == KV offset holds.
                guard
                    let storedRenderTokens = await Self.measureStoredTokenSequence(
                        sessions: sessions,
                        conversation: storedConversation,
                        toolSpecs: canonicalTools,
                        renderContext: renderContext
                    )
                else {
                    diagnosticsContext.logSkip(
                        stage: "leafStore",
                        reason: "tokenization-failed",
                        level: .warning
                    )
                    break leafBlock
                }
                let storedTokens: [Int]
                switch mlxStart.keySpace.translate(renderTokens: storedRenderTokens) {
                case .success(let translated):
                    storedTokens = translated
                case .failure(let failure):
                    diagnosticsContext.logSkip(
                        stage: "leafStore",
                        reason: "render-translation-failed",
                        level: .warning,
                        extraFields: [("failure", "\(failure)")]
                    )
                    break leafBlock
                }

                let leafStoreMode = Self.selectHTTPLeafStoreMode(
                    promptStartsThinking: driver.startsInsideThinkBlock,
                    emittedToolCalls: !toolCalls.isEmpty
                )
                diagnosticsContext.log(
                    PrefixCacheDiagnostics.LeafModeEvent(
                        mode: leafStoreMode.rawValue,
                        continuation: toolCalls.isEmpty
                            ? HTTPLeafContinuationKind.userTurn.rawValue
                            : HTTPLeafContinuationKind.toolResult.rawValue
                    ))

                // directLeaf snapshots the live final KV cache (below) and
                // needs none of the builder's probe/boundary/tokenizer work;
                // only the boundary modes route through the GPU-free plan.
                // This mapping is the one place that knows directLeaf is the
                // live-cache path, so a future `HTTPLeafStoreMode` surfaces as
                // a compile error here rather than a silently missed branch.
                let boundaryMode: BoundaryLeafMode? =
                    switch leafStoreMode {
                    case .directToolLeaf: .directTool
                    case .canonicalUserLeaf: .canonical
                    case .directLeaf: nil
                    }
                if let boundaryMode {
                    let transientBoundary: HybridCacheSnapshot? =
                        switch boundaryMode {
                        case .directTool: mlxStart.transientLastMessageBoundarySnapshot
                        case .canonical: mlxStart.transientLastUserBoundarySnapshot
                        }
                    let leafTokenizer = try await sessions.withSession { $0.tokenizer }
                    let leafPlan = await LeafAdmissionBuilder.plan(
                        mode: boundaryMode,
                        storedConversation: storedConversation,
                        storedTokens: storedTokens,
                        toolSpecs: canonicalTools,
                        transientBoundary: transientBoundary,
                        tokenizer: leafTokenizer,
                        keySpace: mlxStart.keySpace,
                        renderContext: renderContext,
                        resolveBoundary: { tokens in
                            // Drive Snapshot Resolution inside the Model
                            // Session so the SSD `loadSync` stays
                            // off-MainActor (ADR-0001). Session entry cannot
                            // fail with a non-throwing body; the hypothetical
                            // failure degrades to "no boundary snapshot".
                            let resolved = try? await sessions.withSession { _ in
                                await prefixCache.resolve(
                                    tokens: tokens,
                                    promptTokenCount: tokens.count,
                                    partitionKey: mlxStart.partitionKey,
                                    modelFingerprint: mlxStart.partitionKey.modelFingerprint,
                                    diagnostics: diagnosticsContext,
                                    pinningRestorePathFor: diagnosticsContext.requestID
                                ).lookup.snapshot
                            }
                            return resolved.flatMap { $0 }
                        }
                    )

                    // One exhaustive switch over the boundary plan: `.skip`
                    // logs the decidable reason; `.fromBoundary` runs the
                    // shared restore→reprefill→capture executor. Both leave
                    // the leaf block — only directLeaf reaches the live
                    // final-cache capture below.
                    switch leafPlan {
                    case .skip(let reason):
                        Self.logLeafSkip(
                            reason, mode: boundaryMode, diagnosticsContext: diagnosticsContext
                        )
                        break leafBlock
                    case .fromBoundary(let boundarySnapshot, let boundaryStoredTokens):
                        // The boundary sits past the image prefix (builder
                        // guard), so the residual is real tokens in both
                        // spaces and the anchor delta is always defined; on
                        // the vision container the residual reprefill must
                        // resume with it seeded.
                        var positionAnchorRopeDelta: Int?
                        if mlxStart.seedsPositionAnchor {
                            guard
                                let delta = mlxStart.keySpace.positionAnchorDelta(
                                    upTo: boundarySnapshot.tokenOffset
                                )
                            else {
                                diagnosticsContext.logSkip(
                                    stage: Self.leafStages(for: boundaryMode).store,
                                    reason: "boundary-splits-image-run",
                                    level: .warning,
                                    extraFields: [("offset", "\(boundarySnapshot.tokenOffset)")]
                                )
                                break leafBlock
                            }
                            positionAnchorRopeDelta = delta
                        }
                        let stages = Self.leafStages(for: boundaryMode)
                        // Seed the **Speculative Canonical Prefill** before
                        // the GPU-side boundary store: the seed spawns the
                        // future-path probe immediately, so its CPU
                        // render+tokenize overlaps the store (#76's earlier
                        // start). Kept only if the leaf store below succeeds.
                        // The worth-it floor differs by trigger: a canonical
                        // leaf IS the strip floor; a tool stretch measures
                        // its rewind span from the last-user boundary.
                        let seedPlan = Self.speculativeSeedPlan(
                            boundaryMode: boundaryMode,
                            renderContext: renderContext
                        )
                        let pendingSeed: SpeculativeCanonicalPrefill.Seed? =
                            seedPlan.map { plan in
                                SpeculativeCanonicalPrefill.makeSeed(
                                    storedConversation: storedConversation,
                                    toolSpecs: canonicalTools,
                                    tokenizer: leafTokenizer,
                                    keySpace: mlxStart.keySpace,
                                    partitionKey: mlxStart.partitionKey,
                                    prefillStepSize: mlxStart.prefillStepSize,
                                    ssdEnabled: mlxStart.ssdEnabled,
                                    seedsPositionAnchor: mlxStart.seedsPositionAnchor,
                                    canonicalLeafOffset: boundaryMode == .canonical
                                        ? boundaryStoredTokens.count
                                        : mlxStart.transientLastUserBoundarySnapshot?
                                            .tokenOffset ?? 0,
                                    renderContext: renderContext,
                                    idleDelay: plan.idleDelay,
                                    ramOnlySpine: plan.ramOnlySpine,
                                    diagnostics: diagnosticsContext
                                )
                            }
                        leafStoreForTuner = await Self.captureStructuredLeafFromBoundary(
                            sessions: sessions,
                            storedTokens: boundaryStoredTokens,
                            boundarySnapshot: boundarySnapshot,
                            positionAnchorRopeDelta: positionAnchorRopeDelta,
                            partitionKey: mlxStart.partitionKey,
                            prefillStepSize: mlxStart.prefillStepSize,
                            tokenNDim: mlxStart.tokenNDim,
                            requestID: requestID,
                            prefixCache: prefixCache,
                            diagnosticsContext: diagnosticsContext,
                            ssdEnabled: mlxStart.ssdEnabled,
                            storeStage: stages.store,
                            captureStage: stages.capture,
                            admissionStage: stages.admission,
                            captureSource: stages.source
                        )
                        // A stored canonical leaf still ends at the
                        // think-strip divergence; everything past it would
                        // re-prefill interactively on the next user message —
                        // hand the seed to the post-finish hook so the pass
                        // can extend the leaf while the GPU is idle (#76).
                        if leafStoreForTuner != nil {
                            speculativeSeed = pendingSeed
                        } else {
                            pendingSeed?.discard()
                        }
                        break leafBlock
                    }
                }

                // The module owns the cache array the generation ran on; the
                // loop's completion task has been awaited by `loop.run`, so
                // the array is no longer being mutated (ADR-0006 — this read
                // replaced the fork's FinalizedKVCacheHandle hand-off).
                guard !Task.isCancelled else {
                    break leafBlock
                }
                let finalCache = mlxStart.finalCache

                let cacheOffsets = httpPrefixCacheOffsets(finalCache)
                guard httpPrefixCacheHasReusableState(finalCache) else {
                    diagnosticsContext.logSkip(
                        stage: "store",
                        reason: "no-reusable-cache-state",
                        extraFields: [("cacheOffsets", "\(cacheOffsets)")]
                    )
                    break leafBlock
                }

                // 3. Offset-alignment guard: if normalization shortened the
                //    stored conversation (whitespace-only assistant content → ""),
                //    we can only trim attention K/V — Mamba's recurrent state
                //    can't be unwound (`canTrimPromptCache` returns `false`).
                //    Trimming the cache and capturing it as a leaf
                //    produces a snapshot whose attention is aligned to
                //    `storedTokens.count` but whose Mamba state
                //    is from the full pre-trim offset. On Qwen3.5 the
                //    resulting leaf hit perturbs raw logits by ~10 even at
                //    trim=1: argmax stays stable (greedy decoding survives),
                //    but the rest of the distribution drifts in a way that
                //    affects sampled decoding. Since the HTTP server
                //    propagates the request's `temperature`/`top_p` and we
                //    can't predict future request sampling params at store
                //    time, the safe choice is to skip the leaf store
                //    entirely when normalization would require any trim.
                //    Lost cache hits on whitespace-normalized conversations
                //    are the trade-off for sampler-agnostic correctness.
                //    Verified by `HybridCacheCorrectnessRunner` test 9 — see
                //    the `leafHitWithNormalizationDivergence...` diagnostics
                //    for the empirical drift measurements.
                let actualCacheOffset = httpPrefixCacheReportedTokenCount(finalCache)
                if actualCacheOffset > storedTokens.count {
                    let trimAmount = actualCacheOffset - storedTokens.count
                    diagnosticsContext.logSkip(
                        stage: "leafStore",
                        reason: "normalization-trim",
                        extraFields: [
                            ("trimAmount", "\(trimAmount)"),
                            ("offsetBefore", "\(actualCacheOffset)"),
                            ("canonicalCount", "\(storedTokens.count)"),
                        ]
                    )
                    break leafBlock
                }

                // 4. Capture leaf snapshot and derive its admission
                //    storage inside a Metal-affine Model Session so any
                //    per-array `asData()` calls run on the inference
                //    thread. `finalCache` is non-`Sendable`
                //    `[any KVCache]` — reached through the boxed
                //    `mlxStart` instead of a direct capture. The offset
                //    guard above ensures no per-layer trimming is
                //    needed before capture.
                let ssdEnabled = mlxStart.ssdEnabled
                let extensionBase = await Self.resolveExtensionBase(
                    ssdEnabled: ssdEnabled,
                    tokens: storedTokens,
                    partitionKey: mlxStart.partitionKey,
                    prefixCache: prefixCache
                )
                let (maybeLeaf, maybeLeafAdmission): (HybridCacheSnapshot?, SnapshotAdmission?) =
                    try await sessions.withSession { session in
                        let cache = mlxStartBox.value.finalCache
                        guard
                            let snap = session.captureSnapshot(
                                cache: cache,
                                offset: storedTokens.count,
                                type: .leaf
                            )
                        else {
                            return (nil, nil)
                        }
                        let storage = Self.snapshotAdmissionStorage(
                            for: snap,
                            ssdEnabled: ssdEnabled,
                            extending: extensionBase
                        )
                        let leafAdmission = SnapshotAdmission.leaf(
                            storedTokens: storedTokens,
                            snapshot: snap,
                            storage: storage,
                            partitionKey: mlxStart.partitionKey,
                            requestID: requestID
                        )
                        return (snap, leafAdmission)
                    }
                guard let leafSnapshot = maybeLeaf else {
                    diagnosticsContext.logSkip(
                        stage: "leafCapture",
                        reason: "unsupported-cache-type",
                        extraFields: [("cacheOffsets", "\(cacheOffsets)")]
                    )
                    break leafBlock
                }
                guard let leafAdmission = maybeLeafAdmission else {
                    diagnosticsContext.logSkip(
                        stage: "leafAdmission",
                        reason: "invalid-path",
                        extraFields: [
                            ("offset", "\(leafSnapshot.tokenOffset)"),
                            ("storedLen", "\(storedTokens.count)"),
                        ]
                    )
                    break leafBlock
                }
                diagnosticsContext.log(
                    PrefixCacheDiagnostics.CaptureEvent(
                        offset: leafSnapshot.tokenOffset,
                        checkpointType: leafSnapshot.checkpointType,
                        bytes: leafSnapshot.memoryBytes,
                        duringPrefill: false,
                        source: "leaf"
                    ))

                // Coalesce admit + stats read in one MainActor
                // hop — saves one cross-actor switch on the success
                // path (the request hot path). Includes the post-store
                // budget/total snapshot so the admission diagnostic can
                // be logged from this actor without another hop.
                let (diagnostics, postStoreBudgetBytes, postStoreSnapshotBytes) =
                    await MainActor.run { () -> (PrefixCacheManager.StoreDiagnostics, Int, Int) in
                        let d = prefixCache.admit(leafAdmission)
                        return (d, prefixCache.memoryBudgetBytes, prefixCache.totalSnapshotBytes)
                    }
                trace.ingest(evictions: diagnostics.evictions, diagnostics: diagnosticsContext)
                trace.logSupersessions(
                    diagnostics.supersededLeaves, diagnostics: diagnosticsContext)
                let directAdmissionEvicted = diagnostics.evictions.contains { event in
                    event.offset == leafSnapshot.tokenOffset
                        && event.checkpointType == .leaf
                }
                if directAdmissionEvicted {
                    diagnosticsContext.logSkip(
                        stage: "leafAdmission",
                        reason: "capturedThenEvicted",
                        level: .warning,
                        extraFields: [
                            ("offset", "\(leafSnapshot.tokenOffset)"),
                            ("bytes", "\(leafSnapshot.memoryBytes)"),
                            ("budgetBytes", "\(postStoreBudgetBytes)"),
                            ("snapshotBytesAfter", "\(postStoreSnapshotBytes)"),
                        ]
                    )
                } else {
                    leafStoreForTuner = AlphaTuner.LeafStore(
                        storedTokens: storedTokens,
                        bytes: leafSnapshot.memoryBytes
                    )
                }

                // Release the MLX free buffer pool back to the OS so it
                // doesn't accumulate transient prefill intermediates
                // across requests.
                Memory.clearCache()
            }

            // Record the request lifecycle for the alpha tuner. Fires
            // for every request, including the leaf-skipped paths
            // — the tuner needs the full workload trace, not just
            // successful leaf stores.
            let capturedSnapshots = storedSnapshotsForTuner
            let leafCapture = leafStoreForTuner
            let unkeyed = mlxStart.unkeyedReason != nil
            let keyPath = mlxStart.keySpace.keyPath
            let (finalStats, finalBudgetBytes, finalEstimates) = await MainActor.run {
                // Unkeyed Completions stay out of the tuner's workload trace —
                // they never participated in the cache this trace models.
                if !unkeyed {
                    prefixCache.recordRequest(
                        partitionKey: mlxStart.partitionKey,
                        promptTokens: keyPath,
                        capturedSnapshots: capturedSnapshots,
                        leafStore: leafCapture,
                        requestID: requestID
                    )
                }
                return (
                    prefixCache.stats,
                    prefixCache.memoryBudgetBytes,
                    prefixCache.evictionConfig.estimates
                )
            }
            diagnosticsContext.log(
                PrefixCacheDiagnostics.MemoryEvent(
                    stats: finalStats,
                    budgetBytes: finalBudgetBytes,
                    modelWeightBytes: loadedModelWeightBytes,
                    activeMlxBytes: Int64(clamping: Memory.activeMemory),
                    peakMlxBytes: Int64(clamping: Memory.peakMemory),
                    mlxCacheLimitBytes: Int64(clamping: Memory.cacheLimit)
                ))

            // Per-completion trace record (PRD #82, slice #83): one line in
            // the replay corpus for every finished cache-aware completion.
            // Unkeyed Completions return nil from `make`; requests whose
            // stream closed without an `.info` event have no TTFT and emit
            // nothing — same condition as the live `ttft` event.
            if let completionInfo = outcome.completionInfo {
                let record = trace.makeRecord(
                    timestamp: Date().timeIntervalSinceReferenceDate,
                    requestID: requestID,
                    modelID: diagnosticsContext.modelID,
                    start: CompletionTraceAccumulator.StartFacts(
                        partitionDigest: mlxStart.partitionKey.partitionDigest,
                        unkeyedReason: mlxStart.unkeyedReason,
                        keyPath: keyPath,
                        lookupReason: mlxStart.lookupReason,
                        restoredFromSSD: mlxStart.restoredFromSSD,
                        hitTokens: mlxStart.skippedPrefillTokens,
                        sharedPrefixLength: mlxStart.sharedPrefixLength,
                        lookupSeconds: mlxStart.lookupMs,
                        restoreSeconds: mlxStart.restoreMs,
                        hydrationSeconds: mlxStart.hydrationSeconds,
                        prefillSeconds: mlxStart.prefillMs
                    ),
                    capturedSnapshots: capturedSnapshots,
                    leafStore: leafCapture,
                    ramBudgetBytes: finalBudgetBytes,
                    residualPromptSeconds: completionInfo.promptTime,
                    deviceEstimates: finalEstimates
                )
                if let record {
                    traceLog.append(record)
                }
            }

            continuation.finish()
        } catch is CancellationError {
            continuation.finish()
        } catch {
            continuation.finish(
                throwing: AgentEngineError.generationFailed(
                    error.localizedDescription
                ))
        }

        // **Stretch Abandonment**, abort arm (issue #100): a client abort or
        // disconnect mid-generation seeds the canonical pass immediately
        // from the request's *completed* messages — the half-generated
        // assistant turn never enters the speculated path. RAM-only spine:
        // if the client merely reconnects and continues, nothing was
        // written to SSD. A drain-driven cancel is also Task.isCancelled
        // here; the scheduler's drain-generation guard discards the seed
        // then, so teardown never runs a pass.
        // An Unkeyed Completion never participates in the cache (by contract),
        // and a seed with no last-user boundary would build a spine from
        // offset 0 — wasted render/tokenize + scheduler churn the downstream
        // `boundary.tokenOffset > 0` resolve guard discards anyway. Gate the
        // arm on the same keyed-with-boundary preconditions the stop-finish
        // seed path already requires.
        if speculativeSeed == nil,
            Task.isCancelled,
            mlxStart.unkeyedReason == nil,
            mlxStart.transientLastUserBoundarySnapshot != nil,
            !renderContext.preservesThinking
        {
            // Session entry cannot fail with a non-throwing body; the
            // hypothetical failure just skips the abort-arm seed.
            let abortTokenizer = try? await sessions.withSession { $0.tokenizer }
            if let abortTokenizer {
                speculativeSeed = SpeculativeCanonicalPrefill.makeSeed(
                    storedConversation: conversation,
                    toolSpecs: canonicalTools,
                    tokenizer: abortTokenizer,
                    keySpace: mlxStart.keySpace,
                    partitionKey: mlxStart.partitionKey,
                    prefillStepSize: mlxStart.prefillStepSize,
                    ssdEnabled: mlxStart.ssdEnabled,
                    seedsPositionAnchor: mlxStart.seedsPositionAnchor,
                    canonicalLeafOffset: mlxStart.transientLastUserBoundarySnapshot?
                        .tokenOffset ?? 0,
                    renderContext: renderContext,
                    ramOnlySpine: true,
                    diagnostics: diagnosticsContext
                )
            }
        }

        // Release this request's Budget Floor restore pins on every exit
        // path (this tail runs after natural finish, cancellation, and
        // error alike — same guarantee as `finishHook`). From here on the
        // turn's protection is the freshest-leaf floor member, not the
        // in-flight pin (ADR-0019).
        await MainActor.run { prefixCache.completeRequest(requestID: requestID) }
        await finishHook()
        // After the registry slot is released: hand the speculative seed to
        // the actor, which schedules it only if the module is still quiescent
        // (a newer start, or a drain since this request entered, wins).
        if let speculativeSeed {
            await scheduleSpeculative(speculativeSeed)
        }
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_body_length cyclomatic_complexity function_parameter_count
    /// Build the lower-level MLX generation pipeline using the radix-tree prefix cache.
    ///
    /// Flow: tokenize full conversation → extract flat token sequence → detect stable
    /// prefix boundary → radix tree lookup → plan checkpoints → slice suffix on hit →
    /// app-driven chunked prefill (captures snapshots at the planned offsets) →
    /// quantize the module-owned cache → TokenIterator on the final token →
    /// start the app-owned generation stream (ADR-0006).
    ///
    /// Bypasses `ChatSession` because its `init(cache:)` path renders only the new
    /// message and drops intermediate history, which produces incoherent output when
    /// the cached state corresponds to a strict prefix of the request rather than the
    /// most recent turn.
    private func makeHTTPPrefixCacheGeneration(
        on actor: isolated LLMActor,
        sessions: any ModelSessionProviding,
        conversation: HTTPPrefixCacheConversation,
        requestID: UUID,
        modelID: String,
        parameters: GenerateParameters,
        toolSpecs: [ToolSpec]?,
        prefixCache: PrefixCacheManager,
        renderContext: TemplateRenderContext = .canonical,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> HTTPPrefixCacheGeneration {
        // swiftlint:enable function_body_length cyclomatic_complexity function_parameter_count
        // Canonicalize tools once so the stable-prefix detector and the real
        // prefill tokenize against identical dict representations. Historically
        // swift-jinja <2.3.5 had non-deterministic `tojson` key ordering; the
        // canonicalization is kept as defense-in-depth and costs almost nothing.
        let canonicalTools = LLMActor.canonicalizeToolSpecs(toolSpecs)

        // Capture module state for the non-MainActor closure below —
        // the closure runs on the **Model Session**'s isolation and cannot
        // sync-read the actor-confined module.
        let promptStartsThinking = self.promptStartsThinking
        let modelFingerprint = self.modelFingerprint
        let imageKeying = self.modelIdentity?.imageKeying
        let flopProfile = self.modelIdentity?.flopProfile ?? .fallback
        let fullAttentionScratchProfile = self.modelIdentity?.fullAttentionScratchProfile
        let visionAttentionScratchProfile = self.modelIdentity?.visionAttentionScratchProfile
        let ssdEnabled = self.ssdConfig?.enabled == true
        let diagnosticsContext = PrefixCacheDiagnostics.Context(
            requestID: requestID,
            modelID: modelID,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize
        )

        return try await sessions.withSession { session in
            func measure<T>(_ work: () throws -> T) rethrows -> (T, TimeInterval) {
                let started = Date.timeIntervalSinceReferenceDate
                let value = try work()
                return (value, Date.timeIntervalSinceReferenceDate - started)
            }

            // 1. Tokenize the full conversation (BEFORE cache lookup). Images
            // ride along positionally: the renderer emits one `"image"` part
            // per attachment and the processor matches them in order.
            // `MessageConverter` already proved each payload `CIImage`-decodable.
            let requestImages = conversation.images
            let userInputImages: [UserInput.Image] = try requestImages.map { image in
                guard let decoded = CIImage(data: image.data) else {
                    throw AgentEngineError.generationFailed(
                        "image attachment no longer decodes (digest \(image.digest.hexString.prefix(8)))"
                    )
                }
                return .ciImage(decoded)
            }
            let fullInput = try await session.prepare(
                UserInput(
                    messages: conversation.promptMessages,
                    images: userInputImages,
                    tools: canonicalTools,
                    additionalContext: renderContext.additionalContext()
                )
            )
            // Sequence length is always the LAST dim. For LLM models tokens are
            // 1D [seq], for VLM models (ParoQuant Qwen35) they are 2D [batch, seq].
            let fullTokenCount = fullInput.text.tokens.dim(-1)
            let tokenNDim = fullInput.text.tokens.ndim

            // 2. Extract flat token sequence for radix tree operations.
            let fullTokens = LLMActor.extractTokenSequence(fullInput.text.tokens)

            // 3. Build the global Marconi partition key for this model
            //    configuration. Cross-session sharing is intentional:
            //    identical prompts under the same model config should
            //    reuse the same radix tree. The conversation's template-
            //    context digest separates render modes (issue #98) — it is
            //    the same digest the handler derived from `renderContext`.
            let partitionKey = CachePartitionKey(
                modelID: modelID,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                modelFingerprint: modelFingerprint,
                templateContextDigest: conversation.templateContextDigest
            )

            // 3b. Build the request's **Cache Key Space** from the prepared
            // tokens, the conversation's images, and the family's image
            // keying. Identity (and free) for text-only requests. A
            // construction failure degrades the whole request to an **Unkeyed
            // Completion** — served normally, zero cache participation.
            let keySpace: CacheKeySpace
            switch CacheKeySpace.make(
                preparedTokens: fullTokens,
                imageDigests: requestImages.map(\.digest),
                imageGrids: (fullInput.image?.frames ?? []).map { frame in
                    let (t, h, w) = frame.values
                    return (t: t, height: h, width: w)
                },
                imageKeying: imageKeying
            ) {
            case .success(let space):
                keySpace = space
            case .failure(let reason):
                return try await Self.makeUnkeyedGeneration(
                    session: session,
                    fullInput: fullInput,
                    fullTokens: fullTokens,
                    reason: reason,
                    parameters: parameters,
                    toolSpecs: canonicalTools,
                    partitionKey: partitionKey,
                    fullAttentionScratchProfile: fullAttentionScratchProfile,
                    visionAttentionScratchProfile: visionAttentionScratchProfile,
                    ssdEnabled: ssdEnabled,
                    diagnosticsContext: diagnosticsContext,
                    progressHandler: progressHandler
                )
            }
            // Grid instrumentation (ADR-0007 phase 2): the processed image grid
            // is the ground truth for the M-RoPE span and the pad-run length the
            // Cache Key Path expands. Logging it cheaply catches the deferred
            // "one screenshot → ~43,500 pad tokens" anomaly with real numbers if
            // it recurs — the chunked continuation already makes such an image
            // non-fatal, so this is observe-only, not a gate.
            if let frames = fullInput.image?.frames, !frames.isEmpty {
                let merge = imageKeying?.spatialMergeSize ?? 1
                let mergeArea = max(1, merge * merge)
                for (index, frame) in frames.enumerated() {
                    let (t, h, w) = frame.values
                    Log.image.debug(
                        "vision grid #\(index): t=\(t) h=\(h) w=\(w) "
                            + "patches=\(frame.product) padRun=\(frame.product / mergeArea) "
                            + "merge=\(merge)"
                    )
                }
            }
            // The recognized vision container mis-positions M-RoPE on any
            // nil-state warm forward — text-only restores included — so the
            // Position Anchor is seeded whenever the family is recognized,
            // not just when this request carries images.
            let seedsPositionAnchor = imageKeying != nil

            // 4. Detect the prefill boundaries (stable prefix + last-message +
            // last-user). The Prefill Planner owns this tokenizer-affine work —
            // including the generation-prompt suffix subtraction and the
            // last-user re-render — in one tested place, against the key
            // space's own path so boundary offsets are key-space offsets by
            // construction.
            let boundaries = try PrefillPlanner.detectBoundaries(
                conversation: conversation,
                toolSpecs: canonicalTools,
                promptStartsThinking: promptStartsThinking,
                tokenizer: session.tokenizer,
                keySpace: keySpace,
                renderContext: renderContext
            )
            if let failure = boundaries.lastUserTranslationFailure {
                diagnosticsContext.logSkip(
                    stage: "lastUserBoundary",
                    reason: "render-translation-failed",
                    level: .warning,
                    extraFields: [("failure", "\(failure)")]
                )
            }

            // 5–6. Resolve the best cached prefix (lookup + lazy SSD hydration),
            // then plan checkpoints against the settled tree. Every radix-tree
            // token path is the **Cache Key Path** — image runs as
            // digest-derived pseudo-tokens, identical to the prepared text
            // everywhere else.
            await progressHandler?(.cacheLookupStarted)
            let lookupStarted = Date.timeIntervalSinceReferenceDate
            // Resolve the best usable snapshot in one place: radix lookup plus
            // lazy SSD hydration (consumed internally — only `.hit`/miss surface
            // here). `loadSync` stays off-MainActor inside this scope per
            // ADR-0001; promote/clear hop to MainActor inside `resolve`.
            let resolved = await prefixCache.resolve(
                tokens: keySpace.keyPath,
                promptTokenCount: fullTokenCount,
                partitionKey: partitionKey,
                modelFingerprint: modelFingerprint,
                diagnostics: diagnosticsContext,
                pinningRestorePathFor: diagnosticsContext.requestID
            )
            let lookupResult = resolved.lookup
            // Plan AFTER resolution, against the settled tree: any promote or
            // forgiving clear has already happened, so the post-hydration-failure
            // replan becomes the ordinary single plan. `resolved.alignmentLookup`
            // carries the SSD-hydrated-hit special case — it aligns against
            // nothing, matching the pre-carve ordering against the unhydrated
            // `.ssdHit`.
            let checkpointPlan = await MainActor.run {
                prefixCache.planCheckpoints(
                    tokens: keySpace.keyPath,
                    stablePrefixOffset: boundaries.stablePrefixOffset,
                    partitionKey: partitionKey,
                    alignTo: resolved.alignmentLookup
                )
            }
            let lookupMs = Date.timeIntervalSinceReferenceDate - lookupStarted

            // 7. Fold resolution + plan into the request's Prefill Plan:
            // restore-vs-cold, the suffix checkpoint filter, the transient
            // boundary offsets, and the single `prefillBaseOffset` (which
            // collapses the old `skippedTokens` / `checkpointBaseOffset` pair).
            // The key space governs image-bearing requests: a hit below the end
            // of the last image run is continued warm through that image
            // (ADR-0007 phase 2), not degraded to cold, while cold checkpoints
            // inside the image prefix are still dropped (uncapturable there).
            let prefillPlan = PrefillPlanner.plan(
                boundaries: boundaries,
                lookupResult: lookupResult,
                checkpointPlan: checkpointPlan,
                promptTokenCount: fullTokenCount,
                keySpace: keySpace
            )
            if !keySpace.isIdentity,
                case .cold = prefillPlan.restore,
                checkpointPlan.count > prefillPlan.checkpointsToCapture.count
            {
                diagnosticsContext.logSkip(
                    stage: "checkpointPlan",
                    reason: "inside-image-prefix",
                    extraFields: [
                        (
                            "dropped",
                            "\(checkpointPlan.count - prefillPlan.checkpointsToCapture.count)"
                        ),
                        ("minimumWarmOffset", "\(keySpace.minimumWarmOffset)"),
                    ]
                )
            }

            // Execute the restore decision (Metal): on a hit, slice the suffix
            // and restore the KV cache; otherwise run cold. Four shapes:
            // - warm, image-free remainder: restore the snapshot, chunk-prefill
            //   the remainder with a seeded **Position Anchor**;
            // - warm below a new image: restore, continue through the image span
            //   `[restore, minimumWarmOffset)` with the windowed vision
            //   continuation anchored at the restored Position Anchor, then
            //   chunk-prefill the text tail (ADR-0007 phase 2);
            // - cold, text-only: chunk-prefill everything from zero;
            // - cold with images: drive the same windowed vision continuation
            //   over the image prefix `[0, minimumWarmOffset)` anchored at zero
            //   (pixels in, M-RoPE correct by construction, scratch bounded to
            //   `[heads, chunk, L]`), then chunk-prefill the text tail with the
            //   continuation's state threaded — the spike's bitwise-verified
            //   cold chain (ADR-0007).
            let inputForGeneration: LMInput
            let cacheToUse: [any KVCache]?
            let restoreMs: TimeInterval
            /// Offset already covered when the (text-tail) executor starts: the
            /// restore offset on an image-free warm restore, the end of the
            /// vendor-continued image span (`minimumWarmOffset`) on an
            /// image-bearing plan, or 0 on a text-only cold run.
            let executionBaseOffset: Int
            /// The image-bearing span `[restore, minimumWarmOffset)` the vendor
            /// continuation forwards (chunked) before the text tail; nil unless
            /// the plan carries an image in the remainder.
            var imagePrefixInput: LMInput?
            /// Position Anchor seeded into the *text* executor on an image-free
            /// warm restore (the continuation path seeds its own anchor, below).
            var executorInitialState: LMOutput.State?
            /// Position Anchor seeded into the vendor continuation — the rope
            /// delta of the images cached before the restore offset (nil ⇒ 0,
            /// the crash-safe cold-from-zero image prefill).
            var imageContinuationAnchor: LMOutput.State?

            // Build the image-bearing span `[restoreOffset, minimumWarmOffset)`
            // carrying only the images whose runs fall in it — those fully
            // before `restoreOffset` are already in the restored cache, so their
            // pixels must not be re-fed (the **Cache Key Space** selects them by
            // index, and the pre-merge `THW.product` rows are skipped from the
            // concatenated pixel tensor). nil for an image-free remainder.
            func imageSpan(from restoreOffset: Int) -> LMInput? {
                let prefixEnd = keySpace.minimumWarmOffset
                guard restoreOffset < prefixEnd,
                    let remainderRange = keySpace.remainderImageIndices(from: restoreOffset),
                    !remainderRange.isEmpty,
                    let image = fullInput.image,
                    let allFrames = image.frames
                else { return nil }
                let spanTokens = fullInput.text.tokens[0..., restoreOffset..<prefixEnd]
                let spanFrames = Array(allFrames[remainderRange])
                let skipPatches = allFrames[..<remainderRange.lowerBound]
                    .reduce(0) { $0 + $1.product }
                let spanPixels =
                    skipPatches == 0 ? image.pixels : image.pixels[skipPatches..., 0...]
                return LMInput(
                    text: LMInput.Text(tokens: spanTokens, mask: nil),
                    image: LMInput.ProcessedImage(pixels: spanPixels, frames: spanFrames)
                )
            }

            switch prefillPlan.restore {
            case .restore(let cacheOffset, let anchorDelta):
                let (restoredCache, measuredRestoreMs) = measure {
                    Self.restoreCache(lookupResult, session: session)
                }
                cacheToUse = restoredCache
                restoreMs = measuredRestoreMs
                if !keySpace.isIdentity,
                    cacheOffset < keySpace.minimumWarmOffset,
                    let span = imageSpan(from: cacheOffset)
                {
                    // Warm restore *below* a new image (ADR-0007 phase 2):
                    // continue through the image span chunked, anchored at the
                    // restored prefix's Position Anchor, then the text tail.
                    let prefixEnd = keySpace.minimumWarmOffset
                    imagePrefixInput = span
                    inputForGeneration = LMInput(
                        text: LMInput.Text(
                            tokens: fullInput.text.tokens[0..., prefixEnd...], mask: nil))
                    executionBaseOffset = prefixEnd
                    if seedsPositionAnchor {
                        imageContinuationAnchor = PositionAnchor.seededState(ropeDelta: anchorDelta)
                    }
                } else {
                    // Image-free remainder: suffix-only prefill. Layers restore
                    // with their absolute logical offset intact, and each
                    // layer's `makeMask` recreates the suffix's causal mask.
                    let slicedTokens: MLXArray =
                        tokenNDim <= 1
                        ? fullInput.text.tokens[cacheOffset...]
                        : fullInput.text.tokens[0..., cacheOffset...]
                    inputForGeneration = LMInput(
                        text: LMInput.Text(tokens: slicedTokens, mask: nil))
                    executionBaseOffset = cacheOffset
                    if seedsPositionAnchor {
                        executorInitialState = PositionAnchor.seededState(ropeDelta: anchorDelta)
                    }
                }
            case .cold where !keySpace.isIdentity:
                // No valid restore: cold image prefill, but driven through the
                // same windowed continuation (anchored at zero) so even the
                // fallback is crash-safe.
                let prefixEnd = keySpace.minimumWarmOffset
                imagePrefixInput =
                    imageSpan(from: 0)
                    ?? LMInput(
                        text: LMInput.Text(
                            tokens: fullInput.text.tokens[0..., ..<prefixEnd], mask: nil),
                        image: fullInput.image)
                inputForGeneration = LMInput(
                    text: LMInput.Text(
                        tokens: fullInput.text.tokens[0..., prefixEnd...], mask: nil))
                cacheToUse = nil
                restoreMs = 0
                executionBaseOffset = prefixEnd
            case .cold:
                inputForGeneration = fullInput
                cacheToUse = nil
                restoreMs = 0
                executionBaseOffset = 0
            }
            let skippedTokens = prefillPlan.prefillBaseOffset
            let newTokensToPrefill = fullTokenCount - skippedTokens
            await progressHandler?(
                .cacheLookupFinished(
                    .init(
                        reason: String(describing: lookupResult.reason),
                        cachedTokens: skippedTokens,
                        sharedPrefixLength: lookupResult.sharedPrefixLength,
                        promptTokens: fullTokenCount,
                        newTokensToPrefill: newTokensToPrefill,
                        lookupMs: lookupMs * 1000,
                        restoreMs: restoreMs * 1000,
                        divergence: lookupResult.divergence
                    )))
            diagnosticsContext.log(
                PrefixCacheDiagnostics.LookupEvent(
                    reason: lookupResult.reason,
                    promptTokens: fullTokenCount,
                    sharedPrefixLength: lookupResult.sharedPrefixLength,
                    skippedPrefillTokens: skippedTokens,
                    newTokensToPrefill: newTokensToPrefill,
                    lookupMs: lookupMs,
                    restoreMs: restoreMs,
                    plannedCheckpoints: prefillPlan.checkpointsToCapture,
                    hydratedFromSSD: resolved.hydratedFromSSD,
                    chainPrefixRestore: resolved.wasChainPrefixRestore,
                    divergence: lookupResult.divergence
                ))

            // 8. Fold the plan's checkpoints plus the transient boundary
            // helpers (captured as leaves; a planned checkpoint at the same
            // offset wins) into one capture map for the prefill driver.
            // Planner guarantees offset uniqueness, so uniqueKeysWithValues
            // traps loudly on a planner-side invariant break instead of
            // silently dropping a candidate.
            let genParams = parameters
            let plannedCheckpoints = Dictionary(
                uniqueKeysWithValues: prefillPlan.checkpointsToCapture.map { ($0.offset, $0.type) }
            )
            let transientOffsets = prefillPlan.transientCheckpointOffsets
            let helperCheckpoints = Dictionary(
                uniqueKeysWithValues: transientOffsets.map {
                    ($0, HybridCacheSnapshot.CheckpointType.leaf)
                }
            )
            let allCheckpoints = plannedCheckpoints.merging(helperCheckpoints) { stored, _ in stored
            }

            // 9. App-owned prefill (ADR-0006): drive chunked forward passes
            // over the suffix, capturing snapshots at the checkpoint offsets,
            // quantize the module-owned cache once, then hand it to a
            // TokenIterator holding only the final prompt token. Quantizing
            // *before* the iterator (with `kvBits` stripped from its
            // parameters) guarantees the iterator never swaps cache elements
            // during decode, so the array this module retains stays the live
            // final cache for the post-generation leaf capture.
            // Shared begin-prefill step. The ADR-0014 guard prices the
            // patches actually fed THIS forward: all images when cold; only
            // the newly-added images on a warm restore, since earlier images
            // are already in the restored cache and not re-fed.
            let begin = try await Self.beginPrefill(
                session: session,
                restoredCache: cacheToUse,
                parameters: genParams,
                promptTokens: fullTokenCount,
                cachedTokens: skippedTokens,
                pricedImage: imagePrefixInput?.image,
                visionAttentionScratchProfile: visionAttentionScratchProfile,
                guardLabel: "keyed",
                diagnosticsContext: diagnosticsContext,
                progressHandler: progressHandler
            )
            var liveCache = begin.cache
            let prefillResult:
                (iterator: StateThreadedTokenIterator, snapshots: [HybridCacheSnapshot])
            do {
                prefillResult =
                    try MLXCheckedEvaluation.withErrors { error in
                        var initialState = executorInitialState
                        var prefixSnapshots: [HybridCacheSnapshot] = []
                        if let imagePrefixInput {

                            // Crash-safe by construction: the continuation chunks
                            // the forward, so the peak full-attention scratch is
                            // bounded to `[heads, window, executionBaseOffset]`,
                            // not the single-shot `[heads, L, L]`.
                            try Self.checkChunkedVisionBackstop(
                                windowSize: genParams.prefillStepSize,
                                contextTokens: executionBaseOffset,
                                profile: fullAttentionScratchProfile,
                                diagnosticsContext: diagnosticsContext
                            )

                            // Warm/cold image span (ADR-0007 phase 2): the windowed
                            // continuation runs the vision tower once, positions the
                            // new image from the restored Position Anchor
                            // (`imageContinuationAnchor`; nil ⇒ anchored at zero, a
                            // crash-safe cold prefill), and chunks the forward so the
                            // scratch is bounded. Its returned state anchors the
                            // chunked text tail. A non-identity key space implies the
                            // recognized vision container, which conforms to
                            // `WindowedVisionContinuation`.
                            guard let continuation = session.windowedVisionContinuation
                            else {
                                throw AgentEngineError.generationFailed(
                                    "loaded model does not support windowed vision continuation"
                                )
                            }
                            guard
                                case .logits(let prepared) = try continuation.prepareContinuation(
                                    imagePrefixInput,
                                    cache: liveCache,
                                    state: imageContinuationAnchor,
                                    windowSize: genParams.prefillStepSize
                                )
                            else {
                                throw AgentEngineError.generationFailed(
                                    "vision container returned .tokens from prepareContinuation"
                                )
                            }
                            try error.check()
                            initialState = prepared.state
                            // A checkpoint at exactly the prefix end is capturable here
                            // (the executor's relative-checkpoint loop only captures
                            // strictly past its base) — capture needs materialized
                            // arrays, so only that branch pays the capture cost.
                            // The previous async scheduling path let MLX errors
                            // escape Swift's scoped handler and terminate the app.
                            // Keep this crash-sensitive image prefix on checked
                            // synchronous evaluation so failures become throws.
                            if let type = allCheckpoints[executionBaseOffset] {
                                try MLXCheckedEvaluation.eval(liveCache)
                                if let snap = session.captureSnapshot(
                                    cache: liveCache, offset: executionBaseOffset, type: type
                                ) {
                                    prefixSnapshots.append(snap)
                                }
                            } else {
                                try MLXCheckedEvaluation.eval(liveCache)
                            }
                        }
                        // Pipeline the image-free text path for TTFT; keep the
                        // image-text-tail (its cache already holds a large image,
                        // so the per-chunk score matrix is large) on checked
                        // synchronous eval so an MLX failure throws not crashes.
                        let warmed = try prefixCache.storageActivityGate.withPrefillMarked {
                            try session.prefill(
                                text: inputForGeneration.text,
                                cache: liveCache,
                                checkpoints: allCheckpoints,
                                checkpointBaseOffset: executionBaseOffset,
                                prefillStepSize: genParams.prefillStepSize,
                                consumeAll: false,
                                initialState: initialState,
                                evalPolicy: imagePrefixInput == nil
                                    ? .pipelined : .checkedSynchronous
                            )
                        }
                        try error.check()
                        session.quantizeKVCache(&liveCache, parameters: genParams)
                        var iteratorParams = genParams
                        iteratorParams.kvBits = nil
                        // The iterator seeds any configured penalty processors with
                        // the full suffix — its own input is only the final prompt
                        // token, which would otherwise be the entire
                        // repetition/presence/frequency context. It threads the last
                        // prefill chunk's state through the prime forward and every
                        // decode step (PRD #72 — upstream's iterator drops it).
                        let iterator = session.makeDecodeIterator(
                            remainder: warmed.remainder,
                            fullText: inputForGeneration.text,
                            cache: liveCache,
                            state: warmed.state,
                            parameters: iteratorParams
                        )
                        return (iterator: iterator, snapshots: prefixSnapshots + warmed.snapshots)
                    }
            } catch is CancellationError {
                // **Salvage-on-cancel** (issue #97): the client is gone and
                // the GPU just went idle at a chunk boundary — keep the
                // progress instead of discarding it. RAM-only, after the
                // cancellation landed, so the cancel path's perceived
                // latency is unchanged; a re-sent request (or an
                // abort-seeded speculative pass) resumes from the salvaged
                // offset instead of the restore floor.
                await Self.salvageCancelledPrefill(
                    cache: liveCache,
                    keySpace: keySpace,
                    restoreBaseOffset: executionBaseOffset,
                    partitionKey: partitionKey,
                    requestID: requestID,
                    prefixCache: prefixCache,
                    diagnostics: diagnosticsContext
                )
                Memory.clearCache()
                throw CancellationError()
            }
            let prefillMs = Date.timeIntervalSinceReferenceDate - begin.startedAt
            let iterator = prefillResult.iterator
            await progressHandler?(
                .prefillFinished(
                    .init(
                        promptTokens: fullTokenCount,
                        cachedTokens: skippedTokens,
                        newTokensToPrefill: newTokensToPrefill,
                        prefillMs: prefillMs * 1000
                    )))

            // Fold the observed prefill into the rolling FLOPs/s estimate
            // (slice #84) — a real measured operation on this device.
            // Tiny residuals are timer noise, not throughput signal.
            if newTokensToPrefill >= 64, prefillMs > 0 {
                let prefillFlops = EvictionPolicy.parentRelativeFlops(
                    nodeOffset: fullTokenCount,
                    parentOffset: skippedTokens,
                    profile: flopProfile
                )
                await MainActor.run {
                    prefixCache.recordPrefillMeasurement(
                        flops: prefillFlops, seconds: prefillMs
                    )
                }
            }

            // 10. Split the driver's snapshots into stored checkpoints vs the
            // request-local transient boundary helpers, then extract payloads
            // inside this `container.perform` so `MLXArray.asData()` runs on
            // the Metal-affine thread before the later MainActor store hop.
            var capturedSnapshots: [HybridCacheSnapshot] = []
            var transientSnapshots: [Int: HybridCacheSnapshot] = [:]
            for snapshot in prefillResult.snapshots {
                if transientOffsets.contains(snapshot.tokenOffset) {
                    transientSnapshots[snapshot.tokenOffset] = snapshot
                } else {
                    capturedSnapshots.append(snapshot)
                }
            }
            let transientLastMessageBoundarySnapshot = prefillPlan.transientBoundaries.lastMessage
                .flatMap { offset in
                    transientSnapshots[offset]
                        ?? capturedSnapshots.first(where: { $0.tokenOffset == offset })
                }
            let transientLastUserBoundarySnapshot = prefillPlan.transientBoundaries.lastUser.flatMap
            { offset in
                transientSnapshots[offset]
                    ?? capturedSnapshots.first(where: { $0.tokenOffset == offset })
            }
            let checkpointCandidates = Self.extractCheckpointAdmissionCandidates(
                capturedSnapshots,
                ssdEnabled: ssdEnabled
            )
            let snapshotAdmission = SnapshotAdmission.checkpoints(
                fullPromptTokens: keySpace.keyPath,
                candidates: checkpointCandidates,
                partitionKey: partitionKey,
                requestID: requestID
            )
            for snapshot in capturedSnapshots {
                diagnosticsContext.log(
                    PrefixCacheDiagnostics.CaptureEvent(
                        offset: snapshot.tokenOffset,
                        checkpointType: snapshot.checkpointType,
                        bytes: snapshot.memoryBytes,
                        duringPrefill: true,
                        source: "prefill"
                    ))
            }

            // 11. Start the app-owned generation stream.
            let (stream, task) = TokenGenerationLoop.start(
                promptTokenCount: fullTokenCount,
                modelConfiguration: session.configuration,
                tokenizer: session.tokenizer,
                iterator: iterator,
                tools: canonicalTools
            )

            return HTTPPrefixCacheGeneration(
                stream: stream,
                completion: task,
                finalCache: liveCache,
                diagnosticsContext: diagnosticsContext,
                lookupMs: lookupMs,
                restoreMs: restoreMs,
                prefillMs: prefillMs,
                hydrationSeconds: resolved.hydrationSeconds,
                restoredFromSSD: resolved.hydratedFromSSD,
                promptTokenCount: fullTokenCount,
                skippedPrefillTokens: skippedTokens,
                lookupReason: lookupResult.reason,
                sharedPrefixLength: lookupResult.sharedPrefixLength,
                fullTokens: fullTokens,
                keySpace: keySpace,
                unkeyedReason: nil,
                seedsPositionAnchor: seedsPositionAnchor,
                snapshotAdmission: snapshotAdmission,
                ssdEnabled: ssdEnabled,
                partitionKey: partitionKey,
                transientLastMessageBoundarySnapshot: transientLastMessageBoundarySnapshot,
                transientLastUserBoundarySnapshot: transientLastUserBoundarySnapshot,
                prefillStepSize: parameters.prefillStepSize,
                tokenNDim: tokenNDim
            )
        }
    }

    /// The shared **begin-prefill step** (PRD #137, PR B): progress event,
    /// cache creation, the vision-tower patch guard, and the prefill timer —
    /// one place for both generation arms, so the ADR-0014 guard invariant
    /// lives once: the global ViT attends over every fed image's patches
    /// jointly in one `[vision_heads, ΣP, ΣP]` matrix no matter how prefill
    /// is driven, so price the combined patch count of THIS forward's images
    /// and reject *before* the tower allocates — a many-image corner
    /// degrades to a typed error instead of an OOM abort.
    ///
    /// `pricedImage` is the image actually fed to this forward (`nil` for a
    /// text-only forward, or when a warm restore already covers every image);
    /// an image with no frames prices as 0 patches, which would silently
    /// disarm the guard — the fail-open tripwire logs loudly instead.
    // Shared preamble mirrors both arms' parameter needs one-to-one.
    // swiftlint:disable:next function_parameter_count
    private static func beginPrefill(
        session: any ModelSession,
        restoredCache: [any KVCache]?,
        parameters: GenerateParameters,
        promptTokens: Int,
        cachedTokens: Int,
        pricedImage: LMInput.ProcessedImage?,
        visionAttentionScratchProfile: ModelIdentity.FullAttentionScratchProfile?,
        guardLabel: String,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> (cache: [any KVCache], startedAt: TimeInterval) {
        await progressHandler?(
            .prefillStarted(
                .init(
                    promptTokens: promptTokens,
                    cachedTokens: cachedTokens,
                    newTokensToPrefill: promptTokens - cachedTokens,
                    prefillMs: nil
                )))
        if let pricedImage {
            let visionFrames = pricedImage.frames ?? []
            if visionFrames.isEmpty {
                Log.server.error(
                    "vision guard (\(guardLabel)): image present but no frames to price — "
                        + "ViT OOM guard inert this forward")
            }
            let visionPatches = visionFrames.reduce(0) { $0 + $1.product }
            if let rejection = VisionPrefixMemoryGuard.visionRejection(
                totalPatches: visionPatches,
                profile: visionAttentionScratchProfile,
                maxBufferBytes: Self.currentMaxMetalBufferBytes()
            ) {
                diagnosticsContext.logSkip(
                    stage: "prefill",
                    reason: "vision-tower-too-large",
                    level: .warning,
                    extraFields: [
                        ("totalPatches", "\(rejection.totalPatches)"),
                        ("estimatedAttentionBytes", "\(rejection.estimatedBytes)"),
                        ("maxBufferBytes", "\(rejection.maxBufferBytes)"),
                    ]
                )
                throw AgentEngineError.generationFailed(rejection.message)
            }
        }
        let cache = restoredCache ?? session.newCache(parameters: parameters)
        return (cache: cache, startedAt: Date.timeIntervalSinceReferenceDate)
    }

    /// The windowed-continuation backstop (ADR-0007 phase 2): reject when
    /// even a single `[heads, window, context]` chunk cannot fit the Metal
    /// buffer limit. Effectively unreachable — the continuation exists to
    /// bound the scratch — but both arms keep it, so the pricing+diagnostic
    /// shape lives once.
    private static func checkChunkedVisionBackstop(
        windowSize: Int,
        contextTokens: Int,
        profile: ModelIdentity.FullAttentionScratchProfile?,
        diagnosticsContext: PrefixCacheDiagnostics.Context
    ) throws {
        guard
            let rejection = VisionPrefixMemoryGuard.chunkedRejection(
                windowSize: windowSize,
                contextTokens: contextTokens,
                profile: profile,
                maxBufferBytes: Self.currentMaxMetalBufferBytes()
            )
        else { return }
        diagnosticsContext.logSkip(
            stage: "prefill",
            reason: "vision-prefix-too-large",
            level: .warning,
            extraFields: [
                ("prefixTokens", "\(rejection.prefixTokens)"),
                ("estimatedAttentionBytes", "\(rejection.estimatedBytes)"),
                ("maxBufferBytes", "\(rejection.maxBufferBytes)"),
            ]
        )
        throw AgentEngineError.generationFailed(rejection.message)
    }

    /// The restore verb with `LookupResult.restoreCache`'s degrade-to-miss
    /// contract: a snapshot whose persisted layers fail restoration is a
    /// cache miss (`nil`), never a crashed request — routed through the
    /// **Model Session** so the sequencing suite observes restore ordering.
    private static func restoreCache(
        _ lookup: PrefixCacheManager.LookupResult,
        session: any ModelSession
    ) -> [any KVCache]? {
        guard let snapshot = lookup.snapshot, lookup.partitionKey != nil else { return nil }
        do {
            return try session.restore(snapshot)
        } catch {
            Log.server.error(
                "snapshot restore failed — treating as cache miss: \(error)"
            )
            return nil
        }
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_body_length function_parameter_count
    /// Serve an **Unkeyed Completion**: a valid Cache Key Path could not be
    /// built for an image-bearing request (unrecognized family, or the
    /// prepared sequence disagreed with the conversation's images), so the
    /// request is served correctly with zero cache participation — no lookup,
    /// no checkpoints, no admission, never a route bounce.
    ///
    /// Image-bearing requests on a `WindowedVisionContinuation` model (today
    /// only the Qwen3.5/3.6 vision container) prefill through the windowed
    /// `prepareContinuation` from zero (state nil ⇒ anchored at offset 0),
    /// bounding the full-attention scratch to `[heads, chunk, L]` under a scoped
    /// MLX error handler with a `VisionPrefixMemoryGuard` backstop — so the
    /// mismatch corner can no longer crash on the single-shot `[heads, L, L]`
    /// allocation (ADR-0007 phase 2). The image-free (or non-conforming)
    /// fallback runs the vendor single-shot `prepare`. Either way decode runs on
    /// the state-threaded iterator so a `.logits` prefill keeps its returned
    /// state. `kvBits` quantization is skipped on this path — there is no
    /// capture to protect, and the degraded corner is not worth a per-step
    /// quantization loop.
    ///
    /// Converted to the **Model Session** seam (ADR-0016): the arm consumes
    /// the port's verbs, so the sequencing suite drives it with the
    /// toy-model-backed session. Internal (not `private`) for that suite;
    /// production reaches it only through `makeHTTPPrefixCacheGeneration`.
    static func makeUnkeyedGeneration(
        session: any ModelSession,
        fullInput: LMInput,
        fullTokens: [Int],
        reason: CacheKeySpace.UnkeyedReason,
        parameters: GenerateParameters,
        toolSpecs: [ToolSpec]?,
        partitionKey: CachePartitionKey,
        fullAttentionScratchProfile: ModelIdentity.FullAttentionScratchProfile?,
        visionAttentionScratchProfile: ModelIdentity.FullAttentionScratchProfile?,
        ssdEnabled: Bool,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> HTTPPrefixCacheGeneration {
        // swiftlint:enable function_body_length function_parameter_count
        diagnosticsContext.logSkip(
            stage: "cacheKeySpace",
            reason: reason.rawValue,
            level: .warning,
            extraFields: [("promptTokens", "\(fullTokens.count)")]
        )

        let fullTokenCount = fullInput.text.tokens.dim(-1)
        var iteratorParams = parameters
        iteratorParams.kvBits = nil
        // Shared begin-prefill step. The ADR-0014 guard runs here — ABOVE the
        // `WindowedVisionContinuation` cast, so a vision model that does NOT
        // conform (and takes the single-shot else-path below) is guarded too.
        // This unkeyed path prefills the whole prompt from zero, so the whole
        // prompt's image is priced.
        let begin = try await beginPrefill(
            session: session,
            restoredCache: nil,
            parameters: parameters,
            promptTokens: fullTokenCount,
            cachedTokens: 0,
            pricedImage: fullInput.image,
            visionAttentionScratchProfile: visionAttentionScratchProfile,
            guardLabel: "unkeyed",
            diagnosticsContext: diagnosticsContext,
            progressHandler: progressHandler
        )
        let cache = begin.cache

        let iterator: StateThreadedTokenIterator
        if fullInput.image != nil,
            let continuation = session.windowedVisionContinuation
        {
            // Image-bearing **Unkeyed Completion** (ADR-0007 phase 2): cache
            // keying failed (e.g. a placeholder/grid mismatch), but the prompt
            // still carries pixels — the vendor's single-shot `prepare` would
            // allocate the crash-prone `[heads, L, L]` full-attention scratch.
            // Drive the windowed vision continuation from zero instead (state
            // nil ⇒ anchored at offset 0), so even this fallback prefills in
            // bounded `[heads, chunk, L]` windows. The backstop guard fires only
            // if a single window cannot fit (effectively unreachable). The whole
            // continuation runs under a scoped MLX error handler so a runtime
            // failure surfaces as a throw, not a process-fatal dispatch.
            try checkChunkedVisionBackstop(
                windowSize: parameters.prefillStepSize,
                contextTokens: fullTokenCount,
                profile: fullAttentionScratchProfile,
                diagnosticsContext: diagnosticsContext
            )
            iterator = try MLXCheckedEvaluation.withErrors { error in
                let built = try session.makePreparingDecodeIterator(
                    fullInput,
                    cache: cache,
                    parameters: iteratorParams,
                    prepare: { input, cache, windowSize in
                        try continuation.prepareContinuation(
                            input, cache: cache, state: nil, windowSize: windowSize
                        )
                    }
                )
                try error.check()
                return built
            }
        } else {
            iterator = try session.makePreparingDecodeIterator(
                fullInput,
                cache: cache,
                parameters: iteratorParams,
                prepare: nil
            )
        }
        let prefillMs = Date.timeIntervalSinceReferenceDate - begin.startedAt
        await progressHandler?(
            .prefillFinished(
                .init(
                    promptTokens: fullTokenCount,
                    cachedTokens: 0,
                    newTokensToPrefill: fullTokenCount,
                    prefillMs: prefillMs * 1000
                )))

        let (stream, task) = TokenGenerationLoop.start(
            promptTokenCount: fullTokenCount,
            modelConfiguration: session.configuration,
            tokenizer: session.tokenizer,
            iterator: iterator,
            tools: toolSpecs
        )

        return HTTPPrefixCacheGeneration(
            stream: stream,
            completion: task,
            finalCache: cache,
            diagnosticsContext: diagnosticsContext,
            lookupMs: 0,
            restoreMs: 0,
            prefillMs: prefillMs,
            hydrationSeconds: 0,
            restoredFromSSD: false,
            promptTokenCount: fullTokenCount,
            skippedPrefillTokens: 0,
            lookupReason: .missNoEntries,
            sharedPrefixLength: 0,
            fullTokens: fullTokens,
            keySpace: .identity(keyPath: fullTokens),
            unkeyedReason: reason,
            seedsPositionAnchor: false,
            snapshotAdmission: nil,
            ssdEnabled: ssdEnabled,
            partitionKey: partitionKey,
            transientLastMessageBoundarySnapshot: nil,
            transientLastUserBoundarySnapshot: nil,
            prefillStepSize: parameters.prefillStepSize,
            tokenNDim: fullInput.text.tokens.ndim
        )
    }

    // MARK: - Prefix Cache Admin

    /// Lazily creates and returns the `PrefixCacheManager`. Initialization requires
    /// a MainActor hop because PrefixCacheManager is `@MainActor`.
    /// The production cache attaches an `AlphaTuner` so eviction `alpha`
    /// adapts to the workload after the first eviction fires. Each cache
    /// owns its **Eviction Configuration**, so a fresh cache starts at the
    /// LRU default (`alpha = 0`) and reads the model's `flopProfile` from
    /// **Model Identity** — there is no global to reset or leak.
    ///
    /// When `ssdConfig?.enabled == true` the manager is composed over
    /// a `TieredSnapshotStore` owning an `SSDSnapshotStore`, and
    /// `warmStart` restores the radix-tree structure from the on-disk
    /// manifest. Warm start is fingerprint-gated: partitions from a
    /// different model layout get their descriptors skipped and
    /// their directories scheduled for async cleanup.
    private func ensurePrefixCache(on actor: isolated LLMActor) async -> PrefixCacheManager {
        if let existing = _prefixCache { return existing }
        let budget = defaultPrefixCacheMemoryBudgetBytes
        let ssdConfigSnapshot = self.ssdConfig
        let fingerprint = self.modelFingerprint
        let flopProfile: ModelFlopProfile
        if let identity = self.modelIdentity {
            flopProfile = identity.flopProfile
        } else {
            // Normally unreachable: the model load installs the identity and
            // `installLoadedModelFacts` nils any pre-load cache, so the cache
            // is built (or rebuilt) once the identity is known. A nil identity
            // here means a pre-load caller (e.g. the E2E budget/alpha tooling)
            // built the cache early; it gets the shared fallback profile until
            // the next load rebuilds it.
            flopProfile = .fallback
            Log.agent.info(
                "PrefixCacheManager built before model identity is known — "
                    + "using the fallback FLOP profile; the cache is rebuilt after load."
            )
        }
        let admin = cacheAdmin
        let ramCap = ramBudgetCapBytes
        let headroom = headroomSource
        // The Storage Activity Gate (PRD #150): shared busy signal
        // between the prefill/hydration paths and the SSD writer's
        // deferred-class scheduling. Created here so the writer and
        // the prefill marks observe the same instance.
        let activityGate = StorageActivityGate()
        let cache = await MainActor.run { () -> PrefixCacheManager in
            let tieredStore = TieredSnapshotStore(
                ssdConfig: ssdConfigSnapshot, activityGate: activityGate
            )
            let cache = PrefixCacheManager(
                memoryBudgetBytes: budget,
                evictionConfig: EvictionConfiguration(flopProfile: flopProfile),
                alphaTuner: AlphaTuner(flopProfile: flopProfile),
                tieredStore: tieredStore,
                // Snapshot Demotion's write-through extraction. Snapshot
                // arrays are deep copies (`HybridCacheSnapshot.capture`),
                // so extracting on MainActor here matches the AlphaTuner
                // replay precedent rather than the container.perform rule
                // for live model state.
                demotionPayloadExtractor: { snapshot in
                    Self.extractSnapshotPayload(snapshot)
                },
                // The Pressure-Reactive Budget's event feed. The manager
                // holds the adapter strongly, so a model unload (which
                // drops the cache) cancels the OS dispatch source too.
                pressureSource: DispatchMemoryPressureSource(),
                // Dynamic Budget Ceilings (ADR-0018): the load-time
                // `budget` above is only the bootstrap — the first
                // admission-driven headroom measurement replaces it.
                // `nil` (test fixtures) keeps the bootstrap static.
                headroomSource: headroom,
                ramBudgetCapBytes: ramCap,
                // Adaptive Write Eagerness (ADR-0019, PRD #150): skip
                // redundant SSD copies while RAM is comfortable; reuse
                // earns a deferred-class promotion write instead.
                adaptiveWriteEagerness: true
            )
            // The current-cache accessor holds it weakly: dropping this
            // module (model unload) reads as "no live cache" over there.
            admin.publish(cache)
            return cache
        }
        if ssdConfigSnapshot?.enabled == true, let fingerprint {
            do {
                try await cache.warmStart(modelFingerprint: fingerprint)
            } catch {
                Log.agent.error(
                    "PrefixCacheManager.warmStart failed: \(String(describing: error))"
                )
            }
        }
        _prefixCache = cache
        return cache
    }

    // MARK: - Snapshot payload extraction statics

    /// Pre-extract checkpoint snapshots into Snapshot Admission
    /// candidates, attaching storage intent to each entry at the
    /// Metal-affine extraction edge.
    ///
    /// **Metal-affinity contract.** Must be called from inside
    /// ``ModelContainer/perform(_:)`` on `LLMActor` — calling it
    /// outside a live Metal-affine scope risks re-issuing command-queue
    /// work on a non-inference thread. The method is `static` so callers
    /// can invoke it synchronously from inside a `container.perform`
    /// closure without an `await`; the Metal affinity is enforced by
    /// convention, not the type system.
    static func extractCheckpointAdmissionCandidates(
        _ snapshots: [HybridCacheSnapshot],
        ssdEnabled: Bool
    ) -> [SnapshotAdmission.CheckpointCandidate] {
        snapshots.map { snapshot in
            return SnapshotAdmission.CheckpointCandidate(
                snapshot: snapshot,
                storage: snapshotAdmissionStorage(
                    for: snapshot,
                    ssdEnabled: ssdEnabled
                )
            )
        }
    }

    /// Resolve the **Leaf Extension Admission** base for a leaf about
    /// to be captured: one hop to the MainActor (radix tree + ledger),
    /// made *before* entering the Metal-affine `container.perform` so
    /// capture closures stay free of cross-actor hops. Every
    /// leaf-capture path (direct, boundary, speculative) funnels
    /// through here. `nil` when `ssdEnabled` is false — the leaf
    /// admits full.
    static func resolveExtensionBase(
        ssdEnabled: Bool,
        tokens: [Int],
        partitionKey: CachePartitionKey,
        prefixCache: PrefixCacheManager
    ) async -> SnapshotExtension? {
        guard ssdEnabled else { return nil }
        return await MainActor.run {
            prefixCache.extensionBase(tokens: tokens, partitionKey: partitionKey)
        }
    }

    /// Internal (not private): the **Speculative Canonical Prefill** executor
    /// derives its leaf admission storage through the same policy.
    ///
    /// `extending` carries the **Leaf Extension Admission** base (the
    /// deepest SSD-backed ancestor leaf, resolved by
    /// `PrefixCacheManager.extensionBase`); when the payload's sliceable
    /// layers can carry just the suffix past it — and that suffix is
    /// worth writing (see `extensionMaxSuffixFraction`) — the payload
    /// admits as an extension. `nil` (every checkpoint, and leaves with
    /// no usable base) admits full.
    static func snapshotAdmissionStorage(
        for snapshot: HybridCacheSnapshot,
        ssdEnabled: Bool,
        extending: SnapshotExtension? = nil
    ) -> SnapshotAdmission.Storage {
        guard ssdEnabled else { return .ramOnly }
        return .ramAndSSD(extractSnapshotPayload(snapshot, extending: extending))
    }

    /// Worth-it gate for a **Leaf Extension Admission**: when the
    /// estimated suffix payload exceeds this fraction of the full
    /// payload (a model whose layers are mostly non-sliceable, or a
    /// near-root base), the leaf admits full — a "delta" that rivals
    /// the full write buys chain complexity for nothing.
    static let extensionMaxSuffixFraction = 0.9

    /// Cache classes whose state arrays carry the token axis at dim −2
    /// and slice cleanly per token range: `KVCacheSimple` trims its
    /// state to `offset`, and `QuantizedKVCache` packs quantization
    /// groups along the head dim (last axis), so a token-axis slice
    /// never splits a group. Rotating (buffer order), chunked
    /// (`startPosition`), and recurrent (whole-prefix state) classes
    /// ride whole in every segment instead.
    private static let suffixSliceableClassNames: Set<String> = [
        "KVCache", "KVCacheSimple", "QuantizedKVCache",
    ]

    /// True when `layer`'s arrays provably cover `[0..snapshotOffset]`
    /// along the token axis so a `(baseOffset..snapshotOffset]` slice is
    /// exact. Defensive shape checks — a layer that fails them simply
    /// rides whole.
    private static func layerIsSuffixSliceable(
        _ layer: HybridCacheSnapshot.LayerState,
        snapshotOffset: Int
    ) -> Bool {
        guard suffixSliceableClassNames.contains(layer.className),
            layer.offset == snapshotOffset,
            !layer.state.isEmpty
        else { return false }
        return layer.state.allSatisfy { array in
            array.ndim >= 3 && array.dim(-2) == snapshotOffset
        }
    }

    /// The validated, worth-it extension for `snapshot`, or `nil` when
    /// the payload should admit full. Pure metadata arithmetic — no
    /// array bytes move here.
    private static func validatedExtension(
        _ extending: SnapshotExtension?,
        for snapshot: HybridCacheSnapshot
    ) -> SnapshotExtension? {
        guard let extending,
            extending.baseOffset > 0,
            extending.baseOffset < snapshot.tokenOffset
        else { return nil }

        var fullBytes = 0
        var suffixBytes = 0
        let suffixFraction =
            Double(snapshot.tokenOffset - extending.baseOffset)
            / Double(snapshot.tokenOffset)
        for layer in snapshot.layers {
            let layerBytes = layer.state.reduce(0) { $0 + $1.nbytes }
            fullBytes += layerBytes
            if layerIsSuffixSliceable(layer, snapshotOffset: snapshot.tokenOffset) {
                suffixBytes += Int(Double(layerBytes) * suffixFraction)
            } else {
                suffixBytes += layerBytes
            }
        }
        guard fullBytes > 0,
            Double(suffixBytes) <= extensionMaxSuffixFraction * Double(fullBytes)
        else { return nil }
        return extending
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_parameter_count
    /// Shared admission tail for structured-leaf executors (the boundary
    /// leaf store and the **Speculative Canonical Prefill**): wrap a
    /// captured leaf in a leaf admission, log the capture, admit on
    /// MainActor, fan out the eviction/supersession diagnostics, and report
    /// whether the admission survived its own eviction pass. Same
    /// Metal-affinity contract as `extractCheckpointAdmissionCandidates`:
    /// call from inside ``ModelContainer/perform(_:)``.
    static func admitStructuredLeaf(
        _ leaf: HybridCacheSnapshot,
        storedTokens: [Int],
        storage: SnapshotAdmission.Storage,
        partitionKey: CachePartitionKey,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnostics: PrefixCacheDiagnostics.Context,
        admissionStage: String,
        captureSource: String
    ) async -> Bool {
        // swiftlint:enable function_parameter_count
        guard
            let admission = SnapshotAdmission.leaf(
                storedTokens: storedTokens,
                snapshot: leaf,
                storage: storage,
                partitionKey: partitionKey,
                requestID: requestID
            )
        else {
            diagnostics.logSkip(
                stage: admissionStage,
                reason: "invalid-path",
                extraFields: [
                    ("offset", "\(leaf.tokenOffset)"),
                    ("storedLen", "\(storedTokens.count)"),
                ]
            )
            return false
        }

        diagnostics.log(
            PrefixCacheDiagnostics.CaptureEvent(
                offset: leaf.tokenOffset,
                checkpointType: leaf.checkpointType,
                bytes: leaf.memoryBytes,
                duringPrefill: false,
                source: captureSource
            ))

        // Coalesce admit + stats read in one MainActor hop; the post-store
        // budget/total snapshot feeds the capturedThenEvicted diagnostic
        // without another hop.
        let (storeDiagnostics, postStoreBudgetBytes, postStoreSnapshotBytes) =
            await MainActor.run { () -> (PrefixCacheManager.StoreDiagnostics, Int, Int) in
                let d = prefixCache.admit(admission)
                return (d, prefixCache.memoryBudgetBytes, prefixCache.totalSnapshotBytes)
            }
        for event in storeDiagnostics.evictions {
            diagnostics.log(PrefixCacheDiagnostics.EvictionEvent(event))
            if let id = event.bodyDroppedSnapshotRefID {
                diagnostics.log(PrefixCacheDiagnostics.SSDBodyDropEvent(id: id))
            }
        }
        for supersession in storeDiagnostics.supersededLeaves {
            diagnostics.log(
                PrefixCacheDiagnostics.LeafSupersessionEvent(
                    offset: supersession.offset,
                    snapshotRefID: supersession.bodyDroppedSnapshotRefID,
                    mode: supersession.mode
                ))
        }
        let admissionEvicted = storeDiagnostics.evictions.contains { event in
            event.offset == leaf.tokenOffset && event.checkpointType == .leaf
        }
        if admissionEvicted {
            diagnostics.logSkip(
                stage: admissionStage,
                reason: "capturedThenEvicted",
                level: .warning,
                extraFields: [
                    ("offset", "\(leaf.tokenOffset)"),
                    ("bytes", "\(leaf.memoryBytes)"),
                    ("budgetBytes", "\(postStoreBudgetBytes)"),
                    ("snapshotBytesAfter", "\(postStoreSnapshotBytes)"),
                ]
            )
            return false
        }
        return true
    }

    static func extractSnapshotPayload(
        _ snapshot: HybridCacheSnapshot,
        extending: SnapshotExtension? = nil
    ) -> SnapshotPayload {
        let activeExtension = validatedExtension(extending, for: snapshot)

        var layers: [SnapshotPayload.LayerPayload] = []
        layers.reserveCapacity(snapshot.layers.count)

        for layer in snapshot.layers {
            var suffixBaseOffset: Int?
            var stateToExtract = layer.state
            if let activeExtension,
                layerIsSuffixSliceable(layer, snapshotOffset: snapshot.tokenOffset)
            {
                suffixBaseOffset = activeExtension.baseOffset
                stateToExtract = layer.state.map { array in
                    array[.ellipsis, activeExtension.baseOffset..<snapshot.tokenOffset, 0...]
                }
            }

            var arrays: [SnapshotPayload.ArrayPayload] = []
            arrays.reserveCapacity(stateToExtract.count)
            for array in stateToExtract {
                let extracted = array.asData(access: .copy)
                arrays.append(
                    SnapshotPayload.ArrayPayload(
                        data: extracted.data,
                        dtype: dtypeWireString(extracted.dType),
                        shape: extracted.shape
                    ))
            }
            layers.append(
                SnapshotPayload.LayerPayload(
                    className: layer.className,
                    state: arrays,
                    metaState: layer.metaState,
                    offset: layer.offset,
                    suffixBaseOffset: suffixBaseOffset
                ))
        }

        return SnapshotPayload(
            tokenOffset: snapshot.tokenOffset,
            checkpointType: snapshot.checkpointType,
            layers: layers,
            extending: activeExtension
        )
    }

    /// Stable wire-format name for an MLX `DType`. Load-bearing: the
    /// result is written into the SSD snapshot header at
    /// `encodePlaceholderContainer(payload:descriptor:)` (in
    /// `PlaceholderContainer.swift`), so the mapping is part of the
    /// on-disk contract. A vendor-side rename of any `DType` case label
    /// would silently corrupt files without this explicit table.
    ///
    /// `@unknown default` traps via `fatalError` rather than inventing
    /// a placeholder string, because reaching it means the vendor
    /// shipped a new case that this table hasn't audited — inventing
    /// a wire name would persist an unreadable header under a claim of
    /// success. The remediation is always "add the case", not "paper
    /// over with a sentinel." Mirrors `DType.init(_ cmlxDtype:)` at
    /// `Vendor/.../mlx-swift/Source/MLX/DType.swift:61`, which uses
    /// the same loud-failure pattern for the C → Swift direction.
    static func dtypeWireString(_ dtype: DType) -> String {
        switch dtype {
        case .bool: return "bool"
        case .uint8: return "uint8"
        case .uint16: return "uint16"
        case .uint32: return "uint32"
        case .uint64: return "uint64"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .float16: return "float16"
        case .float32: return "float32"
        case .bfloat16: return "bfloat16"
        case .complex64: return "complex64"
        case .float64: return "float64"
        @unknown default:
            fatalError(
                "dtypeWireString missing case for MLX DType \(dtype) — "
                    + "extend the switch to preserve the SSD wire-format contract."
            )
        }
    }

    /// Inverse of ``dtypeWireString``. Must stay exhaustive against
    /// the forward table so round-tripping an SSD-resident snapshot
    /// cannot silently lose dtype information; every branch in
    /// `dtypeWireString` has a matching branch here. Returns `nil`
    /// for unknown wire strings so the `SSDSnapshotStore` decoder
    /// can distinguish a parse error from a supported dtype.
    static func dtypeFromWireString(_ wire: String) -> DType? {
        switch wire {
        case "bool": return .bool
        case "uint8": return .uint8
        case "uint16": return .uint16
        case "uint32": return .uint32
        case "uint64": return .uint64
        case "int8": return .int8
        case "int16": return .int16
        case "int32": return .int32
        case "int64": return .int64
        case "float16": return .float16
        case "float32": return .float32
        case "bfloat16": return .bfloat16
        case "complex64": return .complex64
        case "float64": return .float64
        default: return nil
        }
    }

    private static func currentMaxMetalBufferBytes() -> UInt64 {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return UInt64.max
        }
        return UInt64(device.maxBufferLength)
    }

    // MARK: - Leaf store helpers

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

    /// The diagnostics stage labels for a `.fromBoundary` capture, by boundary
    /// leaf mode — the exact strings the dissolved `captureDirectToolLeaf` /
    /// `captureCanonicalTemplateLeaf` helpers passed to the shared executor.
    private static func leafStages(
        for mode: BoundaryLeafMode
            // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
            // swiftlint:disable:next large_tuple
    ) -> (store: String, capture: String, admission: String, source: String) {
        switch mode {
        case .directTool:
            (
                "directToolLeafStore", "directToolLeafCapture", "directToolLeafAdmission",
                "directToolLeaf"
            )
        case .canonical:
            (
                "canonicalLeafStore", "canonicalLeafCapture", "canonicalLeafAdmission",
                "canonicalLeaf"
            )
        }
    }

    /// The exact `logSkip` record a decidable `LeafSkipReason` reproduces — the
    /// stage/reason/level/fields the dissolved capture helpers logged.
    struct LeafSkipLog: Sendable {
        let stage: String
        let reason: String
        let level: PrefixCacheDiagnostics.Level
        let extraFields: [(String, String)]
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

    /// Emit the mapped wire record for a decidable `LeafSkipReason` the **Leaf
    /// Admission Builder** returned, so existing dashboards and the diagnostics
    /// net keep working byte-for-byte.
    private static func logLeafSkip(
        _ reason: LeafSkipReason,
        mode: BoundaryLeafMode,
        diagnosticsContext: PrefixCacheDiagnostics.Context
    ) {
        let record = leafSkipLog(for: reason, mode: mode)
        diagnosticsContext.logSkip(
            stage: record.stage,
            reason: record.reason,
            level: record.level,
            extraFields: record.extraFields
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

    /// Re-tokenize the stored conversation (prompt + generated response) and return
    /// the flat token sequence. The HTTP prefix cache uses raw prompt messages here
    /// so assistant `reasoning_content` and `tool_calls` survive template rendering.
    /// Used for storing the leaf snapshot under the correct radix path.
    /// Returns `nil` on tokenization failure.
    private static func measureStoredTokenSequence(
        sessions: any ModelSessionProviding,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        renderContext: TemplateRenderContext = .canonical
    ) async -> [Int]? {
        do {
            return try await sessions.withSession { session in
                try session.tokenizer.applyChatTemplate(
                    messages: conversation.promptMessages,
                    tools: toolSpecs,
                    additionalContext: renderContext.additionalContext(
                        merging: ["add_generation_prompt": false]
                    )
                )
            }
        } catch {
            Log.agent.warning(
                "Stored token sequence measurement failed — error=\(error.localizedDescription)"
            )
            return nil
        }
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_parameter_count
    /// Restore the boundary snapshot, prefill the residual stored-token suffix,
    /// capture a `.leaf`, and admit it under the given token path. The pure
    /// model-affine executor for a `.fromBoundary` **Leaf Capture Plan**, shared
    /// by the direct-tool and canonical-user modes so both align to the
    /// structured template render, not the raw generated bytes.
    ///
    /// The **Leaf Admission Builder** only emits `.fromBoundary` when
    /// `storedTokens.count > boundary.tokenOffset`, so the residual is non-empty
    /// and no caller-side trim is required. The leaf is captured at
    /// `storedTokens.count` after a clean extension prefill, which works because
    /// each cache type's `update(...)` extends its own state at the absolute
    /// offset.
    private static func captureStructuredLeafFromBoundary(
        sessions: any ModelSessionProviding,
        storedTokens: [Int],
        boundarySnapshot: HybridCacheSnapshot,
        positionAnchorRopeDelta: Int?,
        partitionKey: CachePartitionKey,
        prefillStepSize: Int,
        tokenNDim: Int,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        ssdEnabled: Bool,
        storeStage: String,
        captureStage: String,
        admissionStage: String,
        captureSource: String
    ) async -> AlphaTuner.LeafStore? {
        // swiftlint:enable function_parameter_count
        // The residual is guaranteed non-empty by the builder's offset guard
        // (it only emits `.fromBoundary` when `storedTokens.count > tokenOffset`).
        let boundaryOffset = boundarySnapshot.tokenOffset

        let extensionBase = await Self.resolveExtensionBase(
            ssdEnabled: ssdEnabled,
            tokens: storedTokens,
            partitionKey: partitionKey,
            prefixCache: prefixCache
        )

        do {
            return try await sessions.withSession { session in
                let restoredCache = try session.restore(boundarySnapshot)

                let residual = Array(storedTokens[boundaryOffset...])
                let prefillStart = Date.timeIntervalSinceReferenceDate
                // Qwen3.5 is a `Qwen3_5ForConditionalGeneration` (VLM)
                // whose `prepare` indexes tokens with two axes
                // (`y[0..., ..<step]`) — 1D crashes in `getRopeIndex` on
                // `inputIds.dim(1)`. Pure LLMs use the default
                // `LLMModel.prepare`, which adds the batch dim itself via
                // `.newAxis` and would promote a pre-batched 2D chunk to
                // 3D. Match the processor's original rank.
                let flatInput = MLXArray(residual.map { Int32($0) })
                let inputArr =
                    tokenNDim >= 2
                    ? flatInput.expandedDimensions(axis: 0)
                    : flatInput
                _ = try prefixCache.storageActivityGate.withPrefillMarked {
                    try session.prefill(
                        text: .init(tokens: inputArr, mask: nil),
                        cache: restoredCache,
                        checkpoints: [:],
                        checkpointBaseOffset: boundaryOffset,
                        prefillStepSize: prefillStepSize,
                        consumeAll: true,
                        initialState: positionAnchorRopeDelta.map(PositionAnchor.seededState),
                        evalPolicy: .pipelined
                    )
                }
                let prefillMs = Date.timeIntervalSinceReferenceDate - prefillStart

                guard
                    let leaf = session.captureSnapshot(
                        cache: restoredCache,
                        offset: storedTokens.count,
                        type: .leaf
                    )
                else {
                    diagnosticsContext.logSkip(
                        stage: captureStage,
                        reason: "unsupported-cache-type"
                    )
                    return nil
                }
                Log.agent.info(
                    "\(captureSource) captured — offset=\(leaf.tokenOffset) "
                        + "residualTokens=\(residual.count) "
                        + "prefillMs=\(String(format: "%.3f", prefillMs * 1000)) "
                        + "storedLen=\(storedTokens.count)"
                )

                let survived = await Self.admitStructuredLeaf(
                    leaf,
                    storedTokens: storedTokens,
                    storage: Self.snapshotAdmissionStorage(
                        for: leaf,
                        ssdEnabled: ssdEnabled,
                        extending: extensionBase
                    ),
                    partitionKey: partitionKey,
                    requestID: requestID,
                    prefixCache: prefixCache,
                    diagnostics: diagnosticsContext,
                    admissionStage: admissionStage,
                    captureSource: captureSource
                )
                Memory.clearCache()
                guard survived else { return nil }
                return AlphaTuner.LeafStore(
                    storedTokens: storedTokens,
                    bytes: leaf.memoryBytes
                )
            }
        } catch {
            diagnosticsContext.logSkip(
                stage: storeStage,
                reason: "prefill-threw",
                level: .warning,
                extraFields: [("error", error.localizedDescription)]
            )
            return nil
        }
    }

    // MARK: - Salvage-on-cancel (issue #97)

    /// The offset a cancelled foreground prefill may admit its progress
    /// at, or `nil` when the progress is below the capture threshold
    /// (shared with speculative preempt capture), the cache reports an
    /// offset past the key path (mid-flight inconsistency — never admit),
    /// or the offset sits inside the image prefix (unanchorable).
    /// Pure — unit-tested directly.
    static func salvageableOffset(
        cacheOffset: Int,
        restoreBaseOffset: Int,
        keyPathCount: Int,
        minimumWarmOffset: Int
    ) -> Int? {
        guard
            cacheOffset - restoreBaseOffset
                >= SpeculativeCanonicalPrefill.minimumPreemptCaptureTokens,
            cacheOffset > 0,
            cacheOffset <= keyPathCount,
            cacheOffset >= minimumWarmOffset
        else { return nil }
        return cacheOffset
    }

    /// **Salvage-on-cancel** (PRD #94, issue #97): a client cancel or
    /// disconnect interrupted the foreground prefill between chunks —
    /// capture the cache at the last completed chunk boundary and admit
    /// it RAM-only, so a re-sent request or an abort-seeded speculative
    /// pass resumes there instead of the restore floor. Runs after the
    /// cancellation landed (the GPU is already idle, nobody is waiting on
    /// this request) inside the same Metal-affine scope as the prefill.
    /// RAM-only by design: the imminent retry supersedes this leaf with
    /// its own SSD-backed one, so the payload extraction and disk churn
    /// are both skipped — the same economics as speculative preempt
    /// capture. Below the progress threshold nothing is admitted, leaving
    /// the cancellation contract (no leaf, no trace record) unchanged.
    static func salvageCancelledPrefill(
        cache: [any KVCache],
        keySpace: CacheKeySpace,
        restoreBaseOffset: Int,
        partitionKey: CachePartitionKey,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnostics: PrefixCacheDiagnostics.Context
    ) async {
        let reportedOffset = httpPrefixCacheReportedTokenCount(cache)
        guard
            salvageableOffset(
                cacheOffset: reportedOffset,
                restoreBaseOffset: restoreBaseOffset,
                keyPathCount: keySpace.keyPath.count,
                minimumWarmOffset: keySpace.minimumWarmOffset
            ) != nil
        else {
            diagnostics.logSkip(
                stage: "salvageOnCancel",
                reason: "below-progress-threshold",
                extraFields: [
                    ("cacheOffset", "\(reportedOffset)"),
                    ("restoreBase", "\(restoreBaseOffset)"),
                ]
            )
            return
        }

        // Settle the pipelined chunks only after the cheap offset check
        // proves the cancel progressed far enough to be worth capturing.
        eval(cache)
        let settledOffset = httpPrefixCacheReportedTokenCount(cache)
        guard
            let offset = salvageableOffset(
                cacheOffset: settledOffset,
                restoreBaseOffset: restoreBaseOffset,
                keyPathCount: keySpace.keyPath.count,
                minimumWarmOffset: keySpace.minimumWarmOffset
            )
        else {
            diagnostics.logSkip(
                stage: "salvageOnCancel",
                reason: "settled-offset-not-salvageable",
                extraFields: [
                    ("cacheOffset", "\(settledOffset)"),
                    ("restoreBase", "\(restoreBaseOffset)"),
                ]
            )
            return
        }
        guard
            let leaf = HybridCacheSnapshot.capture(
                cache: cache, offset: offset, type: .leaf
            )
        else {
            diagnostics.logSkip(
                stage: "salvageOnCancel",
                reason: "unsupported-cache-type"
            )
            return
        }
        let survived = await admitStructuredLeaf(
            leaf,
            storedTokens: Array(keySpace.keyPath[0..<offset]),
            storage: .ramOnly,
            partitionKey: partitionKey,
            requestID: requestID,
            prefixCache: prefixCache,
            diagnostics: diagnostics,
            admissionStage: "salvageOnCancel",
            captureSource: "cancelledPrefillSalvage"
        )
        if survived {
            Log.agent.info(
                "Salvage-on-cancel admitted — offset=\(offset) "
                    + "restoreBase=\(restoreBaseOffset) "
                    + "salvagedTokens=\(offset - restoreBaseOffset)"
            )
        }
    }
}
// swiftlint:enable type_body_length
