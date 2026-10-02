import Foundation
import Metal
import MLX
import MLXLMCommon
import MLXNN
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
    let finalCacheOwner: FinalGenerationCache
    var finalCache: [any KVCache] { finalCacheOwner.cache }
    /// The iterator this request actually used. Loaded drafter capability
    /// alone does not imply engagement (warm or sampled MTP requests, for
    /// example, decode ordinarily and remain eligible for leaf handoff).
    let speculativeArm: SpeculativeArm?
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
    var promptTokenCount: Int { facts.promptTokenCount }
    /// Number of leading tokens skipped because the cache already covered them.
    let skippedPrefillTokens: Int
    /// Lookup outcome classification, surfaced for in-app observability.
    let lookupReason: PrefixCacheManager.LookupReason
    /// Shared-prefix length in tokens between the request and the best cache entry.
    let sharedPrefixLength: Int
    /// The turn's maximum advance (`CacheClaim.maximumAdvance`): the
    /// prompt tokens prefilled past the restore offset plus the output
    /// ceiling plus the speculative allowance, `Int.max` when the output is
    /// unbounded. What check-out eligibility was judged against, and what
    /// the **Active-Inference Reserve** prices the turn's growth at (#522).
    let maximumAdvance: Int

    // -- Post-generation store context (radix tree flow) --

    /// What **Request Keying** made of the request (ADR-0070): a **Keyed
    /// Request**, whose identities and facts the post-generation phases take
    /// whole, or an **Unkeyed Completion**'s request, which has facts and no
    /// keys, so the store flow cannot reach the radix tree with a placeholder.
    enum Keying: Sendable {
        case keyed(KeyedRequest)
        case unkeyed(UnkeyedRequest)

        var facts: RequestFacts {
            switch self {
            case .keyed(let request): request.facts
            case .unkeyed(let request): request.facts
            }
        }

        var keyed: KeyedRequest? {
            guard case .keyed(let request) = self else { return nil }
            return request
        }

        var unkeyedReason: CacheKeySpace.UnkeyedReason? {
            guard case .unkeyed(let request) = self else { return nil }
            return request.reason
        }
    }

    let keying: Keying
    var facts: RequestFacts { keying.facts }

    /// The wire string for `Diagnostics.cacheReason`: the lookup outcome, or
    /// the unkeyed degradation when the request never reached the lookup.
    var cacheReasonDescription: String {
        keying.unkeyedReason.map { "unkeyed(\($0.rawValue))" } ?? String(describing: lookupReason)
    }
    /// Validated mid-prefill checkpoint admission, if any checkpoints survived
    /// extraction-edge path validation.
    let snapshotAdmission: SnapshotAdmission?
    /// Request-local helper snapshot captured at the end of the last history
    /// message. Never stored or persisted; used to synthesize the direct
    /// tool-continuation leaf for tool-call turns.
    let transientLastMessageBoundarySnapshot: HybridCacheSnapshot?
    /// Request-local helper snapshot captured at the end of the last real
    /// user message. Never stored or persisted; used to synthesize the
    /// canonical user-continuation leaf for templates that rewrite the
    /// assistant/tool suffix after the last user.
    let transientLastUserBoundarySnapshot: HybridCacheSnapshot?
    /// The ids the decode loop fed past the prompt (stop token included),
    /// filled by the generation task and read by the **Leaf Store** phase
    /// once `completion` has finished — the tail of the turn's **Emitted
    /// Path**, which the live leaf is keyed on and the index registers.
    let generatedTokens: GeneratedTokenRecorder
    /// How the request restored (`cold`, `copy`, `failedCopy`, `handoff`),
    /// as the claim's check-out decided it — one input to the stored
    /// leaf's source.
    var restoreMode = "cold"
    /// Why a copy restore did not take the leaf; `nil` unless the check-out
    /// answered copy.
    var restoreCopy: CacheClaim.Copy?
    /// Seconds a Pending-Payload Wait cost, whether it ended in a copy or
    /// a handoff (#523).
    var restoreWaitSeconds: TimeInterval = 0
}

/// All copies of the generation handle share this one reference. Handoff
/// empties it for every phase, including the drive's retained start handle.
/// Access follows the generation's existing discipline: inspect only after
/// awaiting completion; transfer only inside the Metal-affine Model Session.
/// A checked-out leaf's lease belongs to the request's **Cache Claim**, not
/// to this handle.
nonisolated final class FinalGenerationCache: @unchecked Sendable {
    private(set) var cache: [any KVCache]

    init(_ cache: [any KVCache]) {
        self.cache = cache
    }

    func moveSnapshot(offset: Int) -> HybridCacheSnapshot? {
        HybridCacheSnapshot.captureMoving(cache: &cache, offset: offset)
    }

    /// **Leaf Rewind** of checked-out objects, in the Model Session once
    /// generation has quiesced.
    func rewind(with state: LeafRewind) {
        state.rewind(&cache)
    }

    func recoverUnadmitted(_ snapshot: HybridCacheSnapshot) {
        precondition(cache.isEmpty)
        guard let (returnedCache, _) = snapshot.takeMovingCache() else {
            preconditionFailure("an unadmitted moved snapshot must still own its cache")
        }
        cache = returnedCache
    }
}

extension GenerationStreamLoop.RawGenerationHandle {
    /// The server's rich prefill handle collapses to `{ stream, cancel, wait }`
    /// plus the request's Generation Prompt; the prefill/cache metadata stays
    /// with the **Server Completion** module and never crosses the seam.
    fileprivate nonisolated init(_ generation: HTTPPrefixCacheGeneration) {
        self.init(
            stream: generation.stream, completion: generation.completion,
            generationPrompt: generation.facts.generationPrompt)
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

/// The two decode iterators the keyed path constructs after its app-owned
/// prefill: the ordinary state-threaded decode, or the **Speculation
/// Plan**'s iterator over the same warmed cache (its prefill covers only the
/// prompt tail past the app driver's checkpoint captures).
private nonisolated enum KeyedDecodeIterator {
    case standard(StateThreadedTokenIterator)
    case speculative(SpeculativeDecodeIterator)

    /// Start the app-owned generation stream over whichever iterator the
    /// keyed path built — one `TokenGenerationLoop.start` per case because
    /// the loop's entry is generic over the concrete iterator type.
    consuming func startGeneration(
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: any MLXLMCommon.Tokenizer,
        tools: [ToolSpec]?,
        generatedTokens: GeneratedTokenRecorder
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        switch consume self {
        case .standard(let decode):
            TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount,
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer,
                iterator: decode,
                tools: tools,
                generatedTokens: generatedTokens
            )
        case .speculative(let decode):
            decode.startGeneration(
                promptTokenCount: promptTokenCount,
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer,
                tools: tools,
                generatedTokens: generatedTokens
            )
        }
    }
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
/// split (a second actor was rejected; ADR-0015). The LLM gate, held across
/// the whole HTTP request by `CompletionHandler`, remains the primary guard
/// against unload/reload interleaving.
///
/// The module owns: **Snapshot Resolution** → restore → suffix prefill from
/// the **Prefill Plan** → the **Generation Stream Loop** drive (with the
/// server's sink: accumulator fold + tool-call projection) → **Snapshot
/// Admission** at the MLX edge → **Leaf Capture Plan** execution, plus the
/// prefix-cache admin entries and the snapshot-payload extraction statics.
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
    /// format, Qwen3.5 family/MoE, and flop profile.
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

    /// The Emitted Path Index this module registers into and resolves
    /// against (ADR-0063). Production installs the process-wide `.shared`
    /// (the actor's unload clears that one); a test fixture hands each Server
    /// Completion its own so hermetic scenarios — restart, eviction, two
    /// generations under one key — see exactly the index they built and
    /// no other suite's fingerprint resets.
    private(set) var emittedPathIndex: EmittedPathIndex = .shared

    private var modelWeightBytes: Int64 = 0
    private var defaultPrefixCacheMemoryBudgetBytes =
        LLMActor.Defaults.fallbackPrefixCacheMemoryBudgetBytes

    private var _prefixCache: PrefixCacheManager?

    /// Active-completion registry: the most recent cache-aware start, keyed
    /// by its request ID so the natural-finish clear and the drain can tell
    /// handles apart. The LLM gate — held across the whole HTTP request by
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
        headroomSource: (any MemoryHeadroomSource)? = nil,
        emittedPathIndex: EmittedPathIndex = .shared
    ) {
        self.modelIdentity = modelIdentity
        self.modelFingerprint = fingerprint
        self.ssdConfig = ssdConfig
        self.ramBudgetCapBytes = ramBudgetCapBytes
        self.headroomSource = headroomSource
        self.emittedPathIndex = emittedPathIndex
    }

    /// Container-derived facts, installed by the actor's `verifyAndStore`
    /// after a successful load. Drops any pre-load prefix cache so the next
    /// use rebuilds it with the real FLOP profile and auto-sized budget.
    func installLoadedModelFacts(
        modelWeightBytes: Int64,
        prefixCacheBudgetBytes: Int
    ) {
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
        repeat {
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
            await _prefixCache?.awaitPendingDrain()
        } while inflightStartCount > 0 || activeCompletion != nil || speculativePrefill != nil
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

        let prefixCache = await ensurePrefixCache(on: actor, sessions: sessions)
        let requestID = UUID()
        let requestContext = PrefixCacheDiagnostics.Context(
            requestID: requestID, modelID: modelID,
            kvBits: parameters.kvBits, kvGroupSize: parameters.kvGroupSize)
        let memory = RequestMemoryTelemetry(context: requestContext)
        let memorySampler = memory.startSampling()
        memory.mark(.preparing, facts: ["modelWeightBytes": "\(modelWeightBytes)"])
        var handedToDrive = false
        var startOutcome = "startFailed"
        defer {
            if !handedToDrive {
                memory.finish(outcome: Task.isCancelled ? "cancelledDuringStart" : startOutcome)
                memorySampler.cancel()
                Task.detached(priority: .utility) { await memory.sampleAfterRelease() }
            }
        }
        let genParams = LLMActor.makeGenerateParameters(from: parameters)
        // Canonicalize tools once so the leaf re-tokenization uses the same dict
        // iteration order as the prefill path inside makeHTTPPrefixCacheGeneration.
        let canonicalTools = LLMActor.canonicalizeToolSpecs(toolSpecs)

        // Everything up to the drive holds the request's **Cache Claim**
        // (ADR-0069). A throw concludes it in this scope; a normal return
        // hands it over, and the drive task created below redeems it.
        let (mlxStart, handOver) = try await CacheClaim.withRequestClaim(
            context: requestContext, prefixCache: prefixCache, sessions: sessions, memory: memory
        ) { claim in
            let mlxStart: HTTPPrefixCacheGeneration
            do {
                mlxStart = try await withTaskCancellationHandler {
                    try await makeHTTPPrefixCacheGeneration(
                        on: actor,
                        sessions: sessions,
                        conversation: conversation,
                        requestID: requestID,
                        modelID: modelID,
                        parameters: genParams,
                        toolSpecs: canonicalTools,
                        prefixCache: prefixCache,
                        claim: claim,
                        renderContext: renderContext,
                        progressHandler: progressHandler,
                        memory: memory
                    )
                } onCancel: {
                    memory.recordCancellationSignal(origin: "caller")
                }
            } catch {
                if error is CancellationError { startOutcome = "cancelledDuringStart" }
                throw error
            }

            // A drain ran while restore/prefill was suspended: the model is
            // tearing down, so stop the freshly started generation and bail
            // before wiring up a handle nothing would ever drain. The scope's
            // conclusion returns a leased leaf.
            if Task.isCancelled || drainGeneration != entryDrainGeneration {
                startOutcome = "cancelledDuringStart"
                mlxStart.completion.cancel()
                await mlxStart.completion.value
                Memory.clearCache()
                throw CancellationError()
            }
            return mlxStart
        }

        let (stream, continuation) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
        let loadedModelWeightBytes = modelWeightBytes

        let driver = ManagedGenerationDriver(logContext: "request_id=\(requestID.uuidString)")
        let actorRef = actor

        // The loop owns raw-handle cancellation. Its `cancelCurrent` must be wired into
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
                handOver: handOver,
                mlxStartBox: mlxStartBox,
                conversation: conversation,
                sessions: sessions,
                requestID: requestID,
                loadedModelWeightBytes: loadedModelWeightBytes,
                memory: memory,
                prefixCache: prefixCache,
                renderContext: renderContext,
                traceLog: traceLog,
                driver: driver,
                loopCancel: loopCancel,
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
            memorySampler.cancel()
            Task.detached(priority: .utility) { await memory.sampleAfterRelease() }
        }
        handedToDrive = true

        let completionStart = ManagedGenerationDriver.makeStart(
            stream: stream,
            continuation: continuation,
            cachedTokenCount: cachedTokenCount,
            diagnostics: completionDiagnostics,
            cancelBridge: loopCancel,
            task: task,
            cancellationObserver: { memory.recordCancellationSignal(origin: $0) }
        )
        activeCompletion = (id: requestID, handle: completionStart)
        return completionStart
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_parameter_count
    /// Drive one cache-aware completion to its end. Deliberately nonisolated
    /// — see the comment at the call site's `Task`. The drive owns the
    /// request's **Cache Claim** from the hand-over to its conclusion, which
    /// runs on every exit path (natural finish, cancellation, error): it
    /// rewinds a leaf still leased, then lets go of the Restore Pins and the
    /// reserve lane, so no exit carries cleanup of its own. From there on the
    /// turn's protection is the freshest-leaf floor member, not the
    /// in-flight pin (ADR-0019). `finishHook` then releases this request's
    /// registry slot back on the actor.
    private static func driveCompletion(
        handOver: CacheClaim.HandOver,
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        conversation: HTTPPrefixCacheConversation,
        sessions: any ModelSessionProviding,
        requestID: UUID,
        loadedModelWeightBytes: Int64,
        memory: RequestMemoryTelemetry,
        prefixCache: PrefixCacheManager,
        renderContext: TemplateRenderContext,
        traceLog: CompletionTraceLog,
        driver: ManagedGenerationDriver,
        loopCancel: LateBoundCancel,
        continuation: AsyncThrowingStream<AgentGeneration, Error>.Continuation,
        finishHook: @escaping @Sendable () async -> Void,
        scheduleSpeculative: @escaping @Sendable (SpeculativeCanonicalPrefill.Seed) async -> Void
    ) async {
        // swiftlint:enable function_parameter_count
        let (terminalOutcome, speculativeSeed) = await handOver.withClaim { claim in
            await drive(
                claim: claim,
                mlxStartBox: mlxStartBox,
                conversation: conversation,
                sessions: sessions,
                requestID: requestID,
                loadedModelWeightBytes: loadedModelWeightBytes,
                memory: memory,
                prefixCache: prefixCache,
                renderContext: renderContext,
                traceLog: traceLog,
                driver: driver,
                loopCancel: loopCancel,
                continuation: continuation
            )
        }
        await finishHook()
        // After the registry slot is released: hand the speculative seed to
        // the actor, which schedules it only if the module is still quiescent
        // (a newer start, or a drain since this request entered, wins).
        if let speculativeSeed {
            await scheduleSpeculative(speculativeSeed)
        }
        memory.finish(outcome: terminalOutcome)
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_body_length function_parameter_count
    /// The drive's work under the claim: the stream-loop run with the
    /// server's sink, snapshot admissions, leaf capture, and the request-end
    /// tuner record. Returns the request's terminal outcome and the
    /// Speculative Canonical Prefill seed, if one is to be scheduled.
    private static func drive(
        claim: CacheClaim,
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        conversation: HTTPPrefixCacheConversation,
        sessions: any ModelSessionProviding,
        requestID: UUID,
        loadedModelWeightBytes: Int64,
        memory: RequestMemoryTelemetry,
        prefixCache: PrefixCacheManager,
        renderContext: TemplateRenderContext,
        traceLog: CompletionTraceLog,
        driver: ManagedGenerationDriver,
        loopCancel: LateBoundCancel,
        continuation: AsyncThrowingStream<AgentGeneration, Error>.Continuation
    ) async -> (outcome: String, speculativeSeed: SpeculativeCanonicalPrefill.Seed?) {
        // swiftlint:enable function_body_length function_parameter_count
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
        var terminalOutcome = "completed"
        memory.mark(.decoding)

        drive: do {
            func handle(_ event: AgentGeneration) {
                // Fold shared accumulation (text/thinking)
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
            // completion log and unparsed-tool-call warning.
            let outcome = try await driver.run(
                initial: .init(mlxStart),
                cancelBridge: loopCancel,
                sink: handle
            )
            // Everything from here to `continuation.finish()` is inside the
            // client-visible wait: the terminal SSE chunk goes out only after
            // the drive finishes (Completion Delivery). The `leafStore`
            // report below carries this span so a post-EOS stall is
            // attributable.
            let generationEnded = Date.timeIntervalSinceReferenceDate
            memory.mark(
                .generationQuiescent, facts: ["generationCancelled": "\(outcome.cancelled)"])

            if outcome.cancelled {
                terminalOutcome = "cancelled"
                // The driver has awaited generation. Sample its final cache
                // on-session even though cancellation bypasses leaf capture.
                let facts = await sessions.withSession { _ in
                    RequestMemoryTelemetry.cacheFacts(mlxStartBox.value.finalCache)
                }
                memory.mark(.generationQuiescent, facts: facts)
                memory.mark(.finishingStream)
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

            if outcome.completionInfo == nil, claim.holdsLease {
                terminalOutcome = "failed"
                continuation.finish()
                break drive
            }

            // -- Post-generation: store snapshots in radix tree --

            // Store mid-prefill snapshots (e.g. stable-prefix boundary) unconditionally.
            // These are captured during prefill and independent of the leaf path — if
            // final-cache recovery or leaf capture fails, the stable-prefix checkpoint
            // still saves future requests from a full re-prefill.
            memory.mark(.admittingCheckpoints)
            var storedSnapshotsForTuner: [HybridCacheSnapshot] = []
            if !Task.isCancelled, let admission = mlxStart.snapshotAdmission {
                let diagnostics = await MainActor.run {
                    prefixCache.admit(admission)
                }
                trace.ingest(evictions: diagnostics.evictions, diagnostics: diagnosticsContext)
                storedSnapshotsForTuner = admission.snapshots
            }

            if Task.isCancelled {
                terminalOutcome = "cancelled"
                memory.mark(.finishingStream)
                Memory.clearCache()
                continuation.finish()
                break drive
            }

            // The **Leaf Store** phase: decide-and-execute how (or whether)
            // the finished turn's KV state is admitted as a leaf, and whether
            // it seeds the Speculative Canonical Prefill. Every skip path
            // falls through to the request-end recordRequest call below — the
            // alpha tuner needs to see every request, not just the ones whose
            // leaf store completed.
            memory.mark(
                .storingLeaf, facts: await MainActor.run { prefixCache.memoryTelemetryFacts() })
            // The phase's report, logged below, is now the turn's one
            // `leafStore` event, a rewind in the phase included.
            claim.enterLeafStorePhase()
            let leafStoreStart = Date.timeIntervalSinceReferenceDate
            var leafResult = await LeafStorePhase.run(
                mlxStartBox: mlxStartBox,
                claim: claim,
                conversation: conversation,
                sessions: sessions,
                requestID: requestID,
                prefixCache: prefixCache,
                assistantText: accumulator.text,
                assistantReasoning: accumulator.thinking,
                toolCalls: toolCalls,
                diagnosticsContext: diagnosticsContext,
                trace: &trace,
                memory: memory
            )
            if mlxStart.facts.ssdEnabled, !Task.isCancelled {
                await prefixCache.persistViewCheckpoints(
                    partitionKey: mlxStart.facts.partitionKey, sessions: sessions)
            }
            leafResult.report.restoreMode = mlxStart.restoreMode
            leafResult.report.restoreCopyReason = mlxStart.restoreCopy?.reason
            leafResult.report.restoreCopyRefusal = mlxStart.restoreCopy?.refusal
            leafResult.report.restoreCopyWaitSeconds = mlxStart.restoreWaitSeconds
            leafResult.report.leafStoreSeconds =
                Date.timeIntervalSinceReferenceDate - leafStoreStart
            let leafStoreForTuner = leafResult.leafStore
            if let seed = leafResult.speculativeSeed {
                speculativeSeed = seed
            }

            memory.mark(
                .recordingRequest,
                facts: [
                    "leafSource": leafResult.report.source?.rawValue ?? "skipped",
                    "leafCopyReason":
                        (leafResult.report.restoreCopyReason ?? leafResult.report.copyReason)?
                        .rawValue ?? "none",
                ])
            // Record the request lifecycle for the alpha tuner. Fires
            // for every request, including the leaf-skipped paths
            // — the tuner needs the full workload trace, not just
            // successful leaf stores.
            let capturedSnapshots = storedSnapshotsForTuner
            let leafCapture = leafStoreForTuner
            let unkeyed = mlxStart.keying.unkeyedReason != nil
            let keyPath = mlxStart.keying.keyed?.keySpace.keyPath ?? mlxStart.facts.promptTokens
            let (finalStats, finalBudgetBytes, finalEstimates) = await MainActor.run {
                // Unkeyed Completions stay out of the tuner's workload trace —
                // they never participated in the cache this trace models.
                if !unkeyed {
                    prefixCache.recordRequest(
                        partitionKey: mlxStart.facts.partitionKey,
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

            // The persisted (notice-level) post-EOS account: which leaf path
            // ran, its stage breakdown, and the whole span from generation
            // end to here — the client sees its terminal chunk right after.
            leafResult.report.tailSeconds = Date.timeIntervalSinceReferenceDate - generationEnded
            // ADR-0063: fold the request's Emitted Path resolves (request
            // edge, planner, leaf store) into the same account.
            let emittedPathSummary = mlxStart.keying.keyed?.render.emittedPathTelemetry?.summary
            leafResult.report.emittedPathResolves = emittedPathSummary
            diagnosticsContext.log(leafResult.report, level: .notice)

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
                        partitionDigest: mlxStart.facts.partitionKey.partitionDigest,
                        unkeyedReason: mlxStart.keying.unkeyedReason,
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
                    deviceEstimates: finalEstimates,
                    leafStoreSeconds: leafResult.report.leafStoreSeconds,
                    tailSeconds: leafResult.report.tailSeconds,
                    emittedPath: EmittedPathTraceTelemetry.make(
                        report: leafResult.report, resolves: emittedPathSummary)
                )
                if let record {
                    traceLog.append(record)
                }
            }

            memory.mark(
                .finishingStream, facts: await MainActor.run { prefixCache.memoryTelemetryFacts() })
            continuation.finish()
        } catch is CancellationError {
            terminalOutcome = "cancelled"
            memory.mark(.finishingStream)
            continuation.finish()
        } catch {
            terminalOutcome = "failed"
            memory.mark(.finishingStream)
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
            case .keyed(let request) = mlxStart.keying,
            mlxStart.transientLastUserBoundarySnapshot != nil,
            !renderContext.preservesThinking
        {
            speculativeSeed = SpeculativeCanonicalPrefill.makeSeed(
                storedConversation: conversation,
                request: request,
                canonicalLeafOffset: mlxStart.transientLastUserBoundarySnapshot?
                    .tokenOffset ?? 0,
                transientBoundary: mlxStart.transientLastUserBoundarySnapshot,
                ramOnlySpine: true,
                diagnostics: diagnosticsContext
            )
        }

        return (terminalOutcome, speculativeSeed)
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_body_length function_parameter_count
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
        claim: CacheClaim,
        renderContext: TemplateRenderContext = .canonical,
        progressHandler: ServerInferenceProgressHandler?,
        memory: RequestMemoryTelemetry
    ) async throws -> HTTPPrefixCacheGeneration {
        // swiftlint:enable function_body_length function_parameter_count
        // Canonicalize tools once so the stable-prefix detector and the real
        // prefill tokenize against identical dict representations. Historically
        // swift-jinja <2.3.5 had non-deterministic `tojson` key ordering; the
        // canonicalization is kept as defense-in-depth and costs almost nothing.
        let canonicalTools = LLMActor.canonicalizeToolSpecs(toolSpecs)

        // Capture module state for the non-MainActor closure below —
        // the closure runs on the **Model Session**'s isolation and cannot
        // sync-read the actor-confined module.
        let modelFingerprint = self.modelFingerprint
        let emittedPathIndex = self.emittedPathIndex
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
            memory.mark(.preparing, facts: session.speculation.memoryFacts)
            func measure<T>(_ work: () throws -> T) rethrows -> (T, TimeInterval) {
                let started = Date.timeIntervalSinceReferenceDate
                let value = try work()
                return (value, Date.timeIntervalSinceReferenceDate - started)
            }

            // 1–3b. The Request Keying phase: the prepared input and the
            // **Keyed Request** — the partition key, the request's **Cache Key
            // Space** and Conversation Render, and every per-request fact —
            // or an Unkeyed Completion's request when key-space construction
            // fails, in which case the whole request is served unkeyed.
            let request: KeyedRequest
            let fullInput: LMInput
            switch try await RequestKeyingPhase.run(
                session: session,
                conversation: conversation,
                canonicalTools: canonicalTools,
                renderContext: renderContext,
                parameters: parameters,
                modelID: modelID,
                modelFingerprint: modelFingerprint,
                imageKeying: imageKeying,
                ssdEnabled: ssdEnabled,
                emittedPathIndex: emittedPathIndex,
                diagnostics: diagnosticsContext
            ) {
            case .keyed(let keyed, let input):
                request = keyed
                fullInput = input
            case .unkeyed(let unkeyed, let input):
                return try await Self.makeUnkeyedGeneration(
                    session: session,
                    request: unkeyed,
                    input: input,
                    parameters: parameters,
                    toolSpecs: canonicalTools,
                    fullAttentionScratchProfile: fullAttentionScratchProfile,
                    visionAttentionScratchProfile: visionAttentionScratchProfile,
                    diagnosticsContext: diagnosticsContext,
                    progressHandler: progressHandler
                )
            }
            let facts = request.facts
            let fullTokenCount = facts.promptTokenCount
            let tokenNDim = facts.tokenNDim
            let partitionKey = facts.partitionKey
            let keySpace = request.keySpace
            let seedsPositionAnchor = request.seedsPositionAnchor

            // 4. Detect the prefill boundaries (stable prefix + last-message +
            // last-user). The Prefill Planner owns this tokenizer-affine work —
            // the last-message arithmetic on the request's measured
            // Generation Prompt and the last-user re-render — in one tested
            // place, against the key space's own path so boundary offsets are
            // key-space offsets by construction.
            let boundaries = try PrefillPlanner.detectBoundaries(
                conversation: conversation,
                request: request
            )
            if let unknown = boundaries.generationPromptUnknown {
                diagnosticsContext.logSkip(
                    stage: "lastMessageBoundary",
                    reason: "generation-prompt-unknown",
                    extraFields: [("generationPrompt", unknown.rawValue)]
                )
            }
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
                for: claim
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

            // The Speculation Plan (ADR-0079), decided once from the facts
            // the restore switch below acts on. DFlash2 rides the ordinary
            // restore + checkpoint-capturing prefill (boundary snapshots
            // preserved, warm restores speculate too) and splits the
            // prefill with its iterator; MTP takes the whole prompt from
            // zero in the cold case. Tool emission is unknowable here, so
            // defined tools predict a tool leaf.
            let speculation = session.speculation.plan(
                for: SpeculationRequest(
                    isTextOnly: facts.isTextOnly,
                    kvBits: parameters.kvBits,
                    temperature: parameters.temperature,
                    promptTokens: fullTokenCount,
                    restoresPrefix: prefillPlan.restore.restoresPrefix,
                    storedLeaf: facts.predictedLeafStoreMode
                ))
            memory.mark(
                .restoring, facts: await MainActor.run { prefixCache.memoryTelemetryFacts() })
            if lookupResult.snapshot == nil {
                memory.mark(.restoring, facts: ["restoreMode": "cold"])
            }
            memory.mark(
                .restoring,
                facts: [
                    "promptTokens": "\(fullTokenCount)",
                    "dflash2Engaged": "\(speculation?.arm == .dflash2)",
                    "restoreSnapshotBytes": "\(lookupResult.snapshot?.memoryBytes ?? 0)",
                ])
            // The claim's typed restore outcome: a handoff holds the lease
            // and the leaf's own cache; a copy says why the leaf was not
            // taken.
            var handoff: CacheClaim.Handoff?
            var restoreCopy: CacheClaim.Copy?
            var restoreMode = "cold"
            // The bounded wait for a pending full payload (#523): what it
            // cost, whether it ended in a copy or a handoff.
            var restoreWaitSeconds: TimeInterval = 0
            // The turn's maximum advance: judged at check-out, priced by
            // the Active-Inference Reserve at the leaf store (#522).
            let restoredOffset: Int
            if case .restore(let cacheOffset, _) = prefillPlan.restore {
                restoredOffset = cacheOffset
            } else {
                restoredOffset = 0
            }
            let maximumAdvance = CacheClaim.maximumAdvance(
                newPromptTokens: fullTokenCount - restoredOffset,
                outputCeiling: parameters.maxTokens,
                speculativeAllowance: speculation?.advanceAllowance ?? 0)
            switch prefillPlan.restore {
            case .restore(let cacheOffset, let anchorDelta):
                let restoreStarted = Date.timeIntervalSinceReferenceDate
                let restoredCache: [any KVCache]?
                switch await claim.checkOut(
                    resolved, tokens: keySpace.keyPath, maximumAdvance: maximumAdvance,
                    identityKeySpace: facts.isTextOnly, in: session)
                {
                case .handoff(let taken):
                    handoff = taken
                    restoreWaitSeconds = taken.waitSeconds
                    restoredCache = taken.cache.cache
                case .copy(let copy):
                    restoreCopy = copy
                    restoreWaitSeconds = copy.waitSeconds
                    restoredCache = Self.restoreCache(lookupResult, session: session)
                case .cold:
                    restoredCache = Self.restoreCache(lookupResult, session: session)
                }
                cacheToUse = restoredCache
                restoreMs = Date.timeIntervalSinceReferenceDate - restoreStarted
                restoreMode =
                    handoff != nil
                    ? "handoff" : restoredCache == nil ? "failedCopy" : "copy"
                memory.mark(
                    .restored,
                    facts: RequestMemoryTelemetry.cacheFacts(restoredCache ?? []).merging([
                        "restoreMode": restoreMode,
                        "restoreCopyReason": restoreCopy?.reason.rawValue ?? "none",
                        "restoreCopyRefusal": restoreCopy?.refusal.rawValue ?? "none",
                        "restoreCopyWaitMs": PrefixCacheDiagnostics.milliseconds(
                            restoreWaitSeconds),
                        "recurrentRewindStateBytes": "\(handoff?.rewindStateBytes ?? 0)",
                        "leafLeaseActive": "\(handoff != nil)",
                        "leafLeaseID": handoff?.leaseID.uuidString ?? "none",
                    ]) { _, new in new })
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
                        imageContinuationAnchor = PositionAnchor.seededState(
                            ropeDelta: anchorDelta)
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
                        executorInitialState = PositionAnchor.seededState(
                            ropeDelta: anchorDelta)
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
                // A plan whose iterator prefills the whole prompt (MTP) takes
                // over the cold request here: its unchunked vendor prefill
                // forfeits the checkpoints, which the plan only accepts for a
                // direct leaf (ADR-0056 amendment).
                if let speculation, speculation.prefillsWholePrompt {
                    return try await Self.makeWholePromptSpeculativeGeneration(
                        plan: speculation,
                        session: session,
                        request: request,
                        input: fullInput,
                        parameters: parameters,
                        toolSpecs: canonicalTools,
                        lookupReason: lookupResult.reason,
                        lookupMs: lookupMs,
                        maximumAdvance: maximumAdvance,
                        visionAttentionScratchProfile: visionAttentionScratchProfile,
                        diagnosticsContext: diagnosticsContext,
                        progressHandler: progressHandler
                    )
                }
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
                    divergence: lookupResult.divergence,
                    restoreMode: restoreMode, copyReason: restoreCopy?.reason,
                    copyRefusal: restoreCopy?.refusal,
                    copyWaitSeconds: restoreWaitSeconds,
                    backingLeafOffset: lookupResult.backingLeaf?.tokenOffset,
                    warmBody: lookupResult.snapshot?.isWarm == true,
                    backingLeafWarm: lookupResult.backingLeaf?.isWarm == true
                ))

            // 8. Fold the plan's checkpoints plus the transient boundary
            // helpers (Prefix-View Checkpoints; a planned checkpoint at the same
            // offset wins) into one capture map for the prefill driver.
            // Planner guarantees offset uniqueness, so uniqueKeysWithValues
            // traps loudly on a planner-side invariant break instead of
            // silently dropping a candidate.
            let genParams = parameters
            let plannedCheckpoints = Dictionary(
                uniqueKeysWithValues: prefillPlan.checkpointsToCapture.map {
                    ($0.offset, $0.type)
                }
            )
            // Preserve-thinking turns never synthesize boundary leaves or
            // abandonment seeds. Their boundary helpers serve no consumer.
            let transientOffsets =
                renderContext.preservesThinking && facts.isTextOnly
                ? Set<Int>() : prefillPlan.transientCheckpointOffsets
            let helperCheckpoints = Dictionary(
                uniqueKeysWithValues: transientOffsets.map {
                    ($0, HybridCacheSnapshot.CheckpointType.branchPoint)
                }
            )
            let allCheckpoints = plannedCheckpoints.merging(helperCheckpoints) { stored, _ in
                stored
            }

            // A speculative plan splits the suffix: the app driver prefills
            // (and snapshots) through the split, then the iterator's own
            // prefill takes the tail (DFlash2's hidden-state window).
            let speculativeSplitOffset: Int? = speculation?.prefillSplit(
                checkpointOffsets: allCheckpoints.keys,
                executionBaseOffset: executionBaseOffset,
                promptTokens: fullTokenCount)

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
            memory.mark(.prefilling, facts: RequestMemoryTelemetry.cacheFacts(liveCache))
            let prefillResult: (iterator: KeyedDecodeIterator, snapshots: [HybridCacheSnapshot])
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
                                windowSize: facts.prefillStepSize,
                                contextTokens: executionBaseOffset,
                                profile: fullAttentionScratchProfile,
                                diagnosticsContext: diagnosticsContext
                            )

                            // Warm/cold image span (ADR-0007 phase 2): the anchored
                            // `prepare` runs the vision tower once, positions the
                            // new image from the restored Position Anchor
                            // (`imageContinuationAnchor`; nil ⇒ anchored at zero, a
                            // crash-safe cold prefill), and windows the forward so
                            // the scratch is bounded. Its returned state anchors the
                            // chunked text tail. A non-identity key space implies the
                            // recognized vision container, whose session exposes
                            // the anchored `prepare`.
                            guard let anchoredPrepare = session.anchoredVisionPrepare
                            else {
                                throw AgentEngineError.generationFailed(
                                    "loaded model does not support anchored vision continuation"
                                )
                            }
                            guard
                                case .logits(let prepared) = try anchoredPrepare(
                                    imagePrefixInput,
                                    liveCache,
                                    imageContinuationAnchor,
                                    genParams.prefill.stepSize
                                )
                            else {
                                throw AgentEngineError.generationFailed(
                                    "vision container returned .tokens from anchored prepare"
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
                        // Speculative plan: driver prefill up to the split
                        // (the split sits at or past the deepest capture, so
                        // every checkpoint and boundary snapshot lands),
                        // then the iterator's capture prefill for the tail.
                        // The driver slice keeps one token back so its final
                        // capture still fires; that token stays unconsumed
                        // for the iterator, which prefills [split, end) and
                        // samples the first token exactly like its cold
                        // prepare's own-chunk final position.
                        if let speculation, let splitOffset = speculativeSplitOffset {
                            var snapshots = prefixSnapshots
                            if splitOffset > executionBaseOffset {
                                let prefixTokenCount = splitOffset - executionBaseOffset
                                let prefixText = LMInput.Text(
                                    tokens: inputForGeneration.text.tokens[
                                        ..<(prefixTokenCount + 1)],
                                    mask: nil)
                                let warmed = try prefixCache.storageActivityGate
                                    .withPrefillMarked {
                                        try session.prefill(
                                            text: prefixText,
                                            cache: liveCache,
                                            checkpoints: allCheckpoints,
                                            checkpointBaseOffset: executionBaseOffset,
                                            prefillStepSize: facts.prefillStepSize,
                                            consumeAll: false,
                                            initialState: initialState,
                                            evalPolicy: .pipelined
                                        )
                                    }
                                snapshots += warmed.snapshots
                            }
                            try error.check()
                            memory.mark(
                                .dflashPreparing,
                                facts: RequestMemoryTelemetry.cacheFacts(liveCache))
                            let iterator = try session.makeSpeculativeDecodeIterator(
                                fullInput,
                                cache: liveCache,
                                prefilledPrefixTokens: splitOffset,
                                plan: speculation,
                                parameters: facts.decodeParameters
                            )
                            try error.check()
                            return (iterator: .speculative(iterator), snapshots: snapshots)
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
                                prefillStepSize: facts.prefillStepSize,
                                consumeAll: false,
                                initialState: initialState,
                                evalPolicy: imagePrefixInput == nil
                                    ? .pipelined : .checkedSynchronous
                            )
                        }
                        try error.check()
                        session.quantizeKVCache(&liveCache, parameters: genParams)
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
                            parameters: facts.decodeParameters
                        )
                        return (
                            iterator: .standard(iterator),
                            snapshots: prefixSnapshots + warmed.snapshots
                        )
                    }
            } catch is CancellationError {
                // **Salvage-on-cancel** (issue #97): the client is gone and
                // the GPU just went idle at a chunk boundary — keep the
                // progress instead of discarding it. RAM-only, after the
                // cancellation landed, so the cancel path's perceived
                // latency is unchanged; a re-sent request (or an
                // abort-seeded speculative pass) resumes from the salvaged
                // offset instead of the restore floor.
                if handoff == nil {
                    await Self.salvageCancelledPrefill(
                        cache: liveCache,
                        keySpace: keySpace,
                        restoreBaseOffset: executionBaseOffset,
                        partitionKey: partitionKey,
                        requestID: requestID,
                        prefixCache: prefixCache,
                        diagnostics: diagnosticsContext,
                        session: session
                    )
                }
                Memory.clearCache()
                throw CancellationError()
            }
            // Quantization can replace attention objects. Retain the array
            // that the iterator actually advances, after that replacement.
            let finalCacheOwner = handoff?.cache ?? FinalGenerationCache(liveCache)
            let prefillMs = Date.timeIntervalSinceReferenceDate - begin.startedAt
            let boundarySnapshots = prefillResult.snapshots.filter {
                transientOffsets.contains($0.tokenOffset)
            }
            memory.mark(
                .prefilled,
                facts: RequestMemoryTelemetry.cacheFacts(liveCache).merging([
                    "prefillCheckpointArrayBytes":
                        "\(prefillResult.snapshots.reduce(0) { $0 + $1.memoryBytes })",
                    "boundaryCheckpointCount": "\(boundarySnapshots.count)",
                    "boundaryCheckpointOffsets": boundarySnapshots.isEmpty
                        ? "none"
                        : boundarySnapshots.map(\.tokenOffset).sorted().map(String.init)
                            .joined(separator: ","),
                    "boundaryCheckpointArrayBytes":
                        "\(boundarySnapshots.reduce(0) { $0 + $1.memoryBytes })",
                ]) { _, new in new })
            let iterator = prefillResult.iterator
            if case .speculative(let decode) = iterator {
                // The iterator exists — the request will decode speculatively.
                // Fired before the token loop so the activity surfaces badge
                // the arm live.
                await progressHandler?(.speculationEngaged(decode.arm))
            }
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
            let transientLastMessageBoundarySnapshot = prefillPlan.transientBoundaries
                .lastMessage
                .flatMap { offset in
                    transientSnapshots[offset]
                        ?? capturedSnapshots.first(where: { $0.tokenOffset == offset })
                }
            let transientLastUserBoundarySnapshot = prefillPlan.transientBoundaries.lastUser
                .flatMap { offset in
                    transientSnapshots[offset]
                        ?? capturedSnapshots.first(where: { $0.tokenOffset == offset })
                }
            let checkpointCandidates = SnapshotAdmission.checkpointCandidates(
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
                        source: "prefill",
                        checkpointKind: snapshot.checkpointKind
                    ))
            }

            // 11. Start the app-owned generation stream.
            try Task.checkCancellation()
            let generatedTokens = GeneratedTokenRecorder()
            memory.mark(.decoding)
            let (stream, task) = iterator.startGeneration(
                promptTokenCount: fullTokenCount,
                modelConfiguration: session.configuration,
                tokenizer: session.tokenizer,
                tools: canonicalTools,
                generatedTokens: generatedTokens
            )

            return HTTPPrefixCacheGeneration(
                stream: stream,
                completion: task,
                finalCacheOwner: finalCacheOwner,
                speculativeArm: speculation?.arm,
                diagnosticsContext: diagnosticsContext,
                lookupMs: lookupMs,
                restoreMs: restoreMs,
                prefillMs: prefillMs,
                hydrationSeconds: resolved.hydrationSeconds,
                restoredFromSSD: resolved.hydratedFromSSD,
                skippedPrefillTokens: skippedTokens,
                lookupReason: lookupResult.reason,
                sharedPrefixLength: lookupResult.sharedPrefixLength,
                maximumAdvance: maximumAdvance,
                keying: .keyed(request),
                snapshotAdmission: snapshotAdmission,
                transientLastMessageBoundarySnapshot: transientLastMessageBoundarySnapshot,
                transientLastUserBoundarySnapshot: transientLastUserBoundarySnapshot,
                generatedTokens: generatedTokens,
                restoreMode: restoreMode,
                restoreCopy: restoreCopy,
                restoreWaitSeconds: restoreWaitSeconds
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
    ///
    /// Shared preamble mirrors both arms' parameter needs one-to-one.
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
        let cache = try restoredCache ?? session.newCache(parameters: parameters)
        // Reserve the prompt on both fresh and restored live caches. The output
        // ceiling is not a capacity request; decode grows in the vendor cache.
        for layer in cache { layer.reserveCapacity(promptTokens) }
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
            return try session.restore(snapshot, backingLeaf: lookup.backingLeaf)
        } catch {
            Log.server.error(
                "snapshot restore failed — treating as cache miss: \(error)"
            )
            return nil
        }
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_parameter_count
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
        request: UnkeyedRequest,
        input fullInput: LMInput,
        parameters: GenerateParameters,
        toolSpecs: [ToolSpec]?,
        fullAttentionScratchProfile: ModelIdentity.FullAttentionScratchProfile?,
        visionAttentionScratchProfile: ModelIdentity.FullAttentionScratchProfile?,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> HTTPPrefixCacheGeneration {
        // swiftlint:enable function_body_length function_parameter_count
        let facts = request.facts
        diagnosticsContext.logSkip(
            stage: "cacheKeySpace",
            reason: request.reason.rawValue,
            level: .warning,
            extraFields: [("promptTokens", "\(facts.promptTokenCount)")]
        )

        let fullTokenCount = facts.promptTokenCount
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
            let anchoredPrepare = session.anchoredVisionPrepare
        {
            // Image-bearing **Unkeyed Completion** (ADR-0007 phase 2): cache
            // keying failed (e.g. a placeholder/grid mismatch), but the prompt
            // still carries pixels — a single-shot `prepare` would
            // allocate the crash-prone `[heads, L, L]` full-attention scratch.
            // Drive the anchored vision `prepare` from zero instead (state
            // nil ⇒ anchored at offset 0), so even this fallback prefills in
            // bounded `[heads, chunk, L]` windows. The backstop guard fires only
            // if a single window cannot fit (effectively unreachable). The whole
            // continuation runs under a scoped MLX error handler so a runtime
            // failure surfaces as a throw, not a process-fatal dispatch.
            try checkChunkedVisionBackstop(
                windowSize: facts.prefillStepSize,
                contextTokens: fullTokenCount,
                profile: fullAttentionScratchProfile,
                diagnosticsContext: diagnosticsContext
            )
            iterator = try MLXCheckedEvaluation.withErrors { error in
                let built = try session.makePreparingDecodeIterator(
                    fullInput,
                    cache: cache,
                    parameters: facts.decodeParameters,
                    prepare: { input, cache, windowSize in
                        try anchoredPrepare(input, cache, nil, windowSize)
                    }
                )
                try error.check()
                return built
            }
        } else {
            iterator = try session.makePreparingDecodeIterator(
                fullInput,
                cache: cache,
                parameters: facts.decodeParameters,
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

        let generatedTokens = GeneratedTokenRecorder()
        let (stream, task) = TokenGenerationLoop.start(
            promptTokenCount: fullTokenCount,
            modelConfiguration: session.configuration,
            tokenizer: session.tokenizer,
            iterator: iterator,
            tools: toolSpecs,
            generatedTokens: generatedTokens
        )

        return HTTPPrefixCacheGeneration(
            stream: stream,
            completion: task,
            finalCacheOwner: FinalGenerationCache(cache),
            speculativeArm: nil,
            diagnosticsContext: diagnosticsContext,
            lookupMs: 0,
            restoreMs: 0,
            prefillMs: prefillMs,
            hydrationSeconds: 0,
            restoredFromSSD: false,
            skippedPrefillTokens: 0,
            lookupReason: .missNoEntries,
            sharedPrefixLength: 0,
            maximumAdvance: CacheClaim.maximumAdvance(
                newPromptTokens: fullTokenCount, outputCeiling: parameters.maxTokens,
                speculativeAllowance: 0),
            keying: .unkeyed(request),
            snapshotAdmission: nil,
            transientLastMessageBoundarySnapshot: nil,
            transientLastUserBoundarySnapshot: nil,
            generatedTokens: generatedTokens
        )
    }

    /// The cold keyed path for a **Speculation Plan** whose iterator
    /// prefills the whole prompt (MTP, ADR-0056): the iterator's unchunked
    /// vendor prepare runs over a fresh cache (the head needs one target
    /// hidden row per prompt token), then the token loop. The plan was
    /// decided only for a cold, direct-leaf turn whose single-shot prefill
    /// fits the scratch budget.
    ///
    /// Mid-prefill checkpoints are forfeited, but the returned record keeps
    /// the request's real key space and `finalCache`, so the post-generation
    /// leaf capture admits the whole run and the next turn restores warm.
    // swiftlint:disable:next function_parameter_count
    private static func makeWholePromptSpeculativeGeneration(
        plan: SpeculationPlan,
        session: any ModelSession,
        request: KeyedRequest,
        input fullInput: LMInput,
        parameters: GenerateParameters,
        toolSpecs: [ToolSpec]?,
        lookupReason: PrefixCacheManager.LookupReason,
        lookupMs: TimeInterval,
        maximumAdvance: Int,
        visionAttentionScratchProfile: ModelIdentity.FullAttentionScratchProfile?,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> HTTPPrefixCacheGeneration {
        let arm = plan.arm
        let fullTokenCount = request.facts.promptTokenCount
        diagnosticsContext.logSkip(
            stage: "prefill",
            reason: "\(arm.rawValue)-speculative-arm",
            extraFields: [("promptTokens", "\(fullTokenCount)")]
        )

        let begin = try await beginPrefill(
            session: session,
            restoredCache: nil,
            parameters: parameters,
            promptTokens: fullTokenCount,
            cachedTokens: 0,
            pricedImage: nil,
            visionAttentionScratchProfile: visionAttentionScratchProfile,
            guardLabel: arm.rawValue,
            diagnosticsContext: diagnosticsContext,
            progressHandler: progressHandler
        )
        let cache = begin.cache

        let iterator = try MLXCheckedEvaluation.withErrors { error in
            let built = try session.makeSpeculativeDecodeIterator(
                fullInput,
                cache: cache,
                prefilledPrefixTokens: 0,
                plan: plan,
                parameters: request.facts.decodeParameters
            )
            try error.check()
            return built
        }
        let prefillMs = Date.timeIntervalSinceReferenceDate - begin.startedAt
        // The iterator exists — the request will decode speculatively. Fired
        // before the token loop so the activity surfaces badge the arm live.
        await progressHandler?(.speculationEngaged(arm))
        await progressHandler?(
            .prefillFinished(
                .init(
                    promptTokens: fullTokenCount,
                    cachedTokens: 0,
                    newTokensToPrefill: fullTokenCount,
                    prefillMs: prefillMs * 1000
                )))

        let generatedTokens = GeneratedTokenRecorder()
        let (stream, task) = iterator.startGeneration(
            promptTokenCount: fullTokenCount,
            modelConfiguration: session.configuration,
            tokenizer: session.tokenizer,
            tools: toolSpecs,
            generatedTokens: generatedTokens
        )

        return HTTPPrefixCacheGeneration(
            stream: stream,
            completion: task,
            finalCacheOwner: FinalGenerationCache(cache),
            speculativeArm: arm,
            diagnosticsContext: diagnosticsContext,
            lookupMs: lookupMs,
            restoreMs: 0,
            prefillMs: prefillMs,
            hydrationSeconds: 0,
            restoredFromSSD: false,
            skippedPrefillTokens: 0,
            lookupReason: lookupReason,
            sharedPrefixLength: 0,
            maximumAdvance: maximumAdvance,
            keying: .keyed(request),
            snapshotAdmission: nil,
            transientLastMessageBoundarySnapshot: nil,
            transientLastUserBoundarySnapshot: nil,
            generatedTokens: generatedTokens
        )
    }

    // MARK: - Prefix Cache Admin

    /// Lazily creates and returns the `PrefixCacheManager`. Initialization requires
    /// a MainActor hop because PrefixCacheManager is `@MainActor`.
    /// Production leaves `AlphaTuner` detached (#504). Each cache owns its
    /// **Eviction Configuration**, keeps the static LRU default (`alpha = 0`),
    /// and reads the model's `flopProfile` from
    /// **Model Identity** — there is no global to reset or leak.
    ///
    /// When `ssdConfig?.enabled == true` the manager is composed over
    /// a `TieredSnapshotStore` owning an `SSDSnapshotStore`, and
    /// `warmStart` restores the radix-tree structure from the on-disk
    /// manifest. Warm start is fingerprint-gated: partitions from a
    /// different model layout get their descriptors skipped and
    /// their directories scheduled for async cleanup.
    private func ensurePrefixCache(
        on actor: isolated LLMActor, sessions: any ModelSessionProviding
    ) async -> PrefixCacheManager {
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
                // Disabled pending #504: replay allocates full-size synthetic
                // MLX caches and blocks MainActor. No instance means no tuning
                // history or replay; eviction keeps its static alpha = 0.
                alphaTuner: nil,
                tieredStore: tieredStore,
                // Snapshot Demotion's write-through extraction: **Deferred
                // Payload Extraction**, so this MainActor call reads shapes
                // only and the SSD writer copies the bytes (snapshot arrays
                // are evaluated deep copies, never live model state).
                demotionPayloadExtractor: { snapshot in
                    SnapshotPayload.extract(snapshot)
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
                adaptiveWriteEagerness: true,
                modelSessions: sessions
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

    private static func currentMaxMetalBufferBytes() -> UInt64 {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return UInt64.max
        }
        return UInt64(device.maxBufferLength)
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
    /// this request) inside the same Metal-affine scope as the prefill, whose
    /// `session` its **Leaf Admission** captures in. RAM-only by design: the
    /// imminent retry supersedes this leaf with its own SSD-backed one, so
    /// the payload extraction and disk churn are both skipped — the same
    /// economics as speculative preempt capture. Below the progress
    /// threshold nothing is admitted, leaving the cancellation contract (no
    /// leaf, no trace record) unchanged.
    static func salvageCancelledPrefill(
        cache: [any KVCache],
        keySpace: CacheKeySpace,
        restoreBaseOffset: Int,
        partitionKey: CachePartitionKey,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnostics: PrefixCacheDiagnostics.Context,
        session: any ModelSession
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
        // The prefill's cache is lent: the leaf is a copy, admitted RAM-only,
        // so preparing resolves no extension base and makes no MainActor hop.
        let admission = await LeafAdmission.prepare(
            storedTokens: Array(keySpace.keyPath[0..<offset]),
            partitionKey: partitionKey,
            reachesSSD: false,
            requestID: requestID,
            prefixCache: prefixCache,
            diagnostics: diagnostics
        )
        let outcome = await admission.admit(.lent(cache), in: session, labels: .salvageOnCancel)
        if case .admitted(let admitted) = outcome, admitted.survived {
            Log.agent.info(
                "Salvage-on-cancel admitted — offset=\(offset) "
                    + "restoreBase=\(restoreBaseOffset) "
                    + "salvagedTokens=\(offset - restoreBaseOffset)"
            )
        }
    }
}
// swiftlint:enable type_body_length
