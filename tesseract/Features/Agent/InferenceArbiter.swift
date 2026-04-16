//
//  InferenceArbiter.swift
//  tesseract
//

import Foundation
import MLXLMCommon
import Observation
import os

/// Which model occupies a GPU slot.
///
/// Co-resident slots (`.llm`, `.tts`) can coexist in memory.
/// Exclusive slots (`.imageGen`) evict all co-residents when loaded.
enum ModelSlot: Sendable, Hashable, CustomStringConvertible {
    case llm
    case tts
    case imageGen

    var description: String {
        switch self {
        case .llm: "llm"
        case .tts: "tts"
        case .imageGen: "imageGen"
        }
    }
}

/// Single authority for model ownership and GPU serialization.
///
/// Replaces ad-hoc `prepareForInference`/`prepareForSpeech`/`ensureModelLoaded`
/// callbacks with a scoped lease API. Only one consumer generates at a time.
///
/// Memory residency model:
///   - LLM + TTS are co-resident (independently lazy-loaded, both allowed in
///     memory simultaneously). Neither evicts the other.
///   - ImageGen is exclusive (evicts co-resident models when loaded, and
///     co-resident loads evict ImageGen). Prototype-only, hidden from UI.
///   - STT (WhisperKit) runs on CoreML in a separate memory pool — not managed here.
@Observable @MainActor
final class InferenceArbiter {

    /// Which slots are currently loaded. Derived from engine state so it cannot
    /// desync if engines are loaded/unloaded outside the arbiter (e.g., AppDelegate teardown).
    var loadedSlots: Set<ModelSlot> {
        var slots: Set<ModelSlot> = []
        if agentEngine.isModelLoaded { slots.insert(.llm) }
        if speechEngine.isModelLoaded { slots.insert(.tts) }
        if imageGenEngine.isModelLoaded || zimageGenEngine.isModelLoaded { slots.insert(.imageGen) }
        return slots
    }

    /// Identity of the currently-loaded `.llm` slot — model ID, vision mode,
    /// and the hidden TriAttention runtime snapshot. Kept as a single struct
    /// so reload-relevant keys can never drift out of sync.
    struct LoadedLLMState: Equatable {
        let modelID: String
        let visionMode: Bool
        let requestedTriAttention: TriAttentionConfiguration
        let effectiveTriAttention: TriAttentionConfiguration
        let triAttentionFallbackReason: TriAttentionDenseFallbackReason?

        init(
            modelID: String,
            visionMode: Bool,
            requestedTriAttention: TriAttentionConfiguration = .v1Disabled,
            effectiveTriAttention: TriAttentionConfiguration = .v1Disabled,
            triAttentionFallbackReason: TriAttentionDenseFallbackReason? = nil
        ) {
            self.modelID = modelID
            self.visionMode = visionMode
            self.requestedTriAttention = requestedTriAttention
            self.effectiveTriAttention = effectiveTriAttention
            self.triAttentionFallbackReason = triAttentionFallbackReason
        }
    }

    private(set) var loadedLLMState: LoadedLLMState?

    /// The model ID currently loaded in the `.llm` slot, or `nil` if unloaded.
    /// Thin accessor over `loadedLLMState` — retained for existing call sites.
    var loadedLLMModelID: String? { loadedLLMState?.modelID }

    /// FIFO waiter queue for GPU access. Each entry is identified by UUID
    /// for cancellation-safe removal.
    @ObservationIgnored private var waiters: [(id: UUID, continuation: CheckedContinuation<Void, any Error>)] = []
    @ObservationIgnored private var isLeased: Bool = false

    /// Continuations waiting for the GPU to become fully idle (no lease, no queued waiters).
    /// Used by background tasks to defer to foreground work.
    @ObservationIgnored private var idleWaiters: [(id: UUID, continuation: CheckedContinuation<Void, Never>)] = []

    // MARK: - Dependencies

    private let agentEngine: AgentEngine
    private let speechEngine: SpeechEngine
    private let imageGenEngine: ImageGenEngine
    private let zimageGenEngine: ZImageGenEngine
    private let settingsManager: SettingsManager
    private let modelDownloadManager: ModelDownloadManager

    init(
        agentEngine: AgentEngine,
        speechEngine: SpeechEngine,
        imageGenEngine: ImageGenEngine,
        zimageGenEngine: ZImageGenEngine,
        settingsManager: SettingsManager,
        modelDownloadManager: ModelDownloadManager
    ) {
        self.agentEngine = agentEngine
        self.speechEngine = speechEngine
        self.imageGenEngine = imageGenEngine
        self.zimageGenEngine = zimageGenEngine
        self.settingsManager = settingsManager
        self.modelDownloadManager = modelDownloadManager
    }

    // MARK: - Public API

    /// Scoped exclusive GPU access. Waits for any active lease to complete
    /// (FIFO order), ensures the required model is loaded, runs the closure,
    /// and releases the lease on exit — including on throw.
    ///
    /// Cancellation:
    ///   - While waiting in the queue: the waiter is removed and
    ///     `CancellationError` is thrown without ever acquiring the lease.
    ///   - After resumption but before ownership transfer: `Task.checkCancellation()`
    ///     runs before `isLeased` is set, so a cancelled waiter cannot inherit
    ///     the lease during handoff.
    ///   - Once the lease is acquired: cancellation propagates normally through
    ///     the body and the lease is released via `defer`.
    func withExclusiveGPU<T: Sendable>(
        _ slot: ModelSlot,
        llmModelIDOverride: String? = nil,
        body: () async throws -> T
    ) async throws -> T {
        // Block if lease is held OR waiters exist (prevents queue bypass).
        if isLeased || !waiters.isEmpty {
            let waiterID = UUID()
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                    if Task.isCancelled {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    waiters.append((id: waiterID, continuation: continuation))
                }
            } onCancel: {
                // Runs concurrently — MainActor hop to safely mutate waiters.
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    if let idx = self.waiters.firstIndex(where: { $0.id == waiterID }) {
                        let removed = self.waiters.remove(at: idx)
                        removed.continuation.resume(throwing: CancellationError())
                    }
                    // If already removed by the defer handoff (which removes
                    // before resuming), firstIndex returns nil — no double-resume.
                }
            }
        }

        // A waiter may have been resumed just before cancellation won the race.
        // Re-check before claiming the lease so cancelled waiters never inherit it.
        try Task.checkCancellation()

        // At this point we own the lease — set flag before any await.
        isLeased = true
        Log.general.info("InferenceArbiter: lease acquired for \(slot)")

        defer {
            // Atomic handoff: if waiters exist, keep isLeased = true and
            // resume the next waiter directly. Only set isLeased = false
            // when the queue is drained.
            if !waiters.isEmpty {
                let next = waiters.removeFirst()
                Log.general.debug("InferenceArbiter: handing off lease, \(self.waiters.count) still queued")
                next.continuation.resume()
            } else {
                isLeased = false
                Log.general.info("InferenceArbiter: lease released for \(slot)")
                // Signal background tasks waiting for idle
                if !idleWaiters.isEmpty {
                    let pending = idleWaiters
                    idleWaiters.removeAll()
                    for waiter in pending { waiter.continuation.resume() }
                }
            }
        }

        // Ensure requested model is loaded (co-resident or exclusive)
        try await ensureLoaded(slot, llmModelIDOverride: llmModelIDOverride)

        return try await body()
    }

    /// Like `withExclusiveGPU`, but defers to foreground work.
    ///
    /// The caller waits until no lease is held and no FIFO waiters are queued,
    /// then acquires. If foreground work arrives while waiting, the caller
    /// re-waits rather than competing in FIFO order. This prevents background
    /// tasks from wedging between consecutive user turns.
    func withDeferredGPU<T: Sendable>(
        _ slot: ModelSlot,
        llmModelIDOverride: String? = nil,
        body: () async throws -> T
    ) async throws -> T {
        // Loop: wait for idle, then re-check. If a foreground request arrived
        // between the idle signal and our continuation running, loop back.
        while isLeased || !waiters.isEmpty {
            await suspendUntilIdle()
            try Task.checkCancellation()
        }
        // GPU is idle with no foreground waiters — acquire immediately.
        // On MainActor, no work can interleave between the loop exit and
        // withExclusiveGPU's synchronous isLeased/waiters check.
        return try await withExclusiveGPU(
            slot,
            llmModelIDOverride: llmModelIDOverride,
            body: body
        )
    }

    // MARK: - Idle Signaling

    /// Suspends until the next idle signal. Called inside `withDeferredGPU`'s
    /// retry loop. Does NOT check if currently idle — the caller handles that.
    private func suspendUntilIdle() async {
        let waiterID = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                if Task.isCancelled {
                    continuation.resume()
                    return
                }
                idleWaiters.append((id: waiterID, continuation: continuation))
            }
        } onCancel: {
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let idx = self.idleWaiters.firstIndex(where: { $0.id == waiterID }) {
                    let removed = self.idleWaiters.remove(at: idx)
                    removed.continuation.resume()
                }
            }
        }
    }

    /// Releases whichever image engine the caller does NOT need.
    /// Call inside a `withExclusiveGPU(.imageGen)` body before loading your engine,
    /// so that only one image pipeline is resident at a time.
    enum ImageEngine { case flux, zImage }
    func releaseOtherImageEngine(keeping: ImageEngine) {
        switch keeping {
        case .flux:
            if zimageGenEngine.isModelLoaded { zimageGenEngine.releaseModel() }
        case .zImage:
            if imageGenEngine.isModelLoaded { imageGenEngine.releaseModel() }
        }
    }

    // MARK: - Model Management

    /// Load a model slot. Co-resident slots coexist; ImageGen is exclusive.
    /// For `.llm`: checks if the loaded model ID and vision mode match the
    /// target. The target model ID is `llmModelIDOverride` when the caller
    /// passed one (HTTP requests honoring `request.model`), otherwise the
    /// user's `settingsManager.selectedAgentModelID` (chat UI, background
    /// agents). Vision mode is always sourced from settings — HTTP requests
    /// cannot override it (see docs/HTTP_SERVER_SPEC.md §4.2 Model routing).
    private func ensureLoaded(
        _ slot: ModelSlot,
        llmModelIDOverride: String? = nil
    ) async throws {
        switch slot {
        case .llm:
            let targetModelID = llmModelIDOverride ?? settingsManager.selectedAgentModelID
            let requestedTriAttention = settingsManager.makeTriAttentionConfig()
            let desired = LoadedLLMState(
                modelID: targetModelID,
                visionMode: settingsManager.visionModeEnabled,
                requestedTriAttention: requestedTriAttention
            )
            if loadedSlots.contains(.llm),
                loadedLLMState?.modelID == desired.modelID,
                loadedLLMState?.visionMode == desired.visionMode,
                loadedLLMState?.requestedTriAttention == desired.requestedTriAttention
            {
                return
            }
            // Model or vision mode changed, or not loaded — (re)load
            if loadedSlots.contains(.imageGen) { unload(.imageGen) }
            if loadedSlots.contains(.llm) { unload(.llm) }
            try await loadSlot(
                .llm,
                modelID: desired.modelID,
                visionMode: desired.visionMode,
                triAttention: desired.requestedTriAttention
            )
            let triAttentionRuntimeSelection = agentEngine.triAttentionRuntimeSelection
            loadedLLMState = LoadedLLMState(
                modelID: desired.modelID,
                visionMode: desired.visionMode,
                requestedTriAttention: desired.requestedTriAttention,
                effectiveTriAttention: triAttentionRuntimeSelection.effectiveConfiguration,
                triAttentionFallbackReason: triAttentionRuntimeSelection.fallbackReason
            )
            if let fallbackReason = triAttentionRuntimeSelection.fallbackReason {
                Log.general.notice(
                    "InferenceArbiter: TriAttention dense fallback — "
                    + "model=\(desired.modelID) visionMode=\(desired.visionMode) "
                    + "requestedEnabled=\(desired.requestedTriAttention.enabled) "
                    + "effectiveEnabled=\(triAttentionRuntimeSelection.effectiveConfiguration.enabled) "
                    + "reason=\(fallbackReason.rawValue)"
                )
            }

        case .tts:
            if loadedSlots.contains(.tts) { return }
            if loadedSlots.contains(.imageGen) { unload(.imageGen) }
            try await loadSlot(.tts)

        case .imageGen:
            // Exclusive — evict co-resident models (LLM, TTS).
            // Don't early-return when an image engine is already loaded: the caller
            // may need a different image engine (Flux vs Z-Image) and will load it
            // inside the lease body. We only guarantee co-residents are cleared.
            if loadedSlots.contains(.llm) { unload(.llm) }
            if loadedSlots.contains(.tts) { unload(.tts) }
            loadedLLMState = nil
        }
    }

    private func loadSlot(
        _ slot: ModelSlot,
        modelID: String? = nil,
        visionMode: Bool = false,
        triAttention: TriAttentionConfiguration = .v1Disabled
    ) async throws {
        switch slot {
        case .llm:
            guard let modelID else {
                throw AgentEngineError.modelNotLoaded
            }
            guard case .downloaded = modelDownloadManager.statuses[modelID],
                  let path = modelDownloadManager.modelPath(for: modelID)
            else {
                Log.general.error("InferenceArbiter: LLM model '\(modelID)' not downloaded")
                // Specific error case so HTTP callers can surface 404
                // `model_not_found` instead of a generic 503. Closes the
                // race where a model validated pre-lease is deleted from
                // Settings → Models while a request is queued.
                throw AgentEngineError.modelNotDownloaded(modelID: modelID)
            }
            Log.general.info(
                "InferenceArbiter: loading LLM model '\(modelID)' "
                + "visionMode=\(visionMode) "
                + "triAttentionRequested=\(triAttention.enabled)"
            )
            try await agentEngine.loadModel(
                from: path,
                visionMode: visionMode,
                triAttention: triAttention
            )

        case .tts:
            Log.general.info("InferenceArbiter: loading TTS model")
            try await speechEngine.loadModel()

        case .imageGen:
            // ImageGen loading requires a model path — callers must load manually
            // for now since there's no standardized model ID for image gen.
            // The arbiter manages eviction; ImageGen UI handles its own loading.
            break
        }
    }

    private func unload(_ slot: ModelSlot) {
        switch slot {
        case .llm:
            agentEngine.unloadModel()
            loadedLLMState = nil
            Log.general.info("InferenceArbiter: unloaded LLM")

        case .tts:
            speechEngine.unloadModel()
            Log.general.info("InferenceArbiter: unloaded TTS")

        case .imageGen:
            if imageGenEngine.isModelLoaded {
                imageGenEngine.releaseModel()
            }
            if zimageGenEngine.isModelLoaded {
                zimageGenEngine.releaseModel()
            }
            Log.general.info("InferenceArbiter: unloaded ImageGen")
        }
    }
}
