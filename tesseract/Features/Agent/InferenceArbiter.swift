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
nonisolated enum ModelSlot: Sendable, Hashable, CustomStringConvertible {
    case llm
    case tts

    var description: String {
        switch self {
        case .llm: "llm"
        case .tts: "tts"
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
///   - STT (WhisperKit) runs on CoreML in a separate memory pool — not managed here.
@Observable @MainActor
final class InferenceArbiter: InferenceArbitrating {

    /// Which slots are currently loaded. Derived from engine state so it cannot
    /// desync if engines are loaded/unloaded outside the arbiter (e.g., AppDelegate teardown).
    var loadedSlots: Set<ModelSlot> {
        var slots: Set<ModelSlot> = []
        if agentEngine.isModelLoaded { slots.insert(.llm) }
        if speechEngine.isModelLoaded { slots.insert(.tts) }
        return slots
    }

    /// Identity of the currently-loaded `.llm` slot — model ID and vision mode.
    /// Kept as a single struct so reload-relevant keys can never drift out of
    /// sync.
    struct LoadedLLMState: Equatable {
        let modelID: String
        let visionMode: Bool
    }

    private(set) var loadedLLMState: LoadedLLMState?

    /// The model ID currently loaded in the `.llm` slot, or `nil` if unloaded.
    /// Thin accessor over `loadedLLMState` — retained for existing call sites.
    var loadedLLMModelID: String? { loadedLLMState?.modelID }

    /// The GPU mutual-exclusion lease — FIFO queue, atomic handoff, cancellation
    /// protocol. Owned and tested as its own module (`GPULeaseQueueTests`); the
    /// arbiter composes it with model loading.
    @ObservationIgnored private let lease = GPULeaseQueue()

    // MARK: - Dependencies

    private let agentEngine: AgentEngine
    private let speechEngine: SpeechEngine
    private let settingsManager: SettingsManager
    private let modelDownloadManager: ModelDownloadManager

    init(
        agentEngine: AgentEngine,
        speechEngine: SpeechEngine,
        settingsManager: SettingsManager,
        modelDownloadManager: ModelDownloadManager
    ) {
        self.agentEngine = agentEngine
        self.speechEngine = speechEngine
        self.settingsManager = settingsManager
        self.modelDownloadManager = modelDownloadManager
    }

    // MARK: - Public API

    /// Scoped exclusive GPU access. Waits for any active lease to complete
    /// (FIFO order), ensures the required model is loaded, runs the closure,
    /// and releases the lease on exit — including on throw.
    ///
    /// The lease semantics (FIFO, atomic handoff, cancellation while queued or
    /// during handoff) live in `GPULeaseQueue`; the arbiter's contribution is
    /// holding the lease across `ensureLoaded` *and* the body, so model identity
    /// can never change under a running consumer.
    func withExclusiveGPU<T: Sendable>(
        _ slot: ModelSlot,
        llmModelIDOverride: String? = nil,
        body: () async throws -> T
    ) async throws -> T {
        try await lease.withExclusive {
            Log.general.info("InferenceArbiter: lease acquired for \(slot)")
            try await ensureLoaded(slot, llmModelIDOverride: llmModelIDOverride)
            return try await body()
        }
    }

    /// Propagate a settings change (selected model or vision mode) into an
    /// eager model reload. Acquires the `.llm` lease FIFO-fair,
    /// runs `ensureLoaded(.llm)` — which compares desired state against
    /// `loadedLLMState` and reloads on mismatch — and releases. A no-op when
    /// nothing relevant changed and the model is already loaded. Throws the
    /// same errors as any other `.llm` lease acquisition (including
    /// `modelNotDownloaded` if the currently-selected model is not on disk).
    /// Independent of `isServerEnabled`: internal server-core use must work
    /// without the public HTTP listener enabled.
    func reloadLLMIfNeeded() async throws {
        try await withExclusiveGPU(.llm) { }
    }

    // MARK: - Model Management

    /// Load a model slot. Co-resident slots coexist.
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
            let desired = LoadedLLMState(
                modelID: targetModelID,
                visionMode: settingsManager.visionModeEnabled
            )
            if loadedSlots.contains(.llm),
                loadedLLMState?.modelID == desired.modelID,
                loadedLLMState?.visionMode == desired.visionMode
            {
                return
            }
            // Model or vision mode changed, or not loaded — (re)load
            if loadedSlots.contains(.llm) { unload(.llm) }
            // Drain the detached unload task before the next load. Without
            // this, the actor-level `llmActor.unloadModel()` can interleave
            // after the new `llmActor.loadModel()` and tear down the freshly
            // loaded model, tokenizer, and prefix-cache state.
            await agentEngine.awaitPendingUnload()
            try await loadSlot(
                .llm,
                modelID: desired.modelID,
                visionMode: desired.visionMode
            )
            loadedLLMState = desired

        case .tts:
            if loadedSlots.contains(.tts) { return }
            try await loadSlot(.tts)
        }
    }

    private func loadSlot(
        _ slot: ModelSlot,
        modelID: String? = nil,
        visionMode: Bool = false
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
                + "visionMode=\(visionMode)"
            )
            try await agentEngine.loadModel(
                from: path,
                visionMode: visionMode
            )

        case .tts:
            Log.general.info("InferenceArbiter: loading TTS model")
            try await speechEngine.loadModel()
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
        }
    }
}
