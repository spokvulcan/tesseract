//
//  InferenceArbiter.swift
//  tesseract
//

import Foundation
import MLXLMCommon
import Observation
import os

/// Single authority for the LLM's identity and its one-at-a-time rule, and
/// the language model's holder for **Model Residency** (its unload waits its
/// turn at the gate).
///
/// LLM work — chat turns, the Companion's moments, HTTP requests, `/compact`,
/// reloads — runs inside `withLLM`, which takes the **LLM Gate** (FIFO) and
/// makes sure the selected model is loaded first, so the model can never
/// change under a running generation. Nothing else takes the gate: speech,
/// dictation, the proofreader and the embedder run beside the LLM (ADR-0081).
/// Memory, not the GPU, is the shared limit; **Model Residency** knows what
/// is loaded and working, and frees it.
@Observable @MainActor
final class InferenceArbiter: InferenceArbitrating {

    /// Whether the language model is loaded. Derived from engine state so it
    /// cannot desync if the engine is loaded or unloaded outside the arbiter
    /// (e.g., AppDelegate teardown).
    var isLLMLoaded: Bool { agentEngine.isModelLoaded }

    /// Identity of the currently-loaded `.llm` slot — model ID and vision mode.
    /// Kept as a single struct so reload-relevant keys can never drift out of
    /// sync.
    ///
    /// `nonisolated` so the satisfaction rule is testable as a pure value
    /// decision without a MainActor hop.
    nonisolated struct LoadedLLMState: Equatable, Sendable {
        let modelID: String
        let visionMode: Bool

        /// ADR-0008 satisfaction rule: loads upgrade, never downgrade. A
        /// loaded vision container serves text-only demands for the same
        /// model, so alternating chat (toggle off) and HTTP callers cannot
        /// thrash reloads — and the warm prefix cache survives.
        func satisfies(_ desired: LoadedLLMState) -> Bool {
            modelID == desired.modelID && (visionMode || !desired.visionMode)
        }
    }

    private(set) var loadedLLMState: LoadedLLMState?

    /// The model ID currently loaded in the `.llm` slot, or `nil` if unloaded.
    /// Thin accessor over `loadedLLMState` — retained for existing call sites.
    var loadedLLMModelID: String? { loadedLLMState?.modelID }

    /// Template-declared render flags of the loaded `.llm` model (issue #98).
    /// Empty when nothing is loaded.
    var loadedDeclaredTemplateFlags: Set<TemplateRenderFlag> { agentEngine.declaredTemplateFlags }

    /// Template-default value per declared flag of the loaded `.llm` model.
    /// Empty when nothing is loaded.
    var loadedTemplateFlagDefaults: [TemplateRenderFlag: Bool] { agentEngine.templateFlagDefaults }

    /// Whether the loaded `.llm` model's template declares the **Reasoning
    /// Effort** kwarg (ADR-0060). `false` when nothing is loaded.
    var loadedDeclaresReasoningEffort: Bool { agentEngine.declaresReasoningEffort }

    /// The loaded `.llm` model template's own default effort level. `nil`
    /// when nothing is loaded.
    var loadedReasoningEffortTemplateDefault: ReasoningEffort? {
        agentEngine.reasoningEffortTemplateDefault
    }

    /// Tool-call format of the loaded `.llm` model — the identity the server's
    /// Argument Transcoder keys off. `nil` when nothing is loaded or the model
    /// has no override (vendor JSON default).
    var loadedToolCallFormat: ToolCallFormat? { agentEngine.toolCallFormat }

    /// One LLM generation at a time — FIFO queue, atomic handoff,
    /// cancellation protocol. Owned and tested as its own module
    /// (`LLMGateTests`); the arbiter composes it with model loading.
    @ObservationIgnored private let gate = LLMGate()

    /// Whether LLM work is running right now — the **Proofread Pass**'s
    /// skip-when-busy read (ADR-0034): dictation commits its cleaned raw text
    /// at once rather than share the GPU with a long generation. A
    /// point-in-time read, never a wait.
    var isLLMBusy: Bool { gate.isHeld }

    // MARK: - Dependencies

    private let agentEngine: AgentEngine
    private let settingsManager: SettingsManager
    private let modelDownloadManager: ModelDownloadManager

    init(
        agentEngine: AgentEngine,
        settingsManager: SettingsManager,
        modelDownloadManager: ModelDownloadManager
    ) {
        self.agentEngine = agentEngine
        self.settingsManager = settingsManager
        self.modelDownloadManager = modelDownloadManager
    }

    // MARK: - Public API

    /// Scoped LLM access: waits its FIFO turn at the **LLM Gate**, makes sure
    /// the model is loaded, runs the closure, and releases the gate on exit —
    /// including on throw.
    ///
    /// The gate semantics (FIFO, atomic handoff, cancellation while queued or
    /// during handoff) live in `LLMGate`; the arbiter's contribution is
    /// holding the gate across `ensureLLMLoaded` *and* the body, so model
    /// identity can never change under a running consumer.
    func withLLM<T: Sendable>(
        modelIDOverride: String?,
        vision: LLMVisionRequirement,
        body: () async throws -> T
    ) async throws -> T {
        try await gate.withExclusive {
            Log.general.info("InferenceArbiter: LLM gate taken")
            try await ensureLLMLoaded(modelIDOverride: modelIDOverride, vision: vision)
            return try await body()
        }
    }

    /// Propagate a settings change (selected model or vision mode) into an
    /// eager model reload. Takes its turn at the gate, runs
    /// `ensureLLMLoaded` — which compares desired state against
    /// `loadedLLMState` and reloads on mismatch — and releases. A no-op when
    /// nothing relevant changed and the model is already loaded. Throws the
    /// same errors as any other LLM turn (including `modelNotDownloaded` if
    /// the currently-selected model is not on disk). Independent of
    /// `isServerEnabled`: internal server-core use must work without the
    /// public HTTP listener enabled.
    func reloadLLMIfNeeded() async throws {
        try await withLLM {}
    }

    // MARK: - Model Management

    /// Load the LLM if the loaded state doesn't satisfy the target. The
    /// target model ID is `modelIDOverride` when the caller passed one (HTTP
    /// requests honoring `request.model`), otherwise the user's
    /// `settingsManager.selectedAgentModelID` (chat UI, background agents).
    /// Vision mode follows `vision` (ADR-0008): chat callers honor the global
    /// vision opt-out (`.fromSettings`); HTTP callers demand vision whenever
    /// the target model is capable (`.visionIfCapable`). Satisfaction upgrades
    /// but never downgrades — a loaded vision container also serves text-only
    /// demands. The TTS engine loads itself (ADR-0039).
    private func ensureLLMLoaded(modelIDOverride: String?, vision: LLMVisionRequirement)
        async throws
    {
        let targetModelID = modelIDOverride ?? settingsManager.selectedAgentModelID
        let desiredVision = vision.wantsVision(
            useVisionWhenAvailable: settingsManager.useVisionWhenAvailable,
            isVisionCapable: modelDownloadManager.isVisionCapable(targetModelID)
        )
        let desired = LoadedLLMState(modelID: targetModelID, visionMode: desiredVision)
        if isLLMLoaded, let loaded = loadedLLMState, loaded.satisfies(desired) {
            return
        }
        // Model or vision mode changed, or not loaded — (re)load
        if isLLMLoaded { unloadLLM() }
        // Drain the detached unload task before the next load. Without
        // this, the actor-level `llmActor.unloadModel()` can interleave
        // after the new `llmActor.loadModel()` and tear down the freshly
        // loaded model, tokenizer, and prefix-cache state.
        await agentEngine.awaitPendingUnload()
        try await loadLLM(modelID: desired.modelID, visionMode: desired.visionMode)
        loadedLLMState = desired
    }

    private func loadLLM(modelID: String, visionMode: Bool) async throws {
        guard let path = modelDownloadManager.usableDirectory(for: modelID) else {
            Log.general.error("InferenceArbiter: LLM model '\(modelID)' not downloaded")
            // Specific error case so HTTP callers can surface 404
            // `model_not_found` instead of a generic 503. Closes the race
            // where a model validated before its turn is deleted from
            // Settings → Models while a request is queued.
            throw AgentEngineError.modelNotDownloaded(modelID: modelID)
        }
        Log.general.info(
            "InferenceArbiter: loading LLM model '\(modelID)' visionMode=\(visionMode)")
        try await agentEngine.loadModel(from: path, visionMode: visionMode)
    }

    /// The language model's part of Offload Model: it waits its turn at the
    /// gate, so a running generation finishes first and nothing can
    /// interleave. The next consumer lazy-reloads as usual.
    func offloadLLM() async {
        do {
            try await gate.withExclusive {
                guard isLLMLoaded else { return }
                unloadLLM()
            }
        } catch {
            Log.general.error(
                "InferenceArbiter.offloadLLM: the LLM gate failed: "
                    + "\(error.localizedDescription)")
        }
    }

    private func unloadLLM() {
        agentEngine.unloadModel()
        loadedLLMState = nil
        Log.general.info("InferenceArbiter: unloaded LLM")
    }
}
