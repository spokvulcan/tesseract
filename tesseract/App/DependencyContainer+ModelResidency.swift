//
//  DependencyContainer+ModelResidency.swift
//  tesseract
//
//  The models Tesseract holds, one row per holder, in the order the menu's
//  Models section lists them: each says which model it holds, whether that is
//  loaded and what it is doing, and how Offload Model frees it. **Model
//  Residency** reads them for the menu, the Models page and Offload.
//

import Foundation

extension DependencyContainer {

    func makeModelResidency() -> ModelResidency {
        ModelResidency(
            holders: [
                languageModel, speedUpDraft, voice, dictation, proofreader, memorySearch,
            ],
            sizeOnDisk: { [modelDownloadManager] in modelDownloadManager.sizeOnDisk(of: $0) })
    }

    /// The language model: working while LLM work holds its gate; offloaded
    /// after the running generation.
    private var languageModel: ModelResidency.Holder {
        ModelResidency.Holder(
            id: "llm", role: "Language model",
            model: { [unowned self] in
                inferenceArbiter.loadedLLMModelID ?? settingsManager.selectedAgentModelID
            },
            state: { [unowned self] in
                agentEngine.isLoading
                    ? .loading
                    : !agentEngine.isModelLoaded
                        ? .notLoaded : inferenceArbiter.isLLMBusy ? .working(llmWork) : .loaded
            },
            unload: { [unowned self] in await inferenceArbiter.offloadLLM() })
    }

    /// The DFlash2 draft: loaded only beside its target, freed with it.
    private var speedUpDraft: ModelResidency.Holder {
        ModelResidency.Holder(
            id: "draft", role: "Speed-up draft",
            model: { DFlash2Support.draftModelID },
            state: { [unowned self] in
                !agentEngine.isDFlash2DraftLoaded
                    ? .notLoaded : inferenceArbiter.isLLMBusy ? .working("Drafting") : .loaded
            },
            isListedOnlyWhileLoaded: true)
    }

    /// The voice: freed at once (the engine cancels its own utterance first).
    private var voice: ModelResidency.Holder {
        ModelResidency.Holder(
            id: "tts", role: "Voice",
            model: { ModelDefinition.defaultTextToSpeechModelID },
            state: { [unowned self] in
                let speaking: Bool =
                    switch speechCoordinator.state {
                    case .generating, .streaming, .streamingLongForm, .playing: true
                    case .idle, .capturingText, .paused, .error: false
                    }
                return speechEnginePresenter.isLoading
                    ? .loading
                    : !speechEnginePresenter.isModelLoaded
                        ? .notLoaded : speaking ? .working("Speaking") : .loaded
            },
            unload: { [unowned self] in await speechEnginePresenter.unload() })
    }

    /// Dictation: listening while a take records on any surface, transcribing
    /// while Whisper decodes (a Live Preview's decode too) or a take is written
    /// down. Offload leaves it: always-armed dictation is a product promise
    /// (ADR-0025), and WhisperKit runs on the Neural Engine, in its own memory.
    private var dictation: ModelResidency.Holder {
        ModelResidency.Holder(
            id: "stt", role: "Dictation",
            model: { [unowned self] in settingsManager.selectedSpeechToTextModelID },
            state: { [unowned self] in
                !transcriptionEngine.isModelLoaded
                    ? .notLoaded
                    : transcriptionEngine.isTranscribing || voiceCapture.isTranscribing
                        ? .working("Transcribing")
                        : voiceCapture.isListening
                            ? .working("Listening")
                            : builtVoiceSession?.isActive == true
                                ? .working("Voice session") : .loaded
            },
            hasSize: false)
    }

    /// The Proofread Pass's own small model.
    private var proofreader: ModelResidency.Holder {
        ModelResidency.Holder(
            id: "proofread", role: "Proofreader",
            model: { ModelDefinition.defaultProofreadModelID },
            state: { [unowned self] in
                !proofreadPass.isModelLoaded
                    ? .notLoaded : voiceCapture.isProofreading ? .working("Proofreading") : .loaded
            },
            unload: { [unowned self] in await proofreadPass.unload() })
    }

    /// The recall embedder: loaded on first use; the next recall loads it again.
    private var memorySearch: ModelResidency.Holder {
        ModelResidency.Holder(
            id: "embedder", role: "Memory search",
            model: { ModelDefinition.defaultEmbeddingModelID },
            state: { [unowned self] in recallEmbedder.isModelLoaded ? .loaded : .notLoaded },
            unload: { [unowned self] in await recallEmbedder.unload() })
    }

    /// What the language model is doing: the Companion's moment by name.
    private var llmWork: String {
        if let moment = dayThread.momentRunning { return "Jarvis · \(moment.title)" }
        return "Generating"
    }
}
