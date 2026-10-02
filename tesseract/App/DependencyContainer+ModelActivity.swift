//
//  DependencyContainer+ModelActivity.swift
//  tesseract
//
//  What the menu bar's Models section shows: each model, loaded or not, and
//  what it is doing right now — read from the engine that holds it, plus the
//  app's memory footprint. Sizes are the weights on disk, which is what an
//  MLX model holds in memory; Whisper runs on the Neural Engine, so it has
//  no size here.
//

import Foundation

extension DependencyContainer {

    func modelActivity() -> ModelActivity {
        var rows: [ModelActivityRow] = []

        // The language model: working while LLM work holds its gate.
        let llmID = inferenceArbiter.loadedLLMModelID ?? settingsManager.selectedAgentModelID
        let llmWorking = agentEngine.isModelLoaded && inferenceArbiter.isLLMBusy
        rows.append(
            ModelActivityRow(
                id: "llm", role: "Language model", name: displayName(of: llmID),
                state: agentEngine.isLoading
                    ? .loading
                    : !agentEngine.isModelLoaded
                        ? .notLoaded : llmWorking ? .working(llmWork) : .loaded,
                bytes: agentEngine.isModelLoaded ? weightBytes(of: llmID) : nil))
        if agentEngine.isDFlash2DraftLoaded {
            rows.append(
                ModelActivityRow(
                    id: "draft", role: "Speed-up draft",
                    name: displayName(of: DFlash2Support.draftModelID),
                    state: llmWorking ? .working("Drafting") : .loaded,
                    bytes: weightBytes(of: DFlash2Support.draftModelID)))
        }

        // The voice.
        let speaking: Bool =
            switch speechCoordinator.state {
            case .generating, .streaming, .streamingLongForm, .playing: true
            case .idle, .capturingText, .paused, .error: false
            }
        rows.append(
            ModelActivityRow(
                id: "tts", role: "Voice",
                name: displayName(of: ModelDefinition.defaultTextToSpeechModelID),
                state: speechEnginePresenter.isLoading
                    ? .loading
                    : !speechEnginePresenter.isModelLoaded
                        ? .notLoaded : speaking ? .working("Speaking") : .loaded,
                bytes: speechEnginePresenter.isModelLoaded
                    ? weightBytes(of: ModelDefinition.defaultTextToSpeechModelID) : nil))

        // Dictation: listening while any microphone take records,
        // transcribing while Whisper decodes.
        let voiceInputs = [agentVoiceInput, captureVoiceInput, panelVoiceInput]
        let listening =
            dictationFeed.phase == .recording
            || voiceInputs.contains { $0.voiceState == .recording }
        let transcribing =
            transcriptionEngine.isTranscribing || dictationFeed.phase == .processing
            || voiceInputs.contains { $0.voiceState == .transcribing }
        let speechToTextID = settingsManager.selectedSpeechToTextModelID
        rows.append(
            ModelActivityRow(
                id: "stt", role: "Dictation", name: displayName(of: speechToTextID),
                state: !transcriptionEngine.isModelLoaded
                    ? .notLoaded
                    : transcribing
                        ? .working("Transcribing")
                        : listening
                            ? .working("Listening")
                            : companionVoiceSession.isActive ? .working("Voice session") : .loaded))

        // The proofreader and the recall embedder.
        let proofreadID = ModelDefinition.defaultProofreadModelID
        rows.append(
            ModelActivityRow(
                id: "proofread", role: "Proofreader", name: displayName(of: proofreadID),
                state: !proofreadPass.isModelLoaded
                    ? .notLoaded
                    : dictationFeed.phase == .proofreading ? .working("Proofreading") : .loaded,
                bytes: proofreadPass.isModelLoaded ? weightBytes(of: proofreadID) : nil))
        let embedderID = ModelDefinition.defaultEmbeddingModelID
        rows.append(
            ModelActivityRow(
                id: "embedder", role: "Memory search", name: displayName(of: embedderID),
                state: isEmbedderLoaded ? .loaded : .notLoaded,
                bytes: isEmbedderLoaded ? weightBytes(of: embedderID) : nil))

        let sample = RequestMemoryTelemetry.Sample.current()
        return ModelActivity(
            rows: rows, footprintBytes: sample.footprintBytes, swapBytes: sample.systemSwapUsedBytes
        )
    }

    /// What the language model is doing: the Companion's moment by name.
    private var llmWork: String {
        if let moment = dayThread.momentRunning { return "Jarvis · \(moment.title)" }
        return "Generating"
    }

    private func displayName(of modelID: String) -> String {
        ModelDefinition.withID(modelID)?.displayName ?? modelID
    }

    private func weightBytes(of modelID: String) -> Int64? {
        if case .downloaded(let size) = modelDownloadManager.status(for: modelID) { return size }
        return nil
    }
}
