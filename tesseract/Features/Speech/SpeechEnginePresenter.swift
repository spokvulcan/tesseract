//
//  SpeechEnginePresenter.swift
//  tesseract
//
//  The view-facing residency mirror of the v2 speech engine (ADR-0038/0039).
//  The engine itself is an actor in the TesseractSpeech package; views and
//  the InferenceArbiter need synchronous main-actor reads (`isModelLoaded`,
//  `isLoading`). `isModelLoaded` follows the Readiness the engine publishes,
//  including a load the engine starts by itself for an utterance on a session
//  that outlived an unload. `isLoading` is the SpeechCoordinator's note that
//  it is opening a session. Replaces the v1 `SpeechEngine` facade in the
//  environment.
//

import Foundation
import Observation
import TesseractSpeech

@Observable @MainActor
final class SpeechEnginePresenter {
    /// Written only by the engine's readiness updates, so it can't drift from
    /// what the engine holds (Offload Model decides from it).
    private(set) var isModelLoaded = false
    private(set) var isLoading = false
    private(set) var loadingStatus: String = ""

    let engine: SpeechEngine

    init(engine: SpeechEngine) {
        self.engine = engine
        // Holds the stream, not the engine: the loop ends when the engine
        // goes away and finishes it.
        Task { [weak self] in
            guard let updates = await self?.engine.readinessUpdates() else { return }
            for await readiness in updates {
                guard let self else { return }
                self.isModelLoaded = readiness >= .loaded
            }
        }
    }

    func noteLoading(_ status: String) {
        isLoading = true
        loadingStatus = status
    }

    func noteReady() {
        isLoading = false
        loadingStatus = ""
    }

    func noteFailed() {
        isLoading = false
        loadingStatus = ""
    }

    /// Deterministic release (ADR-0039): the active utterance's stream has
    /// terminated before this returns; weights, KV, and caches are freed and
    /// the GPU stream synced. Sessions survive as ingredient values.
    func unload() async {
        await engine.unload()
        isLoading = false
        loadingStatus = ""
    }
}
