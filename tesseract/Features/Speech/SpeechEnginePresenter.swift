//
//  SpeechEnginePresenter.swift
//  tesseract
//
//  The view-facing residency mirror of the v2 speech engine (ADR-0038/0039).
//  The engine itself is an actor in the TesseractSpeech package; views and
//  the InferenceArbiter need synchronous main-actor reads (`isModelLoaded`,
//  `isLoading`), so the SpeechCoordinator reports transitions here as it
//  drives sessions. Replaces the v1 `SpeechEngine` facade in the environment.
//

import Foundation
import Observation
import TesseractSpeech

@Observable @MainActor
final class SpeechEnginePresenter {
    private(set) var isModelLoaded = false
    private(set) var isLoading = false
    private(set) var loadingStatus: String = ""

    let engine: SpeechEngine

    init(engine: SpeechEngine) {
        self.engine = engine
    }

    func noteLoading(_ status: String) {
        isLoading = true
        loadingStatus = status
    }

    func noteReady() {
        isModelLoaded = true
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
        isModelLoaded = false
        isLoading = false
        loadingStatus = ""
    }
}
