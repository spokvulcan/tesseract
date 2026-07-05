//
//  AgentInputStatusTests.swift
//  tesseractTests
//
//  The agent input strip's status derivation, lifted to a pure function
//  (PRD #171): errors outrank voice errors outrank loading outrank the
//  download states. The new progress-aware case shows a calm in-flight line
//  while the selected agent model is actively downloading; warning yellow
//  stays for the genuinely-missing case. Prior art: `OnboardingModelPickTests`
//  (pure derivation), `AgentGenerationErrorSurfacingTests` (status strip).
//

import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentInputStatusTests {

    private func derive(
        error: String? = nil,
        voiceErrorMessage: String? = nil,
        isEngineLoading: Bool = false,
        loadingStatus: String = "",
        isModelLoaded: Bool = false,
        selectedModelDisplayName: String = "Qwen",
        selectedModelStatus: ModelStatus = .notDownloaded
    ) -> AgentInputStatus? {
        AgentInputStatus.derive(
            error: error,
            voiceErrorMessage: voiceErrorMessage,
            isEngineLoading: isEngineLoading,
            loadingStatus: loadingStatus,
            isModelLoaded: isModelLoaded,
            selectedModelDisplayName: selectedModelDisplayName,
            selectedModelStatus: selectedModelStatus
        )
    }

    // MARK: - Priority order

    @Test func errorOutranksEverything() {
        let status = derive(
            error: "boom", voiceErrorMessage: "mic", isEngineLoading: true,
            selectedModelStatus: .downloading(progress: 0.5))
        #expect(status == .error("boom"))
    }

    @Test func voiceErrorOutranksLoadingAndDownloads() {
        let status = derive(
            voiceErrorMessage: "mic", isEngineLoading: true,
            selectedModelStatus: .downloading(progress: 0.5))
        #expect(status == .voiceError("mic"))
    }

    @Test func loadingUsesTheEngineStatusLineWithAFallback() {
        #expect(
            derive(isEngineLoading: true, loadingStatus: "Loading weights")
                == .loading("Loading weights"))
        #expect(derive(isEngineLoading: true) == .loading("Loading model\u{2026}"))
    }

    // MARK: - Download states

    @Test func activeDownloadShowsCalmProgressNotAWarning() {
        let status = derive(
            selectedModelDisplayName: "Qwen3.6-27B PARO",
            selectedModelStatus: .downloading(progress: 0.68))
        #expect(
            status
                == .downloadingModel(displayName: "Qwen3.6-27B PARO", progress: 0.68))
    }

    @Test func missingModelWithNoActiveDownloadStaysTheWarning() {
        #expect(derive(selectedModelStatus: .notDownloaded) == .notDownloaded)
        #expect(
            derive(selectedModelStatus: .error("network down")) == .notDownloaded,
            "a stalled download is a missing model, not an in-flight one")
    }

    @Test func downloadedButNotYetLoadedShowsNothing() {
        // Auto-load owns this gap; the strip stays quiet.
        #expect(derive(selectedModelStatus: .downloaded(sizeOnDisk: 1)) == nil)
    }

    @Test func loadedModelShowsNothing() {
        #expect(
            derive(isModelLoaded: true, selectedModelStatus: .downloaded(sizeOnDisk: 1)) == nil)
    }
}
