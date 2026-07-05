//
//  AgentInputStatus.swift
//  tesseract
//

import Foundation

/// The agent input strip's status, derived as a pure function of the facts the
/// strip observes (PRD #171 lifted this out of the view so the download-aware
/// variant is testable). Priority: generation errors, then voice errors, then
/// engine loading, then the selected model's download state.
///
/// `downloadingModel` is the calm in-flight line ("on its way") shown while
/// the selected agent model is actively downloading — e.g. right after the
/// Onboarding Tour hands off mid-download; `notDownloaded` stays the warning
/// for a genuinely missing (or stalled: `.error`) model.
nonisolated enum AgentInputStatus: Equatable {
    case error(String)
    case voiceError(String)
    case loading(String)
    case downloadingModel(displayName: String, progress: Double)
    case notDownloaded

    static func derive(
        error: String?,
        voiceErrorMessage: String?,
        isEngineLoading: Bool,
        loadingStatus: String,
        isModelLoaded: Bool,
        selectedModelDisplayName: String,
        selectedModelStatus: ModelStatus
    ) -> AgentInputStatus? {
        if let error {
            return .error(error)
        }
        if let voiceErrorMessage {
            return .voiceError(voiceErrorMessage)
        }
        if isEngineLoading {
            return .loading(loadingStatus.isEmpty ? "Loading model\u{2026}" : loadingStatus)
        }
        guard !isModelLoaded else { return nil }

        switch selectedModelStatus {
        case .downloading(let progress):
            return .downloadingModel(
                displayName: selectedModelDisplayName, progress: progress)
        case .notDownloaded, .error:
            return .notDownloaded
        case .downloaded, .verifying:
            // On disk; auto-load owns the load gap and the strip stays quiet.
            return nil
        }
    }
}
