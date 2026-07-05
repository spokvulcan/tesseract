//
//  OnboardingTryIt.swift
//  tesseract
//

import Foundation

/// Try-it availability (PRD #171, `CONTEXT.md` → Onboarding tour): whether a
/// Chapter's live demo slot activates. Pure, in the `ModelCatalog` style — the
/// caller supplies the permission and download facts; the tour never blocks on
/// an inactive Try-it, it shows the chapter's scripted animation instead.
nonisolated enum OnboardingTryIt {

    /// Dictating a real sentence inside the tour needs the microphone grant
    /// and the speech-to-text model on disk.
    static func dictationIsLive(
        microphone: PermissionState, speechModelDownloaded: Bool
    ) -> Bool {
        microphone == .granted && speechModelDownloaded
    }

    /// Speaking the voice chapter's headline needs only the voice model.
    static func voiceIsLive(voiceModelDownloaded: Bool) -> Bool {
        voiceModelDownloaded
    }
}
