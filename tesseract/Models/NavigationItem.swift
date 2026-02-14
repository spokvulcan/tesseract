//
//  NavigationItem.swift
//  tesseract
//

import SwiftUI

enum NavigationItem: String, Equatable, Hashable, Identifiable, CaseIterable {
    case dictation
    case speech
    case image
    case general
    case model
    case recording

    var id: String { rawValue }

    static let mainPages: [NavigationItem] = [.dictation, .speech, .image]
    static let settingsPages: [NavigationItem] = [.general, .model, .recording]

    var name: LocalizedStringResource {
        switch self {
        case .dictation: "Dictation"
        case .speech: "Speech"
        case .image: "Image"
        case .general: "General"
        case .model: "Models"
        case .recording: "Recording"
        }
    }

    var symbolName: String {
        switch self {
        case .dictation: "mic.fill"
        case .speech: "speaker.wave.3.fill"
        case .image: "photo.fill"
        case .general: "gear"
        case .model: "brain"
        case .recording: "waveform"
        }
    }

    @MainActor @ViewBuilder
    func viewForPage(
        coordinator: DictationCoordinator,
        transcriptionEngine: TranscriptionEngine,
        history: TranscriptionHistory,
        permissionsManager: PermissionsManager,
        audioCapture: AudioCaptureEngine,
        speechCoordinator: SpeechCoordinator,
        speechEngine: SpeechEngine,
        imageGenEngine: ImageGenEngine
    ) -> some View {
        switch self {
        case .dictation:
            DictationContentView(
                coordinator: coordinator,
                transcriptionEngine: transcriptionEngine,
                history: history,
                permissionsManager: permissionsManager,
                audioCapture: audioCapture
            )
        case .speech:
            SpeechContentView(
                speechCoordinator: speechCoordinator,
                speechEngine: speechEngine
            )
        case .image:
            ImageGenContentView(
                imageGenEngine: imageGenEngine
            )
        case .general:
            GeneralSettingsSection()
        case .model:
            ModelsPageView()
        case .recording:
            RecordingSettingsSection()
        }
    }
}
