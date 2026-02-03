//
//  NavigationItem.swift
//  tesseract
//

import SwiftUI

enum NavigationItem: String, Equatable, Hashable, Identifiable, CaseIterable {
    case dictation
    case general
    case model
    case recording

    var id: String { rawValue }

    static let mainPages: [NavigationItem] = [.dictation]
    static let settingsPages: [NavigationItem] = [.general, .recording]

    var name: LocalizedStringResource {
        switch self {
        case .dictation: "Dictation"
        case .general: "General"
        case .model: "Model"
        case .recording: "Recording"
        }
    }

    var symbolName: String {
        switch self {
        case .dictation: "mic.fill"
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
        audioCapture: AudioCaptureEngine
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
        case .general:
            GeneralSettingsSection()
        case .model:
            ModelSettingsSection()
        case .recording:
            RecordingSettingsSection()
        }
    }
}
