//
//  NavigationItem.swift
//  whisper-on-device
//

import SwiftUI

enum NavigationItem: String, Equatable, Hashable, Identifiable, CaseIterable {
    case dictation
    case general
    case audio
    case model
    case advanced

    var id: String { rawValue }

    static let mainPages: [NavigationItem] = [.dictation]
    static let settingsPages: [NavigationItem] = [.general, .audio, .advanced]

    var name: LocalizedStringResource {
        switch self {
        case .dictation: "Dictation"
        case .general: "General"
        case .audio: "Audio"
        case .model: "Model"
        case .advanced: "Advanced"
        }
    }

    var symbolName: String {
        switch self {
        case .dictation: "mic.fill"
        case .general: "gear"
        case .audio: "waveform"
        case .model: "brain"
        case .advanced: "slider.horizontal.3"
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
        case .audio:
            AudioSettingsSection()
        case .model:
            ModelSettingsSection()
        case .advanced:
            AdvancedSettingsSection()
        }
    }
}
