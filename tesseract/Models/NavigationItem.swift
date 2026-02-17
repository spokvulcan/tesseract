//
//  NavigationItem.swift
//  tesseract
//

import SwiftUI

enum NavigationItem: String, Equatable, Hashable, Identifiable, CaseIterable {
    case dictation
    case speech
    case agent
    case image
    case zimage
    case general
    case model
    case recording

    var id: String { rawValue }

    static let mainPages: [NavigationItem] = [.dictation, .speech, .agent]
    static let settingsPages: [NavigationItem] = [.general, .model, .recording]

    var name: LocalizedStringResource {
        switch self {
        case .dictation: "Dictation"
        case .speech: "Speech"
        case .agent: "Agent"
        case .image: "Image"
        case .zimage: "Z-Image"
        case .general: "General"
        case .model: "Models"
        case .recording: "Recording"
        }
    }

    var symbolName: String {
        switch self {
        case .dictation: "mic.fill"
        case .speech: "speaker.wave.3.fill"
        case .agent: "brain.head.profile"
        case .image: "photo.fill"
        case .zimage: "photo.artframe"
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
        agentCoordinator: AgentCoordinator,
        agentEngine: AgentEngine,
        agentConversationStore: AgentConversationStore,
        imageGenEngine: ImageGenEngine,
        zimageGenEngine: ZImageGenEngine
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
        case .agent:
            AgentContentView(
                coordinator: agentCoordinator,
                agentEngine: agentEngine,
                conversationStore: agentConversationStore,
                transcriptionEngine: transcriptionEngine,
                audioCapture: audioCapture
            )
        case .image:
            ImageGenContentView(
                imageGenEngine: imageGenEngine
            )
        case .zimage:
            ZImageGenContentView(
                zimageGenEngine: zimageGenEngine
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
