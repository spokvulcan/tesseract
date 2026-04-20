//
//  NavigationItem.swift
//  tesseract
//

import SwiftUI

enum NavigationItem: String, Equatable, Hashable, Identifiable, CaseIterable {
    case dictation
    case speech
    case agent
    case scheduled
    case image
    case zimage
    case general
    case serverDashboard
    case serverConfiguration
    case serverPromptCache
    case model
    case recording

    var id: String { rawValue }

    static let mainPages: [NavigationItem] = [.agent, .scheduled, .dictation, .speech]
    static let settingsPages: [NavigationItem] = [.general, .recording, .model]
    static let serverPages: [NavigationItem] = [.serverDashboard, .serverConfiguration, .serverPromptCache]

    var name: LocalizedStringResource {
        switch self {
        case .dictation: "Dictation"
        case .speech: "Speech"
        case .agent: "Agent"
        case .scheduled: "Scheduled"
        case .image: "Image"
        case .zimage: "Z-Image"
        case .general: "General"
        case .serverDashboard: "Dashboard"
        case .serverConfiguration: "Configuration"
        case .serverPromptCache: "Prompt Cache"
        case .model: "Models"
        case .recording: "Preferences"
        }
    }

    var symbolName: String {
        switch self {
        case .dictation: "mic.fill"
        case .speech: "speaker.wave.3.fill"
        case .agent: "brain.head.profile"
        case .scheduled: "calendar.badge.clock"
        case .image: "photo.fill"
        case .zimage: "photo.artframe"
        case .general: "gear"
        case .serverDashboard: "gauge"
        case .serverConfiguration: "server.rack"
        case .serverPromptCache: "tray.2.fill"
        case .model: "brain"
        case .recording: "slider.horizontal.3"
        }
    }

    @MainActor @ViewBuilder
    var destinationView: some View {
        switch self {
        case .dictation:
            DictationContentView()
        case .speech:
            SpeechContentView()
        case .agent:
            AgentContentView()
        case .scheduled:
            ScheduledTasksView()
        case .image:
            ImageGenContentView()
        case .zimage:
            ZImageGenContentView()
        case .general:
            GeneralSettingsSection()
        case .serverDashboard:
            ServerDashboardView()
        case .serverConfiguration:
            ServerConfigurationView()
        case .serverPromptCache:
            ServerPromptCacheView()
        case .model:
            ModelsPageView()
        case .recording:
            RecordingSettingsSection()
        }
    }
}
