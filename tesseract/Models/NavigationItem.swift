//
//  NavigationItem.swift
//  tesseract
//

import SwiftUI

/// The main-window sidebar's content-only pages (map #211): settings live in
/// the native Settings window, not here. Models files under Server — the
/// download manager is server-side infrastructure (#213) — but stays a
/// main-window task surface (live progress, storage, verify).
enum NavigationItem: String, Equatable, Hashable, Identifiable, CaseIterable {
    case dictation
    case speech
    case agent
    case serverDashboard
    case serverPromptCache
    case model

    var id: String { rawValue }

    static let mainPages: [NavigationItem] = [.agent, .dictation, .speech]
    static let serverPages: [NavigationItem] = [
        .serverDashboard, .serverPromptCache, .model,
    ]

    var name: LocalizedStringResource {
        switch self {
        case .dictation: "Dictation"
        case .speech: "Speech"
        case .agent: "Agent"
        case .serverDashboard: "Dashboard"
        case .serverPromptCache: "Prompt Cache"
        case .model: "Models"
        }
    }

    var symbolName: String {
        switch self {
        case .dictation: "mic.fill"
        case .speech: "speaker.wave.3.fill"
        case .agent: "brain.head.profile"
        case .serverDashboard: "gauge"
        case .serverPromptCache: "tray.2.fill"
        case .model: "brain"
        }
    }
}
