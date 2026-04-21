//
//  ServerStatusFormatting.swift
//  tesseract
//

import AppKit
import Foundation
import MLXLMCommon
import SwiftUI

enum ServerRunState: Equatable {
    case stopped
    case starting
    case running
    case failed(String)

    init(enabled: Bool, isRunning: Bool, isStarting: Bool, lastStartError: String?) {
        switch (enabled, isRunning, isStarting, lastStartError) {
        case (false, _, _, _): self = .stopped
        case (true, true, _, _): self = .running
        case (true, false, true, _): self = .starting
        case (true, false, false, let message?): self = .failed(message)
        case (true, false, false, nil): self = .stopped
        }
    }

    var displayLabel: String {
        switch self {
        case .stopped: return "Stopped"
        case .starting: return "Starting…"
        case .running: return "Running"
        case .failed: return "Failed"
        }
    }

    var failureMessage: String? {
        if case .failed(let message) = self { return message }
        return nil
    }

    var dotColor: Color {
        switch self {
        case .running: return .green
        case .starting: return .orange
        case .stopped: return .secondary
        case .failed: return .red
        }
    }
}

func serverEndpointURL(port: Int) -> String {
    "http://127.0.0.1:\(port)"
}

@MainActor
func copyServerEndpointToPasteboard(port: Int) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(serverEndpointURL(port: port), forType: .string)
}

@MainActor
func triAttentionModeDescription(for arbiter: InferenceArbiter) -> String {
    guard let state = arbiter.loadedLLMState else {
        return "Pending — no model loaded yet"
    }
    if let reason = state.triAttentionFallbackReason {
        return "Dense (fallback: \(reason.displayLabel))"
    }
    return state.effectiveTriAttention.enabled ? "TriAttention" : "Dense"
}
