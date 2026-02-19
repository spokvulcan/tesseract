//
//  AgentNotchState.swift
//  tesseract
//

import Foundation

/// Phases of the agent voice interaction shown in the Dynamic Island overlay.
enum AgentNotchPhase: Equatable {
    case hidden
    case listening(audioLevel: Float)
    case transcribing(preview: String)
    case thinking
    case toolCall(name: String)
    case responding(text: String)
    case complete(text: String)
    case error(String)

    var isVisible: Bool {
        if case .hidden = self { return false }
        return true
    }

    /// Display width for the island pill at this phase.
    var displayWidth: CGFloat {
        switch self {
        case .hidden: return 200
        case .listening: return 220
        case .transcribing: return 220
        case .thinking: return 200
        case .toolCall: return 250
        case .responding: return 340
        case .complete: return 340
        case .error: return 280
        }
    }

    /// Display height for the content area (below menu bar).
    var contentHeight: CGFloat {
        switch self {
        case .hidden: return 0
        case .listening, .transcribing, .thinking, .toolCall: return 40
        case .responding: return 72
        case .complete: return 56
        case .error: return 44
        }
    }

    /// Human-readable tool name for display.
    static func toolDisplayName(_ rawName: String) -> String {
        switch rawName {
        case "get_current_time": return "Checking time"
        case "remember": return "Remembering"
        case "recall": return "Recalling"
        case "create_goal": return "Creating goal"
        case "list_goals": return "Listing goals"
        case "update_goal": return "Updating goal"
        case "create_task": return "Creating task"
        case "list_tasks": return "Listing tasks"
        case "complete_task": return "Completing task"
        case "create_habit": return "Creating habit"
        case "log_habit": return "Logging habit"
        case "habit_status": return "Checking habits"
        case "mood_log": return "Logging mood"
        case "list_moods": return "Listing moods"
        case "set_reminder": return "Setting reminder"
        default:
            // Convert snake_case to title case
            return rawName.split(separator: "_").map { $0.capitalized }.joined(separator: " ")
        }
    }
}

/// Observable state driving the agent Dynamic Island overlay.
@Observable
@MainActor
final class AgentNotchState {
    var phase: AgentNotchPhase = .hidden
    var shouldDismiss = false
}
