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
        case "time_get": return "Checking time"
        case "memory_save": return "Saving memory"
        case "memory_search": return "Searching memory"
        case "goal_create": return "Creating goal"
        case "goal_list": return "Listing goals"
        case "goal_update": return "Updating goal"
        case "task_create": return "Creating task"
        case "task_list": return "Listing tasks"
        case "task_complete": return "Completing task"
        case "habit_create": return "Creating habit"
        case "habit_log": return "Logging habit"
        case "habit_status": return "Checking habits"
        case "mood_log": return "Logging mood"
        case "mood_list": return "Listing moods"
        case "reminder_set": return "Setting reminder"
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
