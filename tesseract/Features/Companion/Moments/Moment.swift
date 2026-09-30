//
//  Moment.swift
//  tesseract
//
//  A Moment is one model call at a point in the day when there is something
//  to judge: the Morning Plan, a Breakpoint, a Triage, the Evening Wrap-up,
//  the Night Reflection. Each asks for a structured card as JSON; code
//  validates it, retries once, and falls back to a deterministic card built
//  from the snapshot, so the owner always gets the essentials.
//
//  Every moment runs at the owner's own reasoning effort. On the current
//  checkpoints effort is written into the first system block, so a
//  per-moment effort would give each moment its own prefix and re-read the
//  whole Day Thread every time; one effort keeps one cached root. Each
//  moment is bounded by its own output cap instead.
//

import Foundation

nonisolated enum MomentKind: String, Sendable, Equatable, Hashable, Codable, CaseIterable {
    case morningPlan
    case breakpoint
    case triage
    case eveningWrapUp
    case nightReflection

    /// The owner-facing name.
    var title: String {
        switch self {
        case .morningPlan: "Morning Plan"
        case .breakpoint: "Welcome back"
        case .triage: "Triage"
        case .eveningWrapUp: "Evening Wrap-up"
        case .nightReflection: "Night Reflection"
        }
    }

    /// The output cap (thinking included). A generation that hits it falls
    /// back to the deterministic card.
    var maxTokens: Int {
        switch self {
        case .morningPlan: 6_000
        case .breakpoint: 3_000
        case .triage: 2_500
        case .eveningWrapUp: 4_000
        case .nightReflection: 8_000
        }
    }

    /// Non-urgent moments wait for a cool Mac on power; the rest run now.
    var isDeferrable: Bool { self == .nightReflection || self == .triage }
}

/// Why a moment ran, for the trace and the request.
nonisolated enum MomentTrigger: String, Sendable, Equatable, Codable {
    case firstPresence
    case todayOpened
    case presenceReturned
    case meetingEnded
    case notifications
    case eveningTime
    case night
    case retry
    case ownerAsked
}

/// A moment the engine wants run.
nonisolated struct MomentRequest: Sendable, Equatable, Codable {
    var kind: MomentKind
    var trigger: MomentTrigger
    /// The request text appended to the Day Thread (the Now Tag rides the
    /// message itself).
    var text: String
    /// 0 for the first try, 1 for the one retry.
    var attempt: Int = 0
    /// Card-specific facts the engine needs back with the result (the away
    /// span of a Breakpoint, the notification ids a Triage covered).
    var context: MomentContext = .init()
}

nonisolated struct MomentContext: Sendable, Equatable, Codable {
    var awayFrom: Date?
    var awayUntil: Date?
    /// The notifications the request showed the model, in the order its
    /// short ids ("n1"…) number them.
    var notificationIDs: [String] = []
    /// The card this moment refines (a Breakpoint shows a code-built card at
    /// once; the model's version replaces it in place).
    var cardID: String?
}

/// What a moment's model call produced, fed back to the engine as a signal.
nonisolated enum MomentOutcome: Sendable, Equatable {
    /// The raw reply text (thinking removed) and the call's measurements.
    case reply(String, MomentMeasure)
    /// The call failed: no model, cancelled, an error. Carries the reason.
    case failed(String, MomentMeasure?)
}

nonisolated struct MomentMeasure: Sendable, Equatable, Codable {
    var promptTokens: Int
    var outputTokens: Int
    var prefillSeconds: Double
    var generateSeconds: Double
    var latencySeconds: Double
    var hitCap: Bool
    var modelID: String
}
