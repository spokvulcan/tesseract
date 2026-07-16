//
//  CompanionVoiceFeed.swift
//  tesseract
//

import Foundation
import Observation

/// PROTOTYPE — the Companion voice-overlay explorations (map #301, ticket #328).
///
/// The variant-agnostic surface of *mocked* voice-conversation signals every
/// overlay concept renders from — the Companion-voice mirror of dictation's
/// `DictationFeed`. One writer (the scripted `CompanionVoiceDemoDriver`), many
/// readers (the concept views). Nothing here talks to ASR, TTS, or the agent:
/// the ticket's question is choreography, so the conversation mechanics are
/// scripted theatre. Deleted or rewritten by the Voice-conversation-loop
/// ticket (#310) and the exit PRDs.
@Observable
@MainActor
final class CompanionVoiceFeed {

    /// The five faces of the choreography (#328): what the overlay is *doing*.
    enum State: Equatable {
        case idle
        /// The summons — annoying by design (anchor #302): `escalation` steps
        /// 0 → 2 while unanswered, and concepts render each step more insistent.
        case summoning(escalation: Int)
        case listening
        case thinking
        case speaking
    }

    /// One settled exchange in the running talk.
    struct Line: Equatable, Identifiable {
        enum Role: Equatable { case companion, owner }
        let id: Int
        let role: Role
        let text: String
    }

    private(set) var state: State = .idle
    /// The summons line (the knock's words) while `.summoning`.
    private(set) var summonsLine: String = ""
    /// Scene name for headers ("Morning", "Midday pulse", …).
    private(set) var sceneTitle: String = ""

    /// The Companion's current line while `.speaking`, revealed word by word.
    private(set) var spokenWords: [String] = []
    private(set) var revealedWordCount = 0

    /// The owner's in-progress words while `.listening` (mock ASR partial).
    private(set) var partial: String?

    /// Settled exchanges, oldest first. Concepts render as much or as little
    /// of this as their form wants.
    private(set) var transcript: [Line] = []

    /// The day contract once stamped (anchor #302's one hard step) — the beat
    /// concepts celebrate.
    private(set) var contract: String?

    /// Synthetic loudness + spectrum, same shapes as the dictation meter so
    /// concepts can reuse waveform idioms. Driven while listening/speaking.
    private(set) var level: Float = 0
    private(set) var spectrum: [Float] = MeterFrame.zeroBands

    /// The revealed prefix of the Companion's current line.
    var revealedText: String {
        spokenWords.prefix(revealedWordCount).joined(separator: " ")
    }

    var isActive: Bool { state != .idle }

    // MARK: - Driver side (the demo driver only; never concept views)

    private var nextLineID = 0

    func setState(_ newState: State) {
        state = newState
        if case .speaking = newState {} else { spokenWords = []; revealedWordCount = 0 }
        if case .listening = newState {} else { partial = nil }
    }

    func setScene(title: String) {
        sceneTitle = title
    }

    func setSummons(_ line: String) {
        summonsLine = line
    }

    func beginSpokenLine(_ text: String) {
        spokenWords = text.split(separator: " ").map(String.init)
        revealedWordCount = 0
    }

    func revealNextWord() {
        revealedWordCount = min(revealedWordCount + 1, spokenWords.count)
    }

    func setPartial(_ text: String?) {
        partial = text
    }

    func settle(role: Line.Role, text: String) {
        nextLineID += 1
        transcript.append(Line(id: nextLineID, role: role, text: text))
    }

    func stampContract(_ text: String?) {
        contract = text
    }

    func setMeter(level newLevel: Float, spectrum newSpectrum: [Float]) {
        level = newLevel
        spectrum = newSpectrum
    }

    /// Back to nothing — a fresh scene starts clean.
    func reset() {
        state = .idle
        summonsLine = ""
        sceneTitle = ""
        spokenWords = []
        revealedWordCount = 0
        partial = nil
        transcript = []
        contract = nil
        level = 0
        spectrum = MeterFrame.zeroBands
    }
}

/// The few sanctioned clicks a concept view can send back (mirror of
/// dictation's `OverlayActions`): concepts never see the driver.
@MainActor
struct CompanionVoiceActions {
    /// Answer the summons (also ends the demo's auto-engage wait).
    let engage: @MainActor () -> Void
    /// Interrupt the Companion mid-line — the barge-in the anchor demands.
    let bargeIn: @MainActor () -> Void
    /// One-action dismissal — recorded, never silent (anchor #302).
    let dismiss: @MainActor () -> Void
    /// Hand-off: pull this talk into the chat window.
    let openChat: @MainActor () -> Void

    static let none = CompanionVoiceActions(
        engage: {}, bargeIn: {}, dismiss: {}, openChat: {})
}
