//
//  CompanionSummons.swift
//  tesseract
//
//  The spoken-summons choreography (ADR-0040 §10, #328): speak the line,
//  raise the picked overlay concept when the wearing toggle routes summonses
//  through it, route the owner's answer, and keep §11's first guarantee — an
//  unanswered summons falls back to a notification banner; no delivery
//  evaporates silently. Conduct lives here so the composition root stays
//  pure wiring; the container hands this type doors, never behavior.
//

import Foundation

@MainActor
final class CompanionSummons {

    private let settings: SettingsManager
    private let presence: CompanionPresence
    private let recorder: CompanionFlightRecorder
    private let context: CompanionTurnContext
    /// Plain TTS with the normal notch overlay — the toggle-off spoken rung.
    private let speakPlain: (String) -> Void
    /// TTS without its own overlay — the summons surface is the one visual.
    private let speakUnderOverlay: (String) -> Void
    /// Raises the picked overlay concept and waits out the owner's answer.
    private let summonOverlay:
        (_ title: String, _ line: String) async -> CompanionBeatSummonsOutcome
    /// Engaging opens a summoned dialogue chat (ADR-0046 #372) — its own
    /// conversation, seeded with the summons line, owing a Report-Back —
    /// never Mission Control, which is the loop's record, not a chat.
    private let beginDialogue: (String) -> Void
    private let enterVoiceSession: (String) -> Void
    /// The loop's banner door for the §11 fallback, correlation passed in
    /// because the overlay's give-up can outlive the turn that raised it.
    private let postFallbackBanner:
        (_ line: String, _ wakeID: UUID?, _ conversationID: UUID?) async -> Void

    init(
        settings: SettingsManager,
        presence: CompanionPresence,
        recorder: CompanionFlightRecorder,
        context: CompanionTurnContext,
        speakPlain: @escaping (String) -> Void,
        speakUnderOverlay: @escaping (String) -> Void,
        summonOverlay:
            @escaping (_ title: String, _ line: String) async -> CompanionBeatSummonsOutcome,
        beginDialogue: @escaping (String) -> Void,
        enterVoiceSession: @escaping (String) -> Void,
        postFallbackBanner:
            @escaping (_ line: String, _ wakeID: UUID?, _ conversationID: UUID?) async -> Void
    ) {
        self.settings = settings
        self.presence = presence
        self.recorder = recorder
        self.context = context
        self.speakPlain = speakPlain
        self.speakUnderOverlay = speakUnderOverlay
        self.summonOverlay = summonOverlay
        self.beginDialogue = beginDialogue
        self.enterVoiceSession = enterVoiceSession
        self.postFallbackBanner = postFallbackBanner
    }

    /// The `speak` rung's delivery: plain TTS, unless the #328 wearing toggle
    /// routes spoken summonses through the picked overlay concept.
    func deliver(line: String) {
        guard settings.companionBeatsUseOverlay else {
            speakPlain(line)
            return
        }
        summon(line: line)
    }

    /// The overlay summons proper — also the `summon_overlay` rung (ADR-0040
    /// §10), which raises it regardless of the wearing toggle. Speaks the line
    /// audio-only under the overlay, stands the summons until the owner
    /// engages or dismisses, and never lets an unanswered one vanish:
    /// engaging opens a summoned dialogue chat and enters a live voice
    /// session (#310 §1, ADR-0046 #372); unanswered falls back to the
    /// notification banner.
    func summon(line: String) {
        // Snapshot the correlation now — the turn may end before the answer.
        let conversationID = context.conversationID
        let wakeID = context.wakeIDs.first
        speakUnderOverlay(line)
        Task { @MainActor in
            presence.beginSummons()
            defer { presence.endSummons() }
            let outcome = await summonOverlay("Jarvis", line)
            switch outcome {
            case .engaged:
                recorder.record(
                    "reaction.engaged", conversationID: conversationID, note: "overlay")
                // Dialogue out (ADR-0046 #372): the engagement opens its own
                // chat with the summons line as the entity's first words, and
                // the voice session rides it.
                beginDialogue(line)
                enterVoiceSession("summons-engage")
            case .dismissed:
                recorder.record(
                    "reaction.dismissed", conversationID: conversationID, note: "overlay")
            case .unanswered:
                recorder.record(
                    "reaction.unheard", conversationID: conversationID,
                    note: "overlay summons unanswered")
                // §11 guarantee 1: the missed overlay leaves the same line as
                // a banner in Notification Center — a visible artifact, and a
                // late reaction still routes through its correlation.
                await postFallbackBanner(line, wakeID, conversationID)
            }
        }
    }
}
