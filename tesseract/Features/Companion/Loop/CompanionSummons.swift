//
//  CompanionSummons.swift
//  tesseract
//
//  The spoken-summons choreography (ADR-0040 §10, #328): speak the line,
//  raise the picked overlay concept when the wearing toggle routes summonses
//  through it, route the owner's answer, and keep §11's first guarantee — an
//  unanswered summons falls back to a notification banner; no delivery
//  evaporates silently. Choreography only: an answered summons is a
//  **Reaction**, reported through the loop's one reaction door so the fold
//  reducer decides every write (#391) — this type never touches wake state
//  or mints dialogues itself. Conduct lives here so the composition root
//  stays pure wiring; the container hands this type doors, never behavior.
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
    /// The loop's one **Reaction** door (#391): an engage or dismiss is
    /// reported here — the fold reducer stamps heard, upgrades the wake,
    /// and mints the summoned dialogue chat (ADR-0046 #372, ADR-0052);
    /// awaited so choreography can follow the decided writes.
    private let reportReaction: (CompanionPingReaction) async -> Void
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
        reportReaction: @escaping (CompanionPingReaction) async -> Void,
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
        self.reportReaction = reportReaction
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
            await conclude(
                outcome: outcome, line: line,
                wakeID: wakeID, conversationID: conversationID)
        }
    }

    /// Route the overlay's answer. An engage or dismiss is a **Reaction** —
    /// reported through the loop's one door with the snapshotted
    /// correlation, so the wake is stamped heard (and upgraded on engage)
    /// exactly as a banner reaction would be (#391); the voice session is
    /// entered only after the report so the engage's minted dialogue is the
    /// one it rides. An unanswered summons reached no one — not a Reaction —
    /// and keeps §11's fallback-banner guarantee instead.
    func conclude(
        outcome: CompanionBeatSummonsOutcome,
        line: String,
        wakeID: UUID?,
        conversationID: UUID?
    ) async {
        switch outcome {
        case .engaged:
            await reportReaction(
                CompanionPingReaction(
                    outcome: .engaged, note: "overlay",
                    wakeID: wakeID, conversationID: conversationID, line: line))
            enterVoiceSession("summons-engage")
        case .dismissed:
            await reportReaction(
                CompanionPingReaction(
                    outcome: .dismissed, note: "overlay",
                    wakeID: wakeID, conversationID: conversationID, line: line))
        case .unanswered:
            recorder.record(
                "reaction.unheard", wakeID: wakeID, conversationID: conversationID,
                note: "overlay summons unanswered")
            // §11 guarantee 1: the missed overlay leaves the same line as
            // a banner in Notification Center — a visible artifact, and a
            // late reaction still routes through its correlation.
            await postFallbackBanner(line, wakeID, conversationID)
        }
    }
}
