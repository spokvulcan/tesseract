//
//  CompanionAttentionGate.swift
//  tesseract
//
//  The owner's attention is the one resource the loop must never contend
//  for: while he is using the app — a voice session, a running generation,
//  TTS he is listening to, or simply typing with Tesseract frontmost — no
//  background turn starts, and none starts until the app has been quiet for
//  a full window after. Deferred wakes stay booked; due-ness holds; the next
//  tick fires them the moment the gate opens. The 21:35 evening journal that
//  ran headless *during* his live voice session is the incident this exists
//  to make impossible.
//
//  A lift is the explicit exception: when the owner himself summons a turn
//  (Book Test Wake, a notification reply), being engaged is the point.
//

import Foundation

@MainActor
final class CompanionAttentionGate {

    /// How long the app must be untouched after owner activity before a
    /// background turn may start (the owner's call: "wait at least two
    /// minutes to unblock the queue").
    static let quietWindow: TimeInterval = 120
    /// How long an explicit summons holds the gate open — enough for the
    /// next tick to grant the turn, short enough to never linger.
    static let liftWindow: TimeInterval = 150

    /// Live owner activity: voice session, interactive generation, speech
    /// playback, dictation capture, or Tesseract frontmost with recent input.
    private let isOwnerEngaged: () -> Bool
    /// The GPU is spoken for (someone else's generation, sleep consolidating)
    /// — yield the slot, but no quiet window: the machine isn't the owner.
    private let isMachineBusy: () -> Bool

    /// Briefing evidence: the last moment the gate saw the owner engaged.
    private(set) var lastOwnerEngagedAt: Date?
    private var liftedUntil: Date?

    init(
        isOwnerEngaged: @escaping () -> Bool,
        isMachineBusy: @escaping () -> Bool
    ) {
        self.isOwnerEngaged = isOwnerEngaged
        self.isMachineBusy = isMachineBusy
    }

    /// The evaluator's one question, asked before granting any turn class.
    func mayRunTurn(now: Date = Date()) -> Bool {
        if isOwnerEngaged() { lastOwnerEngagedAt = now }
        if let lifted = liftedUntil, now < lifted { return true }
        if let last = lastOwnerEngagedAt,
            now.timeIntervalSince(last) < Self.quietWindow
        {
            return false
        }
        return !isMachineBusy()
    }

    /// The owner explicitly asked for a turn — hold the gate open for him.
    func lift(now: Date = Date()) {
        liftedUntil = now.addingTimeInterval(Self.liftWindow)
    }
}
