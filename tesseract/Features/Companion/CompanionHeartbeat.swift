//
//  CompanionHeartbeat.swift
//  tesseract
//

import Foundation

// MARK: - CompanionHeartbeat

/// PROTOTYPE — the Companion walking skeleton (map #301, ticket #303).
///
/// A deliberately crude heartbeat the owner lives with while the Companion
/// architecture grillings run: three fixed-time daily beats, each posted as a
/// local notification (optionally spoken) that can be engaged, replied to
/// inline, or dismissed — never silently ignored without the log noticing.
/// Every event lands in the flight-recorder v0 JSONL (#326) via
/// `CompanionHeartbeatLog`, so lived data feeds the decision tickets.
///
/// An instrument the owner wears, not the product: absorbed or retired by the
/// map's exit PRDs. The real rhythm (wake-linked, presence-gated summonses —
/// ticket #302) is out of scope here by design; fixed times are the crudeness
/// that makes the instrument cheap to wear from week one.
///
/// Known crude gaps, accepted: an unanswered ping is forgotten across an app
/// restart (its `fired` line without an outcome still marks it ignored for
/// mining); stale banners linger in Notification Center as evidence.
final class CompanionHeartbeat {

    struct Beat: Sendable {
        let id: String
        let title: String
        let prompt: String
        let hour: Int
        let minute: Int
    }

    /// The crude first rhythm: fixed local times, no presence gating.
    static let beats: [Beat] = [
        Beat(
            id: "morning", title: "Morning",
            prompt: "Morning. What's the one hard thing today?",
            hour: 9, minute: 0),
        Beat(
            id: "midday", title: "Midday pulse",
            prompt: "Midday pulse — still on the morning's one thing?",
            hour: 13, minute: 30),
        Beat(
            id: "evening", title: "Evening",
            prompt: "Evening. How did the day actually go?",
            hour: 21, minute: 30),
    ]

    /// Wall-clock polling keeps the schedule honest across system sleep — a
    /// timer armed for 09:00 would otherwise drift past a lid-closed morning.
    static let tickInterval: Duration = .seconds(30)

    /// A beat woken up to this late still fires (a late summons on Mac-wake is
    /// anchor-flavored); later than this it is logged `missed`, not posted.
    static let staleAfterSeconds: TimeInterval = 45 * 60

    private let isEnabled: () -> Bool
    private let speaks: () -> Bool
    private let speak: (String) -> Void
    private let onEngage: () -> Void
    /// The #328 wearing instrument: when enabled, beats summon the voice
    /// overlay instead of posting a banner; the banner remains the fallback
    /// for an unanswered overlay, so no beat is ever silent.
    private let overlaySummonsEnabled: () -> Bool
    private let summonOverlay:
        (@MainActor (_ title: String, _ body: String) async -> CompanionBeatSummonsOutcome)?
    private let notifier: CompanionNotifier
    private let log: CompanionHeartbeatLog

    /// What this beat actually *says* — the memory system's job (ADR-0035; the
    /// acceptance bar of #302).
    ///
    /// The skeleton shipped with three hardcoded prompts, and "Morning. What's
    /// the one hard thing today?" is a thing a cron job says. What makes a
    /// companion a companion is that it knows which hard thing you said it was
    /// last week. So the body is composed, from memory, at fire time — and falls
    /// back to the hardcoded line whenever memory has nothing true to say, which
    /// is the only acceptable failure mode: generic is survivable, invented is
    /// not.
    private let composeBody: @MainActor (Beat) async -> String

    private var tickTask: Task<Void, Never>?
    private var next: (beat: Beat, at: Date)?
    /// The last ping still waiting for an outcome; the next fire expires it.
    private var outstanding: (pingID: UUID, beatID: String)?
    private var didRequestAuthorization = false

    init(
        isEnabled: @escaping () -> Bool,
        speaks: @escaping () -> Bool,
        speak: @escaping (String) -> Void,
        onEngage: @escaping () -> Void,
        overlaySummonsEnabled: @escaping () -> Bool = { false },
        summonOverlay: (
            @MainActor (_ title: String, _ body: String) async -> CompanionBeatSummonsOutcome
        )? = nil,
        notifier: CompanionNotifier = CompanionNotifier(),
        log: CompanionHeartbeatLog = CompanionHeartbeatLog(),
        composeBody: @escaping @MainActor (Beat) async -> String = { $0.prompt }
    ) {
        self.isEnabled = isEnabled
        self.speaks = speaks
        self.speak = speak
        self.onEngage = onEngage
        self.overlaySummonsEnabled = overlaySummonsEnabled
        self.summonOverlay = summonOverlay
        self.notifier = notifier
        self.log = log
        self.composeBody = composeBody
    }

    /// Arms the tick loop for the app's lifetime — a sleeping task and one
    /// date comparison per tick unless the experimental toggle is on.
    func start() {
        guard tickTask == nil else { return }
        notifier.onOutcome = { [weak self] pingID, beatID, outcome, note in
            self?.recordOutcome(pingID: pingID, beatID: beatID, outcome: outcome, note: note)
        }
        next = Self.nextBeat(after: Date())
        if isEnabled() { requestAuthorizationOnce() }
        tickTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: CompanionHeartbeat.tickInterval)
                self?.tick(now: Date())
            }
        }
        if let next {
            Log.companion.info(
                "Heartbeat armed — next beat '\(next.beat.id)' at \(next.at); enabled: \(isEnabled())"
            )
        }
    }

    /// The Settings toggle-on path: request notification authorization right
    /// when the owner opts in, not at some later beat.
    func activate() {
        requestAuthorizationOnce()
    }

    /// The Settings "Send Test Ping" lever — exercises the whole pipe
    /// (notification, actions, log, optional speech) on demand.
    func sendTestPing() {
        requestAuthorizationOnce()
        let beat = Beat(
            id: "test", title: "Test ping",
            prompt: "Test ping — engage, reply, or dismiss; each lands in the log.",
            hour: 0, minute: 0)
        fire(beat: beat, scheduledFor: Date(), now: Date(), trigger: "manual test from Settings")
    }

    // MARK: - Ticking

    private func tick(now: Date) {
        guard let due = next, now >= due.at else { return }
        next = Self.nextBeat(after: now)
        guard isEnabled() else { return }
        let lateSeconds = Int(now.timeIntervalSince(due.at))
        if TimeInterval(lateSeconds) > Self.staleAfterSeconds {
            log.append(
                event: "missed", beat: due.beat.id, scheduledFor: due.at,
                lateSeconds: lateSeconds, trigger: Self.trigger(for: due.beat))
            Log.companion.info(
                "Beat '\(due.beat.id)' missed (\(lateSeconds)s late — machine asleep?)")
            return
        }
        fire(beat: due.beat, scheduledFor: due.at, now: now, trigger: Self.trigger(for: due.beat))
    }

    private func fire(beat: Beat, scheduledFor: Date, now: Date, trigger: String) {
        expireOutstanding()
        let pingID = UUID()
        outstanding = (pingID, beat.id)
        log.append(
            event: "fired", beat: beat.id, ping: pingID, scheduledFor: scheduledFor,
            lateSeconds: Int(now.timeIntervalSince(scheduledFor)), trigger: trigger)
        Log.companion.info("Ping '\(beat.id)' fired (\(pingID.uuidString))")
        Task { [weak self] in
            guard let self else { return }
            // Composed before the post, and spoken as the *same words* — a banner
            // that says one thing while the voice says another is two companions,
            // not one.
            let body = await self.composeBody(beat)
            self.log.append(event: "composed", beat: beat.id, ping: pingID, note: body)
            if self.speaks() {
                self.speak(body)
                self.log.append(event: "spoken", beat: beat.id, ping: pingID)
            }
            if self.overlaySummonsEnabled(), let summonOverlay = self.summonOverlay {
                self.log.append(event: "overlaySummoned", beat: beat.id, ping: pingID)
                switch await summonOverlay(beat.title, body) {
                case .engaged:
                    self.recordOutcome(
                        pingID: pingID, beatID: beat.id, outcome: .engaged, note: nil)
                case .dismissed:
                    self.recordOutcome(
                        pingID: pingID, beatID: beat.id, outcome: .dismissed, note: nil)
                case .unanswered:
                    // Never a silent give-up: the unanswered overlay falls back
                    // to the banner, which still lands in Notification Center.
                    self.log.append(event: "overlayUnanswered", beat: beat.id, ping: pingID)
                    await self.notifier.post(
                        pingID: pingID, beatID: beat.id, title: beat.title, body: body)
                }
            } else {
                await self.notifier.post(
                    pingID: pingID, beatID: beat.id, title: beat.title, body: body)
            }
        }
    }

    private func recordOutcome(
        pingID: UUID, beatID: String, outcome: CompanionPingOutcome, note: String?
    ) {
        log.append(event: outcome.rawValue, beat: beatID, ping: pingID, note: note)
        Log.companion.info("Ping \(pingID.uuidString) outcome: \(outcome.rawValue)")
        if outstanding?.pingID == pingID { outstanding = nil }
        if outcome == .engaged { onEngage() }
    }

    private func expireOutstanding() {
        guard let outstanding else { return }
        log.append(event: "expired", beat: outstanding.beatID, ping: outstanding.pingID)
        self.outstanding = nil
    }

    private func requestAuthorizationOnce() {
        guard !didRequestAuthorization else { return }
        didRequestAuthorization = true
        Task { [notifier, log] in
            let granted = await notifier.activate()
            if !granted { log.append(event: "authDenied") }
        }
    }

    // MARK: - Schedule

    static func nextBeat(after date: Date) -> (beat: Beat, at: Date)? {
        let calendar = Calendar.current
        let candidates = beats.compactMap { beat -> (beat: Beat, at: Date)? in
            var components = DateComponents()
            components.hour = beat.hour
            components.minute = beat.minute
            guard
                let at = calendar.nextDate(
                    after: date, matching: components, matchingPolicy: .nextTime)
            else { return nil }
            return (beat, at)
        }
        return candidates.min { $0.at < $1.at }
    }

    private static func trigger(for beat: Beat) -> String {
        String(
            format: "fixed-time %02d:%02d daily (skeleton v0)", beat.hour, beat.minute)
    }
}
