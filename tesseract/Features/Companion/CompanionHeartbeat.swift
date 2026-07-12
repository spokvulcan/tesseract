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
    private let notifier: CompanionNotifier
    private let log: CompanionHeartbeatLog

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
        notifier: CompanionNotifier = CompanionNotifier(),
        log: CompanionHeartbeatLog = CompanionHeartbeatLog()
    ) {
        self.isEnabled = isEnabled
        self.speaks = speaks
        self.speak = speak
        self.onEngage = onEngage
        self.notifier = notifier
        self.log = log
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
        Task { [notifier] in
            await notifier.post(
                pingID: pingID, beatID: beat.id, title: beat.title, body: beat.prompt)
        }
        if speaks() {
            speak(beat.prompt)
            log.append(event: "spoken", beat: beat.id, ping: pingID)
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
