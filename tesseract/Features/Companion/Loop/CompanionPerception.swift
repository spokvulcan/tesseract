//
//  CompanionPerception.swift
//  tesseract
//
//  The fold's perception substrate (ADR-0046, #368): the v1 Event producers.
//  Every digital input this type sees becomes exactly one Event in the queue
//  — day and system transitions observed directly (Mac-wake, calendar-day
//  rollover, this launch), power and sustained-app-session verdicts arriving
//  through the sensed-observation pipeline's doors, and day-start through the
//  loop's existing detection. Producers perceive and admit; they never decide
//  anything — the clock that grants turns over pending Events is #371's, and
//  until it lands Events accumulate on the record unconsumed.
//
//  Once-per-occasion perceptions (a day's start, a day's end) mint
//  deterministic ids, so a repeated signal collapses at admission instead of
//  needing dedupe state here.
//

import AppKit
import Foundation

@MainActor
final class CompanionPerception {

    private let store: MemoryStore
    private let recorder: CompanionFlightRecorder
    private let isEnabled: () -> Bool
    /// Test hosts bootstrap the full container against the real app container
    /// (issue #360), so without this gate every test run would admit phantom
    /// launch events into the owner's queue — the same reason durable
    /// telemetry diverts under tests (issue #159). Proven live: one focused
    /// run left five phantom `launch-catch-up` rows. Injected so the
    /// perception's own tests can open the door they are testing.
    private let isTestHost: Bool

    private var macWakeObserver: NSObjectProtocol?
    private var dayChangeObserver: NSObjectProtocol?
    private var started = false

    init(
        store: MemoryStore,
        recorder: CompanionFlightRecorder,
        isEnabled: @escaping () -> Bool,
        isTestHost: Bool = ProcessEnvironment.isRunningTests
    ) {
        self.store = store
        self.recorder = recorder
        self.isEnabled = isEnabled
        self.isTestHost = isTestHost
    }

    func start() {
        guard !started else { return }
        started = true

        // This launch is itself one input: everything before it went
        // unobserved, and the entity should know a gap sits behind.
        admit(
            CompanionEvent(
                kind: .launchCatchUp,
                content: "The app launched — anything earlier went unobserved."))

        macWakeObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.admit(CompanionEvent(kind: .macWake, content: "The Mac woke from sleep."))
            }
        }

        // The system's own day-rollover signal; delivered on wake when the
        // Mac slept through midnight.
        dayChangeObserver = NotificationCenter.default.addObserver(
            forName: .NSCalendarDayChanged, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.dayRolled(now: Date()) }
        }
    }

    func stop() {
        if let macWakeObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(macWakeObserver)
            self.macWakeObserver = nil
        }
        if let dayChangeObserver {
            NotificationCenter.default.removeObserver(dayChangeObserver)
            self.dayChangeObserver = nil
        }
        started = false
    }

    // MARK: - Doors (the signals other machinery already detects)

    /// The loop's day-start decision fired — first presence of the calendar
    /// day (the detector stays with the evaluator until #371 moves the clock).
    func dayStarted(at now: Date) {
        let day = TrackingDay.key(for: now)
        admit(
            CompanionEvent(
                id: CompanionEvent.deterministicID("day-start:\(day)"),
                kind: .dayStart,
                content: "His day started — first presence of \(day).",
                payload: Self.json(["day": day]),
                occurredAt: now))
    }

    /// The sensed-observation pipeline saw external power appear or vanish.
    func powerChanged(onACPower: Bool) {
        admit(
            CompanionEvent(
                kind: .powerChange,
                content: onACPower
                    ? "Power changed: on AC power." : "Power changed: on battery.",
                payload: Self.json(["power": onACPower ? "ac" : "battery"])))
    }

    /// The sensed-observation pipeline closed an app session that proved
    /// sustained (its threshold, its verdict) — brief flips never reach here.
    /// The payload is the pipeline's own span shape, not a second encoding.
    func sustainedAppSession(app: String, start: Date, end: Date) {
        let span = SensedObservationRecorder.SpanValue(start: start, end: end, app: app)
        admit(
            CompanionEvent(
                kind: .appSwitch,
                content: "Sustained app session: \(app), \(span.minutes) min.",
                payload: Self.json(span),
                occurredAt: end))
    }

    /// Internal, not private: the calendar-day rollover handler — tests drive
    /// it with controlled dates.
    func dayRolled(now: Date) {
        let day = TrackingDay.yesterdayKey(from: now)
        admit(
            CompanionEvent(
                id: CompanionEvent.deterministicID("day-end:\(day)"),
                kind: .dayEnd,
                content: "The calendar day \(day) ended.",
                payload: Self.json(["day": day]),
                occurredAt: now))
    }

    // MARK: - Admission

    /// One door to the queue: gate on the Companion toggle (and the test-host
    /// gate above), admit exactly once, and put fresh admissions on the
    /// record. A duplicate (collapsed deterministic id) is silent — the
    /// occasion was already perceived.
    private func admit(_ event: CompanionEvent) {
        guard isEnabled(), !isTestHost else { return }
        Task { [store, recorder] in
            do {
                guard try await store.admitEvent(event) else { return }
                recorder.record(
                    "event.admitted",
                    snapshot: ["kind": event.kind.rawValue, "eventID": event.id.uuidString],
                    note: event.content)
            } catch {
                Log.companion.error("Event admission failed: \(error.localizedDescription)")
            }
        }
    }

    private static func json(_ value: some Encodable) -> String? {
        (try? JSONEncoder().encode(value)).flatMap { String(data: $0, encoding: .utf8) }
    }
}
