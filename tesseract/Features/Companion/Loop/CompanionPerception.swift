//
//  CompanionPerception.swift
//  tesseract
//
//  The fold's perception substrate (ADR-0046, #368): the v1 Event producers.
//  Every digital input this type sees becomes exactly one Event in the queue
//  — day and system transitions observed directly (Mac-wake, calendar-day
//  rollover, this launch), power and sustained-app-session verdicts arriving
//  through the sensed-observation pipeline's doors, and day-start detected
//  here from the facts the loop's tick hands over (#371 moved the detector in
//  with the producers). Producers perceive and admit; they never decide
//  anything — the clock that grants turns over pending Events is #371's.
//
//  Once-per-occasion perceptions (a day's start, a day's end) mint
//  deterministic ids, so a repeated signal collapses at admission instead of
//  needing dedupe state here.
//

import AppKit
import Foundation

@MainActor
final class CompanionPerception {

    /// Day start needs the calendar day to have begun in earnest — a 1 a.m.
    /// tail counts as yesterday.
    static let dayStartEarliestHour = 4

    private let store: MemoryStore
    private let recorder: CompanionFlightRecorder
    private let isEnabled: () -> Bool
    /// Display names that are Tesseract's own, so its banners never become
    /// Events (#378's self-exclusion invariant — the AX tree has no bundle
    /// ids, only display names). Injected so tests can pin the set.
    private let selfDisplayNames: Set<String>
    /// Test hosts bootstrap the full container against the real app container
    /// (issue #360), so without this gate every test run would admit phantom
    /// launch events into the owner's queue — the same reason durable
    /// telemetry diverts under tests (issue #159). Proven live: one focused
    /// run left five phantom `launch-catch-up` rows. Injected so the
    /// perception's own tests can open the door they are testing.
    private let isTestHost: Bool

    private var macWakeObserver: NSObjectProtocol?
    private var dayChangeObserver: NSObjectProtocol?
    /// The Notification Hub's AX watcher (#378) — created on `start`, never in
    /// a test host. Nil until then.
    private var notificationWatcher: NotificationCenterWatcher?
    private var started = false
    /// The day key already admitted this process — a cheap edge on the tick's
    /// level signal, so a quiet afternoon isn't a stream of collapsed
    /// duplicates. The deterministic id below stays the once-only truth
    /// (relaunches re-attempt once and collapse at the store).
    private var admittedDayStartKey: String?

    init(
        store: MemoryStore,
        recorder: CompanionFlightRecorder,
        isEnabled: @escaping () -> Bool,
        selfDisplayNames: Set<String> = CompanionPerception.ownDisplayNames,
        isTestHost: Bool = ProcessEnvironment.isRunningTests
    ) {
        self.store = store
        self.recorder = recorder
        self.isEnabled = isEnabled
        self.selfDisplayNames = selfDisplayNames
        self.isTestHost = isTestHost
    }

    /// Tesseract's own banner display names — what its `UNUserNotificationCenter`
    /// posts appear as in the NC tree (`CFBundleDisplayName` is "Tesseract
    /// Agent"). The bare "Tesseract" is kept for robustness.
    static var ownDisplayNames: Set<String> {
        var names: Set<String> = ["Tesseract Agent", "Tesseract"]
        for key in ["CFBundleDisplayName", "CFBundleName"] {
            if let value = Bundle.main.object(forInfoDictionaryKey: key) as? String,
                !value.isEmpty
            {
                names.insert(value)
            }
        }
        return names
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

        // The Notification Hub's watcher (#378): AX observation of the live
        // NotificationCenter process. Never in a test host — there is no live
        // banner renderer to read, and the #360 container-sharing rule that
        // gates admission applies to perception too. Admission still rides the
        // Companion toggle through `admit`.
        if !isTestHost {
            let watcher = NotificationCenterWatcher(
                isEnabled: isEnabled,
                onNotification: { [weak self] captured in
                    self?.notificationArrived(captured)
                })
            watcher.start()
            notificationWatcher = watcher
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
        notificationWatcher?.stop()
        notificationWatcher = nil
        started = false
    }

    // MARK: - Doors (the signals other machinery already detects)

    /// First presence of the calendar day — the detector (#371): the loop's
    /// tick hands over the facts every beat; once the day has begun in
    /// earnest and he is actually there, the occasion becomes one Event.
    func dayStartIfDue(now: Date, ownerPresent: Bool) {
        let day = TrackingDay.key(for: now)
        guard admittedDayStartKey != day,
            ownerPresent,
            Calendar.current.component(.hour, from: now) >= Self.dayStartEarliestHour
        else { return }
        admittedDayStartKey = day
        admit(
            CompanionEvent(
                id: CompanionEvent.deterministicID("day-start:\(day)"),
                kind: .dayStart,
                content: "His day started — first presence of \(day).",
                payload: CompanionEvent.payloadJSON(["day": day]),
                occurredAt: now))
    }

    /// The sensed-observation pipeline saw external power appear or vanish.
    func powerChanged(onACPower: Bool) {
        admit(
            CompanionEvent(
                kind: .powerChange,
                content: onACPower
                    ? "Power changed: on AC power." : "Power changed: on battery.",
                payload: CompanionEvent.payloadJSON(["power": onACPower ? "ac" : "battery"])))
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
                payload: CompanionEvent.payloadJSON(span),
                occurredAt: end))
    }

    /// The Notification Hub's producer (#378): a banner the watcher read
    /// becomes exactly one Event through the same admission door as every other
    /// kind. Self-exclusion, the body cap, and the deterministic id (so a
    /// re-observed banner collapses at admission) live in the factory; nil
    /// there — a self-banner or an empty read — never admits.
    func notificationArrived(_ captured: CapturedNotification) {
        guard
            let event = CompanionEvent.notification(
                from: captured, selfDisplayNames: selfDisplayNames)
        else { return }
        admit(event)
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
                payload: CompanionEvent.payloadJSON(["day": day]),
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
                var snapshot = [
                    "kind": event.kind.rawValue, "eventID": event.id.uuidString,
                ]
                // The source app rides the record for notification and
                // app-switch kinds, so the Hub aggregator can pair a held
                // notification with a later switch to its app (#380). The
                // switch's *start* rides too (`at`), so the pairing keys off
                // when the owner moved to the app, not when he later left it —
                // the record's own stamp lands at the session's close.
                if let app = event.appHint { snapshot["app"] = app }
                if let at = event.spanStartHint { snapshot["at"] = String(at) }
                recorder.record("event.admitted", snapshot: snapshot, note: event.content)
            } catch {
                Log.companion.error("Event admission failed: \(error.localizedDescription)")
            }
        }
    }
}
