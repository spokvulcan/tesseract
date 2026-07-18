//
//  SensedObservationRecorder.swift
//  tesseract
//
//  The zero-dialog sensing tier (#308, #304): presence spans, frontmost-app
//  sessions, and power transitions, written straight to the observation
//  stream. No LLM between a sensor and the disk, ever — these rows are what
//  the situation briefing and the pulse's skip judgment read.
//
//  Deliberately coarse: an app session below one minute is noise, not a
//  session; a presence span is bounded by the idle threshold the IdleMonitor
//  already applies. The Companion toggle gates writing, not observing — when
//  off, nothing lands.
//

import AppKit
import Foundation
import IOKit.ps

@MainActor
final class SensedObservationRecorder {

    /// An app stint shorter than this never becomes a row (#308's >= 60 s).
    static let minimumAppSessionSeconds: TimeInterval = 60
    /// Power state is polled — transitions are rare and a minute's lag is fine.
    static let powerPollInterval: Duration = .seconds(60)

    private let store: MemoryStore
    private let isEnabled: () -> Bool

    /// Perception doors (ADR-0046, #368): the Event producers ride the same
    /// sensing pipeline instead of duplicating pollers and thresholds. Fired
    /// beside the observation write, under the same sustained/transition
    /// verdicts.
    var onPowerTransition: (@MainActor (_ onACPower: Bool) -> Void)?
    var onSustainedAppSession: (@MainActor (_ app: String, _ start: Date, _ end: Date) -> Void)?

    /// When the current presence span began; nil while idle/locked/asleep.
    private var presenceSpanStart: Date?
    /// The frontmost app and when it took the front.
    private var currentApp: (name: String, since: Date)?
    /// Last observed external-power verdict; transitions get a row.
    private(set) var isOnACPower: Bool = SensedObservationRecorder.readACPower()

    private var powerPollTask: Task<Void, Never>?
    private var workspaceObserver: NSObjectProtocol?

    init(store: MemoryStore, isEnabled: @escaping () -> Bool) {
        self.store = store
        self.isEnabled = isEnabled
    }

    func start() {
        guard workspaceObserver == nil else { return }
        presenceSpanStart = Date()
        if let app = NSWorkspace.shared.frontmostApplication?.localizedName {
            currentApp = (app, Date())
        }
        workspaceObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] note in
            let name =
                (note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication)?
                .localizedName
            MainActor.assumeIsolated { self?.appBecameFrontmost(name) }
        }
        powerPollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: SensedObservationRecorder.powerPollInterval)
                self?.pollPower()
            }
        }
        Log.companion.info("Sensed observations armed (presence, app sessions, power)")
    }

    func stop() {
        if let workspaceObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(workspaceObserver)
            self.workspaceObserver = nil
        }
        powerPollTask?.cancel()
        powerPollTask = nil
    }

    // MARK: - Presence (driven by IdleMonitor's composed callbacks)

    /// The IdleMonitor fired `onIdle` — the span really ended when input
    /// stopped, one idle-threshold ago.
    func ownerWentIdle() {
        let end = Date().addingTimeInterval(-IdleMonitor.idleThreshold)
        closeAppSession(at: end)
        guard let start = presenceSpanStart else { return }
        presenceSpanStart = nil
        writeSpan(kind: "presence-span", start: start, end: max(start, end), detail: nil)
    }

    func ownerReturned() {
        guard presenceSpanStart == nil else { return }
        presenceSpanStart = Date()
        if let app = NSWorkspace.shared.frontmostApplication?.localizedName {
            currentApp = (app, Date())
        }
    }

    // MARK: - App sessions

    /// Internal, not private: the one deterministic seam through the app-
    /// session pipeline — tests drive it with controlled dates to prove the
    /// sustained-vs-brief verdict without NSWorkspace.
    func appBecameFrontmost(_ name: String?, at now: Date = Date()) {
        closeAppSession(at: now)
        if let name { currentApp = (name, now) }
    }

    private func closeAppSession(at end: Date) {
        guard let app = currentApp else { return }
        currentApp = nil
        guard end.timeIntervalSince(app.since) >= Self.minimumAppSessionSeconds else { return }
        writeSpan(kind: "app-session", start: app.since, end: end, detail: app.name)
        onSustainedAppSession?(app.name, app.since, end)
    }

    // MARK: - Power

    private func pollPower() {
        let ac = Self.readACPower()
        guard ac != isOnACPower else { return }
        isOnACPower = ac
        append(
            TrackingObservation(
                domain: .work, kind: "power-transition", value: ac ? "ac" : "battery",
                source: .sensed))
        onPowerTransition?(ac)
    }

    private static func readACPower() -> Bool {
        // "AC Power" per IOPSKeys; an unplugged desktop reads AC too, which is
        // exactly right for the ambient-turn gate. No source info at all means
        // a desktop Mac: treat as AC.
        guard let type = IOPSGetProvidingPowerSourceType(nil)?.takeRetainedValue() else {
            return true
        }
        return (type as String) == kIOPMACPowerKey
    }

    // MARK: - Writing

    /// The one encoded shape of a sensed span. Internal, not private: the
    /// Event producer's `app-switch` payload (ADR-0046, #368) rides the same
    /// shape — one definition of the epoch/minutes math, two consumers.
    struct SpanValue: Codable {
        let start: Int
        let end: Int
        let minutes: Int
        let app: String?

        init(start: Date, end: Date, app: String?) {
            self.start = Int(start.timeIntervalSince1970)
            self.end = Int(end.timeIntervalSince1970)
            self.minutes = Int(end.timeIntervalSince(start) / 60)
            self.app = app
        }
    }

    private func writeSpan(kind: String, start: Date, end: Date, detail: String?) {
        let span = SpanValue(start: start, end: end, app: detail)
        let value =
            (try? JSONEncoder().encode(span)).flatMap { String(data: $0, encoding: .utf8) }
            ?? "{}"
        append(
            TrackingObservation(domain: .work, kind: kind, value: value, source: .sensed))
    }

    private func append(_ observation: TrackingObservation) {
        guard isEnabled() else { return }
        Task { [store] in
            do { try await store.appendObservation(observation) } catch {
                Log.companion.error(
                    "Sensed observation write failed: \(error.localizedDescription)")
            }
        }
    }
}
