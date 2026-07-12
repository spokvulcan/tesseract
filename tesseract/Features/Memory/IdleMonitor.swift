//
//  IdleMonitor.swift
//  tesseract
//
//  "Is he away?" (ADR-0035 §7 — the owner's call: idle-opportunistic sleep.)
//
//  Nothing in the app knew this before. Dictation knows about hotkeys, the
//  agent knows about turns, but nobody was watching for the absence of a person
//  — which is the only thing consolidation is allowed to run in.
//
//  Two signals, because one is not enough:
//
//    - `CGEventSource.secondsSinceLastEventType` — HID idle. Cheap, polled, and
//      the one that catches "he walked away mid-sentence".
//    - Screen lock and system sleep/wake notifications. Instant, and they catch
//      the case the idle timer cannot: a locked screen is *definitely* away,
//      immediately, with no three-minute wait.
//
//  The return signal is the one that must never be slow, so it does not wait for
//  a poll: any HID activity at all, or an unlock, or a wake, fires `onReturn`
//  synchronously. Sleep's whole contract with the owner is that coming back to
//  the machine costs him nothing.
//

import AppKit
import Foundation

@MainActor
@Observable
final class IdleMonitor {

    /// How long with no keyboard or mouse before the machine counts as idle.
    ///
    /// Three minutes is a compromise between two failure modes: too short and
    /// consolidation starts while he is reading the screen; too long and a
    /// coffee break is never long enough to consolidate anything. A locked
    /// screen bypasses it entirely.
    static let idleThreshold: TimeInterval = 180

    private(set) var isIdle = false
    private(set) var isScreenLocked = false

    /// Fired when the machine becomes idle enough to work in.
    var onIdle: (@MainActor () -> Void)?
    /// Fired the instant the owner is back. **Nothing here may be slow.**
    var onReturn: (@MainActor () -> Void)?

    private var pollTask: Task<Void, Never>?
    private var observers: [any NSObjectProtocol] = []

    /// The idle poll interval. Fifteen seconds: this decides how late sleep can
    /// *start*, never how late it stops — waking is event-driven.
    private let pollInterval: Duration

    /// Injected for tests, which cannot move the real HID clock.
    private let secondsSinceLastEvent: @MainActor () -> TimeInterval

    init(
        pollInterval: Duration = .seconds(15),
        secondsSinceLastEvent: @escaping @MainActor () -> TimeInterval = IdleMonitor.hidIdleSeconds
    ) {
        self.pollInterval = pollInterval
        self.secondsSinceLastEvent = secondsSinceLastEvent
    }

    func start() {
        guard pollTask == nil else { return }

        let distributed = DistributedNotificationCenter.default()
        observers.append(
            distributed.addObserver(
                forName: .init("com.apple.screenIsLocked"), object: nil, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.screenLocked() }
            })
        observers.append(
            distributed.addObserver(
                forName: .init("com.apple.screenIsUnlocked"), object: nil, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.ownerReturned() }
            })

        let workspace = NSWorkspace.shared.notificationCenter
        observers.append(
            workspace.addObserver(
                forName: NSWorkspace.willSleepNotification, object: nil, queue: .main
            ) { [weak self] _ in
                // The machine is going to sleep. Whatever we were doing, stop —
                // the GPU is about to go away underneath us.
                MainActor.assumeIsolated { self?.ownerReturned() }
            })
        observers.append(
            workspace.addObserver(
                forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.ownerReturned() }
            })

        let interval = pollInterval
        pollTask = Task { [weak self] in
            // `guard let self` rather than `self?.` so the loop ends when the
            // monitor does — there is no `deinit` to cancel it from (a nonisolated
            // `deinit` cannot touch main-actor state).
            while !Task.isCancelled {
                guard let self else { return }
                self.poll()
                try? await Task.sleep(for: interval)
            }
        }
        Log.memory.info("Idle monitor started")
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
        for observer in observers {
            DistributedNotificationCenter.default().removeObserver(observer)
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
        }
        observers = []
        isIdle = false
    }

    /// Exposed for the poll's test seam and for a caller that wants to force a
    /// re-evaluation (the sleep scheduler does, after a run ends).
    func poll() {
        // A locked screen is idle regardless of what the HID clock says — and it
        // says zero right after the lock keystroke.
        let idle = isScreenLocked || secondsSinceLastEvent() >= Self.idleThreshold
        if idle {
            guard !isIdle else { return }
            isIdle = true
            Log.memory.info("Machine idle — consolidation may run")
            onIdle?()
        } else {
            // `ownerReturned` owns the flag and the idempotence — clearing it
            // here first would make its own guard swallow the notification.
            ownerReturned()
        }
    }

    private func screenLocked() {
        isScreenLocked = true
        guard !isIdle else { return }
        isIdle = true
        Log.memory.info("Screen locked — consolidation may run")
        onIdle?()
    }

    /// The one path back. Idempotent: the poll, the unlock, and the wake can all
    /// fire for the same return, and only the first does anything.
    private func ownerReturned() {
        isScreenLocked = false
        guard isIdle else { return }
        isIdle = false
        Log.memory.info("Owner is back — yielding")
        onReturn?()
    }

    /// Seconds since the last keyboard, mouse, or trackpad event, across the
    /// whole login session — not just this app.
    ///
    /// `~0` is `kCGAnyInputEventType`: not a named Swift case, but constructible
    /// (verified on macOS 26). If that ever stops being true, this reports zero
    /// — "the owner is here" — which is the safe direction to be wrong in: sleep
    /// simply never runs, rather than running over his shoulder.
    static func hidIdleSeconds() -> TimeInterval {
        guard let anyInput = CGEventType(rawValue: ~0) else { return 0 }
        return CGEventSource.secondsSinceLastEventType(
            .combinedSessionState, eventType: anyInput)
    }
}
