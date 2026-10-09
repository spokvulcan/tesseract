//
//  IdleMonitor.swift
//  tesseract
//
//  Is the owner at the Mac? The Companion's presence signal: coming back after
//  being away is when a Breakpoint (or the day's first Morning Plan) is due.
//
//  Two signals, because one is not enough:
//
//    - `CGEventSource.secondsSinceLastEventType` — HID idle. Cheap, polled, and
//      the one that catches walking away mid-sentence.
//    - Screen lock and system sleep/wake notifications. Instant, and they catch
//      the case the idle timer cannot: a locked screen is *definitely* away,
//      immediately, with no three-minute wait.
//
//  The return signal is the one that must never be slow. Unlock and wake enqueue
//  `onReturn` directly onto MainActor; HID return has no event to subscribe to
//  without Input Monitoring entitlements, so it is caught by the poll — which
//  tightens to one second the moment the machine goes idle.
//

import AppKit
import Foundation

@MainActor
@Observable
final class IdleMonitor {

    /// How long with no keyboard or mouse before the owner counts as away. A
    /// Breakpoint needs a much longer absence (Settings, 10 minutes by
    /// default), measured from `awaySince`; a locked screen is away at once.
    static let idleThreshold: TimeInterval = 180

    private(set) var isIdle = false
    private(set) var isScreenLocked = false
    /// When the owner was last active before going idle: the last input, or
    /// the moment the screen locked. nil while present.
    private(set) var awaySince: Date?

    /// The one home of "he is with the machine": recent input AND unlocked.
    /// The Companion's evaluator and the Situation Briefing both read this —
    /// the rule must never be derived twice.
    var isOwnerPresent: Bool { !isIdle && !isScreenLocked }

    /// Fired when the machine becomes idle enough to work in.
    var onIdle: (@MainActor () -> Void)?
    /// Fired the instant the owner is back. **Nothing here may be slow.**
    var onReturn: (@MainActor () -> Void)?

    private var pollTask: Task<Void, Never>?
    private var observers: [any NSObjectProtocol] = []

    /// The poll interval while the owner is present. Fifteen seconds only
    /// decides how late sleep can *start* — cheap to be lazy about.
    private let pollInterval: Duration
    /// The poll interval while idle — this is the HID half of the return
    /// latency, so it is tight. One `secondsSinceLastEventType` call per second
    /// is negligible.
    private let returnPollInterval: Duration

    /// Injected so the AppKit wake bridge can be exercised without putting the
    /// test machine to sleep.
    private let workspaceNotificationCenter: NotificationCenter

    /// Injected for tests, which cannot move the real HID clock.
    private let secondsSinceLastEvent: @MainActor () -> TimeInterval

    init(
        pollInterval: Duration = .seconds(15),
        returnPollInterval: Duration = .seconds(1),
        workspaceNotificationCenter: NotificationCenter = NSWorkspace.shared.notificationCenter,
        secondsSinceLastEvent: @escaping @MainActor () -> TimeInterval = IdleMonitor.hidIdleSeconds
    ) {
        self.pollInterval = pollInterval
        self.returnPollInterval = returnPollInterval
        self.workspaceNotificationCenter = workspaceNotificationCenter
        self.secondsSinceLastEvent = secondsSinceLastEvent
    }

    func start() {
        guard pollTask == nil else { return }

        let distributed = DistributedNotificationCenter.default()
        observers.append(
            distributed.addObserver(
                forName: .init("com.apple.screenIsLocked"), object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, self.pollTask != nil else { return }
                    self.screenLocked()
                }
            })
        observers.append(
            distributed.addObserver(
                forName: .init("com.apple.screenIsUnlocked"), object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, self.pollTask != nil else { return }
                    self.ownerReturned()
                }
            })

        let workspace = workspaceNotificationCenter
        observers.append(
            workspace.addObserver(
                forName: NSWorkspace.willSleepNotification, object: nil, queue: .main
            ) { [weak self] _ in
                // The machine is going to sleep: the owner is away from here on.
                Task { @MainActor [weak self] in
                    guard let self, self.pollTask != nil else { return }
                    self.machineSleeps()
                }
            })
        observers.append(
            workspace.addObserver(
                forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, self.pollTask != nil else { return }
                    self.ownerReturned()
                }
            })

        pollTask = Task { [weak self] in
            // `guard let self` rather than `self?.` so the loop ends when the
            // monitor does — there is no `deinit` to cancel it from (a nonisolated
            // `deinit` cannot touch main-actor state).
            while !Task.isCancelled {
                guard let self else { return }
                self.poll()
                // Tight while idle: the poll is how a keyboard return gets
                // noticed, and a return should reach the Companion at once.
                try? await Task.sleep(
                    for: self.isIdle ? self.returnPollInterval : self.pollInterval)
            }
        }
        Log.companion.info("Idle monitor started")
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
        for observer in observers {
            DistributedNotificationCenter.default().removeObserver(observer)
            workspaceNotificationCenter.removeObserver(observer)
        }
        observers = []
        isIdle = false
    }

    /// Exposed as the poll's test seam.
    func poll() {
        // A locked screen is idle regardless of what the HID clock says — and it
        // says zero right after the lock keystroke.
        let idle = isScreenLocked || secondsSinceLastEvent() >= Self.idleThreshold
        if idle {
            guard !isIdle else { return }
            isIdle = true
            awaySince = Date().addingTimeInterval(-secondsSinceLastEvent())
            Log.companion.info("Owner away (idle)")
            onIdle?()
        } else {
            // `ownerReturned` owns the flag and the idempotence — clearing it
            // here first would make its own guard swallow the notification.
            ownerReturned()
        }
    }

    /// The Mac is going to sleep: from here the owner is away, if they weren't
    /// already (a lid closed mid-work), since their last input. Never a
    /// return: sleep read as one — a leftover of the retired memory
    /// consolidation, which stopped its GPU work that way — made a Breakpoint
    /// for nobody, stamped presence at the sleep, and cut the night short.
    func machineSleeps() {
        guard !isIdle else { return }
        isIdle = true
        awaySince = Date().addingTimeInterval(-secondsSinceLastEvent())
        Log.companion.info("Owner away (sleep)")
        onIdle?()
    }

    private func screenLocked() {
        isScreenLocked = true
        guard !isIdle else { return }
        isIdle = true
        awaySince = Date()
        Log.companion.info("Owner away (screen locked)")
        onIdle?()
    }

    /// The one path back. Idempotent: the poll, the unlock, and the wake can all
    /// fire for the same return, and only the first does anything.
    private func ownerReturned() {
        isScreenLocked = false
        guard isIdle else { return }
        isIdle = false
        defer { awaySince = nil }
        Log.companion.info("Owner is back")
        onReturn?()
    }

    /// Seconds since the last keyboard, mouse, or trackpad event, across the
    /// whole login session — not just this app.
    ///
    /// `~0` is `kCGAnyInputEventType`: not a named Swift case, but constructible
    /// (verified on macOS 26). If that ever stops being true, this reports zero
    /// — "the owner is here" — which is the safe direction to be wrong in: no
    /// Breakpoint fires, rather than one firing while they are at the Mac.
    static func hidIdleSeconds() -> TimeInterval {
        guard let anyInput = CGEventType(rawValue: ~0) else { return 0 }
        return CGEventSource.secondsSinceLastEventType(
            .combinedSessionState, eventType: anyInput)
    }
}
