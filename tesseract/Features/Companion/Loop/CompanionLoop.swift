//
//  CompanionLoop.swift
//  tesseract
//
//  The harness's spine (ADR-0040): a ticking evaluator with event
//  accelerants. It computes due-ness and eligibility — never judgment — and
//  grants the entity turns: due wakes, transition wakes (day start, launch
//  catch-up), and ambient cognition when the eligibility gate passes. The
//  single correctness invariant lives here: a wake is consumed only by a
//  completed turn; anything less re-presents it.
//
//  Replaces the walking skeleton (`CompanionHeartbeat`, #303): the tick loop
//  and the never-silent delivery guarantees survive; the fixed schedule, the
//  staleness cutoff, and the forgotten-outstanding-ping gap die.
//

import AppKit
import Foundation

@MainActor
final class CompanionLoop {

    /// Wall-clock ticking keeps the schedule honest across system sleep — the
    /// skeleton proved a timer armed for 09:00 dies with a closed lid.
    static let tickInterval: Duration = .seconds(30)
    /// Overdue past this: the wake goes to a catch-up triage turn instead of
    /// firing as if the moment were now.
    static let catchUpGrace: TimeInterval = 30 * 60
    /// Ambient turns: at most one per this interval (ADR-0040 §7, revisable).
    static let ambientSpacing: TimeInterval = 30 * 60
    /// A failing turn retries this many times before the generic fallback.
    static let maxTurnAttempts = 2

    private let store: MemoryStore
    private let recorder: CompanionFlightRecorder
    private let runner: CompanionTurnRunner
    private let notifier: CompanionNotifier
    private let idleMonitor: IdleMonitor
    private let sensed: SensedObservationRecorder
    /// Read-only calendar for the briefing (stage G); access asked on enable.
    private let calendar: CompanionCalendarReader
    /// Reads the concrete arbiter's lease state (the protocol seam carries only
    /// the lease itself) — ambient turns yield to any in-flight generation.
    private let isGPUBusy: () -> Bool
    private let isEnabled: () -> Bool
    private let speak: @MainActor (String) -> Void
    private let openConversation: @MainActor (UUID) -> Void

    private var tickTask: Task<Void, Never>?
    private var evaluating = false
    private var didRequestAuthorization = false
    private var didRunLaunchRecovery = false
    /// Failed attempts per wake batch, keyed by the earliest wake id.
    private var turnAttempts: [UUID: Int] = [:]
    /// Posted notification pings → their correlation, for reaction routing.
    private var postedPings: [UUID: (wakeID: UUID?, conversationID: UUID?)] = [:]

    init(
        store: MemoryStore,
        recorder: CompanionFlightRecorder,
        runner: CompanionTurnRunner,
        notifier: CompanionNotifier,
        idleMonitor: IdleMonitor,
        sensed: SensedObservationRecorder,
        calendar: CompanionCalendarReader,
        isGPUBusy: @escaping () -> Bool,
        isEnabled: @escaping () -> Bool,
        speak: @escaping @MainActor (String) -> Void,
        openConversation: @escaping @MainActor (UUID) -> Void
    ) {
        self.store = store
        self.recorder = recorder
        self.runner = runner
        self.notifier = notifier
        self.idleMonitor = idleMonitor
        self.sensed = sensed
        self.calendar = calendar
        self.isGPUBusy = isGPUBusy
        self.isEnabled = isEnabled
        self.speak = speak
        self.openConversation = openConversation
    }

    // MARK: - Lifecycle

    func start() {
        guard tickTask == nil else { return }
        notifier.onOutcome = { [weak self] pingID, _, outcome, note in
            self?.handleReaction(pingID: pingID, outcome: outcome, note: note)
        }
        // Mac-wake is the one transition the tick can sleep through the moment
        // of — force an immediate evaluation instead of waiting half a minute.
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.evaluateSoon() }
        }
        tickTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: CompanionLoop.tickInterval)
                await self?.evaluate()
            }
        }
        Log.companion.info("Companion loop armed; enabled: \(self.isEnabled())")
    }

    /// The Settings toggle-on path: authorization right when the owner opts
    /// in, plus the v0 cutover and crash recovery.
    func activate() {
        requestAuthorizationOnce()
        evaluateSoon()
    }

    /// Event accelerant — presence transitions and Mac-wake call this so
    /// reactions feel instant instead of up-to-a-tick late.
    func evaluateSoon() {
        Task { await evaluate() }
    }

    /// The Settings test lever: a wake due now exercises the entire pipe —
    /// evaluator, turn, delivery, recorder.
    func bookTestWake() {
        Task {
            let wake = CompanionWake(
                content:
                    "Test wake from Settings — greet him briefly and confirm the loop works.",
                due: Date(), wakeClass: .followup)
            try? await store.upsertWake(wake)
            recorder.record("wake.booked", wakeID: wake.id, snapshot: ["class": "followup"])
            await evaluate()
        }
    }

    // MARK: - The evaluator

    /// Pure due-ness and eligibility — every judgment belongs to the turn this
    /// grants. Serialized: one evaluation, and at most one turn, at a time.
    private func evaluate() async {
        guard isEnabled(), !evaluating, !runner.isRunning else { return }
        evaluating = true
        defer { evaluating = false }

        requestAuthorizationOnce()
        runLaunchRecoveryOnce()

        let now = Date()
        let todayKey = TrackingDay.key(for: now)

        // 1. Due wakes — the entity's booked present.
        if let due = try? await store.dueWakes(asOf: now), !due.isEmpty {
            let overdue = due.filter { now.timeIntervalSince($0.due) > Self.catchUpGrace }
            let batch = overdue.isEmpty ? due : overdue
            let origin = overdue.isEmpty ? "wake" : "catchup"
            await runWakeTurn(batch, origin: origin, now: now)
            return
        }

        // 2. Day start — first presence of the calendar day (after 04:00, so a
        // 1 a.m. tail counts as yesterday).
        var dayState = (try? await store.loopDayState(todayKey)) ?? CompanionLoopDayState()
        let hour = Calendar.current.component(.hour, from: now)
        if dayState.dayStartedAt == nil, hour >= 4,
            !idleMonitor.isIdle, !idleMonitor.isScreenLocked
        {
            dayState.dayStartedAt = now
            try? await store.setLoopDayState(todayKey, dayState)
            recorder.record("loop.day-start", snapshot: ["at": CompanionWakeTime.format(now)])
            await runTransitionTurn(origin: "wake", template: Self.dayStartTemplate, now: now)
            return
        }

        // 3. Ambient cognition — eligibility, not judgment (ADR-0040 §7).
        if dayState.dayStartedAt != nil,
            sensed.isOnACPower,
            !isGPUBusy(),
            now.timeIntervalSince(dayState.lastAmbientAt ?? .distantPast) > Self.ambientSpacing
        {
            dayState.lastAmbientAt = now
            try? await store.setLoopDayState(todayKey, dayState)
            await runTransitionTurn(origin: "ambient", template: Self.ambientTemplate, now: now)
        }
    }

    // MARK: - Turns

    private func runWakeTurn(_ wakes: [CompanionWake], origin: String, now: Date) async {
        // Fired = presented to the entity. Recorded before the turn so a crash
        // between here and completion is visible as fired-but-unconsumed.
        for var wake in wakes {
            wake.state = .fired
            wake.firedAt = wake.firedAt ?? now
            try? await store.upsertWake(wake)
            recorder.record(
                "wake.fired", wakeID: wake.id,
                snapshot: [
                    "class": wake.wakeClass.rawValue,
                    "due": CompanionWakeTime.format(wake.due),
                    "present": idleMonitor.isIdle ? "idle" : "active",
                ],
                note: wake.content)
        }

        let template = origin == "catchup" ? Self.catchUpTemplate : Self.wakeTemplate
        let opening = await composeOpening(template: template, dueWakes: wakes, now: now)
        let outcome = await runner.run(
            origin: origin, opening: opening, wakeIDs: wakes.map(\.id))

        guard let outcome else {
            await handleTurnFailure(wakes)
            return
        }
        turnAttempts.removeAll()

        // Consumed only by this completed turn — unless the turn itself moved
        // the wake (a reschedule flips it back to booked; respect that).
        for wake in wakes {
            guard var current = try? await store.wake(id: wake.id), current.state == .fired
            else { continue }
            current.state = .delivered
            current.consumedAt = Date()
            try? await store.upsertWake(current)
            recorder.record(
                "wake.consumed", wakeID: wake.id, turnID: outcome.turnID,
                conversationID: outcome.conversationID)
        }
    }

    private func runTransitionTurn(origin: String, template: String, now: Date) async {
        _ = await runner.run(
            origin: origin,
            opening: await composeOpening(template: template, dueWakes: [], now: now))
    }

    private func composeOpening(
        template: String, dueWakes: [CompanionWake], now: Date
    ) async -> String {
        // The entity's own standing document rides first (ADR-0040 §12); the
        // unseeded fallback only exists for a turn racing first-run recovery.
        let instructions =
            (try? await store.currentInstructions())
            ?? CompanionInstructionsVersion(
                version: 0, text: CompanionInstructions.seed, author: "seed",
                note: nil, createdAt: now)
        let inputs = await CompanionBriefing.gather(
            store: store, idleMonitor: idleMonitor, sensed: sensed,
            dueWakes: dueWakes, recorder: recorder, calendar: calendar, now: now)
        return [
            CompanionInstructions.wrap(instructions),
            CompanionBriefing.render(inputs),
            template,
        ].joined(separator: "\n\n")
    }

    // MARK: - Failure semantics (ADR-0040 §13)

    private func handleTurnFailure(_ wakes: [CompanionWake]) async {
        guard let key = wakes.first?.id else { return }
        let attempts = (turnAttempts[key] ?? 0) + 1
        turnAttempts[key] = attempts

        if attempts < Self.maxTurnAttempts {
            // The wakes stay fired-but-unconsumed; recovery re-books them and
            // the next tick retries. Generic is survivable, invented is not.
            for var wake in wakes {
                wake.state = .booked
                try? await store.upsertWake(wake)
            }
            return
        }

        // Last resort: the brain is offline — deliver each wake's own
        // stateable line as a plain banner so never-silent-give-up holds.
        turnAttempts[key] = nil
        for var wake in wakes {
            let pingID = UUID()
            postedPings[pingID] = (wake.id, nil)
            await notifier.post(
                pingID: pingID, beatID: wake.wakeClass.rawValue, title: "Jarvis",
                body: wake.content)
            wake.state = .delivered
            wake.consumedAt = Date()
            try? await store.upsertWake(wake)
            recorder.record("delivery.fallback", wakeID: wake.id, note: wake.content)
        }
    }

    private func runLaunchRecoveryOnce() {
        guard !didRunLaunchRecovery else { return }
        didRunLaunchRecovery = true
        // v0 cutover (#326): the skeleton's lived JSONL joins the one corpus.
        let v0 = PathSandbox.defaultRoot
            .appendingPathComponent("companion", isDirectory: true)
            .appendingPathComponent("heartbeat.jsonl")
        recorder.importV0IfNeeded(from: v0)
        Task {
            // First run: install the instructions seed as version 1 — from
            // version 2 on, the document is the entity's own (ADR-0040 §12).
            if (try? await store.seedInstructionsIfNeeded(CompanionInstructions.seed)) == true {
                recorder.record("instructions.seeded", snapshot: ["version": "1"])
            }
            // Crash recovery: fired-but-unconsumed wakes re-present (the
            // invariant). Their half-finished conversations stay visible.
            guard let orphans = try? await store.unconsumedFiredWakes(), !orphans.isEmpty
            else { return }
            for var wake in orphans {
                wake.state = .booked
                try? await store.upsertWake(wake)
                recorder.record("wake.represented", wakeID: wake.id, note: wake.content)
            }
        }
    }

    // MARK: - Delivery plumbing (the tools' closures land here)

    /// The `notify` tool's door: posts under the turn's correlation so the
    /// owner's reaction routes back to the right wake and conversation.
    func deliverNotification(title: String, body: String) async {
        let pingID = UUID()
        let context = runner.context
        postedPings[pingID] = (context.wakeIDs.first, context.conversationID)
        await notifier.post(
            pingID: pingID, beatID: context.origin ?? "companion", title: title, body: body)
        recorder.record(
            "delivery.notification",
            wakeID: context.wakeIDs.first,
            turnID: context.turnID,
            conversationID: context.conversationID,
            snapshot: ["title": title],
            note: body)
    }

    /// The `speak` tool's door — the words are the verbatim snapshot (#326).
    func deliverSpoken(_ text: String) {
        let context = runner.context
        speak(text)
        recorder.record(
            "delivery.spoken",
            wakeID: context.wakeIDs.first,
            turnID: context.turnID,
            conversationID: context.conversationID,
            note: text)
    }

    // MARK: - Reactions

    private func handleReaction(pingID: UUID, outcome: CompanionPingOutcome, note: String?) {
        let correlation = postedPings.removeValue(forKey: pingID)
        recorder.record(
            "reaction.\(outcome.rawValue)",
            wakeID: correlation?.wakeID,
            conversationID: correlation?.conversationID,
            note: note)

        Task { [weak self] in
            guard let self else { return }
            switch outcome {
            case .engaged:
                if let wakeID = correlation?.wakeID,
                    var wake = try? await self.store.wake(id: wakeID)
                {
                    wake.state = .engaged
                    try? await self.store.upsertWake(wake)
                }
                if let conversationID = correlation?.conversationID {
                    self.openConversation(conversationID)
                }
            case .replied:
                // His words become a followup wake due now: the next turn sees
                // them with full situation context — one machinery, no side
                // channel.
                guard let text = note, !text.isEmpty else { return }
                let wake = CompanionWake(
                    content: "He replied to your notification: \"\(text)\" — respond.",
                    due: Date(), wakeClass: .followup,
                    conversationID: correlation?.conversationID)
                try? await self.store.upsertWake(wake)
                self.evaluateSoon()
            case .dismissed:
                break  // The dismissal record above is the whole point.
            }
        }
    }

    private func requestAuthorizationOnce() {
        guard !didRequestAuthorization, isEnabled() else { return }
        didRequestAuthorization = true
        Task { [notifier, recorder, calendar] in
            let granted = await notifier.activate()
            if !granted { recorder.record("loop.auth-denied") }
            // Calendar is briefing material (stage G) — denial just means the
            // situation block carries no schedule; nothing else changes.
            let calendarGranted = await calendar.requestAccessIfNeeded()
            if !calendarGranted { recorder.record("loop.calendar-denied") }
        }
    }

    // MARK: - Turn templates — the harness's occasion framing. The standing
    // conduct lives in the entity's own instructions document (ADR-0040 §12).

    private static let wakeTemplate = """
        <turn>
        The wakes listed as DUE NOW are why you are awake. Act on them per your \
        instructions: deliver, hold with a re-booking, or fold together — your \
        judgment. Anything you choose not to act on, say so in one line here (the \
        transcript is your record). Book whatever future wakes this implies before \
        you finish.
        </turn>
        """

    private static let catchUpTemplate = """
        <turn>
        These wakes are OVERDUE — the Mac was asleep or the app was closed. \
        Triage, don't pretend it is earlier than it is: a late morning summons is \
        better than none if his day is young; a stale pulse is better folded into \
        the next beat; a promise still fires quietly. For anything you skip, one \
        recorded line of reasoning here. Re-book the rest of today's rhythm if the \
        gap swallowed it.
        </turn>
        """

    private static let dayStartTemplate = """
        <turn>
        His day is starting — first presence after the overnight gap. This is not \
        the morning summons itself: check what is booked. If your rhythm for today \
        is missing (fresh install, wiped state, or you never booked it), establish \
        it now with book_wake: morning planning shortly, and the rest as your \
        instructions say. If the morning beat is already booked and near, silence \
        is correct.
        </turn>
        """

    private static let ambientTemplate = """
        <turn>
        Nothing is due — this is ambient time, your own. Think: is the day on \
        track, is something worth noticing in the observations, is there something \
        genuinely useful to prepare or research (read-only web is yours)? Acting \
        is allowed but rare: silence is the usual, correct end of an ambient turn. \
        Never manufacture a touchpoint to seem busy.
        </turn>
        """
}
