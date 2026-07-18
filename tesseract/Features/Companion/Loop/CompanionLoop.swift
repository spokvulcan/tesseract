//
//  CompanionLoop.swift
//  tesseract
//
//  The harness's spine (ADR-0040, the fold's clock since ADR-0046 #371): a
//  ticking evaluator with event accelerants. Each tick gathers one `Signals`
//  snapshot — the pending Event queue, the due wakes, eligibility — asks the
//  Wake Evaluator (the pure decider) for at most one `Decision`, and performs
//  it. A granted turn drains everything pending into Mission Control; there
//  is no cadence and no safety tick — a quiet queue grants no turns,
//  indefinitely, by design. The correctness invariant lives here: a wake or
//  an Event is consumed only by a completed turn; anything less re-presents.
//
//  The tick is a coalescing clock, not a cadence: it never grants a turn by
//  itself — it only notices what the queue and the wake table already hold.
//

import AppKit
import Foundation

@MainActor
final class CompanionLoop {

    /// Wall-clock ticking keeps due-ness honest across system sleep — the
    /// skeleton proved a timer armed for 09:00 dies with a closed lid. The
    /// tick decides nothing: an empty queue ticks forever in silence.
    static let tickInterval: Duration = .seconds(30)
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
    /// Reads the concrete arbiter's lease state — the fold's one mechanical
    /// eligibility: the model slot (#371).
    private let isGPUBusy: () -> Bool
    /// Live owner activity (voice session, generation, dictation, app use) —
    /// sampled each tick as briefing evidence ("he last used this app…"),
    /// never as a gate: the attention gate's granting role died with #371.
    private let isOwnerEngaged: () -> Bool
    private let isEnabled: () -> Bool
    private let speak: @MainActor (String) -> Void
    private let openConversation: @MainActor (UUID) -> Void
    /// The perception substrate's day-start door (ADR-0046, #368): the
    /// evaluator detects first presence; the producer admits the Event.
    private let perceiveDayStart: @MainActor (Date) -> Void

    private var tickTask: Task<Void, Never>?
    private var evaluating = false
    /// The Wake Evaluator — the whole clock, as one pure decider.
    private var evaluator = CompanionEvaluator()
    private var didRequestAuthorization = false
    private var didRunLaunchRecovery = false
    /// Briefing evidence: the last tick that saw the owner engaged.
    private var lastOwnerEngagedAt: Date?
    /// Failed attempts per fold batch, keyed by the earliest wake id (or the
    /// earliest event id for an event-only batch).
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
        isOwnerEngaged: @escaping () -> Bool,
        isEnabled: @escaping () -> Bool,
        speak: @escaping @MainActor (String) -> Void,
        openConversation: @escaping @MainActor (UUID) -> Void,
        perceiveDayStart: @escaping @MainActor (Date) -> Void
    ) {
        self.store = store
        self.recorder = recorder
        self.runner = runner
        self.notifier = notifier
        self.idleMonitor = idleMonitor
        self.sensed = sensed
        self.calendar = calendar
        self.isGPUBusy = isGPUBusy
        self.isOwnerEngaged = isOwnerEngaged
        self.isEnabled = isEnabled
        self.speak = speak
        self.openConversation = openConversation
        self.perceiveDayStart = perceiveDayStart
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

    /// Event accelerant — Mac-wake and reaction paths call this so reactions
    /// feel instant instead of up-to-a-tick late.
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

    /// Gather → decide → execute. Serialized: one evaluation, and at most one
    /// turn, at a time.
    private func evaluate() async {
        guard isEnabled(), !evaluating, !runner.isRunning else { return }
        evaluating = true
        defer { evaluating = false }

        requestAuthorizationOnce()
        runLaunchRecoveryOnce()

        let now = Date()
        if isOwnerEngaged() { lastOwnerEngagedAt = now }
        let todayKey = TrackingDay.key(for: now)

        // A booked wake coming due is itself a perception (ADR-0046): admit
        // it into the queue before reading, exactly once per wake by
        // deterministic id — a busy tick re-admitting is a collapsed
        // duplicate, not a second event.
        let dueWakes = (try? await store.dueWakes(asOf: now)) ?? []
        for wake in dueWakes {
            let event = CompanionEvent(
                id: CompanionEvent.deterministicID("wake-due:\(wake.id.uuidString)"),
                kind: .wakeDue,
                content: "[\(wake.wakeClass.rawValue)] \(wake.content)",
                occurredAt: wake.due)
            if (try? await store.admitEvent(event)) == true {
                recorder.record(
                    "event.admitted",
                    wakeID: wake.id,
                    snapshot: ["kind": "wake-due", "eventID": event.id.uuidString],
                    note: wake.content)
            }
        }

        let signals = CompanionEvaluator.Signals(
            now: now,
            localHour: Calendar.current.component(.hour, from: now),
            pendingEvents: (try? await store.pendingEvents()) ?? [],
            dueWakes: dueWakes,
            dayState: (try? await store.loopDayState(todayKey)) ?? CompanionLoopDayState(),
            ownerPresent: idleMonitor.isOwnerPresent,
            onACPower: sensed.isOnACPower,
            gpuBusy: isGPUBusy())

        switch evaluator.decide(signals) {
        case .wait:
            return
        case .recordDeferral(let pendingCount, let firstWakeID):
            recorder.record(
                "turn.deferred", wakeID: firstWakeID,
                snapshot: ["pending": String(pendingCount), "reason": "model-slot-busy"])
        case .foldTurn(let dueWakes, let origin, let carriesBeat):
            await runFoldTurn(
                dueWakes: dueWakes, origin: origin, carriesBeat: carriesBeat, now: now)
        case .perceiveDayStart(let updated):
            try? await store.setLoopDayState(todayKey, updated)
            recorder.record("loop.day-start", snapshot: ["at": CompanionWakeTime.format(now)])
            perceiveDayStart(now)
        }
    }

    // MARK: - The fold turn

    /// One granted turn drains everything pending — Events in total order,
    /// due wakes fired — into Mission Control.
    private func runFoldTurn(
        dueWakes wakes: [CompanionWake], origin: TurnOrigin, carriesBeat: Bool, now: Date
    ) async {
        // Drain: the whole pending queue becomes this turn's batch, marked
        // presented in one transaction. Anything admitted after this instant
        // waits for the next turn.
        let batch = (try? await store.drainPendingEvents(at: now)) ?? []
        guard !batch.isEmpty || !wakes.isEmpty else { return }

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

        // A firing beat advances the ignored-promise ladder (#309): what the
        // last beat resurfaced and he never heard dies delivered-unheard, and
        // newly ignored promises join this beat's agenda. `carriesBeat` is the
        // evaluator's call — true even for a rhythm wake firing as catch-up.
        var resurfaced: [CompanionWake] = []
        if carriesBeat {
            resurfaced = await CompanionResurfacing.pass(
                store: store, recorder: recorder, now: now)
        }

        let template = origin == .catchup ? Self.catchUpTemplate : Self.foldTemplate
        let opening = await composeOpening(
            template: template, events: batch, dueWakes: wakes, resurfaced: resurfaced,
            now: now)
        let outcome = await runner.run(
            origin: origin, opening: opening, wakeIDs: wakes.map(\.id))

        guard let outcome else {
            await handleTurnFailure(events: batch, wakes: wakes)
            return
        }
        turnAttempts.removeAll()

        // Consumed only by this completed turn — the invariant, for Events
        // and wakes alike. A wake the turn itself moved (revise_wake flips it
        // back to booked) is respected.
        try? await store.consumeEvents(ids: batch.map(\.id), turnID: outcome.turnID)
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

    private func composeOpening(
        template: String, events: [CompanionEvent], dueWakes: [CompanionWake],
        resurfaced: [CompanionWake] = [], now: Date
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
            dueWakes: dueWakes, resurfacedWakes: resurfaced, recorder: recorder,
            calendar: calendar, lastAppUse: lastOwnerEngagedAt, now: now)
        return [
            CompanionInstructions.wrap(instructions),
            CompanionBriefing.render(inputs),
            CompanionEventBatch.render(events, now: now),
            template,
        ].filter { !$0.isEmpty }.joined(separator: "\n\n")
    }

    // MARK: - Failure semantics (ADR-0040 §13)

    private func handleTurnFailure(events: [CompanionEvent], wakes: [CompanionWake]) async {
        guard let key = wakes.first?.id ?? events.first?.id else { return }
        let attempts = (turnAttempts[key] ?? 0) + 1
        turnAttempts[key] = attempts

        if attempts < Self.maxTurnAttempts {
            // Everything re-presents: wakes back to booked, Events back to
            // pending, order untouched — the next tick retries the fold.
            for var wake in wakes {
                wake.state = .booked
                try? await store.upsertWake(wake)
            }
            try? await store.representEvents(ids: events.map(\.id))
            return
        }

        // Last resort: the brain is offline. Each wake's own stateable line
        // delivers as a plain banner so never-silent-give-up holds. The
        // Events stay presented — out of the retry path, recovered into the
        // queue at next launch (the recorded-failed half of the invariant:
        // `turn.failed` is already on the record for every attempt).
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
            // Crash recovery — the invariant, both tables: fired-but-
            // unconsumed wakes re-book, presented-but-unconsumed Events go
            // back to pending in their original order.
            if let orphanEvents = try? await store.unconsumedPresentedEvents(),
                !orphanEvents.isEmpty
            {
                try? await store.representEvents(ids: orphanEvents.map(\.id))
                recorder.record(
                    "event.represented", snapshot: ["count": String(orphanEvents.count)])
            }
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
            pingID: pingID, beatID: context.origin?.rawValue ?? "companion",
            title: title, body: body)
        recorder.record(
            "delivery.notification",
            wakeID: context.wakeIDs.first,
            turnID: context.turnID,
            conversationID: context.conversationID,
            snapshot: ["title": title],
            note: body)
    }

    /// A summons that lapsed unanswered leaves its line as a banner — §11
    /// guarantee 1: no delivery evaporates silently. Correlation is passed in
    /// (not read from the runner) because the overlay's give-up can outlive
    /// the turn that raised it.
    func deliverUnansweredFallback(line: String, wakeID: UUID?, conversationID: UUID?) async {
        let pingID = UUID()
        postedPings[pingID] = (wakeID, conversationID)
        await notifier.post(
            pingID: pingID, beatID: "summons-fallback", title: "Jarvis", body: line)
        recorder.record(
            "delivery.notification",
            wakeID: wakeID,
            conversationID: conversationID,
            snapshot: ["reason": "summons-unanswered"],
            note: line)
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
            // Any reaction is proof the delivery reached him (#309): stamp
            // heard first so the resurfacing ladder never nags about a wake
            // he engaged, answered, or explicitly waved off.
            if let wakeID = correlation?.wakeID {
                try? await self.store.stampWakeHeard(id: wakeID, at: Date())
            }
            switch outcome {
            case .engaged:
                if let wakeID = correlation?.wakeID,
                    var wake = try? await self.store.wake(id: wakeID)
                {
                    wake.state = .engaged
                    try? await self.store.upsertWake(wake)
                }
                // Him engaging any companion banner means the beat that
                // carried the resurfacing reached him — spare its agenda.
                try? await self.store.stampResurfacedHeard(at: Date())
                if let conversationID = correlation?.conversationID {
                    self.openConversation(conversationID)
                }
            case .replied:
                try? await self.store.stampResurfacedHeard(at: Date())
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

    private static let foldTemplate = """
        <turn>
        The events above are why you are awake — everything that reached you \
        since your last turn, in order. Reason over the whole batch at once, \
        per your instructions: act, hold, fold together, or stay silent — your \
        judgment. Anything you choose not to act on, say so in one line here \
        (this conversation is your record). Book whatever future wakes this \
        implies before you finish — no one wakes you but yourself.

        A beat that needs his participation — the evening journal, morning \
        planning, any review — is a conversation, not a monologue. Summon him \
        (notify; speak too only if he is demonstrably present), say what it is \
        time for, then END the turn and wait: his reply reaches you as an \
        event. Never write his side of a ritual, and never close his day \
        without him. If a summons lapses unanswered, re-book the beat once, \
        30-45 minutes out; if it lapses again, note it and fold the ritual \
        into the next natural beat.
        </turn>
        """

    private static let catchUpTemplate = """
        <turn>
        Some of the wakes in this batch are badly OVERDUE — the Mac was \
        asleep, the app was closed, or the queue was long. Triage, don't \
        pretend it is earlier than it is: a late morning summons is better \
        than none if his day is young; a stale pulse is better folded into \
        the next beat; a promise still fires quietly. For anything you skip, \
        one recorded line of reasoning here. Re-book the rest of today's \
        rhythm if the gap swallowed it. A participatory beat still runs WITH \
        him: summon, end the turn, and wait — never run it solo because it \
        is late.
        </turn>
        """
}
