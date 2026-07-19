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
//  indefinitely, by design. The correctness invariant — a wake or an Event
//  is consumed only by a completed turn; anything less re-presents — is
//  decided by the **Companion Fold Reducer** (ADR-0051); this loop gathers
//  its inputs and performs its effect values.
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
    /// A banner engage with no live conversation behind it mints a dialogue
    /// seeded with the banner's line (ADR-0052) — the same door the overlay
    /// summons engage uses.
    private let beginDialogue: @MainActor (String?) -> Void
    /// The perception substrate's day-start door (ADR-0046, #368/#371): the
    /// tick hands over the facts; the producer detects and admits.
    private let perceiveDayStart: @MainActor (_ now: Date, _ ownerPresent: Bool) -> Void
    /// Mission Control's estimated size — the ceiling's signal (#373).
    private let foldTokens: () -> Int
    /// The intraday fold-down (#373): the digest engine, behind the
    /// evaluator's `.compactFold` grant.
    private let earlyFold: @MainActor () async -> Void

    private var tickTask: Task<Void, Never>?
    private var evaluating = false
    /// The Wake Evaluator — the whole clock, as one pure decider.
    private var evaluator = CompanionEvaluator()
    /// The fold's write side — presentation, settlement, and reaction
    /// writes as ordered effect values this loop performs (ADR-0051).
    private var reducer = CompanionFoldReducer()
    private var didRequestAuthorization = false
    private var didRunLaunchRecovery = false
    /// Briefing evidence: the last tick that saw the owner engaged.
    private var lastOwnerEngagedAt: Date?

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
        beginDialogue: @escaping @MainActor (String?) -> Void = { _ in },
        perceiveDayStart: @escaping @MainActor (Date, Bool) -> Void,
        foldTokens: @escaping () -> Int = { 0 },
        earlyFold: @escaping @MainActor () async -> Void = {}
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
        self.beginDialogue = beginDialogue
        self.perceiveDayStart = perceiveDayStart
        self.foldTokens = foldTokens
        self.earlyFold = earlyFold
    }

    // MARK: - Lifecycle

    func start() {
        guard tickTask == nil else { return }
        notifier.onOutcome = { [weak self] reaction in
            self?.handleReaction(reaction)
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

        // First presence of the calendar day is itself a perception (#371):
        // the tick hands the producer the facts; the deterministic id makes
        // the admission once-per-day, and the turn follows over the queue.
        perceiveDayStart(now, idleMonitor.isOwnerPresent)

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
            pendingEvents: (try? await store.pendingEvents()) ?? [],
            dueWakes: dueWakes,
            gpuBusy: isGPUBusy(),
            foldTokens: foldTokens())

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
        case .compactFold:
            // The ceiling fold (#373): awaited here so no turn can interleave
            // with it — the digest engine records the outcome.
            await earlyFold()
        }
    }

    // MARK: - The fold turn

    /// One granted turn drains everything pending — Events in total order,
    /// due wakes fired — into Mission Control. Gather → decide (the
    /// reducer) → perform: every write here is an effect value first.
    private func runFoldTurn(
        dueWakes: [CompanionWake], origin: TurnOrigin, carriesBeat: Bool, now: Date
    ) async {
        // Drain: the whole pending queue becomes this turn's batch, marked
        // presented in one transaction. Anything admitted after this instant
        // waits for the next turn.
        let batch = (try? await store.drainPendingEvents(at: now)) ?? []

        // From presentation on, the plan's fired copies are the turn's wake
        // values: settlement must see the stamped `firedAt`, or a retry's
        // rebook (and the exhausted fallback) would clobber it back to nil.
        let wakes: [CompanionWake]
        let resurfaced: [CompanionWake]
        switch reducer.begin(
            batch: batch, dueWakes: dueWakes, carriesBeat: carriesBeat, now: now)
        {
        case .skip:
            return
        case .present(let effects):
            wakes = effects.compactMap {
                guard case .fireWake(let wake) = $0 else { return nil }
                return wake
            }
            resurfaced = await perform(effects, now: now)
        }

        let template = origin == .catchup ? Self.catchUpTemplate : Self.foldTemplate
        let opening = await composeOpening(
            template: template, events: batch, dueWakes: wakes, resurfaced: resurfaced,
            now: now)
        let outcome = await runner.run(
            origin: origin, opening: opening, wakeIDs: wakes.map(\.id))

        let settlement = reducer.settle(
            batch: batch,
            wakes: wakes,
            outcome: outcome.map { ($0.turnID, $0.conversationID) },
            now: Date())
        await perform(settlement, now: now)
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

    // MARK: - The performer

    /// Execute the reducer's effect values, in order. The one home of the
    /// fold's store/notifier writes — no decision is made here beyond the
    /// re-reads an effect's own contract names. Returns the resurfacing
    /// pass's wakes for the opening composition.
    @discardableResult
    private func perform(
        _ effects: [CompanionFoldReducer.Effect], now: Date
    ) async -> [CompanionWake] {
        var resurfaced: [CompanionWake] = []
        for effect in effects {
            switch effect {
            case .fireWake(let wake):
                try? await store.upsertWake(wake)
                recorder.record(
                    "wake.fired", wakeID: wake.id,
                    snapshot: [
                        "class": wake.wakeClass.rawValue,
                        "due": CompanionWakeTime.format(wake.due),
                        "present": idleMonitor.isIdle ? "idle" : "active",
                    ],
                    note: wake.content)
            case .runResurfacingPass:
                // What the last beat resurfaced and he never heard dies
                // delivered-unheard; newly ignored promises join this
                // beat's agenda (#309).
                resurfaced = await CompanionResurfacing.pass(
                    store: store, recorder: recorder, now: now)
            case .consumeEvents(let ids, let turnID):
                try? await store.consumeEvents(ids: ids, turnID: turnID)
            case .deliverFiredWake(let id, let turnID, let conversationID):
                guard var current = try? await store.wake(id: id), current.state == .fired
                else { continue }
                current.state = .delivered
                current.consumedAt = Date()
                try? await store.upsertWake(current)
                recorder.record(
                    "wake.consumed", wakeID: id, turnID: turnID,
                    conversationID: conversationID)
            case .rebookWake(let wake):
                try? await store.upsertWake(wake)
            case .representEvents(let ids):
                try? await store.representEvents(ids: ids)
            case .fallbackBanner(let wake):
                await notifier.post(
                    pingID: UUID(), beatID: wake.wakeClass.rawValue, title: "Jarvis",
                    body: wake.content, wakeID: wake.id)
                try? await store.upsertWake(wake)
                recorder.record("delivery.fallback", wakeID: wake.id, note: wake.content)
            case .stampWakeHeard(let id):
                try? await store.stampWakeHeard(id: id, at: Date())
            case .engageWake(let id):
                if var wake = try? await store.wake(id: id) {
                    wake.state = .engaged
                    try? await store.upsertWake(wake)
                }
            case .stampResurfacedHeard:
                try? await store.stampResurfacedHeard(at: Date())
            case .openConversation(let id):
                openConversation(id)
            case .beginDialogue(let line):
                beginDialogue(line)
            case .bookReplyFollowup(let content, let conversationID):
                let wake = CompanionWake(
                    content: content, due: Date(), wakeClass: .followup,
                    conversationID: conversationID)
                try? await store.upsertWake(wake)
            case .accelerateEvaluation:
                evaluateSoon()
            }
        }
        return resurfaced
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
    /// owner's reaction routes back to the right wake and conversation —
    /// carried in the notification itself (ADR-0052), so the route survives
    /// relaunch.
    func deliverNotification(title: String, body: String) async {
        let context = runner.context
        await notifier.post(
            pingID: UUID(), beatID: context.origin?.rawValue ?? "companion",
            title: title, body: body,
            wakeID: context.wakeIDs.first, conversationID: context.conversationID)
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
        await notifier.post(
            pingID: UUID(), beatID: "summons-fallback", title: "Jarvis", body: line,
            wakeID: wakeID, conversationID: conversationID)
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

    private func handleReaction(_ reaction: CompanionPingReaction) {
        recorder.record(
            "reaction.\(reaction.outcome.rawValue)",
            wakeID: reaction.wakeID,
            conversationID: reaction.conversationID,
            note: reaction.note)

        let effects = reducer.reaction(
            outcome: reaction.outcome,
            wakeID: reaction.wakeID,
            conversationID: reaction.conversationID,
            line: reaction.line,
            note: reaction.note)
        Task { [weak self] in
            await self?.perform(effects, now: Date())
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
