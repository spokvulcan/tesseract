//
//  CompanionRuntime.swift
//  tesseract
//
//  The Companion's loop: it gathers a snapshot, feeds the Day Engine one
//  signal at a time, and performs the effects the engine returns. All the
//  deciding lives in the engine; this type only does the I/O — the clock,
//  presence, the agenda, the Day Thread, the notification center, the trace.
//
//  It rides the Companion switch: on, it asks for Reminders, Calendar and
//  notification access once and starts the clock; off, it withdraws every
//  nudge it scheduled and stops.
//

import Foundation
import Observation

/// How cards reach the owner, as closures the composition root wires: the
/// Jarvis panel, the voice, apps to bring forward.
@MainActor
struct CompanionDelivery {
    var showPanel: (DayCard) -> Void = { _ in }
    var showStep: (StepCue) -> Void = { _ in }
    var retractPanel: (String) -> Void = { _ in }
    var closePanel: () -> Void = {}
    /// The panel is up with something the owner hasn't closed.
    var isPanelUp: () -> Bool = { false }
    var speak: (String) -> Void = { _ in }
    var openApp: (String) -> Void = { _ in }
}

@Observable @MainActor
final class CompanionRuntime {

    /// The day as the engine sees it: cards, plan, must-do. Today reads it.
    private(set) var state: DayState
    private(set) var isActive = false

    @ObservationIgnored private let settings: SettingsManager
    @ObservationIgnored private let agenda: Agenda
    @ObservationIgnored private let notifier: CompanionNotifier
    @ObservationIgnored private let trace: CompanionTrace
    @ObservationIgnored private let idleMonitor: IdleMonitor
    @ObservationIgnored private let presence: CompanionPresence
    @ObservationIgnored private let thread: DayThread
    @ObservationIgnored private let stateStore: DayStateStore
    @ObservationIgnored private let frontmost: FrontmostAppTracker
    @ObservationIgnored private let power: PowerMonitor
    @ObservationIgnored private let delivery: CompanionDelivery
    @ObservationIgnored private let profile: ProfileStore?
    @ObservationIgnored private let now: @MainActor () -> Date
    @ObservationIgnored private var watcher: NotificationCenterWatcher?

    @ObservationIgnored private var clockTask: Task<Void, Never>?
    @ObservationIgnored private var toggleTask: Task<Void, Never>?
    /// Signals are decided strictly one after another.
    @ObservationIgnored private var chain: Task<Void, Never>?

    init(
        settings: SettingsManager, agenda: Agenda, notifier: CompanionNotifier,
        trace: CompanionTrace, idleMonitor: IdleMonitor, presence: CompanionPresence,
        thread: DayThread, stateStore: DayStateStore, frontmost: FrontmostAppTracker,
        power: PowerMonitor, delivery: CompanionDelivery, profile: ProfileStore? = nil,
        now: @escaping @MainActor () -> Date = Date.init
    ) {
        self.settings = settings
        self.agenda = agenda
        self.notifier = notifier
        self.trace = trace
        self.idleMonitor = idleMonitor
        self.presence = presence
        self.thread = thread
        self.stateStore = stateStore
        self.frontmost = frontmost
        self.power = power
        self.delivery = delivery
        self.profile = profile
        self.now = now
        // A moment in flight when the app quit never finished: the engine
        // picks it up when the Companion comes on.
        self.state = (stateStore.load() ?? DayState(day: DayKey(for: now()))).relaunched()
        agenda.addListener { [weak self] in self?.send(.agendaChanged) }
        thread.openingProvider = { [weak self] in self?.dayOpening() ?? "" }
    }

    /// Follow the Companion switch for the life of the app.
    func start() {
        guard toggleTask == nil else { return }
        idleMonitor.onIdle = { [weak self] in self?.send(.presenceLeft) }
        idleMonitor.onReturn = { [weak self] in
            guard let self else { return }
            let awayFrom = self.idleMonitor.awaySince ?? self.now()
            self.send(.presenceReturned(awayFrom: awayFrom))
        }
        idleMonitor.start()
        frontmost.onActivate = { [weak self] name, bundleID in
            self?.send(.appActivated(name: name, bundleID: bundleID))
        }
        frontmost.start()
        power.onChange = { [weak self] in self?.send(.powerChanged) }
        power.start()
        let selfNames: Set<String> = ["Tesseract Agent", "Tesseract"]
        let watcher = NotificationCenterWatcher(
            isEnabled: { [settings] in settings.companionHeartbeatEnabled },
            onNotification: { [weak self] captured in
                guard let self,
                    let notification = captured.admitted(selfDisplayNames: selfNames)
                else { return }
                // Code decides who it's from (a person, an app, noise)
                // before the engine or a model sees it.
                let source = NotificationSources.classify(
                    notification, app: self.frontmost.identity(named: notification.app))
                self.send(.notificationArrived(notification.classified(source)))
            })
        watcher.start()
        self.watcher = watcher
        thread.show(day: state.day)
        toggleTask = Task { [weak self] in
            guard let self else { return }
            for await enabled in Observations({ self.settings.companionHeartbeatEnabled }) {
                if enabled { await self.activate() } else { await self.deactivate() }
            }
        }
    }

    // MARK: Owner actions

    /// The owner opened the Today page.
    func todayOpened() {
        Task { await agenda.refresh() }
        send(.todayOpened)
    }

    func act(_ action: CardAction) {
        send(.cardAction(action))
    }

    /// A coding agent's hook reported in (the local route).
    func receive(_ signal: AgentSignal) {
        send(.agentSignal(signal))
    }

    /// The owner's notification rules.
    var rules: [TriageRule] { TriageRules.decode(settings.companionTriageRulesJSON) }

    // MARK: Lifecycle

    private func activate() async {
        guard !isActive else { return }
        isActive = true
        notifier.onNudgeDelivered = { [weak self] _ in
            Task { await self?.readDeliveredNudges() }
        }
        await notifier.activate()
        await agenda.requestAccessIfNeeded()
        send(.companionEnabled)
        clockTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(60))
                guard !Task.isCancelled else { return }
                await self?.agenda.refresh()
                self?.send(.tick)
                await self?.readDeliveredNudges()
            }
        }
        Log.companion.info("Companion on")
    }

    /// macOS shows the event nudges itself; read back which ones it
    /// delivered, so the trace records each one once.
    private func readDeliveredNudges() async {
        send(.nudgesDelivered(await notifier.deliveredNudges()))
    }

    private func deactivate() async {
        clockTask?.cancel()
        clockTask = nil
        guard isActive else {
            // Off at launch: still withdraw anything a previous run scheduled.
            notifier.cancel(nudgeIDs: Array(await notifier.scheduledNudgeIDs()))
            return
        }
        send(.companionDisabled)
        isActive = false
        presence.setClock(nil)
        delivery.closePanel()
        Log.companion.info("Companion off")
    }

    // MARK: Signals

    func send(_ signal: DaySignal) {
        let alwaysHeard: Bool =
            switch signal {
            case .companionDisabled, .momentOutcome, .cardAction: true
            default: false
            }
        guard isActive || alwaysHeard else { return }
        let previous = chain
        chain = Task { [weak self] in
            await previous?.value
            await self?.process(signal)
        }
    }

    private func process(_ signal: DaySignal) async {
        let snapshot = snapshot()
        let decision = DayEngine.decide(signal, snapshot: snapshot, state: state)
        if decision.state != state {
            state = decision.state
            stateStore.save(state)
        }
        // The thread turns to the new day only between moments: one in
        // flight at 04:00 finishes in its own day's thread, not the next.
        if thread.momentRunning == nil { thread.show(day: state.day) }
        presence.setClock(isActive ? DayEngine.clock(snapshot: snapshot, state: state) : nil)
        for effect in decision.effects { await perform(effect) }
    }

    private func snapshot() -> DaySnapshot {
        DaySnapshot(
            now: now(), settings: daySettings, agenda: agenda.snapshot, areas: agenda.areas,
            inboxListID: agenda.inbox?.id, ownerPresent: idleMonitor.isOwnerPresent,
            chatBusy: thread.isChatBusy, frontmostAppName: frontmost.name,
            frontmostBundleID: frontmost.bundleID, frontmostIsGame: frontmost.isGame,
            lastTerminalFrontAt: frontmost.lastTerminalFrontAt(now: now()), power: power.state,
            profile: profile?.facts.map(\.text) ?? [], panelUp: delivery.isPanelUp())
    }

    private var daySettings: DaySettings {
        DaySettings(
            nudgeLeadMinutes: settings.companionNudgeLeadMinutes,
            morningStartHour: settings.companionMorningStartHour,
            morningEndHour: settings.companionMorningEndHour,
            eveningMinutes: settings.companionEveningMinutes,
            breakpointAwayMinutes: settings.companionBreakpointAwayMinutes,
            speaks: settings.companionSpeaks,
            quietStartMinutes: settings.companionQuietStartMinutes,
            quietEndMinutes: settings.companionQuietEndMinutes,
            windDown: settings.companionWindDown, stepCues: settings.companionStepCues,
            rules: rules)
    }

    private func dayOpening() -> String {
        let snapshot = snapshot()
        return MomentPrompts.dayOpening(
            facts: snapshot.facts(state: state), profile: snapshot.profile,
            carryOver: state.carryOver, draft: state.draft)
    }

    // MARK: Effects

    private func perform(_ effect: DayEffect) async {
        switch effect {
        case .syncNudges(let desired):
            await syncNudges(desired)

        case .runMoment(let request):
            // Off the signal chain: a generation takes seconds to minutes, and
            // the loop must keep hearing the owner meanwhile.
            presence.beginThinking()
            Task { [weak self] in
                guard let self else { return }
                let outcome = await self.thread.runMoment(request)
                self.presence.endThinking()
                self.send(.momentOutcome(request, outcome))
            }

        case .presentCard(let card, let rung):
            await present(card, on: rung)

        case .presentStep(let cue):
            if isActive { delivery.showStep(cue) }

        case .retractCard(let cardID):
            delivery.retractPanel(cardID)

        case .speak(let line):
            // A moment that finishes after the Companion went off stays quiet.
            if isActive { delivery.speak(line) }

        case .postBanner(let title, let body):
            if isActive { await notifier.post(title: title, body: body) }

        case .openApp(let name):
            delivery.openApp(name)

        case .mutateAgenda(let mutation):
            await mutate(mutation)

        case .setWaiting(let count):
            presence.setWaiting(count: count)

        case .proposeFacts(let drafts):
            profile?.propose(drafts, source: "night-reflection")

        case .trace(let event, let fields):
            trace.record(
                event, conversationID: AgentConversation.dayThreadID(for: state.day), fields: fields
            )
        }
    }

    private func syncNudges(_ desired: [Nudge]) async {
        let scheduled = await notifier.scheduledNudgeIDs()
        let (add, remove) = NudgePlanner.diff(desired: desired, scheduled: scheduled)
        notifier.cancel(nudgeIDs: remove)
        for nudge in add { await notifier.schedule(nudge) }
        for id in remove { trace.record(.nudgeCancelled, fields: ["id": .string(id)]) }
        for nudge in add {
            trace.record(
                .nudgeScheduled,
                fields: [
                    "id": .string(nudge.id), "title": .string(nudge.title),
                    "fireAt": .double(nudge.fireAt.timeIntervalSince1970),
                ])
        }
    }

    private func present(_ card: DayCard, on rung: DeliveryRung) async {
        // Switched off: the card stays in Today, and nothing else reaches out.
        guard isActive || rung == .today else { return }
        switch rung {
        case .today:
            break  // Today renders the day's cards from `state`.
        case .panel:
            delivery.showPanel(card)
        case .banner:
            await notifier.post(title: card.kind.title, body: card.line, cardID: card.id)
        case .voice:
            delivery.speak(card.line)
        }
    }

    private func mutate(_ mutation: AgendaMutation) async {
        let calendar = Calendar.current
        do {
            switch mutation {
            case .dueTomorrow(let id):
                // The owner's tomorrow: after a wrap-up at 01:30 it is today's date.
                let tomorrow =
                    DayKey(for: now(), calendar: calendar).next(calendar: calendar).date(
                        calendar: calendar)
                    ?? now()
                try await moveReminder(id, toDay: tomorrow)
            case .dueOn(let id, let day):
                try await moveReminder(id, toDay: calendar.startOfDay(for: day))
            case .dueAt(let id, let at):
                _ = try await agenda.updateReminder(
                    id: id, due: .set(at, hasTime: true), source: "card")
            case .clearDue(let id):
                _ = try await agenda.updateReminder(id: id, due: .clear, source: "wrapUp")
            case .delete(let id):
                try await agenda.deleteReminder(id: id, source: "wrapUp")
            case .complete(let id):
                _ = try await agenda.updateReminder(id: id, completed: true, source: "card")
            case .followUp(let title, let at):
                _ = try agenda.addReminder(title: title, due: at, dueHasTime: true, source: "card")
            case .add(let title, let due):
                _ = try agenda.addReminder(title: title, due: due, source: "proposal")
            }
        } catch {
            Log.companion.error("Agenda change failed: \(error.localizedDescription)")
        }
    }

    /// Re-date a reminder to `day`, keeping its time of day if it had one.
    private func moveReminder(_ id: String, toDay day: Date) async throws {
        let calendar = Calendar.current
        let reminder = await agenda.store.reminder(id: id, now: now())
        var target = day
        if let due = reminder?.due, reminder?.dueHasTime == true {
            let time = calendar.dateComponents([.hour, .minute], from: due)
            target =
                calendar.date(
                    bySettingHour: time.hour ?? 9, minute: time.minute ?? 0, second: 0, of: day)
                ?? day
        }
        _ = try await agenda.updateReminder(
            id: id, due: .set(target, hasTime: reminder?.dueHasTime ?? false), source: "wrapUp")
    }
}

// MARK: - State store

/// The day's engine state on disk, so a relaunch neither repeats a moment
/// nor forgets a card.
@MainActor
final class DayStateStore {
    private let url: URL?

    /// nil keeps the state in memory only (tests).
    init(url: URL?) {
        self.url = url
    }

    static var production: DayStateStore {
        DayStateStore(
            url: StorageEnvironment.applicationSupport
                .appendingPathComponent("Tesseract Agent/companion", isDirectory: true)
                .appendingPathComponent("day-state.json"))
    }

    func load() -> DayState? {
        guard let url, let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(DayState.self, from: data)
    }

    func save(_ state: DayState) {
        guard let url else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(state) else { return }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }
}
