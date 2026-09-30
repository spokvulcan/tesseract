//
//  CompanionRuntime.swift
//  tesseract
//
//  The Companion's loop: it gathers a snapshot, feeds the Day Engine one
//  signal at a time, and performs the effects the engine returns. All the
//  deciding lives in the engine; this type only does the I/O — the clock,
//  the agenda, the notification center, the trace.
//
//  It rides the Companion switch: on, it asks for Reminders, Calendar and
//  notification access once and starts the clock; off, it withdraws every
//  nudge it scheduled and stops.
//

import Foundation
import Observation

@MainActor
final class CompanionRuntime {

    private let settings: SettingsManager
    private let agenda: Agenda
    private let notifier: CompanionNotifier
    private let trace: CompanionTrace
    private let now: @MainActor () -> Date

    private var state: DayState
    private var clockTask: Task<Void, Never>?
    private var toggleTask: Task<Void, Never>?
    /// Signals are decided strictly one after another.
    private var chain: Task<Void, Never>?
    private(set) var isActive = false

    init(
        settings: SettingsManager, agenda: Agenda, notifier: CompanionNotifier,
        trace: CompanionTrace, now: @escaping @MainActor () -> Date = Date.init
    ) {
        self.settings = settings
        self.agenda = agenda
        self.notifier = notifier
        self.trace = trace
        self.now = now
        self.state = DayState(day: DayKey(for: now()))
        agenda.addListener { [weak self] in self?.send(.agendaChanged) }
    }

    /// Follow the Companion switch for the life of the app.
    func start() {
        guard toggleTask == nil else { return }
        toggleTask = Task { [weak self] in
            guard let self else { return }
            for await enabled in Observations({ self.settings.companionHeartbeatEnabled }) {
                if enabled { await self.activate() } else { await self.deactivate() }
            }
        }
    }

    // MARK: Lifecycle

    private func activate() async {
        guard !isActive else { return }
        isActive = true
        await notifier.activate()
        await agenda.requestAccessIfNeeded()
        send(.tick)
        clockTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(60))
                guard !Task.isCancelled else { return }
                await self?.agenda.refresh()
                self?.send(.tick)
            }
        }
        Log.companion.info("Companion on")
    }

    private func deactivate() async {
        clockTask?.cancel()
        clockTask = nil
        guard isActive else {
            // Off at launch: still withdraw anything a previous run scheduled.
            notifier.cancel(nudgeIDs: Array(await notifier.scheduledNudgeIDs()))
            return
        }
        isActive = false
        send(.companionDisabled)
        Log.companion.info("Companion off")
    }

    // MARK: Signals

    func send(_ signal: DaySignal) {
        guard isActive || signal == .companionDisabled else { return }
        let previous = chain
        chain = Task { [weak self] in
            await previous?.value
            await self?.process(signal)
        }
    }

    private func process(_ signal: DaySignal) async {
        let snapshot = DaySnapshot(
            now: now(), settings: daySettings, agenda: agenda.snapshot)
        let decision = DayEngine.decide(signal, snapshot: snapshot, state: state)
        state = decision.state
        for effect in decision.effects { await perform(effect) }
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
            quietEndMinutes: settings.companionQuietEndMinutes)
    }

    // MARK: Effects

    private func perform(_ effect: DayEffect) async {
        switch effect {
        case .syncNudges(let desired):
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
        case .trace(let event, let fields):
            trace.record(event, fields: fields)
        }
    }
}
