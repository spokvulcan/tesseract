//
//  DayTypes.swift
//  tesseract
//
//  The Day Engine's vocabulary: the signals it hears, the effects it asks
//  for, the snapshot the loop gathers, and the state it keeps for the day.
//

import Foundation

// MARK: - Signals

nonisolated enum DaySignal: Sendable, Equatable {
    /// The loop's clock, about once a minute.
    case tick
    /// Reminders or Calendar changed.
    case agendaChanged
    /// The Companion was switched on (or the app started with it on).
    case companionEnabled
    /// The Companion was switched off: everything it scheduled is withdrawn.
    case companionDisabled
    /// The owner came back after being away since `awayFrom`.
    case presenceReturned(awayFrom: Date)
    /// The owner walked away (idle or locked).
    case presenceLeft
    /// The owner opened the Today page.
    case todayOpened
    /// A moment's model call finished.
    case momentOutcome(MomentRequest, MomentOutcome)
    /// The owner acted on a card.
    case cardAction(CardAction)
}

/// What the owner did on a card.
nonisolated enum CardAction: Sendable, Equatable {
    case dismiss(cardID: String)
    case setMustDo(reminderID: String?)
    case removeFromPlan(reminderID: String)
    /// Put a task into today's plan at a time ("Find a time").
    case place(reminderID: String, start: Date, minutes: Int)
    case leftover(cardID: String, reminderID: String, Leftover.Suggestion)
    /// Every remaining leftover, each with its own suggestion.
    case allLeftovers(cardID: String)
    /// Plan the day now, whatever the hour.
    case planNow
    /// Wrap up now, whatever the hour.
    case wrapUpNow
}

// MARK: - Effects

nonisolated enum DayEffect: Sendable, Equatable {
    /// Make the OS have exactly these event nudges scheduled.
    case syncNudges([Nudge])
    /// Run one moment: append its request to the Day Thread and generate.
    case runMoment(MomentRequest)
    /// Show a card on a delivery rung.
    case presentCard(DayCard, DeliveryRung)
    /// Change the owner's Reminders.
    case mutateAgenda(AgendaMutation)
    /// How many things are waiting on the owner, for the glyph.
    case setWaiting(Int)
    /// Write one Companion Trace event.
    case trace(CompanionTraceEvent, [String: CompanionTraceValue])
}

/// Where a card reaches the owner. Code picks the rung, never the model.
nonisolated enum DeliveryRung: String, Sendable, Equatable, Codable {
    /// Only on the Today page (and the glyph).
    case today
    /// The floating Jarvis panel over the current app.
    case panel
    /// A Notification Center banner.
    case banner
    /// A spoken line.
    case voice
}

nonisolated enum AgendaMutation: Sendable, Equatable {
    /// Due tomorrow, keeping a time of day if it had one.
    case dueTomorrow(reminderID: String)
    /// No date: it waits in its Area or the Inbox.
    case clearDue(reminderID: String)
    /// Let it go.
    case delete(reminderID: String)
}

// MARK: - Snapshot

/// What the loop gathered before this signal.
nonisolated struct DaySnapshot: Sendable, Equatable {
    var now: Date
    var calendar: Calendar
    var settings: DaySettings
    var agenda: AgendaSnapshot
    var areas: [Area]
    var inboxListID: String?
    /// The owner is at the Mac (not idle, not locked).
    var ownerPresent: Bool
    /// A Today chat turn is running; moments wait for it.
    var chatBusy: Bool

    init(
        now: Date, calendar: Calendar = .current, settings: DaySettings,
        agenda: AgendaSnapshot, areas: [Area] = [], inboxListID: String? = nil,
        ownerPresent: Bool = true, chatBusy: Bool = false
    ) {
        self.now = now
        self.calendar = calendar
        self.settings = settings
        self.agenda = agenda
        self.areas = areas
        self.inboxListID = inboxListID
        self.ownerPresent = ownerPresent
        self.chatBusy = chatBusy
    }

    func facts(state: DayState) -> DayFacts {
        DayFacts(
            snapshot: agenda, areas: areas, inboxListID: inboxListID, now: now,
            calendar: calendar, mustDoID: state.mustDoID, plan: state.plan)
    }

    /// Minutes after local midnight.
    var minuteOfDay: Int {
        let parts = calendar.dateComponents([.hour, .minute], from: now)
        return (parts.hour ?? 0) * 60 + (parts.minute ?? 0)
    }
}

/// The owner's Companion settings, as the engine reads them.
nonisolated struct DaySettings: Sendable, Equatable {
    var nudgeLeadMinutes: Int = 10
    var morningStartHour: Int = 4
    var morningEndHour: Int = 12
    var eveningMinutes: Int = 21 * 60
    var breakpointAwayMinutes: Int = 10
    var speaks: Bool = true
    var quietStartMinutes: Int = 23 * 60
    var quietEndMinutes: Int = 8 * 60

    /// The overnight gap that makes a return the day's first sit-down.
    static let overnightGap: TimeInterval = 4 * 3600
}

// MARK: - State

/// What the engine remembers about the day, persisted so a relaunch
/// neither repeats a moment nor forgets a card.
nonisolated struct DayState: Sendable, Equatable, Codable {
    var day: DayKey
    /// The nudges last handed to the OS, by id.
    var syncedNudgeIDs: Set<String>?
    /// When the owner was last at the Mac (carried across days: it is how the
    /// overnight gap is measured).
    var lastPresentAt: Date?
    var morningPlanAt: Date?
    var eveningWrapUpAt: Date?
    var nightReflectionAt: Date?
    /// The moment in flight, if any.
    var running: MomentKind?
    var cards: [DayCard] = []
    var mustDoID: String?
    var plan: [Placement] = []
    /// Last night's carry-over note, for this day's opening.
    var carryOver: String?
    /// Tonight's note, handed to the next day.
    var carryOverForNextDay: String?

    init(day: DayKey, syncedNudgeIDs: Set<String>? = nil) {
        self.day = day
        self.syncedNudgeIDs = syncedNudgeIDs
    }

    /// The next day's state: what must survive the rollover survives.
    func rolledOver(to day: DayKey) -> DayState {
        var next = DayState(day: day, syncedNudgeIDs: syncedNudgeIDs)
        next.lastPresentAt = lastPresentAt
        next.carryOver = carryOverForNextDay
        return next
    }

    /// Cards the owner hasn't dismissed.
    var openCards: [DayCard] { cards.filter { !$0.dismissed } }
}
