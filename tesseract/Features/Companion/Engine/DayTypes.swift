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
    /// Another app's banner appeared.
    case notificationArrived(ObservedNotification)
    /// An app came to the front (its display name and bundle id).
    case appActivated(name: String, bundleID: String?)
    /// A coding agent's hook reported in.
    case agentSignal(AgentSignal)
    /// Power or thermal state changed.
    case powerChanged
    /// A moment's model call finished.
    case momentOutcome(MomentRequest, MomentOutcome)
    /// The owner acted on a card.
    case cardAction(CardAction)
    /// The event nudges macOS has delivered so far (the OS shows them, with
    /// Tesseract in front or not, so the loop reads them back on its tick).
    case nudgesDelivered([DeliveredNudge])
}

/// An event nudge the OS delivered.
nonisolated struct DeliveredNudge: Sendable, Equatable {
    let id: String
    let title: String
    let at: Date
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
    /// A card item was opened in its app.
    case openItem(cardID: String, itemID: String)
    /// A card item is handled: the notification or agent is resolved, the
    /// reminder is completed.
    case itemDone(cardID: String, itemID: String)
    /// Not now: a notification becomes a follow-up reminder in half an hour;
    /// a reminder moves half an hour on.
    case itemLater(cardID: String, itemID: String)
    /// A leftover moves to a chosen day.
    case leftoverOn(cardID: String, reminderID: String, day: Date)
    /// A waiting coding agent was dealt with, from Today.
    case agentHandled(agentID: String)
    /// Plan the day now, whatever the hour.
    case planNow
    /// Wrap up now, whatever the hour.
    case wrapUpNow
    /// The owner answered a Step Cue.
    case step(reminderID: String, StepChoice)
    /// The owner took a card in on the panel (Looks Good, Good Night): it
    /// leaves the panel and stays in Today.
    case keep(cardID: String)
    /// The owner added a proposed task to Reminders, or let it go.
    case taskProposal(id: String, add: Bool)
}

/// The step the owner started and is in now.
nonisolated struct StepFocus: Sendable, Equatable {
    var reminderID: String
    var title: String
    var end: Date
}

/// What the menu bar counts down beside the glyph, so time can be seen from
/// any app: the started step's time left, or the time until what comes next.
nonisolated struct MenuBarClock: Sendable, Equatable {
    enum Kind: Sendable, Equatable {
        /// The step the owner started: "25m" left.
        case focus
        /// An event starts: "in 12m".
        case event
        /// Time to leave for an event in person: "leave in 12m".
        case leave
    }

    var kind: Kind
    var title: String
    var until: Date
}

/// What the owner chose on a Step Cue.
nonisolated enum StepChoice: String, Sendable, Equatable {
    /// Doing it now: the slot starts this minute.
    case start
    /// Just five minutes, then decide: offered once a step was put off
    /// twice. The slot starts this minute, five minutes long.
    case startSmall
    /// Not yet: the slot moves a quarter of an hour on, and is cued again then.
    case later
    /// Not finished: the slot runs a quarter of an hour longer, and checks in
    /// again at its new end.
    case extend
    /// Not today: due tomorrow, off today's plan.
    case tomorrow
    /// Already done.
    case done
    /// Closed: nothing changes.
    case dismiss
}

/// A planned step at its start or, once the owner started it, at its end,
/// as the Jarvis Panel shows it.
nonisolated struct StepCue: Sendable, Equatable {
    enum Phase: String, Sendable, Equatable {
        /// Its slot starts now.
        case start
        /// The slot the owner started is over: done, longer, or another day?
        case end
    }

    var reminderID: String
    var title: String
    var start: Date
    var minutes: Int
    var areaName: String
    var isMustDo: Bool
    /// What comes after it ("Design review at 15:00").
    var next: String?
    var phase: Phase = .start
    /// Shown well after its moment — the owner was away, busy or behind
    /// another panel — so it says so ("Still time for", "How did it go?").
    var late = false
    /// How often today the owner put this task off ("In 15 min"): from the
    /// second time, its start is offered as five minutes.
    var putOff = 0
    /// A five-minute start ("Start 5 min"): its end asks to keep going.
    var small = false

    /// The start is offered small: starting is the hard part.
    var offersSmallStart: Bool { phase == .start && putOff >= 2 }

    var end: Date { start.addingTimeInterval(TimeInterval(minutes * 60)) }

    /// One slot, one cue: a task moved to another time is a new slot.
    static func key(_ placement: Placement) -> String {
        "\(placement.reminderID)@\(Int(placement.start.timeIntervalSince1970))"
    }

    /// One end, one check-in: a slot made longer has a new end.
    static func endKey(_ placement: Placement) -> String {
        "\(key(placement))+\(placement.minutes)"
    }
}

// MARK: - Effects

nonisolated enum DayEffect: Sendable, Equatable {
    /// Make the OS have exactly these event nudges scheduled.
    case syncNudges([Nudge])
    /// Run one moment: append its request to the Day Thread and generate.
    case runMoment(MomentRequest)
    /// Show a card on a delivery rung (voice is `speak`).
    case presentCard(DayCard, DeliveryRung)
    /// Put a planned step that starts now on the Jarvis Panel.
    case presentStep(StepCue)
    /// Take a card off the panel.
    case retractCard(cardID: String)
    /// Say one line aloud.
    case speak(String)
    /// Post one banner of Jarvis's own (the wind-down).
    case postBanner(title: String, body: String)
    /// Bring an app to the front (a card item's "Open").
    case openApp(name: String)
    /// Change the owner's Reminders.
    case mutateAgenda(AgendaMutation)
    /// "Should I remember this?" proposals for the Profile.
    case proposeFacts([ProposalDraft])
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
    /// Done.
    case complete(reminderID: String)
    /// Due on this day, keeping a time of day if it had one.
    case dueOn(reminderID: String, day: Date)
    /// Due at this moment.
    case dueAt(reminderID: String, at: Date)
    /// A new reminder: a follow-up for something that can't be handled now.
    case followUp(title: String, at: Date)
    /// A new reminder the owner accepted from a proposal: due that day, or
    /// in the Inbox.
    case add(title: String, due: Date?)
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
    /// The app in front.
    var frontmostAppName: String?
    var frontmostBundleID: String?
    /// The app in front is a game: no Triage, no panel, nothing spoken.
    var frontmostIsGame: Bool
    /// When a terminal (where coding agents live) was last in front; now
    /// when one is in front.
    var lastTerminalFrontAt: Date?
    var power: PowerState
    /// The owner's Profile facts (for the reflection's "don't propose again").
    var profile: [String]
    /// The Jarvis Panel is up, with a card or a cue the owner hasn't closed.
    var panelUp: Bool

    init(
        now: Date, calendar: Calendar = .current, settings: DaySettings,
        agenda: AgendaSnapshot, areas: [Area] = [], inboxListID: String? = nil,
        ownerPresent: Bool = true, chatBusy: Bool = false, frontmostAppName: String? = nil,
        frontmostBundleID: String? = nil, frontmostIsGame: Bool = false,
        lastTerminalFrontAt: Date? = nil, power: PowerState = .nominal, profile: [String] = [],
        panelUp: Bool = false
    ) {
        self.now = now
        self.calendar = calendar
        self.settings = settings
        self.agenda = agenda
        self.areas = areas
        self.inboxListID = inboxListID
        self.ownerPresent = ownerPresent
        self.chatBusy = chatBusy
        self.frontmostAppName = frontmostAppName
        self.frontmostBundleID = frontmostBundleID
        self.frontmostIsGame = frontmostIsGame
        self.lastTerminalFrontAt = lastTerminalFrontAt
        self.power = power
        self.profile = profile
        self.panelUp = panelUp
    }

    func facts(state: DayState) -> DayFacts {
        var facts = DayFacts(
            snapshot: agenda, areas: areas, inboxListID: inboxListID, now: now,
            calendar: calendar, mustDoID: state.mustDoID, plan: state.plan,
            weekFocus: state.weekFocus)
        facts.mustDoDays = state.mustDoDays
        facts.departures = state.departures
        if state.mustDoID != nil {
            facts.mustDoDays[state.day.rawValue] = state.mustDoDoneAt != nil
        }
        return facts
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
    /// Say when tomorrow starts, once, as quiet hours begin.
    var windDown: Bool = true
    /// Put a planned step on the panel at its start, and check in at the end
    /// of one the owner started.
    var stepCues: Bool = true
    /// The owner's notification rules.
    var rules: [TriageRule] = []

    /// The overnight gap that makes a return the day's first sit-down.
    static let overnightGap: TimeInterval = 4 * 3600
    /// Triage runs at most this often.
    static let triageInterval: TimeInterval = 10 * 60
    /// A waiting agent is spoken about once the terminal has been out of
    /// sight this long.
    static let agentSpeakAfter: TimeInterval = 2 * 60
    /// Agents that never report back stop waiting after this.
    static let agentExpiry: TimeInterval = 12 * 3600
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
    /// When to leave for the day's events in person, from the plan.
    var departures: [Departure] = []
    /// Last night's carry-over note, for this day's opening.
    var carryOver: String?
    /// Tonight's note, handed to the next day.
    var carryOverForNextDay: String?
    /// Last night's first draft of this day, for its opening.
    var draft: [String] = []
    /// Tonight's first draft of tomorrow, handed to the next day.
    var draftForNextDay: [String] = []

    /// Other apps' banners and what the owner has seen (carried across days:
    /// unresolved items expire on their own).
    var ledger = SeenLedger()
    /// Coding agents that reported in, latest signal per session (carried).
    var agents: [AgentSignal] = []
    /// When each waiting agent was last spoken about.
    var agentSpokenAt: [String: Date] = [:]
    /// The previous clock tick, to notice meetings that just ended.
    var lastTickAt: Date?
    var lastTriageAt: Date?
    /// Where the owner left off: the app in front when they walked away.
    var whereYouWere: String?
    /// Moments the governor held back, to run when the Mac allows.
    var deferred: Set<MomentKind> = []
    /// Event nudges already recorded as delivered (carried: a nudge stays in
    /// Notification Center past midnight).
    var firedNudgeIDs: Set<String> = []
    /// Planned slots whose start was cued, by `StepCue.key`, and whose end
    /// was checked in, by `StepCue.endKey`, with when.
    var cuedSteps: [String: Date] = [:]
    /// Slots the owner started (Start on a cue, Start now on Today), by
    /// `StepCue.key`: their end checks in.
    var startedSteps: Set<String> = []
    /// How often today each task was put off on its cue ("In 15 min").
    var putOff: [String: Int] = [:]
    /// Five-minute starts, by `StepCue.key`: their end asks to keep going.
    var smallStarts: Set<String> = []
    /// The cue on the panel now, by its key, until the owner answers it.
    var cueOnPanel: String?
    /// Tasks the Night Reflection proposed, until the owner decides (kept
    /// that night and the next day).
    var taskProposals: [TaskProposal] = []
    /// When the owner sat down to start this day (the first sit-down after
    /// the night): for them, the morning's end of quiet hours is over.
    var satDownAt: Date?
    /// The week's one focus, from the last week's look-back, and when it was
    /// set (carried a week).
    var weekFocus: String?
    var weekFocusSetAt: Date?
    /// When today's must-do was seen done.
    var mustDoDoneAt: Date?
    /// The last days' must-dos, by day: done or not (days with none are
    /// absent). Carried a week, for the week's look-back.
    var mustDoDays: [String: Bool] = [:]
    /// A moment the app quit in the middle of, until the engine picks it up.
    var interrupted: MomentKind?
    /// The Morning Plan was run again once after a quit cut it short.
    var morningPlanResumed = false
    /// When tonight's wind-down banner went out.
    var windDownAt: Date?

    init(day: DayKey, syncedNudgeIDs: Set<String>? = nil) {
        self.day = day
        self.syncedNudgeIDs = syncedNudgeIDs
    }

    private enum CodingKeys: String, CodingKey {
        case day, syncedNudgeIDs, lastPresentAt, morningPlanAt, eveningWrapUpAt, nightReflectionAt
        case running, cards, mustDoID, plan, carryOver, carryOverForNextDay, ledger, agents
        case agentSpokenAt, lastTickAt, lastTriageAt, whereYouWere, deferred, firedNudgeIDs
        case cuedSteps, startedSteps, interrupted, morningPlanResumed, windDownAt
        case draft, draftForNextDay, departures, satDownAt, weekFocus
        case weekFocusSetAt, mustDoDoneAt, mustDoDays, cueOnPanel, taskProposals
        case putOff, smallStarts
    }

    /// Every field but the day is optional on disk, so a state saved by an
    /// earlier build still loads.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        day = try c.decode(DayKey.self, forKey: .day)
        syncedNudgeIDs = try c.decodeIfPresent(Set<String>.self, forKey: .syncedNudgeIDs)
        lastPresentAt = try c.decodeIfPresent(Date.self, forKey: .lastPresentAt)
        morningPlanAt = try c.decodeIfPresent(Date.self, forKey: .morningPlanAt)
        eveningWrapUpAt = try c.decodeIfPresent(Date.self, forKey: .eveningWrapUpAt)
        nightReflectionAt = try c.decodeIfPresent(Date.self, forKey: .nightReflectionAt)
        running = try c.decodeIfPresent(MomentKind.self, forKey: .running)
        cards = (try? c.decodeIfPresent([DayCard].self, forKey: .cards)) ?? []
        mustDoID = try c.decodeIfPresent(String.self, forKey: .mustDoID)
        plan = (try? c.decodeIfPresent([Placement].self, forKey: .plan)) ?? []
        carryOver = try c.decodeIfPresent(String.self, forKey: .carryOver)
        carryOverForNextDay = try c.decodeIfPresent(String.self, forKey: .carryOverForNextDay)
        ledger = (try? c.decodeIfPresent(SeenLedger.self, forKey: .ledger)) ?? SeenLedger()
        agents = (try? c.decodeIfPresent([AgentSignal].self, forKey: .agents)) ?? []
        agentSpokenAt = (try? c.decodeIfPresent([String: Date].self, forKey: .agentSpokenAt)) ?? [:]
        lastTickAt = try c.decodeIfPresent(Date.self, forKey: .lastTickAt)
        lastTriageAt = try c.decodeIfPresent(Date.self, forKey: .lastTriageAt)
        whereYouWere = try c.decodeIfPresent(String.self, forKey: .whereYouWere)
        deferred = (try? c.decodeIfPresent(Set<MomentKind>.self, forKey: .deferred)) ?? []
        firedNudgeIDs = (try? c.decodeIfPresent(Set<String>.self, forKey: .firedNudgeIDs)) ?? []
        cuedSteps = (try? c.decodeIfPresent([String: Date].self, forKey: .cuedSteps)) ?? [:]
        startedSteps = (try? c.decodeIfPresent(Set<String>.self, forKey: .startedSteps)) ?? []
        interrupted = try? c.decodeIfPresent(MomentKind.self, forKey: .interrupted)
        morningPlanResumed =
            (try? c.decodeIfPresent(Bool.self, forKey: .morningPlanResumed)) ?? false
        windDownAt = try? c.decodeIfPresent(Date.self, forKey: .windDownAt)
        draft = (try? c.decodeIfPresent([String].self, forKey: .draft)) ?? []
        draftForNextDay = (try? c.decodeIfPresent([String].self, forKey: .draftForNextDay)) ?? []
        departures = (try? c.decodeIfPresent([Departure].self, forKey: .departures)) ?? []
        satDownAt = try? c.decodeIfPresent(Date.self, forKey: .satDownAt)
        weekFocus = try? c.decodeIfPresent(String.self, forKey: .weekFocus)
        weekFocusSetAt = try? c.decodeIfPresent(Date.self, forKey: .weekFocusSetAt)
        mustDoDoneAt = try? c.decodeIfPresent(Date.self, forKey: .mustDoDoneAt)
        mustDoDays = (try? c.decodeIfPresent([String: Bool].self, forKey: .mustDoDays)) ?? [:]
        cueOnPanel = try? c.decodeIfPresent(String.self, forKey: .cueOnPanel)
        taskProposals =
            (try? c.decodeIfPresent([TaskProposal].self, forKey: .taskProposals)) ?? []
        putOff = (try? c.decodeIfPresent([String: Int].self, forKey: .putOff)) ?? [:]
        smallStarts = (try? c.decodeIfPresent(Set<String>.self, forKey: .smallStarts)) ?? []
    }

    /// The day as a relaunch finds it: the moment in flight never finished,
    /// so it is recorded as interrupted for the engine to pick up, a card it
    /// was refining keeps the version code built, and a Step Cue left on the
    /// panel is cued again.
    func relaunched() -> DayState {
        var state = self
        // One still waiting from an earlier launch is kept.
        state.interrupted = running ?? interrupted
        state.running = nil
        for index in state.cards.indices { state.cards[index].isRefining = false }
        // The cue on the panel went with the app: it comes back by its rules.
        if let key = state.cueOnPanel {
            state.cuedSteps[key] = nil
            state.cueOnPanel = nil
        }
        return state
    }

    /// The next day's state: what must survive the rollover survives.
    func rolledOver(to day: DayKey) -> DayState {
        var next = DayState(day: day, syncedNudgeIDs: syncedNudgeIDs)
        next.lastPresentAt = lastPresentAt
        next.carryOver = carryOverForNextDay
        // "Last night's draft" only for the morning after it, and tonight's
        // proposed tasks through tomorrow.
        if self.day.next() == day {
            next.draft = draftForNextDay
            // Only this night's: last night's lapse unless tonight made new ones.
            if nightReflectionAt != nil { next.taskProposals = taskProposals }
        }
        // Quiet hours that begin just before 04:00 are still the same night.
        next.windDownAt = windDownAt
        // Whether this day's must-do got done, kept a week for the look-back.
        var days = mustDoDays
        if mustDoID != nil { days[self.day.rawValue] = mustDoDoneAt != nil }
        next.mustDoDays = days.filter { $0.key > Self.weekBefore(day) }
        // The week's focus holds until the next look-back (a week, a day's
        // grace), counted from the day it was set: a look-back after
        // midnight belongs to the evening before.
        if let setAt = weekFocusSetAt, let from = DayKey(for: setAt).date(),
            let to = day.date(),
            let days = Calendar.current.dateComponents([.day], from: from, to: to).day,
            days <= 8
        {
            next.weekFocus = weekFocus
            next.weekFocusSetAt = setAt
        }
        next.ledger = ledger
        next.agents = agents
        next.agentSpokenAt = agentSpokenAt
        next.lastTickAt = lastTickAt
        next.whereYouWere = whereYouWere
        next.firedNudgeIDs = firedNudgeIDs
        return next
    }

    /// The day key a week before `day`: older must-dos leave the record.
    static func weekBefore(_ day: DayKey) -> String {
        guard let date = day.date(),
            let earlier = Calendar.current.date(byAdding: .day, value: -7, to: date)
        else { return "" }
        return DayKey(for: earlier.addingTimeInterval(12 * 3600)).rawValue
    }

    /// Cards the owner hasn't dismissed.
    var openCards: [DayCard] { cards.filter { !$0.dismissed } }

    /// Agents still waiting on the owner (or finished and unreviewed).
    func agentsWaiting(now: Date) -> [AgentSignal] {
        agents.filter { now.timeIntervalSince($0.at) < DaySettings.agentExpiry }
    }
}
