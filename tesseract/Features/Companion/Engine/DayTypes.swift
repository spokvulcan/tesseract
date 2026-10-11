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
    /// Tomorrow on Today (the Now Card, the Day Line, the Inbox): off today's
    /// plan and due tomorrow, as a Step Cue's Tomorrow leaves it.
    case tomorrow(reminderID: String)
    /// Move to Today on a task due tomorrow: into today's Anytime tasks.
    /// Never a repeating reminder, whose date is its whole series'.
    case today(reminderID: String)
    /// Done on Today, or not done after all: the Now Card's Done, the Day
    /// Line's ring, the Inbox's, the chat's plan.
    case tick(reminderID: String, done: Bool)
    /// Put a task into today's plan at a time ("Find a time").
    case place(reminderID: String, start: Date, minutes: Int)
    /// Tasks that slid, back into today's plan in one go ("Fit all 3 in").
    case placeAll([Placement])
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
    /// The owner answered a Break Cue.
    case breakCue(BreakChoice)
    /// The owner took a card in on the panel (Looks Good, Good Night): it
    /// leaves the panel and stays in Today.
    case keep(cardID: String)
    /// The owner closed the panel on a card: taken in, as with Looks Good —
    /// it stays in Today with whatever it holds still to settle (an
    /// evening's leftovers, someone waiting), never back on the panel.
    case close(cardID: String)
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
    /// A meeting, or the way to one, comes before a quarter hour more
    /// would: the slot moves to when it is over and is cued again then —
    /// the rest of a started step, or the whole of one not begun.
    case resume
    /// Not today: due tomorrow, off today's plan.
    case tomorrow
    /// Already done.
    case done
    /// Not done after all (Undo, as the panel says "Done"): the task reopens
    /// and its cue is back on the panel, to choose again.
    case undo
    /// Closed: nothing changes.
    case dismiss
}

/// A planned step at its start or, once the owner started it, at its end,
/// as the Jarvis Panel shows it.
nonisolated struct StepCue: Sendable, Equatable {
    enum Phase: String, Sendable, Equatable, Codable {
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
    /// What comes after it ("Design review at 15:00"), and when it starts
    /// (nil: a step already under way).
    var next: String?
    var nextAt: Date? = nil
    /// What comes next is a call: its link, to join from the panel.
    var nextLink: URL? = nil
    var phase: Phase = .start
    /// Shown well after its moment — the owner was away, busy or behind
    /// another panel — so it says so ("Still time for", "How did it go?").
    var late = false
    /// How often today the owner put this task off ("In 15 min"): from the
    /// second time, its start is offered as five minutes.
    var putOff = 0
    /// A five-minute start ("Start 5 min"): its end asks to keep going.
    var small = false
    /// A meeting, or the way to one, would cut into "In 15 min" or "15 more
    /// min": when it is over, offered instead ("At 16:00", "Go on at 16:00").
    var resumeAt: Date? = nil

    /// The start is offered small: starting is the hard part.
    var offersSmallStart: Bool { phase == .start && putOff >= 2 }

    var end: Date { start.addingTimeInterval(TimeInterval(minutes * 60)) }
}

/// A step the owner started whole and saw done in time: the minutes it
/// had, and the minutes it took.
nonisolated struct StepRun: Sendable, Equatable, Codable {
    var planned: Int
    var actual: Int
    var at: Date
}

/// What the owner chose on a Break Cue.
nonisolated enum BreakChoice: String, Sendable, Equatable {
    /// Up from the Mac now: the time at it starts again from here.
    case taking
    /// Not yet: asked again in half an hour.
    case later
    /// Closed: not asked again for two hours.
    case dismiss
}

/// Two hours at the Mac with no break, as the Jarvis Panel shows it.
nonisolated struct BreakCue: Sendable, Equatable {
    /// When the owner sat down: the last return from five minutes away or
    /// more.
    var since: Date
    /// How long they have been at it.
    var minutes: Int
    /// The day's how-manyth Break Cue, from 1: each says something different.
    var number: Int
    /// The step the owner started, which runs on through the break, and
    /// when it ends.
    var step: String? = nil
    var stepEnd: Date? = nil
}

// MARK: - Effects

nonisolated enum DayEffect: Sendable, Equatable {
    /// Make the OS have exactly these event nudges scheduled.
    case syncNudges([Nudge])
    /// Run one moment: append its request to the Day Thread and generate.
    case runMoment(MomentRequest)
    /// Stop the moment's generation: it went quiet and was given up on.
    case cancelMoment
    /// Show a card on a delivery rung (voice is `speak`).
    case presentCard(DayCard, DeliveryRung)
    /// Put a planned step that starts now on the Jarvis Panel.
    case presentStep(StepCue)
    /// Suggest a break on the Jarvis Panel.
    case presentBreak(BreakCue)
    /// Take the Break Cue off the panel: the owner took one meanwhile.
    case retractBreak
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

    /// The facts a moment is asked about, read from this snapshot and the
    /// day's state: the one projection of the two (`DayFacts`).
    func facts(state: DayState) -> DayFacts {
        DayFacts(snapshot: self, state: state)
    }

    /// Minutes after local midnight.
    var minuteOfDay: Int {
        let parts = calendar.dateComponents([.hour, .minute], from: now)
        return (parts.hour ?? 0) * 60 + (parts.minute ?? 0)
    }

    /// The Evening Wrap-up's window (`DaySettings.isEvening`).
    var isEvening: Bool { settings.isEvening(minuteOfDay: minuteOfDay) }
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
    /// Suggest a break after two hours at the Mac without one.
    var breakCues: Bool = true
    /// The owner's notification rules.
    var rules: [TriageRule] = []

    /// The evening, from the evening time until 03:00: the Evening Wrap-up's
    /// window, and when the Now Card closes the day.
    func isEvening(minuteOfDay minute: Int) -> Bool {
        minute >= eveningMinutes || minute < 3 * 60
    }

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
    /// Today's plan: a step per task, in time order, each with its life.
    var plan: [Step] = []
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
    /// The moment running, as asked, and since when: one that stays quiet
    /// is given up on, and a reply to one given up on doesn't end another.
    var runningRequest: MomentRequest?
    var runningSince: Date?
    /// The Step Cue on the panel now, until the owner answers it.
    var cueOnPanel: CueOnPanel?
    /// Tasks the Night Reflection proposed, until the owner decides (kept
    /// that night and the next day).
    var taskProposals: [TaskProposal] = []
    /// When the owner sat down to start this day (the first sit-down after
    /// the night): for them, the morning's end of quiet hours is over.
    var satDownAt: Date?
    /// The last clock tick with the owner at the Mac (carried). Unlike
    /// `lastPresentAt`, no presence signal stamps it: the Mac going to sleep
    /// reads as a return, and idle is noticed minutes after the last input.
    var lastActiveAt: Date?
    /// Today's first sit-down measured the night before it (`night.ended`).
    var nightMeasured = false
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
    /// The first sit-down wanted a plan while the model was busy (a moment
    /// running, a chat): it runs at the next tick the model is free.
    var morningPlanWaiting = false
    /// When tonight's wind-down banner went out.
    var windDownAt: Date?
    /// When the owner sat down at the Mac: the last return from five minutes
    /// away or more (carried: 04:00 is no break).
    var sittingSince: Date?
    /// No Break Cue before this: "In 30 min", or one closed (carried).
    var breakNotBefore: Date?
    /// The Break Cue on the panel, since when, until the owner answers it.
    var breakCuedAt: Date?
    /// Break Cues shown today: each says something different.
    var breakCues = 0
    /// Planned against actual minutes of the steps the owner started and saw
    /// done in time (carried two weeks, the last 30): the Morning Plan sizes
    /// slots by them.
    var stepRuns: [StepRun] = []

    init(day: DayKey, syncedNudgeIDs: Set<String>? = nil) {
        self.day = day
        self.syncedNudgeIDs = syncedNudgeIDs
    }

    private enum CodingKeys: String, CodingKey {
        case day, syncedNudgeIDs, lastPresentAt, morningPlanAt, eveningWrapUpAt, nightReflectionAt
        case running, cards, mustDoID, plan, carryOver, carryOverForNextDay, ledger, agents
        case agentSpokenAt, lastTickAt, lastTriageAt, whereYouWere, deferred, firedNudgeIDs
        case interrupted, morningPlanResumed, windDownAt
        case draft, draftForNextDay, departures, satDownAt, weekFocus
        case weekFocusSetAt, mustDoDoneAt, mustDoDays, cueOnPanel, taskProposals
        case lastActiveAt, nightMeasured, morningPlanWaiting
        case runningRequest, runningSince, sittingSince, breakNotBefore, breakCuedAt, breakCues
        case stepRuns
    }

    /// Where a state saved before steps held their life kept it: in maps
    /// beside the plan, by task or by slot (`id@epoch`; an end's check-in
    /// `id@epoch+minutes`). Read once, into the steps; never written.
    private enum EarlierStepKeys: String, CodingKey {
        case cuedSteps, startedSteps, putOff, smallStarts, cutShort, startedMinutes
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
        plan = Self.oneStepPerTask((try? c.decodeIfPresent([Step].self, forKey: .plan)) ?? [])
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
        cueOnPanel = try? c.decodeIfPresent(CueOnPanel.self, forKey: .cueOnPanel)
        taskProposals =
            (try? c.decodeIfPresent([TaskProposal].self, forKey: .taskProposals)) ?? []
        runningRequest = try? c.decodeIfPresent(MomentRequest.self, forKey: .runningRequest)
        runningSince = try? c.decodeIfPresent(Date.self, forKey: .runningSince)
        lastActiveAt = try? c.decodeIfPresent(Date.self, forKey: .lastActiveAt)
        nightMeasured = (try? c.decodeIfPresent(Bool.self, forKey: .nightMeasured)) ?? false
        morningPlanWaiting =
            (try? c.decodeIfPresent(Bool.self, forKey: .morningPlanWaiting)) ?? false
        sittingSince = try? c.decodeIfPresent(Date.self, forKey: .sittingSince)
        breakNotBefore = try? c.decodeIfPresent(Date.self, forKey: .breakNotBefore)
        breakCuedAt = try? c.decodeIfPresent(Date.self, forKey: .breakCuedAt)
        breakCues = (try? c.decodeIfPresent(Int.self, forKey: .breakCues)) ?? 0
        stepRuns = (try? c.decodeIfPresent([StepRun].self, forKey: .stepRuns)) ?? []
        let earlier = try decoder.container(keyedBy: EarlierStepKeys.self)
        readEarlierSteps(
            earlier, cueOnPanel: try? c.decodeIfPresent(String.self, forKey: .cueOnPanel))
    }

    /// The plan holds at most one step per task: of two saved for one, the
    /// first, as every reader took it.
    private static func oneStepPerTask(_ steps: [Step]) -> [Step] {
        var seen = Set<String>()
        return steps.filter { seen.insert($0.reminderID).inserted }
    }

    /// A state saved before steps held their life: what its maps knew of
    /// each slot still in the plan goes into its step — the start's marks
    /// by `id@epoch`, the end's check-in by `id@epoch+minutes`, what was put
    /// off or cut by task — and the cue on the panel, saved by its key,
    /// becomes the step's start or end. Marks of slots no longer planned
    /// are let go: nothing read them but the cue's reaction time.
    private mutating func readEarlierSteps(
        _ earlier: KeyedDecodingContainer<EarlierStepKeys>, cueOnPanel cueKey: String?
    ) {
        func read<T: Decodable>(_ type: T.Type, _ key: EarlierStepKeys) -> T? {
            try? earlier.decodeIfPresent(type, forKey: key)
        }
        let cued = read([String: Date].self, .cuedSteps) ?? [:]
        let started = read(Set<String>.self, .startedSteps) ?? []
        let small = read(Set<String>.self, .smallStarts) ?? []
        let putOff = read([String: Int].self, .putOff) ?? [:]
        let cut = read([String: Int].self, .cutShort) ?? [:]
        let timed = read([String: Int].self, .startedMinutes) ?? [:]
        for index in plan.indices {
            let step = plan[index]
            let key = "\(step.reminderID)@\(Int(step.start.timeIntervalSince1970))"
            let endKey = "\(key)+\(step.minutes)"
            if let at = cued[key] { plan[index].cuedAt = at }
            if started.contains(key) { plan[index].isStarted = true }
            if small.contains(key) { plan[index].isSmall = true }
            if let minutes = timed[key] { plan[index].timedMinutes = minutes }
            if let at = cued[endKey] { plan[index].checkedInAt = at }
            if let count = putOff[step.reminderID] { plan[index].putOff = count }
            if let minutes = cut[step.reminderID] { plan[index].cutMinutes = minutes }
            if cueKey == key { cueOnPanel = CueOnPanel(reminderID: step.reminderID, phase: .start) }
            if cueKey == endKey {
                cueOnPanel = CueOnPanel(reminderID: step.reminderID, phase: .end)
            }
        }
    }

    /// The day as a relaunch finds it: the moment in flight never finished,
    /// so it is recorded as interrupted for the engine to pick up, a card it
    /// was refining keeps the version code built, and a Step or Break Cue
    /// left on the panel is cued again.
    func relaunched() -> DayState {
        var state = self
        // One still waiting from an earlier launch is kept.
        state.interrupted = running ?? interrupted
        state.running = nil
        state.runningRequest = nil
        state.runningSince = nil
        for index in state.cards.indices { state.cards[index].isRefining = false }
        // The cue on the panel went with the app: it comes back by its rules.
        state.uncuePanel()
        state.breakCuedAt = nil
        return state
    }

    /// The step `reminderID` names in today's plan.
    func step(_ reminderID: String) -> Step? {
        plan.first { $0.reminderID == reminderID }
    }

    /// The Step Cue on the panel came off it unanswered (a card took the
    /// panel, the app quit, the Companion went off): its step is cued again
    /// by its rules.
    mutating func uncuePanel() {
        if let cue = cueOnPanel,
            let index = plan.firstIndex(where: { $0.reminderID == cue.reminderID })
        {
            plan[index].uncue(cue.phase)
        }
        cueOnPanel = nil
    }

    /// The next day's state: what must survive the rollover survives.
    func rolledOver(to day: DayKey) -> DayState {
        var next = DayState(day: day, syncedNudgeIDs: syncedNudgeIDs)
        next.lastPresentAt = lastPresentAt
        next.lastActiveAt = lastActiveAt
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
        // Two weeks of how long steps took, for the plan's sizing.
        if let date = day.date(),
            let since = Calendar.current.date(
                byAdding: .day, value: -14, to: date)
        {
            next.stepRuns = stepRuns.filter { $0.at >= since }
        }
        // At the Mac across 04:00: the same sitting.
        next.sittingSince = sittingSince
        next.breakNotBefore = breakNotBefore
        next.breakCuedAt = breakCuedAt
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

    /// What the day holds of a repeating task, moved from its series to the
    /// occurrence ticked off today (`DayEngine.followDoneOccurrences`): the
    /// must-do, its step with all its life, and its cue on the panel.
    mutating func follow(series: String, to occurrence: String) {
        if mustDoID == series { mustDoID = occurrence }
        for index in plan.indices where plan[index].reminderID == series {
            plan[index].follow(to: occurrence)
        }
        if cueOnPanel?.reminderID == series { cueOnPanel?.reminderID = occurrence }
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
