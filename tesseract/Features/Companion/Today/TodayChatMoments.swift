//
//  TodayChatMoments.swift
//  tesseract
//
//  The Day Thread's moments as Today's Chat shows them. The Day Opening
//  heads the thread: the day's date, and what Jarvis started the day with
//  one click away. Each moment is a card in the column — its sign on the
//  day's arc (sunrise to moon), its name and time, Jarvis's line as prose,
//  and behind a +/− row what the card holds: the plan, who needed the
//  owner, what got done and what's left, the first draft of tomorrow. While
//  the day's state still holds the card its choices are live, as on the Now
//  Card, and a card with something to decide opens on its own, its edge in
//  the accent until it's settled.
//
//  Content layer (design-language §1): no glass. The chat's one type size;
//  hierarchy from weight and color. A row that exists to be decided shows
//  its offers as small bordered buttons (§2).
//

import SwiftUI

/// Today's Chat row for one moment turn of the Day Thread.
struct ThreadMomentRow: View {
    let request: UserMessage
    let reply: AssistantMessage?

    var body: some View {
        let moment = ThreadMoment(request: request, reply: reply)
        switch moment.kind {
        case .dayOpening:
            DayOpeningRow(moment: moment, digest: DayOpeningDigest(request.content))
        case .moment, .other:
            MomentCardRow(moment: moment, reply: reply)
        case .retry:
            // Its moment's own row shows the card the retry ended with.
            EmptyView()
        }
    }
}

// MARK: - Layout

private enum MomentLayout {
    /// Between a card's header, line and details.
    static let spacing: CGFloat = 10
    /// Between the sections of a card's details.
    static let sectionSpacing: CGFloat = 14
    /// Between the rows of a list in a card's details.
    static let rowSpacing: CGFloat = 6
    static let padding: CGFloat = 14
    static let cornerRadius: CGFloat = 14
    /// The column the moment's sign sits in, so its words line up under the
    /// name.
    static let signWidth: CGFloat = 22
    /// The time column of the plan's steps.
    static let clockWidth: CGFloat = 50
    /// The most of a list a card shows before "and N more".
    static let listed = 5
}

private nonisolated func plural(_ count: Int, _ word: String) -> String {
    "\(count) \(word)\(count == 1 ? "" : "s")"
}

extension MomentKind {
    /// Its sign on the day's arc.
    var symbol: String {
        switch self {
        case .morningPlan: "sunrise"
        case .breakpoint: "figure.walk.arrival"
        case .triage: "bell.badge"
        case .eveningWrapUp: "sunset"
        case .nightReflection: "moon.stars"
        }
    }

    /// What Jarvis is doing while he writes it.
    var inProgress: String {
        switch self {
        case .morningPlan: "Jarvis is planning the day…"
        case .breakpoint: "Jarvis is catching up on what came in…"
        case .triage: "Jarvis is reading what came in…"
        case .eveningWrapUp: "Jarvis is wrapping up the day…"
        case .nightReflection: "Jarvis is looking back over the day…"
        }
    }
}

// MARK: - Day Opening

/// The thread's head: the day's date across the column, then the Day
/// Opening as one quiet row — last night's note in its preview — that
/// opens to what Jarvis started the day with.
private struct DayOpeningRow: View {
    let moment: ThreadMoment
    let digest: DayOpeningDigest
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            DayDivider(
                day: digest.day
                    ?? moment.at.formatted(.dateTime.weekday(.wide).day().month(.wide)))
            VStack(alignment: .leading, spacing: 8) {
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
                } label: {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        CollapseMarker(isExpanded: isExpanded)
                        Text(ThreadMoment.dayOpeningName)
                            .fontWeight(.medium)
                            .foregroundStyle(.secondary)
                        if !isExpanded {
                            Text(preview)
                                .foregroundStyle(.tertiary)
                                .lineLimit(1)
                                .truncationMode(.tail)
                        }
                        Spacer(minLength: 8)
                        Text(AgendaTime.clock(moment.at))
                            .foregroundStyle(.tertiary)
                            .monospacedDigit()
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(
                    isExpanded
                        ? "Hide what Jarvis started the day with"
                        : "Show what Jarvis started the day with")
                if isExpanded {
                    DayOpeningDetails(digest: digest)
                        .padding(.leading, ChatLayout.markerWidth + 8)
                }
            }
        }
        .font(.system(size: chatBodyFontSize))
    }

    /// Last night's note, else what the opening held.
    private var preview: String {
        if let note = digest.carryOver {
            return "From last night: " + note.chatPreviewLine(head: ChatLayout.previewHeadChars)
        }
        var parts: [String] = []
        if !digest.profile.isEmpty { parts.append("your profile") }
        if !digest.areas.isEmpty { parts.append(plural(digest.areas.count, "area")) }
        parts.append(
            digest.calendar.isEmpty
                ? "nothing on the calendar" : plural(digest.calendar.count, "event"))
        if !digest.due.isEmpty { parts.append("\(plural(digest.due.count, "task")) due") }
        return parts.joined(separator: " · ")
    }
}

/// The day's date between two hairlines.
private struct DayDivider: View {
    let day: String

    var body: some View {
        HStack(spacing: 12) {
            rule
            Text(day)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
                .fixedSize()
            rule
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    private var rule: some View {
        Rectangle().fill(.quaternary).frame(height: 1)
    }
}

/// What Jarvis started the day with, section by section.
private struct DayOpeningDetails: View {
    let digest: DayOpeningDigest

    var body: some View {
        VStack(alignment: .leading, spacing: MomentLayout.sectionSpacing) {
            if let note = digest.carryOver {
                MomentSection("From last night") {
                    Text(note)
                        .lineSpacing(chatLineSpacing)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if !digest.draft.isEmpty {
                MomentSection("Last night's draft of today") { Bullets(digest.draft) }
            }
            if let focus = digest.weekFocus {
                MomentSection("This week") { Text(focus) }
            }
            MomentSection("Calendar") {
                if digest.calendar.isEmpty && digest.allDay.isEmpty {
                    Text("Nothing scheduled.")
                }
                ForEach(Array(digest.calendar.enumerated()), id: \.offset) { _, entry in
                    CalendarLine(entry: entry)
                }
                if !digest.allDay.isEmpty {
                    Text("All day: " + digest.allDay.joined(separator: ", "))
                }
            }
            if !digest.due.isEmpty {
                MomentSection("Due") { Bullets(digest.due) }
            }
            if !digest.profile.isEmpty {
                MomentSection("What you've told him") { Bullets(digest.profile) }
            }
            if !digest.areas.isEmpty {
                MomentSection("Areas") { Text(digest.areas.joined(separator: ", ")) }
            }
        }
        .foregroundStyle(.secondary)
        .textSelection(.enabled)
    }
}

/// "09:30–10:00 Standup": the times in their own column.
private struct CalendarLine: View {
    let entry: String

    var body: some View {
        let split = entry.firstIndex(of: " ")
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            if let split {
                Text(entry[..<split]).monospacedDigit().foregroundStyle(.tertiary)
                Text(entry[entry.index(after: split)...])
            } else {
                Text(entry)
            }
        }
    }
}

// MARK: - Moment card

/// Finds the card a moment made in the day's state and draws it.
private struct MomentCardRow: View {
    @Environment(CompanionRuntime.self) private var runtime
    @Environment(DayThread.self) private var thread
    let moment: ThreadMoment
    let reply: AssistantMessage?

    var body: some View {
        // The day's state holds the cards of the thread's own day only.
        let live =
            runtime.state.day == thread.day
            ? moment.card(in: runtime.state.cards, turns: turns) : nil
        let content = live?.body ?? fromReply
        // A Triage that raised nothing made no card and reached no one.
        if live != nil
            || (moment.kind != .moment(.triage) && (moment.line != nil || content != nil))
        {
            MomentCard(moment: moment, live: live, content: content)
        }
    }

    /// The thread's moment turns, by their headers alone.
    private var turns: [ThreadMoment] {
        thread.chat.items.compactMap { item in
            guard case .moment(let request, _) = item else { return nil }
            return ThreadMoment(request: request, reply: nil)
        }
    }

    /// With no card in the state, what the reply alone can say: a night's
    /// reflection holds all of its card.
    private var fromReply: DayCard.Body? {
        guard moment.kind == .moment(.nightReflection), let reply,
            case .card(let body) = CardParser.nightReflection(reply.text)
        else { return nil }
        return body
    }
}

private struct MomentCard: View {
    @Environment(CompanionRuntime.self) private var runtime
    @Environment(ProfileStore.self) private var profile
    @Environment(Agenda.self) private var agenda
    let moment: ThreadMoment
    /// The card in the day's state: its choices act on it.
    let live: DayCard?
    /// What the card holds.
    let content: DayCard.Body?
    /// The owner's own open or closed; nil follows what waits on them.
    @State private var expanded: Bool?

    var body: some View {
        let waiting = Waiting(
            content: content, state: runtime.state, agenda: agenda, profile: profile)
        let summary = content.map {
            MomentCardSummary(
                $0, live: isLive, tasksWaiting: waiting.tasks.count,
                factsWaiting: waiting.facts.count, remembered: waiting.remembered.count,
                mustDo: mustDoTitle)
        }
        let isExpanded = expanded ?? (summary?.waiting.isEmpty == false)

        MomentCardShell(symbol: symbol, waits: summary?.waiting.isEmpty == false) {
            header
            if let line {
                Text(line)
                    .lineSpacing(chatLineSpacing)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if live?.isRefining == true {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.mini)
                    Text("Jarvis is still thinking it through.").foregroundStyle(.secondary)
                }
            }
            if let content, let summary, !summary.isEmpty {
                disclosure(summary, isExpanded: isExpanded)
                if isExpanded {
                    MomentDetails(
                        content: content, cardID: live?.id, isLive: isLive, waiting: waiting
                    )
                    .padding(.leading, ChatLayout.markerWidth + 8)
                    .transition(.opacity)
                }
            }
        }
        .contextMenu {
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(copyText, forType: .string)
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(title), \(AgendaTime.clock(moment.at))")
    }

    /// Still in the day's state, not replaced or dismissed: its choices act.
    private var isLive: Bool { live.map { !$0.dismissed } ?? false }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(title).fontWeight(.semibold).layoutPriority(1)
            if let away {
                Text(away).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 8)
            Text(AgendaTime.clock(moment.at))
                .foregroundStyle(.tertiary)
                .monospacedDigit()
        }
    }

    private func disclosure(_ summary: MomentCardSummary, isExpanded: Bool) -> some View {
        Button {
            withAnimation(.easeOut(duration: 0.15)) { expanded = !isExpanded }
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                CollapseMarker(isExpanded: isExpanded)
                summaryText(summary)
                Spacer(minLength: 0)
            }
            .lineLimit(1)
            .truncationMode(.tail)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(isExpanded ? "Hide what the card holds" : "Show what the card holds")
    }

    /// What waits first, in the accent, then the rest.
    private func summaryText(_ summary: MomentCardSummary) -> Text {
        let waiting = Text(summary.waiting.joined(separator: " · "))
            .fontWeight(.medium)
            .foregroundStyle(Color.accentColor)
        let parts = Text(summary.parts.joined(separator: " · ")).foregroundStyle(.secondary)
        switch (summary.waiting.isEmpty, summary.parts.isEmpty) {
        case (true, _): return parts
        case (false, true): return waiting
        case (false, false):
            return Text("\(waiting)\(Text(" · ").foregroundStyle(.secondary))\(parts)")
        }
    }

    private var kind: MomentKind? {
        if case .moment(let kind) = moment.kind { return kind }
        return nil
    }

    private var title: String {
        switch moment.kind {
        case .moment(let kind): kind.title
        case .other(let name): name
        case .dayOpening: ThreadMoment.dayOpeningName
        case .retry: ""
        }
    }

    private var symbol: String { kind?.symbol ?? "sparkle" }

    /// What the owner saw: the card's line, else the reply's.
    private var line: String? {
        if let line = live?.line, !line.isEmpty { return line }
        return moment.line
    }

    /// How long a welcome back's absence was: "45 min".
    private var away: String? {
        if case .breakpoint(let card)? = content {
            let minutes = Int(card.awayUntil.timeIntervalSince(card.awayFrom) / 60)
            return MomentPrompts.minutesText(max(minutes, 1))
        }
        // The request says it too: "away 45 min, 13:20–14:05".
        guard moment.kind == .moment(.breakpoint), let detail = moment.detail else { return nil }
        let span = detail.prefix { $0 != "," }
        return String(span.hasPrefix("away ") ? span.dropFirst(5) : span)
    }

    private var mustDoTitle: String? {
        guard case .morningPlan(let plan)? = content, let id = plan.mustDoID else { return nil }
        return (agenda.snapshot.open + agenda.snapshot.doneToday).first { $0.id == id }?.title
    }

    private var copyText: String {
        [title + " · " + AgendaTime.clock(moment.at), line].compactMap(\.self)
            .joined(separator: "\n")
    }
}

/// A moment card's frame: its sign in a column of its own and the words
/// beside it, on the content layer's fill, the edge in the accent while
/// something in it waits on the owner.
private struct MomentCardShell<Content: View>: View {
    let symbol: String
    let waits: Bool
    let content: Content

    init(symbol: String, waits: Bool = false, @ViewBuilder content: () -> Content) {
        self.symbol = symbol
        self.waits = waits
        self.content = content()
    }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: MomentLayout.cornerRadius, style: .continuous)
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: symbol)
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(Color.accentColor)
                .fontWeight(.medium)
                .frame(width: MomentLayout.signWidth)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: MomentLayout.spacing) { content }
        }
        .font(.system(size: chatBodyFontSize))
        .padding(MomentLayout.padding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quinary, in: shape)
        .overlay {
            shape.strokeBorder(
                waits ? AnyShapeStyle(Color.accentColor.opacity(0.45)) : AnyShapeStyle(.quaternary),
                lineWidth: 1)
        }
        .animation(.easeOut(duration: 0.2), value: waits)
    }
}

// MARK: - In progress

/// The moment Jarvis is writing now, at the foot of the thread: its sign and
/// name and what he's doing, until it joins the thread whole. A Triage stays
/// quiet: most raise nothing.
struct MomentInProgressRow: View {
    @Environment(DayThread.self) private var thread

    var body: some View {
        if let kind = thread.momentRunning, kind != .triage {
            MomentCardShell(symbol: kind.symbol) {
                Text(kind.title).fontWeight(.semibold)
                HStack(spacing: 8) {
                    ProgressView().controlSize(.mini)
                    Text(kind.inProgress).foregroundStyle(.secondary)
                }
            }
            .transition(.opacity)
        }
    }
}

// MARK: - Summary

/// What a moment's card says behind its +/− row, in a few words, and what
/// in it waits on the owner (which opens the card, in the accent).
nonisolated struct MomentCardSummary: Sendable, Equatable {
    /// "2 to settle".
    var waiting: [String] = []
    /// "4 done", "tomorrow starts 09:00 Work".
    var parts: [String] = []

    var isEmpty: Bool { waiting.isEmpty && parts.isEmpty }

    /// - Parameters:
    ///   - live: the card is still in the day's state: what it holds still
    ///     waits on the owner.
    ///   - tasksWaiting, factsWaiting: a reflection's tasks and facts the
    ///     owner hasn't decided; remembered: its facts they kept.
    ///   - mustDo: the title of a plan's must-do.
    init(
        _ content: DayCard.Body, live: Bool, tasksWaiting: Int = 0, factsWaiting: Int = 0,
        remembered: Int = 0, mustDo: String? = nil
    ) {
        switch content {
        case .morningPlan(let plan):
            if !plan.placements.isEmpty {
                parts.append(plural(plan.placements.count, "step") + " planned")
            }
            if let mustDo { parts.append("must-do: \(mustDo)") }
            if let leave = plan.departures.first {
                parts.append("leave at \(AgendaTime.clock(leave.at)) for \(leave.title)")
            }
            if !plan.suggestions.isEmpty { parts.append(plural(plan.suggestions.count, "tip")) }
        case .breakpoint(let card):
            let needs = card.needsYou.count
            if needs > 0 {
                if live {
                    waiting.append("\(needs) need\(needs == 1 ? "s" : "") you")
                } else {
                    parts.append("\(needs) needed you")
                }
            }
            if let next = card.next.first { parts.append("next: \(next.title)") }
            if card.canWaitCount > 0 { parts.append("\(card.canWaitCount) could wait") }
        case .triage(let card):
            let raised = card.raise.count
            if raised > 0 {
                if live {
                    waiting.append("\(raised) need\(raised == 1 ? "s" : "") you")
                } else {
                    parts.append("\(raised) raised")
                }
            }
        case .eveningWrapUp(let card):
            if !card.leftovers.isEmpty {
                if live {
                    waiting.append("\(card.leftovers.count) to settle")
                } else {
                    parts.append("\(card.leftovers.count) left")
                }
            }
            parts.append(card.done.isEmpty ? "nothing ticked off" : "\(card.done.count) done")
            if let first = card.tomorrowFirst { parts.append("tomorrow starts \(first)") }
            if let focus = card.focus { parts.append("next week: \(focus)") }
        case .reflection(let card):
            if tasksWaiting > 0 { waiting.append(plural(tasksWaiting, "task") + " to add") }
            if factsWaiting > 0 { waiting.append("\(factsWaiting) to remember") }
            if !card.tomorrow.isEmpty { parts.append("\(card.tomorrow.count) for tomorrow") }
            if remembered > 0 { parts.append("\(remembered) remembered") }
        }
    }
}

// MARK: - What waits

/// A night's tasks and facts as they stand: those still to decide, and the
/// facts already remembered.
private struct Waiting {
    var tasks: [TaskProposal] = []
    var facts: [FactProposal] = []
    var remembered: [FactProposal] = []

    @MainActor
    init(content: DayCard.Body?, state: DayState, agenda: Agenda, profile: ProfileStore) {
        guard case .reflection(let card)? = content else { return }
        let ids = Set(card.tasks.map(\.id))
        // One the owner wrote down meanwhile is no longer a question.
        let open = Set(agenda.snapshot.open.map { $0.title.lowercased() })
        tasks = state.taskProposals.filter {
            ids.contains($0.id) && !open.contains($0.title.lowercased())
        }
        let texts = Set(card.proposals.map { $0.text.lowercased() })
        let proposed = profile.proposals.filter { texts.contains($0.text.lowercased()) }
        facts = proposed.filter { $0.status == .open }
        remembered = proposed.filter { $0.status == .kept || $0.status == .edited }
    }
}

// MARK: - Details

/// What a card holds, by its kind.
private struct MomentDetails: View {
    let content: DayCard.Body
    let cardID: String?
    let isLive: Bool
    let waiting: Waiting

    var body: some View {
        Group {
            switch content {
            case .morningPlan(let plan):
                PlanDetails(plan: plan)
            case .breakpoint(let card):
                AwayDetails(card: card, cardID: cardID, isLive: isLive)
            case .triage(let card):
                if !card.raise.isEmpty {
                    MomentSection(isLive ? "Needs you" : "Raised") {
                        ForEach(card.raise) { item in
                            WaitingItemRow(item: item, cardID: isLive ? cardID : nil)
                        }
                    }
                }
            case .eveningWrapUp(let wrapUp):
                WrapUpDetails(wrapUp: wrapUp, cardID: isLive ? cardID : nil)
            case .reflection(let reflection):
                ReflectionDetails(reflection: reflection, waiting: waiting)
            }
        }
    }
}

/// A titled part of a card's details.
private struct MomentSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MomentLayout.rowSpacing) {
            Text(title).fontWeight(.medium).foregroundStyle(.secondary)
            content
        }
    }
}

private struct Bullets: View {
    let lines: [String]

    init(_ lines: [String]) { self.lines = lines }

    var body: some View {
        ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("·")
                Text(linkedTitle: line).fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// The day's decisions take small bordered buttons, Jarvis's pick in the
/// accent (design-language §2).
private struct ChoiceButtons: ViewModifier {
    func body(content: Content) -> some View {
        content
            .buttonStyle(.bordered)
            .controlSize(.small)
            .font(nil)
            .focusable(false)
    }
}

// MARK: Morning Plan

private struct PlanDetails: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    let plan: MorningPlanCard

    var body: some View {
        let actions = TodayActions(agenda: agenda, runtime: runtime)
        let reminders = Dictionary(
            (agenda.snapshot.open + agenda.snapshot.doneToday).map { ($0.id, $0) },
            uniquingKeysWith: { first, _ in first })
        VStack(alignment: .leading, spacing: MomentLayout.sectionSpacing) {
            let steps = plan.placements.compactMap { slot in
                reminders[slot.reminderID].map { PlanEntry(slot: slot, reminder: $0) }
            }
            if !steps.isEmpty {
                MomentSection("The plan") {
                    ForEach(steps) { step in
                        let slot = step.slot
                        let reminder = step.reminder
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text(AgendaTime.clock(slot.start))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                                .frame(width: MomentLayout.clockWidth, alignment: .leading)
                            Button {
                                actions.setDone(reminder.id, !reminder.isCompleted)
                            } label: {
                                Image(
                                    systemName: reminder.isCompleted
                                        ? "checkmark.circle.fill" : "circle"
                                )
                                .foregroundStyle(.secondary)
                                .contentTransition(.symbolEffect(.replace))
                            }
                            .buttonStyle(.plain)
                            .focusable(false)
                            .help(reminder.isCompleted ? "Mark as not done" : "Mark as done")
                            Text(
                                linkedTitle: reminder.title,
                                linkColor: reminder.isCompleted ? .secondary : .accentColor
                            )
                            .foregroundStyle(reminder.isCompleted ? .secondary : .primary)
                            .lineLimit(1)
                            if slot.reminderID == plan.mustDoID {
                                Image(systemName: "star.fill")
                                    .foregroundStyle(Color.accentColor)
                                    .help("The day's must-do")
                            }
                            Spacer(minLength: 8)
                            Text(MomentPrompts.minutesText(slot.minutes))
                                .foregroundStyle(.tertiary)
                                .lineLimit(1)
                        }
                    }
                }
            }
            if !plan.departures.isEmpty {
                MomentSection("Leave") {
                    ForEach(plan.departures, id: \.self) { leave in
                        Text(
                            "At \(AgendaTime.clock(leave.at)) for \(leave.title)"
                                + (leave.location.map { " · \($0)" } ?? ""))
                    }
                }
            }
            if !plan.suggestions.isEmpty {
                MomentSection("Tips") { Bullets(plan.suggestions) }
            }
        }
    }
}

/// A step of the plan, with the task it placed.
private struct PlanEntry: Identifiable {
    let slot: Placement
    let reminder: AgendaReminder
    var id: String { slot.reminderID }
}

// MARK: Breakpoint and Triage

private struct AwayDetails: View {
    let card: BreakpointCard
    let cardID: String?
    let isLive: Bool
    @State private var showQuiet = false

    var body: some View {
        VStack(alignment: .leading, spacing: MomentLayout.sectionSpacing) {
            if !card.needsYou.isEmpty {
                MomentSection(isLive ? "Needs you" : "Needed you") {
                    ForEach(card.needsYou) { item in
                        WaitingItemRow(item: item, cardID: isLive ? cardID : nil)
                    }
                }
            }
            if !card.next.isEmpty {
                MomentSection("Next") {
                    ForEach(card.next) { next in
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text(next.at.map { AgendaTime.clock($0) } ?? "")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                                .frame(width: MomentLayout.clockWidth, alignment: .leading)
                            Text(linkedTitle: next.title).lineLimit(1)
                            Spacer(minLength: 8)
                            if let minutes = next.minutes {
                                Text(MomentPrompts.minutesText(minutes)).foregroundStyle(.tertiary)
                            }
                        }
                    }
                }
            }
            if let place = card.whereYouWere {
                Text("You were in \(place) when you left.").foregroundStyle(.secondary)
            }
            if card.canWaitCount > 0 {
                VStack(alignment: .leading, spacing: MomentLayout.rowSpacing) {
                    Button {
                        withAnimation(.easeOut(duration: 0.15)) { showQuiet.toggle() }
                    } label: {
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            CollapseMarker(isExpanded: showQuiet)
                            Text(
                                "\(plural(card.canWaitCount, "other notification")) could wait"
                            )
                            .foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    if showQuiet {
                        ForEach(card.canWait) { group in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(group.app).fontWeight(.medium)
                                ForEach(Array(group.lines.enumerated()), id: \.offset) { _, line in
                                    Text(line).foregroundStyle(.secondary).lineLimit(2)
                                }
                            }
                            .padding(.leading, ChatLayout.markerWidth + 8)
                        }
                    }
                }
            }
        }
    }
}

/// Someone or something that needs the owner: who or what, a line of
/// detail, and — while the card is live — its ways on.
private struct WaitingItemRow: View {
    @Environment(CompanionRuntime.self) private var runtime
    let item: WaitingItem
    /// The live card's id: its choices act on it.
    let cardID: String?

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                words
                Spacer(minLength: 8)
                choices
            }
            VStack(alignment: .leading, spacing: MomentLayout.rowSpacing) {
                words
                choices
            }
        }
    }

    private var words: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(item.title).fontWeight(.medium).lineLimit(1)
            Text(item.detail).foregroundStyle(.secondary).lineLimit(2)
        }
    }

    @ViewBuilder private var choices: some View {
        if let cardID {
            HStack(spacing: 6) {
                if item.app != nil {
                    Button("Open") { runtime.act(.openItem(cardID: cardID, itemID: item.id)) }
                        .tint(.accentColor)
                }
                if item.kind != .agent {
                    Button("Later") { runtime.act(.itemLater(cardID: cardID, itemID: item.id)) }
                        .help("Remind me in half an hour")
                }
                Button("Done") { runtime.act(.itemDone(cardID: cardID, itemID: item.id)) }
            }
            .modifier(ChoiceButtons())
            .fixedSize()
        }
    }
}

// MARK: Evening Wrap-up

private struct WrapUpDetails: View {
    @Environment(CompanionRuntime.self) private var runtime
    let wrapUp: EveningWrapUpCard
    /// The live card's id: the leftovers' choices act on it.
    let cardID: String?

    var body: some View {
        VStack(alignment: .leading, spacing: MomentLayout.sectionSpacing) {
            if !wrapUp.leftovers.isEmpty {
                MomentSection(wrapUp.leftoversHeading) {
                    ForEach(wrapUp.leftovers) { leftover in
                        LeftoverChoiceRow(leftover: leftover, cardID: cardID)
                    }
                    if let cardID, wrapUp.leftovers.count > 1 {
                        Button("Do What Jarvis Suggests for All") {
                            runtime.act(.allLeftovers(cardID: cardID))
                        }
                        .modifier(ChoiceButtons())
                        .padding(.top, 2)
                    }
                }
            }
            if !wrapUp.done.isEmpty {
                MomentSection("Done today") {
                    ForEach(
                        Array(wrapUp.done.prefix(MomentLayout.listed).enumerated()), id: \.offset
                    ) {
                        _, title in
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(Color.accentColor)
                            Text(linkedTitle: title).lineLimit(1)
                        }
                    }
                    if wrapUp.done.count > MomentLayout.listed {
                        Text("and \(wrapUp.done.count - MomentLayout.listed) more")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            if wrapUp.week != nil || wrapUp.focus != nil {
                MomentSection("This week") {
                    if let week = wrapUp.week {
                        Text(week).fixedSize(horizontal: false, vertical: true)
                    }
                    if let focus = wrapUp.focus {
                        Text("Next week: \(focus)").foregroundStyle(Color.accentColor)
                    }
                }
            }
            if let first = wrapUp.tomorrowFirst {
                Text("Tomorrow starts with \(first).").foregroundStyle(.secondary)
            }
        }
    }
}

/// One task left from the day, and its ways on: tomorrow, another day,
/// later, or let go — Jarvis's pick in the accent.
private struct LeftoverChoiceRow: View {
    @Environment(CompanionRuntime.self) private var runtime
    let leftover: Leftover
    /// The live card's id: the choices act on it.
    let cardID: String?

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                words.lineLimit(1)
                Spacer(minLength: 8)
                choices
            }
            VStack(alignment: .leading, spacing: MomentLayout.rowSpacing) {
                words.lineLimit(2)
                choices
            }
        }
    }

    private var words: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(linkedTitle: leftover.title)
            if let waiting = leftover.waiting {
                Text(waiting).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var choices: some View {
        if let cardID {
            HStack(spacing: 6) {
                choice("Tomorrow", .tomorrow, cardID)
                Menu("Another day") {
                    // From the owner's today: until 04:00, the day that is ending.
                    let today =
                        DayKey(for: Date()).date() ?? Calendar.current.startOfDay(for: Date())
                    ForEach(2..<8, id: \.self) { offset in
                        let day =
                            Calendar.current.date(byAdding: .day, value: offset, to: today) ?? today
                        Button(day.formatted(.dateTime.weekday(.wide).day().month())) {
                            runtime.act(
                                .leftoverOn(
                                    cardID: cardID, reminderID: leftover.reminderID, day: day))
                        }
                    }
                }
                .menuStyle(.button)
                .fixedSize()
                choice("Later", .later, cardID)
                choice("Let go", .drop, cardID)
            }
            .modifier(ChoiceButtons())
            .fixedSize()
        } else {
            Text("Jarvis suggested \(suggested)").foregroundStyle(.tertiary)
        }
    }

    private func choice(_ title: String, _ suggestion: Leftover.Suggestion, _ cardID: String)
        -> some View
    {
        Button(title) {
            runtime.act(.leftover(cardID: cardID, reminderID: leftover.reminderID, suggestion))
        }
        .tint(leftover.suggestion == suggestion ? .accentColor : nil)
    }

    private var suggested: String {
        switch leftover.suggestion {
        case .tomorrow: "tomorrow"
        case .later: "later"
        case .drop: "letting it go"
        }
    }
}

// MARK: Night Reflection

private struct ReflectionDetails: View {
    @Environment(CompanionRuntime.self) private var runtime
    @Environment(ProfileStore.self) private var profile
    let reflection: ReflectionCard
    let waiting: Waiting

    var body: some View {
        VStack(alignment: .leading, spacing: MomentLayout.sectionSpacing) {
            if !reflection.tomorrow.isEmpty {
                MomentSection("For tomorrow") { Bullets(reflection.tomorrow) }
            }
            if !waiting.tasks.isEmpty {
                MomentSection("Add to your tasks?") {
                    ForEach(waiting.tasks) { task in
                        DecisionRow(
                            text: task.title, detail: due(task), yes: "Add", no: "No",
                            onYes: { runtime.act(.taskProposal(id: task.id, add: true)) },
                            onNo: { runtime.act(.taskProposal(id: task.id, add: false)) })
                    }
                }
            }
            if !waiting.facts.isEmpty || !waiting.remembered.isEmpty {
                MomentSection("Should I remember this?") {
                    ForEach(waiting.facts) { fact in
                        DecisionRow(
                            text: fact.text, detail: fact.reason.isEmpty ? nil : fact.reason,
                            yes: "Remember", no: "Not true",
                            onYes: { profile.decide(fact.id, keep: true) },
                            onNo: { profile.decide(fact.id, keep: false) })
                    }
                    ForEach(waiting.remembered) { fact in
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(Color.accentColor)
                            Text(fact.text).fixedSize(horizontal: false, vertical: true)
                        }
                        .help("Remembered in your Profile")
                    }
                }
            }
        }
    }

    private func due(_ task: TaskProposal) -> String {
        guard let due = task.due else { return "Into the Inbox" }
        return "Due \(due.formatted(.dateTime.weekday(.wide)))"
    }
}

/// Something Jarvis asks the owner to decide: what, why or when, and a yes
/// and a no.
private struct DecisionRow: View {
    let text: String
    let detail: String?
    let yes: String
    let no: String
    let onYes: () -> Void
    let onNo: () -> Void

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                words
                Spacer(minLength: 8)
                choices
            }
            VStack(alignment: .leading, spacing: MomentLayout.rowSpacing) {
                words
                choices
            }
        }
    }

    private var words: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(linkedTitle: text).fixedSize(horizontal: false, vertical: true)
            if let detail {
                Text(detail).foregroundStyle(.secondary).fixedSize(
                    horizontal: false, vertical: true)
            }
        }
    }

    private var choices: some View {
        HStack(spacing: 6) {
            Button(yes, action: onYes).tint(.accentColor)
            Button(no, action: onNo)
        }
        .modifier(ChoiceButtons())
        .fixedSize()
    }
}
