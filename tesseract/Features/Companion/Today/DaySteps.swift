//
//  DaySteps.swift
//  tesseract
//
//  The day as steps. A wide page shows a table (time, task, area, length);
//  a narrow one shows a list shaped for a phone. One Day Line runs down the
//  schedule, walked up to the Now line and ahead after it, with each step's
//  marker on it: a check for a task, the calendar's color for an event, the
//  accent ring for now. The day's Anytime tasks close it; then the line runs
//  on past the day break into tomorrow, so what comes next is always in
//  sight. Row actions live in context menus (design-language §2).
//

import SwiftUI

struct DaySteps: View {
    let timeline: TodayTimeline
    let style: TodayLayout.StepStyle

    var body: some View {
        let steps = Self.steps(timeline)
        let nowIndex = steps.firstIndex { $0.kind == .now } ?? steps.count
        let nowStart = timeline.rows.first { $0.kind == .now }?.start
        // The line runs from the day's first timed step to tomorrow's last.
        let first = steps.firstIndex(where: \.isOnLine)
        let last = steps.lastIndex(where: \.isOnLine)
        VStack(alignment: .leading, spacing: 0) {
            if style == .table {
                ColumnHeader()
            }
            ForEach(Array(steps.enumerated()), id: \.element.id) { index, step in
                let line: DayLine? =
                    if let first, let last, step.isOnLine, index >= first, index <= last {
                        DayLine(
                            above: index == first ? nil : index <= nowIndex ? .walked : .ahead,
                            below: index == last ? nil : index < nowIndex ? .walked : .ahead)
                    } else {
                        nil
                    }
                switch step.kind {
                case .allDay(let event, _):
                    EventStep(event: event, isPast: false, isAllDay: true, style: style, line: line)
                case .event(let event, let isPast):
                    EventStep(
                        event: event, isPast: isPast, isAllDay: false, style: style, line: line)
                case .task(let task, let isTomorrow):
                    TaskStep(task: task, isTomorrow: isTomorrow, style: style, line: line)
                case .free(let start, let minutes):
                    // Free time from now on needs no time: the Now line has it.
                    FreeStep(
                        start: start, minutes: minutes, showsTime: start != nowStart,
                        style: style, line: line)
                case .now:
                    NowStep(time: nowStart ?? Date(), style: style, line: line)
                case .heading(let title):
                    HeadingStep(title: title, line: line)
                case .dayBreak(let date):
                    DayBreakStep(date: date, style: style, line: line)
                case .nothingTomorrow:
                    StepRow(line: nil, highlights: false) {
                        Color.clear
                    } cells: {
                        StepCells(style: style, time: nil) {
                            Text("Nothing on tomorrow yet.").foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    /// The two days as one run of steps: today's all-day events, its
    /// timeline and its Anytime tasks, the day break, then tomorrow the
    /// same way.
    static func steps(_ timeline: TodayTimeline) -> [Step] {
        var steps = timeline.allDayEvents.map {
            Step(id: "allday-\($0.id)", kind: .allDay($0, isTomorrow: false))
        }
        for row in timeline.rows {
            steps.append(Step(row: row, isTomorrow: false))
        }
        if !timeline.anytime.isEmpty {
            steps.append(Step(id: "anytime-today", kind: .heading("Anytime today")))
            steps += timeline.anytime.map {
                Step(id: "task-\($0.id)", kind: .task($0, isTomorrow: false))
            }
        }
        let tomorrow = timeline.tomorrow
        steps.append(Step(id: "day-break", kind: .dayBreak(tomorrow.date)))
        steps += tomorrow.allDayEvents.map {
            Step(id: "tomorrow-allday-\($0.id)", kind: .allDay($0, isTomorrow: true))
        }
        for row in tomorrow.rows {
            steps.append(Step(row: row, isTomorrow: true))
        }
        if !tomorrow.anytime.isEmpty {
            steps.append(Step(id: "anytime-tomorrow", kind: .heading("Anytime tomorrow")))
            steps += tomorrow.anytime.map {
                Step(id: "tomorrow-task-\($0.id)", kind: .task($0, isTomorrow: true))
            }
        }
        if tomorrow.isEmpty {
            steps.append(Step(id: "nothing-tomorrow", kind: .nothingTomorrow))
        }
        return steps
    }

    /// One row of the two days.
    struct Step: Identifiable, Equatable {
        enum Kind: Equatable {
            case allDay(AgendaEvent, isTomorrow: Bool)
            case event(AgendaEvent, isPast: Bool)
            case task(TimelineTask, isTomorrow: Bool)
            case free(start: Date, minutes: Int)
            case now
            /// "Anytime today", "Anytime tomorrow".
            case heading(String)
            /// Where today hands over to tomorrow (at midnight).
            case dayBreak(Date)
            case nothingTomorrow
        }

        let id: String
        let kind: Kind

        /// Today's all-day events sit above the line, which starts at the
        /// day's first timed step; the note under an empty tomorrow sits
        /// below it.
        var isOnLine: Bool {
            switch kind {
            case .allDay(_, let isTomorrow): isTomorrow
            case .nothingTomorrow: false
            default: true
            }
        }
    }
}

extension DaySteps.Step {
    init(row: TimelineRow, isTomorrow: Bool) {
        let kind: Kind =
            switch row.kind {
            case .event(let event): .event(event, isPast: row.isPast)
            case .task(let task): .task(task, isTomorrow: isTomorrow)
            case .free(let minutes): .free(start: row.start, minutes: minutes)
            case .now: .now
            }
        self.init(id: (isTomorrow ? "tomorrow-" : "") + row.id, kind: kind)
    }
}

// MARK: - The Day Line

/// The Day Line's two segments through one row, above and below its marker.
private struct DayLine: Equatable {
    enum Segment {
        /// Before now: the part of the day already walked.
        case walked
        case ahead
    }

    var above: Segment?
    var below: Segment?
}

private struct DayLineSegments: View {
    let line: DayLine?
    /// Free time has no marker: the line runs straight through.
    var straight = false
    var topSpace: CGFloat = 0

    var body: some View {
        VStack(spacing: 0) {
            segment(line?.above).frame(height: topSpace + TodayLayout.markerTop)
            Group {
                if straight { segment(line?.above) } else { Color.clear }
            }
            .frame(height: TodayLayout.markerSize)
            segment(line?.below).frame(maxHeight: .infinity)
        }
        .frame(width: TodayLayout.lineColumnWidth)
    }

    @ViewBuilder private func segment(_ segment: DayLine.Segment?) -> some View {
        switch segment {
        case .walked: Rectangle().fill(.secondary).frame(width: 2)
        case .ahead: Rectangle().fill(.quaternary).frame(width: 2)
        case nil: Color.clear.frame(width: 2)
        }
    }
}

/// One step: its marker on the Day Line, then its cells; highlighted under
/// the pointer, like a table row.
private struct StepRow<Marker: View, Cells: View>: View {
    let line: DayLine?
    var straight = false
    /// Room above the row, with the line running through it: a heading or
    /// the day break opens a group.
    var topSpace: CGFloat = 0
    var highlights = true
    @ViewBuilder let marker: Marker
    @ViewBuilder let cells: Cells
    @State private var hovering = false

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            marker
                .frame(width: TodayLayout.lineColumnWidth, height: TodayLayout.markerSize)
                .padding(.top, TodayLayout.markerTop)
            cells
                .padding(.vertical, TodayLayout.rowPadding)
                .padding(.trailing, 8)
        }
        .padding(.top, topSpace)
        .background(alignment: .leading) {
            if line != nil {
                DayLineSegments(line: line, straight: straight, topSpace: topSpace)
            }
        }
        .background(
            .quaternary.opacity(highlights && hovering ? 0.5 : 0),
            in: RoundedRectangle(cornerRadius: Theme.Radius.small)
        )
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
    }
}

// MARK: - Cells

private struct ColumnHeader: View {
    var body: some View {
        HStack(spacing: TodayLayout.cellSpacing) {
            Text("Time").frame(width: TodayLayout.timeWidth, alignment: .leading)
            Text("Task").frame(maxWidth: .infinity, alignment: .leading)
            Text("Area").frame(width: TodayLayout.areaWidth, alignment: .leading)
            Text("Length").frame(width: TodayLayout.lengthWidth, alignment: .trailing)
        }
        .fontWeight(.medium)
        .foregroundStyle(.secondary)
        .padding(.leading, TodayLayout.lineColumnWidth)
        .padding(.trailing, 8)
        .padding(.bottom, 8)
        .overlay(alignment: .bottom) { Divider() }
        .padding(.bottom, 4)
    }
}

/// A step's cells after its marker: a table row's columns, or a phone row's
/// time and title over a line of details.
private struct StepCells<Title: View>: View {
    let style: TodayLayout.StepStyle
    /// nil for an Anytime task, which has no time.
    let time: Text?
    var area: AreaTag?
    var length: String?
    var lengthStyle: HierarchicalShapeStyle = .secondary
    /// The phone row's second line: area, length, place.
    var details: String?
    @ViewBuilder let title: Title

    var body: some View {
        switch style {
        case .table:
            HStack(alignment: .firstTextBaseline, spacing: TodayLayout.cellSpacing) {
                (time ?? Text(verbatim: " "))
                    .monospacedDigit()
                    .frame(width: TodayLayout.timeWidth, alignment: .leading)
                title.frame(maxWidth: .infinity, alignment: .leading)
                Group {
                    if let area { area } else { Text(verbatim: " ") }
                }
                .frame(width: TodayLayout.areaWidth, alignment: .leading)
                Text(length ?? "")
                    .monospacedDigit()
                    .foregroundStyle(lengthStyle)
                    .frame(width: TodayLayout.lengthWidth, alignment: .trailing)
            }
        case .list:
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                if let time {
                    time.monospacedDigit()
                        .frame(width: TodayLayout.compactTimeWidth, alignment: .leading)
                }
                VStack(alignment: .leading, spacing: 2) {
                    title
                    if let details, !details.isEmpty {
                        Text(details).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

/// An Area (or a calendar) by name, with its color as a small dot.
private struct AreaTag: View {
    let name: String
    let colorHex: String?

    var body: some View {
        let dot = Text(Image(systemName: "circle.fill"))
            .font(.system(size: 7))
            .foregroundStyle(Color(hexString: colorHex))
            .baselineOffset(2)
        Text("\(dot)  \(name)")
            .foregroundStyle(.secondary)
            .lineLimit(1)
    }
}

// MARK: - Steps

private struct EventStep: View {
    @Environment(CompanionRuntime.self) private var runtime
    let event: AgendaEvent
    let isPast: Bool
    let isAllDay: Bool
    let style: TodayLayout.StepStyle
    let line: DayLine?

    /// When the plan says to leave for it, while it is still ahead.
    private var leave: Departure? {
        guard !isPast else { return nil }
        return runtime.state.departures.first {
            $0.eventID == event.id && $0.eventStart == event.start
        }
    }

    var body: some View {
        StepRow(line: line) {
            RoundedRectangle(cornerRadius: 2)
                .fill(Color(hexString: event.colorHex))
                .frame(width: 5, height: 15)
                .opacity(isPast ? 0.4 : 1)
        } cells: {
            StepCells(
                style: style, time: Text(time).foregroundStyle(.secondary),
                area: AreaTag(name: event.calendarTitle, colorHex: event.colorHex),
                length: isAllDay ? nil : MomentPrompts.minutesText(Int(event.duration / 60)),
                details: details
            ) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(event.title)
                        .fontWeight(.medium)
                        .foregroundStyle(isPast ? .secondary : .primary)
                    if style == .table, let leave {
                        Text("Leave at \(AgendaTime.clock(leave.at))")
                            .foregroundStyle(Color.accentColor)
                    }
                    if style == .table, let place = event.place {
                        Text(place).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
            }
        }
    }

    private var time: String {
        if isAllDay { return "All day" }
        let start = AgendaTime.clock(event.start)
        return style == .table ? "\(start)–\(AgendaTime.clock(event.end))" : start
    }

    private var details: String {
        var parts = [event.calendarTitle]
        // Leaving matters more than the end, on a line that may be cut short.
        if let leave { parts.append("leave at \(AgendaTime.clock(leave.at))") }
        if !isAllDay { parts.append("until \(AgendaTime.clock(event.end))") }
        if let place = event.place { parts.append(place) }
        return parts.joined(separator: " · ")
    }
}

private struct TaskStep: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    let task: TimelineTask
    /// Due tomorrow: today's must-do and plan are not its own.
    var isTomorrow = false
    let style: TodayLayout.StepStyle
    let line: DayLine?

    var body: some View {
        let actions = TodayActions(agenda: agenda, runtime: runtime)
        StepRow(line: line) {
            Button {
                actions.setDone(task.id, !task.isDone)
            } label: {
                Image(systemName: symbol)
                    .foregroundStyle(.secondary)
                    .contentTransition(.symbolEffect(.replace))
            }
            .buttonStyle(.plain)
            .focusable(false)
            .help(task.isDone ? "Mark as not done" : "Mark as done")
        } cells: {
            StepCells(
                style: style,
                time: (style == .list && task.start == nil)
                    ? nil
                    : Text(task.start.map { AgendaTime.clock($0) } ?? "")
                        .foregroundStyle(.secondary),
                area: AreaTag(name: task.areaName, colorHex: task.areaColorHex),
                length: length, details: details
            ) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(task.reminder.title)
                        .foregroundStyle(task.isDone ? .secondary : .primary)
                        .lineLimit(2)
                    if task.isMustDo {
                        Image(systemName: "star.fill")
                            .foregroundStyle(Color.accentColor)
                            .help("The day's must-do")
                    }
                    if style == .table, let note {
                        Text(note).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .contextMenu {
            Button(task.isDone ? "Mark as Not Done" : "Mark as Done") {
                actions.setDone(task.id, !task.isDone)
            }
            if isTomorrow {
                if !task.isDone {
                    Button("Move to Today") { actions.moveToToday(task.id) }
                }
            } else {
                if task.isMustDo {
                    Button("Clear Must-Do") { runtime.act(.setMustDo(reminderID: nil)) }
                } else if !task.isDone {
                    Button("Make This the Must-Do") {
                        runtime.act(.setMustDo(reminderID: task.id))
                    }
                }
                if task.isPlanned {
                    Button("Remove from Today's Plan") {
                        runtime.act(.removeFromPlan(reminderID: task.id))
                    }
                }
                if !task.isDone {
                    Button("Move to Tomorrow") { actions.moveToTomorrow(task.id) }
                }
            }
        }
    }

    private var symbol: String {
        if task.isDone { return "checkmark.circle.fill" }
        return task.isSlid ? "circle.dashed" : "circle"
    }

    private var length: String? {
        task.isPlanned || task.start != nil ? MomentPrompts.minutesText(task.minutes) : nil
    }

    /// Slid, or carried over from an earlier day: never "missed".
    private var note: String? {
        if task.isSlid { return "slid" }
        if task.isCarried, let due = task.reminder.due {
            return "from " + due.formatted(.dateTime.weekday(.abbreviated))
        }
        return nil
    }

    private var details: String {
        [task.areaName, length, note].compactMap { $0 }.joined(separator: " · ")
    }
}

private struct FreeStep: View {
    let start: Date
    let minutes: Int
    let showsTime: Bool
    let style: TodayLayout.StepStyle
    let line: DayLine?

    var body: some View {
        StepRow(line: line, straight: true) {
            Color.clear
        } cells: {
            StepCells(
                style: style, time: Text(showsTime ? AgendaTime.clock(start) : ""),
                length: MomentPrompts.minutesText(minutes), lengthStyle: .tertiary
            ) {
                Text(style == .table ? "Free" : "Free · \(MomentPrompts.minutesText(minutes))")
            }
            .foregroundStyle(.tertiary)
        }
    }
}

/// A group's name on the line: "Anytime today", "Anytime tomorrow".
private struct HeadingStep: View {
    let title: String
    let line: DayLine?

    var body: some View {
        StepRow(line: line, straight: true, topSpace: 8, highlights: false) {
            Color.clear
        } cells: {
            Text(title)
                .fontWeight(.semibold)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .accessibilityAddTraits(.isHeader)
    }
}

/// Where today hands over to tomorrow. The line runs on through it, so the
/// evening reads straight into the next day.
private struct DayBreakStep: View {
    let date: Date
    let style: TodayLayout.StepStyle
    let line: DayLine?

    var body: some View {
        StepRow(line: line, topSpace: TodayLayout.rhythm, highlights: false) {
            Circle()
                .strokeBorder(.secondary, lineWidth: 2)
                .frame(width: 12, height: 12)
        } cells: {
            HStack(alignment: .firstTextBaseline, spacing: TodayLayout.cellSpacing) {
                Text("Tomorrow")
                    .fontWeight(.semibold)
                    .frame(
                        width: style == .table ? TodayLayout.timeWidth : nil, alignment: .leading)
                HStack(spacing: 8) {
                    Text(date.formatted(.dateTime.weekday(.wide).day().month(.wide)))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .layoutPriority(1)
                    Rectangle().fill(.quaternary).frame(height: 1)
                }
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }
}

private struct NowStep: View {
    let time: Date
    let style: TodayLayout.StepStyle
    let line: DayLine?

    var body: some View {
        StepRow(line: line) {
            ZStack {
                Circle().fill(Color.accentColor.opacity(0.25)).frame(width: 16, height: 16)
                Circle().fill(Color.accentColor).frame(width: 8, height: 8)
            }
        } cells: {
            HStack(alignment: .firstTextBaseline, spacing: TodayLayout.cellSpacing) {
                Text(AgendaTime.clock(time))
                    .monospacedDigit()
                    .frame(
                        width: style == .table
                            ? TodayLayout.timeWidth : TodayLayout.compactTimeWidth,
                        alignment: .leading)
                HStack(spacing: 8) {
                    Text("Now")
                    Rectangle().fill(Color.accentColor.opacity(0.6)).frame(height: 1)
                }
            }
            .fontWeight(.semibold)
            .foregroundStyle(Color.accentColor)
        }
    }
}

// MARK: - Actions

/// What Today's buttons do to the day: Reminders through the Agenda, the
/// plan through the runtime.
@MainActor
struct TodayActions {
    let agenda: Agenda
    let runtime: CompanionRuntime

    func perform(_ kind: NowAction.Kind) {
        switch kind {
        case .complete(let id): setDone(id, true)
        case .place(let id, let start, let minutes):
            runtime.act(.place(reminderID: id, start: start, minutes: minutes))
        case .placeAll(let placements):
            for slot in placements {
                runtime.act(
                    .place(reminderID: slot.reminderID, start: slot.start, minutes: slot.minutes))
            }
        case .tomorrow(let id): moveToTomorrow(id)
        case .planDay: runtime.act(.planNow)
        case .wrapUp: runtime.act(.wrapUpNow)
        }
    }

    func setDone(_ id: String, _ done: Bool) {
        Task { _ = try? await agenda.updateReminder(id: id, completed: done, source: "today") }
    }

    /// Due on the owner's tomorrow (after midnight and before the day rolls
    /// over at 04:00, that is still today's date).
    func moveToTomorrow(_ id: String) {
        runtime.act(.removeFromPlan(reminderID: id))
        guard let tomorrow = DayKey(for: Date()).next().date() else { return }
        Task {
            _ = try? await agenda.updateReminder(
                id: id, due: .set(tomorrow, hasTime: false), source: "today")
        }
    }

    /// Due on the owner's today, at no set time: off tomorrow and into
    /// today's Anytime tasks.
    func moveToToday(_ id: String) {
        guard let today = DayKey(for: Date()).date() else { return }
        Task {
            _ = try? await agenda.updateReminder(
                id: id, due: .set(today, hasTime: false), source: "today")
        }
    }

    /// Give an Inbox item the slot Jarvis offered.
    func take(_ slot: InboxSlot, for id: String) {
        switch slot {
        case .today(let start):
            runtime.act(.place(reminderID: id, start: start, minutes: InboxSlot.minutes))
        case .tomorrow:
            moveToTomorrow(id)
        }
    }
}
