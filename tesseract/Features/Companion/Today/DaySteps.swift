//
//  DaySteps.swift
//  tesseract
//
//  The day as steps. A wide page shows a table (time, task, area, length);
//  a narrow one shows a list shaped for a phone. One Day Line runs down the
//  schedule, walked up to the Now line and ahead after it, with each step's
//  marker on it: a check for a task, the calendar's color for an event, the
//  accent ring for now. Anytime tasks follow, off the line, because they
//  have no time. Row actions live in context menus (design-language §2).
//

import SwiftUI

struct DaySteps: View {
    let timeline: TodayTimeline
    let style: TodayLayout.StepStyle

    var body: some View {
        let rows = timeline.rows
        let nowIndex = rows.firstIndex { $0.kind == .now } ?? rows.count
        VStack(alignment: .leading, spacing: 0) {
            if style == .table {
                ColumnHeader()
            }
            ForEach(timeline.allDayEvents) { event in
                EventStep(event: event, isPast: false, isAllDay: true, style: style, line: nil)
            }
            ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in
                let line = DayLine(
                    above: index == 0 ? nil : index <= nowIndex ? .walked : .ahead,
                    below: index == rows.count - 1 ? nil : index < nowIndex ? .walked : .ahead)
                switch row.kind {
                case .event(let event):
                    EventStep(
                        event: event, isPast: row.isPast, isAllDay: false, style: style, line: line)
                case .task(let task):
                    TaskStep(task: task, style: style, line: line)
                case .free(let minutes):
                    // Free time from now on needs no time: the Now line has it.
                    FreeStep(
                        start: row.start, minutes: minutes,
                        showsTime: nowIndex == rows.count || row.start != rows[nowIndex].start,
                        style: style, line: line)
                case .now:
                    NowStep(time: row.start, style: style, line: line)
                }
            }
            if !timeline.anytime.isEmpty {
                Text("Anytime today")
                    .fontWeight(.semibold)
                    .padding(.top, TodayLayout.rhythm)
                    .padding(.bottom, 4)
                ForEach(timeline.anytime) { task in
                    TaskStep(task: task, style: style, line: nil)
                }
            }
        }
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

    var body: some View {
        VStack(spacing: 0) {
            segment(line?.above).frame(height: TodayLayout.markerTop)
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
        .background(alignment: .leading) {
            if line != nil { DayLineSegments(line: line, straight: straight) }
        }
        .background(
            .quaternary.opacity(hovering ? 0.5 : 0),
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
    let event: AgendaEvent
    let isPast: Bool
    let isAllDay: Bool
    let style: TodayLayout.StepStyle
    let line: DayLine?

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
                    if style == .table, let location = event.location, !location.isEmpty {
                        Text(location).foregroundStyle(.secondary).lineLimit(1)
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
        if !isAllDay { parts.append("until \(AgendaTime.clock(event.end))") }
        if let location = event.location, !location.isEmpty { parts.append(location) }
        return parts.joined(separator: " · ")
    }
}

private struct TaskStep: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    let task: TimelineTask
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
            if task.isMustDo {
                Button("Clear Must-Do") { runtime.act(.setMustDo(reminderID: nil)) }
            } else if !task.isDone {
                Button("Make This the Must-Do") { runtime.act(.setMustDo(reminderID: task.id)) }
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
