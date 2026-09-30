//
//  TodayView.swift
//  tesseract
//
//  Today: the app's home. The Timeline layout chosen from the prototype — a
//  header (date, Jarvis's line, "N of M done", the must-do), the day as one
//  time-ordered list, and a side column with Capture, Waiting on you, the
//  Inbox and "Jarvis noticed". The day's cards sit at the top when they
//  apply. The Chat mode shows the Day Thread, where the owner talks to
//  Jarvis with the whole day in view.
//
//  Content layer only (design-language §1): no custom glass. One type size;
//  hierarchy from weight and color. Row actions live in context menus.
//

import SwiftUI

struct TodayView: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    @Environment(DayThread.self) private var thread
    @Environment(SettingsManager.self) private var settings

    enum Mode: String, CaseIterable, Identifiable {
        case day = "Day"
        case chat = "Chat"
        var id: String { rawValue }
    }

    @State private var mode: Mode = .day
    @State private var now = Date()
    @State private var speakingMessageID: UUID?

    var body: some View {
        Group {
            switch mode {
            case .day:
                ScrollView {
                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .top, spacing: TodayLayout.columnGap) {
                            mainColumn.frame(minWidth: 440, maxWidth: 760)
                            TodaySideColumn(now: now).frame(width: 300)
                        }
                        VStack(alignment: .leading, spacing: TodayLayout.rhythm) {
                            mainColumn
                            TodaySideColumn(now: now)
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.vertical, 20)
                    .frame(maxWidth: .infinity, alignment: .top)
                }
            case .chat:
                ChatTranscriptView(speakingMessageID: $speakingMessageID, isSpeechActive: false)
                    .environment(thread.chat)
            }
        }
        .font(TodayLayout.font)
        .safeAreaInset(edge: .bottom) {
            AskJarvisBar(onSend: { mode = .chat })
                .padding(.horizontal, 24)
                .padding(.vertical, 12)
        }
        .navigationTitle("Today")
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("View", selection: $mode) {
                    ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .fixedSize()
            }
        }
        .task {
            runtime.todayOpened()
            while !Task.isCancelled {
                now = Date()
                try? await Task.sleep(for: .seconds(30))
            }
        }
    }

    private var facts: DayFacts {
        DayFacts(
            snapshot: agenda.snapshot, areas: agenda.areas, inboxListID: agenda.inbox?.id,
            now: now, mustDoID: runtime.state.mustDoID, plan: runtime.state.plan)
    }

    private var mainColumn: some View {
        let timeline = TimelineBuilder.build(facts: facts)
        return VStack(alignment: .leading, spacing: TodayLayout.rhythm) {
            TodayHeader(timeline: timeline, now: now)
            TodayAccessNotice()
            ForEach(runtime.state.openCards) { card in
                DayCardView(card: card, facts: facts)
            }
            TimelineList(timeline: timeline)
        }
    }
}

// MARK: - Layout

enum TodayLayout {
    /// The page's one type size.
    static let fontSize: CGFloat = 14
    static let font = Font.system(size: fontSize)
    /// The page's one vertical rhythm.
    static let rhythm: CGFloat = 16
    static let rowSpacing: CGFloat = 6
    static let columnGap: CGFloat = 28
    static let timeWidth: CGFloat = 92
}

extension Color {
    /// `#RRGGBB` from EventKit colors; nil falls back to the accent.
    init(hexString: String?) {
        guard let hexString, hexString.count == 7, hexString.hasPrefix("#"),
            let value = UInt32(hexString.dropFirst(), radix: 16)
        else {
            self = .accentColor
            return
        }
        self = Color(
            red: Double((value >> 16) & 0xFF) / 255, green: Double((value >> 8) & 0xFF) / 255,
            blue: Double(value & 0xFF) / 255)
    }
}

// MARK: - Header

private struct TodayHeader: View {
    @Environment(CompanionRuntime.self) private var runtime
    @Environment(SettingsManager.self) private var settings
    let timeline: TodayTimeline
    let now: Date

    var body: some View {
        VStack(alignment: .leading, spacing: TodayLayout.rowSpacing) {
            HStack(alignment: .firstTextBaseline) {
                Text(now.formatted(.dateTime.weekday(.wide).day().month(.wide)))
                    .fontWeight(.semibold)
                Spacer()
                if timeline.totalCount > 0 {
                    Text("\(timeline.doneCount) of \(timeline.totalCount) done")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }
            Text(line)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 8) {
                if let mustDo = timeline.mustDo {
                    MustDoChip(task: mustDo)
                }
                Spacer()
                if runtime.state.running != nil {
                    ProgressView().controlSize(.small)
                    Text("Jarvis is thinking…").foregroundStyle(.secondary)
                } else if settings.companionHeartbeatEnabled {
                    if runtime.state.morningPlanAt == nil {
                        Button("Plan my day") { runtime.act(.planNow) }
                            .focusable(false)
                    }
                    if isEvening, runtime.state.eveningWrapUpAt == nil {
                        Button("Wrap up") { runtime.act(.wrapUpNow) }
                            .focusable(false)
                    }
                }
            }
            .controlSize(.small)
        }
    }

    private var shape: String {
        let events = timeline.rows.filter { if case .event = $0.kind { true } else { false } }.count
        let tasks = timeline.totalCount
        var parts: [String] = []
        if events > 0 { parts.append("\(events) event\(events == 1 ? "" : "s")") }
        if tasks > 0 { parts.append("\(tasks) task\(tasks == 1 ? "" : "s")") }
        return parts.isEmpty ? "A clear day." : parts.joined(separator: " · ") + " today."
    }

    private var isEvening: Bool {
        let parts = Calendar.current.dateComponents([.hour, .minute], from: now)
        let minute = (parts.hour ?? 0) * 60 + (parts.minute ?? 0)
        return minute >= settings.companionEveningMinutes || minute < 3 * 60
    }

    /// Jarvis's one-line note. While a card is open below, the card carries
    /// his words, so the header gives the day's shape instead.
    private var line: String {
        if !runtime.state.openCards.isEmpty { return shape }
        if let card = runtime.state.cards.last { return card.line }
        if !settings.companionHeartbeatEnabled {
            return "Your day from Reminders and Calendar."
        }
        return "Here's your day."
    }
}

private struct MustDoChip: View {
    let task: TimelineTask

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "star.fill")
            Text("Must-do · \(task.reminder.title)")
                .lineLimit(1)
            if let start = task.start {
                Text(AgendaTime.clock(start)).monospacedDigit()
            }
        }
        .fontWeight(.medium)
        .foregroundStyle(Color.accentColor)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(Color.accentColor.opacity(0.12), in: Capsule())
    }
}

// MARK: - Access and switch

private struct TodayAccessNotice: View {
    @Environment(Agenda.self) private var agenda
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        let access =
            agenda.snapshot.takenAt == .distantPast ? agenda.access : agenda.snapshot.access
        if access.needsRequest {
            notice(
                "Jarvis keeps your day in Reminders and Calendar, so it reaches your phone and watch. He asks once.",
                action: "Allow Access"
            ) { Task { await agenda.requestAccessIfNeeded() } }
        } else if !access.isFull {
            notice(
                access.canUseCalendar
                    ? "Without Reminders, Today shows your calendar only. Allow it in System Settings → Privacy & Security → Reminders."
                    : "Without Calendar and Reminders, Today can't show your day. Allow them in System Settings → Privacy & Security.",
                action: "Open Settings"
            ) {
                if let url = URL(
                    string:
                        "x-apple.systempreferences:com.apple.preference.security?Privacy_Reminders")
                {
                    NSWorkspace.shared.open(url)
                }
            }
        } else if !settings.companionHeartbeatEnabled {
            @Bindable var settings = settings
            notice(
                "Jarvis is off. Turn him on for a plan each morning, a card when you come back, and nudges before events.",
                action: "Turn On"
            ) { settings.companionHeartbeatEnabled = true }
        }
    }

    private func notice(_ text: String, action: String, perform: @escaping () -> Void) -> some View
    {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(text).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Button(action, action: perform).focusable(false)
        }
        .padding(12)
        .background(
            .quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: Theme.Radius.medium))
    }
}

// MARK: - Timeline

private struct TimelineList: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    let timeline: TodayTimeline

    var body: some View {
        VStack(alignment: .leading, spacing: TodayLayout.rowSpacing) {
            if !timeline.allDayEvents.isEmpty {
                Text("All day · " + timeline.allDayEvents.map(\.title).joined(separator: ", "))
                    .foregroundStyle(.secondary)
            }
            ForEach(timeline.rows) { row in
                switch row.kind {
                case .event(let event):
                    EventRow(event: event, isPast: row.isPast)
                case .task(let task):
                    TaskRow(task: task, time: task.start.map { AgendaTime.clock($0) })
                case .free(let minutes):
                    HStack(spacing: 0) {
                        Text(AgendaTime.clock(row.start))
                            .frame(width: TodayLayout.timeWidth, alignment: .leading)
                        Text("Free · \(MomentPrompts.minutesText(minutes))")
                    }
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
                    .padding(.vertical, 2)
                case .now:
                    NowLine(time: row.start)
                }
            }
            if !timeline.anytime.isEmpty {
                Text("Anytime today")
                    .fontWeight(.semibold)
                    .padding(.top, 8)
                ForEach(timeline.anytime) { task in
                    TaskRow(task: task, time: nil)
                }
            }
            if timeline.rows.count <= 1, timeline.anytime.isEmpty {
                Text(
                    "Nothing planned yet. Capture what's on your mind, or ask Jarvis to plan your day."
                )
                .foregroundStyle(.secondary)
            }
        }
    }
}

private struct EventRow: View {
    let event: AgendaEvent
    let isPast: Bool

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 0) {
            Text("\(AgendaTime.clock(event.start))–\(AgendaTime.clock(event.end))")
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: TodayLayout.timeWidth, alignment: .leading)
            VStack(alignment: .leading, spacing: 2) {
                Text(event.title).fontWeight(.medium)
                if let location = event.location, !location.isEmpty {
                    Text(location).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(
            Color(hexString: event.colorHex).opacity(isPast ? 0.06 : 0.14),
            in: RoundedRectangle(cornerRadius: Theme.Radius.small)
        )
        .opacity(isPast ? 0.55 : 1)
    }
}

private struct TaskRow: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    let task: TimelineTask
    let time: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 0) {
            Text(time ?? "")
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: time == nil ? 0 : TodayLayout.timeWidth, alignment: .leading)
            Button {
                toggle()
            } label: {
                Image(systemName: task.isDone ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(Color(hexString: task.areaColorHex))
            }
            .buttonStyle(.plain)
            .focusable(false)
            .help(task.isDone ? "Mark as not done" : "Mark as done")
            .padding(.trailing, 8)
            Text(task.reminder.title)
                .strikethrough(task.isDone)
                .foregroundStyle(task.isDone ? .secondary : .primary)
            if task.isMustDo {
                Image(systemName: "star.fill").foregroundStyle(Color.accentColor).padding(
                    .leading, 6)
            }
            Spacer(minLength: 8)
            Text(meta)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(.vertical, 3)
        .opacity(task.isDone ? 0.6 : 1)
        .contentShape(Rectangle())
        .contextMenu {
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
                Button("Move to Tomorrow") {
                    runtime.act(.removeFromPlan(reminderID: task.id))
                    Task { await moveToTomorrow() }
                }
            }
        }
    }

    private var meta: String {
        var parts = [task.areaName]
        if task.isPlanned || task.start != nil { parts.append("\(task.minutes) min") }
        if task.isSlid { parts.append("slid") }
        if task.isCarried, let due = task.reminder.due {
            parts.append("from " + due.formatted(.dateTime.weekday(.abbreviated)))
        }
        return parts.joined(separator: " · ")
    }

    private func toggle() {
        Task {
            _ = try? await agenda.updateReminder(
                id: task.id, completed: !task.isDone, source: "today")
        }
    }

    private func moveToTomorrow() async {
        let calendar = Calendar.current
        guard
            let tomorrow = calendar.date(
                byAdding: .day, value: 1, to: calendar.startOfDay(for: Date()))
        else { return }
        _ = try? await agenda.updateReminder(
            id: task.id, due: .set(tomorrow, hasTime: false), source: "today")
    }
}

private struct NowLine: View {
    let time: Date

    var body: some View {
        HStack(spacing: 8) {
            Text("Now \(AgendaTime.clock(time))")
                .fontWeight(.semibold)
                .monospacedDigit()
                .foregroundStyle(Color.accentColor)
            Rectangle()
                .fill(Color.accentColor)
                .frame(height: 1)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Ask Jarvis

private struct AskJarvisBar: View {
    @Environment(DayThread.self) private var thread
    let onSend: () -> Void
    @State private var text = ""

    var body: some View {
        HStack(spacing: 10) {
            TextField("Ask Jarvis about your day…", text: $text)
                .textFieldStyle(.plain)
                .onSubmit(send)
            if thread.chat.isGenerating || thread.momentRunning != nil {
                ProgressView().controlSize(.small)
            }
            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 20))
            }
            .buttonStyle(.plain)
            .focusable(false)
            .disabled(text.trimmingCharacters(in: .whitespaces).isEmpty || !thread.canSend)
            .help("Send")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.bar, in: RoundedRectangle(cornerRadius: Theme.Radius.large))
        .frame(maxWidth: 760)
    }

    private func send() {
        let message = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty, thread.canSend else { return }
        thread.send(message)
        text = ""
        onSend()
    }
}
