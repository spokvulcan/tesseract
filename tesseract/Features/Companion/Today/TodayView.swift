//
//  TodayView.swift
//  tesseract
//
//  Today: the app's home. At the top the Now Card says where the day is and
//  offers the step that moves it on. Below it the day runs as steps on one
//  Day Line, on into tomorrow, a table on a wide page and a list shaped for
//  a phone on a narrow one, with the Inbox and "Jarvis noticed" beside it
//  (below it when narrow). Until 04:00 the page keeps the day that is
//  ending (DayKey). One field at the bottom, the Today composer, asks Jarvis or
//  adds a task, and confirms every change made from here, with an undo. A
//  picture dropped anywhere on the page goes into it, for Jarvis. The Chat
//  mode shows the Day Thread, its pictures one click from Quick Look.
//
//  Content layer only (design-language §1): no custom glass. One type size;
//  hierarchy from weight and color. Row actions live in context menus.
//

import SwiftUI

struct TodayView: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    @Environment(DayThread.self) private var thread
    @Environment(ComposerDraftController.self) private var draft

    enum Mode: String, CaseIterable, Identifiable {
        case day = "Day"
        case chat = "Chat"
        var id: String { rawValue }
    }

    /// A fixed clock, for the Today gallery; nil follows the real one.
    private let fixedNow: Date?
    @State private var mode: Mode = .day
    @State private var now: Date
    @State private var speakingMessageID: UUID?

    init(fixedNow: Date? = nil) {
        self.fixedNow = fixedNow
        _now = State(initialValue: fixedNow ?? Date())
    }

    var body: some View {
        Group {
            switch mode {
            case .day:
                TodayDayPage(now: now)
            case .chat:
                ChatTranscriptView(speakingMessageID: $speakingMessageID, isSpeechActive: false)
                    .environment(thread.chat)
            }
        }
        .font(TodayLayout.font)
        .safeAreaInset(edge: .bottom) {
            TodayComposer(onAsk: { mode = .chat })
                .padding(Theme.Spacing.md)
        }
        .background(
            QuickLookContainer(
                request: draft.quickLookRequest, onClose: { draft.dismissQuickLook() })
        )
        .imageDropTarget(draft, title: "Drop a picture for Jarvis")
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
            guard fixedNow == nil else { return }
            // Tick on the minute, so the Now line and card read the clock.
            while !Task.isCancelled {
                let day = DayKey(for: now)
                now = Date()
                // A new day at 04:00: fetch its tomorrow.
                if DayKey(for: now) != day { await agenda.refresh() }
                let intoMinute = now.timeIntervalSinceReferenceDate.truncatingRemainder(
                    dividingBy: 60)
                try? await Task.sleep(for: .seconds(60 - intoMinute))
            }
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
    /// The widest the page grows; past it, it centers.
    static let maxWidth: CGFloat = 1240
    static let sideWidth: CGFloat = 300

    // The steps: the Day Line's column, then the table's columns.
    static let lineColumnWidth: CGFloat = 28
    static let markerSize: CGFloat = 18
    /// From a row's top to its marker, so the marker sits on the first line.
    static let markerTop: CGFloat = 6.5
    static let rowPadding: CGFloat = 7
    static let cellSpacing: CGFloat = 12
    static let timeWidth: CGFloat = 104
    static let compactTimeWidth: CGFloat = 46
    static let areaWidth: CGFloat = 140
    static let lengthWidth: CGFloat = 76

    /// How the day's steps are drawn.
    enum StepStyle {
        case table
        case list
    }

    /// The page's width class, from the width it is given.
    nonisolated enum Width: Equatable, Sendable {
        /// A phone's width: the steps as a list, the side column below.
        case compact
        /// The steps as a table, the side column below.
        case regular
        /// The steps as a table, the side column beside them.
        case wide

        init(_ width: CGFloat) {
            switch width {
            case ..<640: self = .compact
            case ..<1060: self = .regular
            default: self = .wide
            }
        }
    }
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

// MARK: - The day page

private struct TodayDayPage: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    @Environment(SettingsManager.self) private var settings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let now: Date
    @State private var width = TodayLayout.Width.regular

    var body: some View {
        let state = runtime.state
        let facts = DayFacts(
            snapshot: agenda.snapshot, areas: agenda.areas, inboxListID: agenda.inbox?.id,
            now: now, mustDoID: state.mustDoID, plan: state.plan, weekFocus: state.weekFocus,
            departures: state.departures)
        let timeline = TimelineBuilder.build(facts: facts)
        let isEvening = NowCardBuilder.isEvening(
            now, eveningMinutes: settings.companionEveningMinutes, calendar: facts.calendar)
        let card = NowCardBuilder.build(
            timeline: timeline, facts: facts,
            context: NowCardBuilder.Context(
                companionOn: settings.companionHeartbeatEnabled,
                planned: state.morningPlanAt != nil, wrappedUp: state.eveningWrapUpAt != nil,
                eveningMinutes: settings.companionEveningMinutes,
                inboxCount: TodaySideColumn.inbox(agenda: agenda, plan: state.plan).count,
                startedSteps: state.startedSteps))
        // The Inbox's offers keep clear of the card's own.
        let offerFacts = card.reserving(facts)
        let motion: Animation? = reduceMotion ? nil : .smooth(duration: 0.25)
        ScrollView {
            VStack(alignment: .leading, spacing: TodayLayout.rhythm) {
                TodayHeader(timeline: timeline, now: now, weekFocus: state.weekFocus)
                TodayAccessNotice()
                if width == .wide {
                    HStack(alignment: .top, spacing: TodayLayout.columnGap) {
                        day(card: card, facts: facts, timeline: timeline)
                        TodaySideColumn(facts: offerFacts, isEvening: isEvening)
                            .frame(width: TodayLayout.sideWidth, alignment: .leading)
                    }
                } else {
                    day(card: card, facts: facts, timeline: timeline)
                    TodaySideColumn(facts: offerFacts, isEvening: isEvening)
                        .padding(.top, 8)
                }
            }
            .padding(.horizontal, width == .compact ? 16 : 24)
            .padding(.vertical, 20)
            .frame(maxWidth: TodayLayout.maxWidth, alignment: .leading)
            .frame(maxWidth: .infinity)
            // Steps move when the owner acts, not as the clock ticks.
            .animation(motion, value: state.plan)
            .animation(motion, value: timeline.doneCount)
        }
        .onGeometryChange(for: TodayLayout.Width.self) { proxy in
            TodayLayout.Width(proxy.size.width)
        } action: {
            width = $0
        }
    }

    private func day(card: NowCard, facts: DayFacts, timeline: TodayTimeline) -> some View {
        VStack(alignment: .leading, spacing: TodayLayout.rhythm + 4) {
            NowCardView(card: card, facts: facts)
            DaySteps(timeline: timeline, style: width == .compact ? .list : .table)
        }
    }
}

// MARK: - Header

private struct TodayHeader: View {
    let timeline: TodayTimeline
    let now: Date
    /// The week's one focus, kept in sight all week.
    let weekFocus: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                // The owner's day: until 04:00, the one that is ending.
                Text(
                    (DayKey(for: now).date() ?? now).formatted(
                        .dateTime.weekday(.wide).day().month(.wide))
                )
                .fontWeight(.semibold)
                if let weekFocus {
                    Text("· This week: \(weekFocus)")
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                if timeline.totalCount > 0 {
                    Text("\(timeline.doneCount) of \(timeline.totalCount) done")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }
            if timeline.totalCount > 0 {
                DayProgress(done: timeline.doneCount, total: timeline.totalCount)
            }
        }
    }
}

/// How far through the day's tasks the owner is: a hairline track, filled
/// in the accent as tasks are done.
private struct DayProgress: View {
    let done: Int
    let total: Int

    var body: some View {
        Capsule()
            .fill(.quaternary)
            .frame(height: 4)
            .overlay(alignment: .leading) {
                GeometryReader { proxy in
                    Capsule()
                        .fill(Color.accentColor)
                        .frame(width: proxy.size.width * CGFloat(done) / CGFloat(max(total, 1)))
                }
            }
            .accessibilityElement()
            .accessibilityLabel("Today's tasks")
            .accessibilityValue("\(done) of \(total) done")
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
