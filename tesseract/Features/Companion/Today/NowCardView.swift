//
//  NowCardView.swift
//  tesseract
//
//  The Now Card on Today: the step the day is on and the one-click
//  proposals that move it on, Jarvis's latest line while it is fresh, and
//  everything that waits on the owner (people, coding agents, the evening's
//  leftovers), each said once, here. A plain content-layer card; nothing is
//  ever labelled missed or failed.
//

import SwiftUI

struct NowCardView: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    let card: NowCard
    let facts: DayFacts

    var body: some View {
        let state = runtime.state
        // Jarvis's word: the day's latest card, until it is dismissed or the
        // day moves past it.
        let jarvis = state.cards.last.flatMap {
            $0.dismissed || !$0.isFresh(at: facts.now) ? nil : $0
        }
        let waiting = Waiting(state: state, now: facts.now)
        VStack(alignment: .leading, spacing: 12) {
            header(jarvis)
            VStack(alignment: .leading, spacing: 3) {
                Text(card.headline)
                    .fontWeight(.semibold)
                    .fixedSize(horizontal: false, vertical: true)
                if let detail = card.detail {
                    Text(detail)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if !card.actions.isEmpty {
                actionRow
            }
            if jarvis != nil || !waiting.isEmpty {
                Divider()
                if let jarvis {
                    JarvisWord(card: jarvis)
                }
                NeedsYou(waiting: waiting)
                Leftovers(wrapUps: waiting.wrapUps)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            .quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: Theme.Radius.medium))
    }

    private func header(_ jarvis: DayCard?) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("Now · \(AgendaTime.clock(facts.now))")
                .fontWeight(.semibold)
                .monospacedDigit()
                .foregroundStyle(Color.accentColor)
            Spacer(minLength: 8)
            if runtime.state.running != nil {
                ProgressView().controlSize(.small)
                Text("Jarvis is thinking…").foregroundStyle(.secondary)
            } else if let jarvis {
                Text(jarvis.kind.title).foregroundStyle(.secondary)
                Button {
                    runtime.act(.dismiss(cardID: jarvis.id))
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .focusable(false)
                .help("Dismiss")
            }
        }
    }

    private var actionRow: some View {
        let actions = TodayActions(agenda: agenda, runtime: runtime)
        let thinking = runtime.state.running != nil
        return HStack(spacing: 8) {
            ForEach(Array(card.actions.enumerated()), id: \.element.id) { index, action in
                let needsJarvis = action.kind == .planDay || action.kind == .wrapUp
                Group {
                    if index == 0 {
                        Button(action.title) { actions.perform(action.kind) }
                            .buttonStyle(.borderedProminent)
                    } else {
                        Button(action.title) { actions.perform(action.kind) }
                            .buttonStyle(.bordered)
                    }
                }
                .disabled(needsJarvis && thinking)
                .focusable(false)
            }
        }
    }
}

// MARK: - Jarvis's word

/// The latest card's line, three lines at most until expanded, with what
/// only that moment knows: the plan's tips, where the owner was, what can
/// wait, the first draft of tomorrow.
private struct JarvisWord: View {
    let card: DayCard
    @State private var expanded = false
    @State private var showQuiet = false
    /// The line's height in full and as shown, to offer More only when the
    /// three lines cut it short.
    @State private var fullHeight: CGFloat = 0
    @State private var shownHeight: CGFloat = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(card.line)
                .lineLimit(expanded ? nil : 3)
                .fixedSize(horizontal: false, vertical: true)
                .onGeometryChange(for: CGFloat.self) {
                    $0.size.height
                } action: {
                    shownHeight = $0
                }
                .background(alignment: .top) {
                    Text(card.line)
                        .fixedSize(horizontal: false, vertical: true)
                        .hidden()
                        .onGeometryChange(for: CGFloat.self) {
                            $0.size.height
                        } action: {
                            fullHeight = $0
                        }
                }
            if expanded || fullHeight > shownHeight + 1 {
                Button(expanded ? "Less" : "More") { expanded.toggle() }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.accentColor)
                    .focusable(false)
            }
            if card.isRefining {
                Text("Jarvis is still thinking it through.").foregroundStyle(.secondary)
            }
            extras
        }
    }

    @ViewBuilder private var extras: some View {
        switch card.body {
        case .morningPlan(let plan):
            if !plan.placements.isEmpty {
                let count = plan.placements.count
                Text("Jarvis placed \(count) task\(count == 1 ? "" : "s") in your day.")
                    .foregroundStyle(.secondary)
            }
            ForEach(plan.suggestions, id: \.self) { tip in
                Text("· \(tip)").foregroundStyle(.secondary)
            }
        case .breakpoint(let breakpoint):
            let minutes = Int(breakpoint.awayUntil.timeIntervalSince(breakpoint.awayFrom) / 60)
            let away =
                "Away \(MomentPrompts.minutesText(minutes)), \(AgendaTime.clock(breakpoint.awayFrom))–\(AgendaTime.clock(breakpoint.awayUntil))"
            Text(breakpoint.whereYouWere.map { "\(away) · you were in \($0)." } ?? "\(away).")
                .foregroundStyle(.secondary)
            if breakpoint.canWaitCount > 0 {
                let count = breakpoint.canWaitCount
                Button(
                    showQuiet
                        ? "Hide what can wait"
                        : "\(count) other notification\(count == 1 ? "" : "s") can wait"
                ) { showQuiet.toggle() }
                .buttonStyle(.plain)
                .foregroundStyle(Color.accentColor)
                .focusable(false)
                if showQuiet {
                    ForEach(breakpoint.canWait) { group in
                        Text("\(group.app): " + group.lines.joined(separator: " · "))
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        case .reflection(let reflection):
            ForEach(reflection.tomorrow, id: \.self) { line in
                Text("· \(line)").foregroundStyle(.secondary)
            }
        case .eveningWrapUp, .triage:
            // Their items wait below, under Needs you and Left from today.
            EmptyView()
        }
    }
}

// MARK: - Needs you

/// What waits on the owner: the waiting coding agents, the open cards'
/// items (a waiting agent once, from the agents themselves), and the
/// evening's leftovers.
private struct Waiting {
    struct Entry: Identifiable {
        let cardID: String
        let item: WaitingItem
        var id: String { "\(cardID)/\(item.id)" }
    }

    let agents: [AgentSignal]
    let entries: [Entry]
    let wrapUps: [(cardID: String, card: EveningWrapUpCard)]

    @MainActor
    init(state: DayState, now: Date) {
        agents = state.agentsWaiting(now: now)
        entries = state.openCards.flatMap { card -> [Entry] in
            switch card.body {
            case .breakpoint(let breakpoint):
                breakpoint.needsYou.filter { $0.kind != .agent }.map {
                    Entry(cardID: card.id, item: $0)
                }
            case .triage(let triage):
                triage.raise.map { Entry(cardID: card.id, item: $0) }
            case .morningPlan, .eveningWrapUp, .reflection:
                []
            }
        }
        wrapUps = state.openCards.compactMap { card in
            guard case .eveningWrapUp(let wrapUp) = card.body, !wrapUp.leftovers.isEmpty else {
                return nil
            }
            return (card.id, wrapUp)
        }
    }

    var isEmpty: Bool { agents.isEmpty && entries.isEmpty && wrapUps.isEmpty }
}

/// People, coding agents and missed notifications that need the owner, one
/// action each.
private struct NeedsYou: View {
    @Environment(CompanionRuntime.self) private var runtime
    let waiting: Waiting

    var body: some View {
        if !waiting.agents.isEmpty || !waiting.entries.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Needs you").fontWeight(.semibold)
                ForEach(waiting.agents) { agent in
                    ItemLine(
                        title: agent.title,
                        detail: agent.kind == .waiting
                            ? agent.message : "Finished: \(agent.message)"
                    ) {
                        Button("Handled") { runtime.act(.agentHandled(agentID: agent.id)) }
                            .foregroundStyle(.secondary)
                    }
                }
                ForEach(waiting.entries) { entry in
                    ItemLine(title: entry.item.title, detail: entry.item.detail) {
                        if entry.item.app != nil {
                            Button("Open") {
                                runtime.act(.openItem(cardID: entry.cardID, itemID: entry.item.id))
                            }
                            .foregroundStyle(Color.accentColor)
                        }
                        if entry.item.kind != .agent {
                            Button("Later") {
                                runtime.act(.itemLater(cardID: entry.cardID, itemID: entry.item.id))
                            }
                            .foregroundStyle(.secondary)
                            .help("Remind me in half an hour")
                        }
                        Button("Done") {
                            runtime.act(.itemDone(cardID: entry.cardID, itemID: entry.item.id))
                        }
                        .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }
}

/// One item that needs the owner: who or what, a line of detail, and its
/// actions as plain words.
private struct ItemLine<Actions: View>: View {
    let title: String
    let detail: String
    @ViewBuilder let actions: Actions

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).fontWeight(.medium).lineLimit(1)
                Text(detail).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer(minLength: 6)
            HStack(spacing: 10) { actions }
                .buttonStyle(.plain)
                .focusable(false)
        }
    }
}

// MARK: - Left from today

/// The Evening Wrap-up's leftovers, each rolled to tomorrow or another day,
/// kept for later, or let go; Jarvis's choice is tinted.
private struct Leftovers: View {
    @Environment(CompanionRuntime.self) private var runtime
    let wrapUps: [(cardID: String, card: EveningWrapUpCard)]

    var body: some View {
        ForEach(wrapUps, id: \.cardID) { cardID, wrapUp in
            VStack(alignment: .leading, spacing: 8) {
                Text("Left from today").fontWeight(.semibold)
                ForEach(wrapUp.leftovers) { leftover in
                    LeftoverRow(cardID: cardID, leftover: leftover)
                }
                Button("Do What Jarvis Suggests for All") {
                    runtime.act(.allLeftovers(cardID: cardID))
                }
                .focusable(false)
            }
        }
    }
}

private struct LeftoverRow: View {
    @Environment(CompanionRuntime.self) private var runtime
    let cardID: String
    let leftover: Leftover

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                Text(leftover.title).lineLimit(1)
                Spacer(minLength: 8)
                choices
            }
            VStack(alignment: .leading, spacing: 6) {
                Text(leftover.title).lineLimit(2)
                HStack(spacing: 8) { choices }
            }
        }
    }

    @ViewBuilder private var choices: some View {
        choice("Tomorrow", .tomorrow)
        Menu("Another day") {
            // From the owner's today: until 04:00, the day that is ending.
            let today = DayKey(for: Date()).date() ?? Calendar.current.startOfDay(for: Date())
            ForEach(2..<8, id: \.self) { offset in
                let day = Calendar.current.date(byAdding: .day, value: offset, to: today) ?? today
                Button(day.formatted(.dateTime.weekday(.wide).day().month())) {
                    runtime.act(
                        .leftoverOn(cardID: cardID, reminderID: leftover.reminderID, day: day))
                }
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .controlSize(.small)
        choice("Later", .later)
        choice("Let go", .drop)
    }

    private func choice(_ title: String, _ suggestion: Leftover.Suggestion) -> some View {
        Button(title) {
            runtime.act(.leftover(cardID: cardID, reminderID: leftover.reminderID, suggestion))
        }
        .controlSize(.small)
        .buttonStyle(.bordered)
        .tint(leftover.suggestion == suggestion ? .accentColor : nil)
        .focusable(false)
    }
}
