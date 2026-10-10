//
//  TodaySideColumn.swift
//  tesseract
//
//  Today's side column: the Inbox, each item with the slot Jarvis offers
//  for it (a free half hour today, clear of his other offers, or tomorrow)
//  and a ring to tick it off, and "Jarvis noticed". On a narrower page it
//  stacks below the day.
//

import SwiftUI

struct TodaySideColumn: View {
    let facts: DayFacts
    let isEvening: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: TodayLayout.rhythm) {
            InboxSection(facts: facts, isEvening: isEvening)
            JarvisNoticed()
        }
    }

    /// Undated captures in the Inbox. One that has a slot in today's plan
    /// has left the Inbox for the day's steps.
    @MainActor
    static func inbox(agenda: Agenda, plan: [Placement]) -> [AgendaReminder] {
        let inboxID = agenda.inbox?.id
        let planned = Set(plan.map(\.reminderID))
        return agenda.snapshot.open.filter {
            $0.due == nil && $0.listID == inboxID && !planned.contains($0.id)
        }
    }
}

private struct InboxSection: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    @Environment(SettingsManager.self) private var settings
    let facts: DayFacts
    let isEvening: Bool

    /// The most the column lists; Reminders holds the rest.
    private static let shown = 12

    var body: some View {
        let items = TodaySideColumn.inbox(agenda: agenda, plan: runtime.state.plan)
        let shown = Array(items.prefix(Self.shown))
        let slots = InboxSlot.suggest(for: shown.map(\.id), facts: facts, evening: isEvening)
        let actions = TodayActions(agenda: agenda, runtime: runtime)
        VStack(alignment: .leading, spacing: TodayLayout.rowSpacing) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("Inbox").fontWeight(.semibold)
                if !items.isEmpty {
                    Text("\(items.count)").foregroundStyle(.secondary).monospacedDigit()
                }
            }
            if items.isEmpty {
                Text("Inbox is clear.").foregroundStyle(.secondary)
                // Where the one-key capture is taught: it went unused for
                // ten days of the trace.
                Text(captureHint)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(shown) { reminder in
                InboxRow(
                    reminder: reminder, slot: slots[reminder.id] ?? .tomorrow, actions: actions)
            }
            if items.count > Self.shown {
                Text("…and \(items.count - Self.shown) more in Reminders.")
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// How to catch a thought from any app: the capture hotkey, by name.
    private var captureHint: String {
        let key = settings.captureHotkey
        return key.isSingleModifier
            ? "Tap \(key.displayString) in any app to write a thought down, or hold it to say one."
            : "Press \(key.displayString) in any app to write a thought down."
    }
}

/// One capture waiting for a home: a ring to tick it off, as on the Day
/// Line, and Jarvis's offer, one click from taking. The context menu holds
/// the other way it can go.
private struct InboxRow: View {
    let reminder: AgendaReminder
    let slot: InboxSlot
    let actions: TodayActions
    @State private var hovering = false

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Button {
                actions.setDone(reminder.id, true)
            } label: {
                Image(systemName: "circle").foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .focusable(false)
            .help("Mark as done")
            Text(linkedTitle: reminder.title)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button(title(of: slot)) { actions.take(slot, for: reminder.id) }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .fixedSize()
                .focusable(false)
                .help(help(for: slot))
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 6)
        .background(
            .quaternary.opacity(hovering ? 0.5 : 0),
            in: RoundedRectangle(cornerRadius: Theme.Radius.small)
        )
        .padding(.horizontal, -6)
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .contextMenu {
            if case .today(let start) = slot {
                Button("Do It at \(AgendaTime.clock(start))") {
                    actions.take(slot, for: reminder.id)
                }
            }
            Button("Move to Tomorrow") { actions.take(.tomorrow, for: reminder.id) }
            Divider()
            Button("Mark as Done") { actions.setDone(reminder.id, true) }
        }
    }

    private func title(of slot: InboxSlot) -> String {
        switch slot {
        case .today(let start): "At \(AgendaTime.clock(start))"
        case .tomorrow: "Tomorrow"
        }
    }

    private func help(for slot: InboxSlot) -> String {
        switch slot {
        case .today(let start):
            "Give it half an hour today at \(AgendaTime.clock(start)), the next free slot"
        case .tomorrow: "Make it due tomorrow; tomorrow's plan finds it a time"
        }
    }
}

/// What Jarvis noticed, until the owner decides: tasks the day showed they
/// must do ("Add to your tasks?"), then "Should I remember this?" facts.
private struct JarvisNoticed: View {
    @Environment(ProfileStore.self) private var profile
    @Environment(CompanionRuntime.self) private var runtime
    @Environment(Agenda.self) private var agenda
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        // One the owner wrote down meanwhile is no longer a question.
        let open = Set(agenda.snapshot.open.map { $0.title.lowercased() })
        let tasks = runtime.state.taskProposals.filter { !open.contains($0.title.lowercased()) }
        if !profile.openProposals.isEmpty || !tasks.isEmpty {
            VStack(alignment: .leading, spacing: TodayLayout.rowSpacing) {
                HStack {
                    Text("Jarvis noticed").fontWeight(.semibold)
                    Spacer()
                    Button("Profile") { openWindow(id: WindowID.profile) }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .focusable(false)
                }
                ForEach(tasks) { task in
                    TaskProposalRow(proposal: task)
                }
                ForEach(profile.openProposals) { proposal in
                    ProposalRow(proposal: proposal)
                }
            }
        }
    }
}

/// A task the day showed, one click from Reminders.
private struct TaskProposalRow: View {
    @Environment(CompanionRuntime.self) private var runtime
    let proposal: TaskProposal

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(linkedTitle: proposal.title)
                .fixedSize(horizontal: false, vertical: true)
            Text(when).foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button("Add") { runtime.act(.taskProposal(id: proposal.id, add: true)) }
                Button("No") { runtime.act(.taskProposal(id: proposal.id, add: false)) }
            }
            .buttonStyle(.plain)
            .foregroundStyle(Color.accentColor)
            .focusable(false)
        }
    }

    private var when: String {
        guard let due = proposal.due else { return "Add to your tasks? Into the Inbox." }
        let day = due.formatted(.dateTime.weekday(.wide))
        return "Add to your tasks? Due \(day)."
    }
}
