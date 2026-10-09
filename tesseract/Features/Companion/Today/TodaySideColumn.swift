//
//  TodaySideColumn.swift
//  tesseract
//
//  Today's side column: the Inbox, each item with the slot Jarvis offers
//  for it (a free half hour today, clear of his other offers, or tomorrow),
//  and "Jarvis noticed". On a narrower page it stacks below the day.
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
                let slot = slots[reminder.id] ?? .tomorrow
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(reminder.title).lineLimit(2)
                    Spacer(minLength: 8)
                    Button(title(of: slot)) { actions.take(slot, for: reminder.id) }
                        .buttonStyle(.plain)
                        .foregroundStyle(Color.accentColor)
                        .focusable(false)
                        .help(help(for: slot))
                }
                .padding(.vertical, 2)
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
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        let tasks = runtime.state.taskProposals
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
            Text(proposal.title).fixedSize(horizontal: false, vertical: true)
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
