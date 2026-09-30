//
//  TodaySideColumn.swift
//  tesseract
//
//  Today's side column: Capture, Waiting on you, the Inbox (with "Find a
//  time") and "Jarvis noticed". At narrow widths it stacks below the
//  Timeline.
//

import SwiftUI

struct TodaySideColumn: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    let now: Date

    var body: some View {
        VStack(alignment: .leading, spacing: TodayLayout.rhythm) {
            CaptureBox()
            section("Waiting on you") {
                Text("Nothing waiting on you.").foregroundStyle(.secondary)
            }
            InboxSection(now: now)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content)
        -> some View
    {
        VStack(alignment: .leading, spacing: TodayLayout.rowSpacing) {
            Text(title).fontWeight(.semibold)
            content()
        }
    }
}

private struct CaptureBox: View {
    @Environment(CaptureService.self) private var capture
    @State private var text = ""

    var body: some View {
        VStack(alignment: .leading, spacing: TodayLayout.rowSpacing) {
            Text("Capture").fontWeight(.semibold)
            TextField("Remind me to…", text: $text)
                .textFieldStyle(.roundedBorder)
                .onSubmit(submit)
            if let outcome = capture.lastOutcome {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(outcome.line)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 0)
                    if case .added = outcome {
                        Button("Undo") { Task { await capture.undoLast() } }
                            .buttonStyle(.plain)
                            .foregroundStyle(Color.accentColor)
                            .focusable(false)
                    }
                }
            }
        }
    }

    private func submit() {
        let captured = text
        text = ""
        Task { await capture.capture(captured, source: "today") }
    }
}

private struct InboxSection: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CompanionRuntime.self) private var runtime
    let now: Date

    var body: some View {
        let inboxID = agenda.inbox?.id
        let items = agenda.snapshot.open.filter { $0.due == nil && $0.listID == inboxID }
        VStack(alignment: .leading, spacing: TodayLayout.rowSpacing) {
            Text("Inbox").fontWeight(.semibold)
            if items.isEmpty {
                Text("Inbox is clear.").foregroundStyle(.secondary)
            }
            ForEach(items.prefix(12)) { reminder in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(reminder.title).lineLimit(2)
                    Spacer(minLength: 4)
                    if runtime.state.plan.contains(where: { $0.reminderID == reminder.id }) {
                        Text("Planned").foregroundStyle(.secondary)
                    } else {
                        Button("Find a time") { findTime(for: reminder) }
                            .buttonStyle(.plain)
                            .foregroundStyle(Color.accentColor)
                            .focusable(false)
                    }
                }
            }
            if items.count > 12 {
                Text("…and \(items.count - 12) more in Reminders.").foregroundStyle(.secondary)
            }
        }
    }

    private func findTime(for reminder: AgendaReminder) {
        let facts = DayFacts(
            snapshot: agenda.snapshot, areas: agenda.areas, inboxListID: agenda.inbox?.id,
            now: now, mustDoID: runtime.state.mustDoID, plan: runtime.state.plan)
        let minutes = 30
        guard let start = TimelineBuilder.firstFreeSlot(minutes: minutes, facts: facts) else {
            return
        }
        runtime.act(.place(reminderID: reminder.id, start: start, minutes: minutes))
    }
}
