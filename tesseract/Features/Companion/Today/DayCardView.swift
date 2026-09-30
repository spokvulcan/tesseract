//
//  DayCardView.swift
//  tesseract
//
//  A moment's card at the top of Today: the Morning Plan and the Evening
//  Wrap-up in this slice. Plain content-layer cards; every action is one
//  click, and nothing is ever labelled missed or failed.
//

import SwiftUI

struct DayCardView: View {
    @Environment(CompanionRuntime.self) private var runtime
    let card: DayCard
    let facts: DayFacts

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(card.kind.title).fontWeight(.semibold)
                if card.isFallback {
                    Text("· Jarvis couldn't think this one through; here are the facts.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    runtime.act(.dismiss(cardID: card.id))
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .focusable(false)
                .help("Dismiss")
            }
            Text(card.line).fixedSize(horizontal: false, vertical: true)
            switch card.body {
            case .morningPlan(let plan):
                MorningPlanBody(card: plan, facts: facts)
            case .eveningWrapUp(let wrapUp):
                EveningWrapUpBody(cardID: card.id, card: wrapUp)
            case .breakpoint, .triage, .reflection:
                EmptyView()
            }
        }
        .padding(14)
        .background(
            .quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: Theme.Radius.medium))
    }
}

private struct MorningPlanBody: View {
    let card: MorningPlanCard
    let facts: DayFacts

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let mustDo = card.mustDoID, let task = facts.task(mustDo) {
                Label("Must-do: \(task.title)", systemImage: "star.fill")
                    .foregroundStyle(Color.accentColor)
            }
            if !card.placements.isEmpty {
                Text(
                    "\(card.placements.count) task\(card.placements.count == 1 ? "" : "s") placed in your day below."
                )
                .foregroundStyle(.secondary)
            }
            ForEach(card.suggestions, id: \.self) { suggestion in
                Text("· \(suggestion)").foregroundStyle(.secondary)
            }
        }
    }
}

private struct EveningWrapUpBody: View {
    @Environment(CompanionRuntime.self) private var runtime
    let cardID: String
    let card: EveningWrapUpCard

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !card.done.isEmpty {
                Text("Done: " + card.done.joined(separator: " · "))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !card.leftovers.isEmpty {
                ForEach(card.leftovers) { leftover in
                    HStack(spacing: 8) {
                        Text(leftover.title).lineLimit(1)
                        Spacer(minLength: 8)
                        choice("Tomorrow", .tomorrow, leftover)
                        choice("Later", .later, leftover)
                        choice("Let go", .drop, leftover)
                    }
                }
                Button("Do What Jarvis Suggests for All") {
                    runtime.act(.allLeftovers(cardID: cardID))
                }
                .controlSize(.small)
                .focusable(false)
            }
            if let first = card.tomorrowFirst {
                Text("Tomorrow starts with \(first).").foregroundStyle(.secondary)
            }
        }
    }

    private func choice(_ title: String, _ suggestion: Leftover.Suggestion, _ leftover: Leftover)
        -> some View
    {
        Button(title) {
            runtime.act(.leftover(cardID: cardID, reminderID: leftover.reminderID, suggestion))
        }
        .controlSize(.small)
        .buttonStyle(.bordered)
        .tint(leftover.suggestion == suggestion ? .accentColor : nil)
        .focusable(false)
    }
}
