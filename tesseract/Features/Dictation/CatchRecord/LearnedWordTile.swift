//
//  LearnedWordTile.swift
//  tesseract
//
//  One **Learned Word** on the catch record (PRD #612): the spelling the
//  owner meant, every way it was heard, when it was taught, the fix that
//  taught it as a before-and-after strip, and what it has caught since.
//  Content layer: a quiet fill and a hairline, no glass (design language
//  §1). Forget lives in the context menu (§2: actions in context menus).
//

import SwiftUI

struct LearnedWordTile: View {
    let tile: CatchRecord.Tile
    /// The page's clock, so "taught today" rolls over with the record.
    let now: Date
    let onForget: () -> Void
    var calendar: Calendar = .current

    @Environment(\.accessibilityDifferentiateWithoutColor) private var differentiateWithoutColor

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            header
            if let example = tile.example {
                Text(
                    Self.strip(
                        example, tile: tile, differentiateWithoutColor: differentiateWithoutColor)
                )
                .foregroundStyle(.secondary)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
            }
            footer
        }
        .font(.system(size: DictationPageStyle.bodySize))
        .padding(DictationPageStyle.rhythm)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.medium)
                .fill(Color.primary.opacity(0.04))
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.medium)
                .strokeBorder(.separator, lineWidth: 1)
        )
        .contentShape(RoundedRectangle(cornerRadius: Theme.Radius.medium))
        .contextMenu {
            Button(action: onForget) {
                Label("Forget", systemImage: "minus.circle")
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Self.accessibilityLabel(for: tile, now: now, calendar: calendar))
        .accessibilityAction(named: "Forget", onForget)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(tile.meant)
                .fontWeight(.semibold)
                .foregroundStyle(.primary)
                .lineLimit(1)
                .layoutPriority(2)
            Text(tile.heard.joined(separator: ", "))
                .foregroundStyle(.secondary)
                .strikethrough()
                .lineLimit(1)
                .truncationMode(.tail)
                .help("Heard as \(tile.heard.formatted(.list(type: .and)))")
        }
    }

    /// What it caught, and when it was taught on the right: the heard forms
    /// get the header's room.
    private var footer: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(Self.stats(for: tile))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Text(Self.taught(tile.learnedAt, now: now, calendar: calendar))
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .fixedSize()
                .help(tile.learnedAt.formatted(date: .long, time: .shortened))
        }
    }

    // MARK: - Words

    /// When the word was taught: "taught today", "taught yesterday",
    /// "taught Tue" within the week, else "taught Sep 12" (with the year
    /// when it isn't this one).
    static func taught(
        _ date: Date, now: Date, calendar: Calendar = .current,
        weekday: Date.FormatStyle.Symbol.Weekday = .abbreviated
    ) -> String {
        let days =
            calendar.dateComponents(
                [.day], from: calendar.startOfDay(for: date), to: calendar.startOfDay(for: now)
            ).day ?? 0
        switch days {
        case ..<1: return "taught today"
        case 1: return "taught yesterday"
        case 2..<7: return "taught \(date.formatted(.dateTime.weekday(weekday)))"
        default:
            let sameYear = calendar.isDate(date, equalTo: now, toGranularity: .year)
            let format =
                sameYear
                ? Date.FormatStyle.dateTime.month(.abbreviated).day()
                : Date.FormatStyle.dateTime.month(.abbreviated).day().year()
            return "taught \(date.formatted(format))"
        }
    }

    /// A stretch of the stats line; numbers wear the primary ink.
    struct StatPiece: Equatable {
        let text: String
        var isNumber = false
    }

    /// The stats line's parts, in order: "Caught 4 times", "2 this week"
    /// (when some but not all were this week), "left alone in Notes".
    static func statParts(for tile: CatchRecord.Tile) -> [[StatPiece]] {
        var parts: [[StatPiece]] = []
        if tile.caught > 0 {
            parts.append([
                StatPiece(text: "Caught "), StatPiece(text: "\(tile.caught)", isNumber: true),
                StatPiece(text: tile.caught == 1 ? " time" : " times"),
            ])
            if tile.caughtThisWeek > 0, tile.caughtThisWeek < tile.caught {
                parts.append([
                    StatPiece(text: "\(tile.caughtThisWeek)", isNumber: true),
                    StatPiece(text: " this week"),
                ])
            }
        } else {
            parts.append([StatPiece(text: "Not caught yet")])
        }
        if !tile.leftAloneIn.isEmpty {
            parts.append([
                StatPiece(text: "left alone in \(tile.leftAloneIn.formatted(.list(type: .and)))")
            ])
        }
        return parts
    }

    /// "Caught 4 times · 2 this week · left alone in Notes", the numbers in
    /// the primary ink and the rest in the text's own (secondary).
    static func stats(for tile: CatchRecord.Tile) -> AttributedString {
        var result = AttributedString()
        for (index, part) in statParts(for: tile).enumerated() {
            if index > 0 { result.append(AttributedString(" · ")) }
            for piece in part {
                var run = AttributedString(piece.text)
                if piece.isNumber {
                    run.foregroundColor = Color.primary
                }
                result.append(run)
            }
        }
        return result
    }

    /// The fix that taught the word: the take before it with the heard form
    /// struck, an arrow, and the take after it with the spelling in the
    /// caught-word treatment.
    static func strip(
        _ example: LearnedWord.Example, tile: CatchRecord.Tile, differentiateWithoutColor: Bool
    ) -> AttributedString {
        var result = AttributedString()
        for run in CatchRecord.runs(example.before, marking: tile.heard) {
            var piece = AttributedString(run.text)
            if run.marked {
                piece.foregroundColor = Color.primary
                piece.strikethroughStyle = .single
            }
            result.append(piece)
        }
        result.append(AttributedString("  →  "))
        for run in CatchRecord.runs(example.after, marking: [tile.meant]) {
            var piece = AttributedString(run.text)
            if run.marked {
                CaughtWordStyle.mark(&piece, differentiateWithoutColor: differentiateWithoutColor)
            }
            result.append(piece)
        }
        return result
    }

    /// The whole tile in one sentence for VoiceOver.
    static func accessibilityLabel(
        for tile: CatchRecord.Tile, now: Date, calendar: Calendar = .current
    ) -> String {
        var sentences = [
            "\(tile.meant), heard as \(tile.heard.formatted(.list(type: .or)))",
            taught(tile.learnedAt, now: now, calendar: calendar, weekday: .wide).capitalizedFirst,
        ]
        if let example = tile.example {
            sentences.append("Was \(example.before), now \(example.after)")
        }
        let stats = statParts(for: tile)
            .map { $0.map(\.text).joined() }
            .joined(separator: ", ")
        sentences.append(stats.capitalizedFirst)
        return sentences.joined(separator: ". ") + "."
    }
}

extension String {
    /// The string with its first letter uppercased, the rest untouched.
    fileprivate var capitalizedFirst: String {
        prefix(1).uppercased() + dropFirst()
    }
}
