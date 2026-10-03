//
//  TodayTakeRow.swift
//  tesseract
//
//  One of today's takes on the catch record (PRD #612): when it was said,
//  its text with the **Catches** marked, and what happened to it. A click
//  opens it in the **Lens** to fix a word. `CaughtText`, the take's text in
//  the Lens's caught-word treatment, is shared with the history.
//

import AppKit
import SwiftUI

/// The treatment the Lens gives a caught word: the accent ink. With
/// Differentiate Without Color a dotted underline joins it, so a catch never
/// rides on color alone.
enum CaughtWordStyle {
    /// `runs` as one attributed string, the marked runs in the treatment.
    static func text(
        _ runs: [CatchRecord.Run], differentiateWithoutColor: Bool
    ) -> AttributedString {
        var result = AttributedString()
        for run in runs {
            var piece = AttributedString(run.text)
            if run.marked {
                mark(&piece, differentiateWithoutColor: differentiateWithoutColor)
            }
            result.append(piece)
        }
        return result
    }

    static func mark(_ piece: inout AttributedString, differentiateWithoutColor: Bool) {
        piece.foregroundColor = Color.accentColor
        if differentiateWithoutColor {
            piece.underlineStyle = Text.LineStyle(pattern: .dot, color: Color.accentColor)
        }
    }
}

/// A take's text with its catches in the caught-word treatment, at the
/// page's one type size. Selectable only where its container turns
/// selection on: the history's rows do; today's rows are buttons and don't.
struct CaughtText: View {
    let text: String
    let catches: [LearnedWordCatch]

    @Environment(\.accessibilityDifferentiateWithoutColor) private var differentiateWithoutColor

    init(text: String, catches: [LearnedWordCatch]) {
        self.text = text
        self.catches = catches
    }

    var body: some View {
        Text(
            CaughtWordStyle.text(
                CatchRecord.runs(text, marking: catches.map(\.tokenRange)),
                differentiateWithoutColor: differentiateWithoutColor)
        )
        .font(.system(size: DictationPageStyle.bodySize))
        .fixedSize(horizontal: false, vertical: true)
        .help(Self.help(for: catches))
    }

    /// What the pointer shows over a take with catches: "Caught cloud as
    /// Claude".
    static func help(for catches: [LearnedWordCatch]) -> String {
        guard !catches.isEmpty else { return "" }
        let pairs = catches.map { "\($0.heard) as \($0.meant)" }
        return "Caught \(pairs.formatted(.list(type: .and)))"
    }
}

/// One of today's takes: a plain button that opens it in the Lens.
struct TodayTakeRow: View {
    let take: CatchRecord.Take

    @Environment(\.fixInLens) private var fixInLens
    @State private var isHovered = false

    /// What a take's trailing tag says: "fixed" once a word in it was fixed
    /// in the Lens, else how many words were caught, else nothing.
    static func tag(for take: CatchRecord.Take) -> String? {
        if take.fixes > 0 { return "fixed" }
        if !take.catches.isEmpty { return "\(take.catches.count) caught" }
        return nil
    }

    private var time: String {
        take.at.formatted(date: .omitted, time: .shortened)
    }

    var body: some View {
        Button(action: fix) {
            HStack(alignment: .firstTextBaseline, spacing: DictationPageStyle.rhythm) {
                Text(time)
                    .font(.system(size: DictationPageStyle.bodySize))
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
                    // A floor, not a cap: a longer day period ("p. m.")
                    // widens the column instead of clipping the time.
                    .fixedSize()
                    .frame(minWidth: 72, alignment: .leading)

                CaughtText(text: take.text, catches: take.catches)
                    .textSelection(.disabled)
                    .foregroundStyle(.primary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                if let tag = Self.tag(for: take) {
                    Text(tag)
                        .font(.system(size: DictationPageStyle.bodySize))
                        .foregroundStyle(take.fixes > 0 ? .secondary : .tertiary)
                        .monospacedDigit()
                        .lineLimit(1)
                        .fixedSize()
                }
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 10)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(isHovered ? Color.primary.opacity(0.04) : Color.clear)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }  // No animation, like the history row.
        .contextMenu {
            Button(action: fix) {
                Label("Fix a Word…", systemImage: "character.cursor.ibeam")
            }
            Button(action: copy) {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }
        .accessibilityLabel("\(time). \(take.text)")
        .accessibilityValue(Self.tag(for: take) ?? "")
        .accessibilityHint("Opens the take in the Lens to fix a word")
        .accessibilityAction(named: "Copy", copy)
    }

    /// The Lens refuses while a take is being recorded or a fix is being
    /// put back in an app: a beep says so.
    private func fix() {
        if !fixInLens(take.lensTake) {
            NSSound.beep()
        }
    }

    private func copy() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(take.text, forType: .string)
    }
}
