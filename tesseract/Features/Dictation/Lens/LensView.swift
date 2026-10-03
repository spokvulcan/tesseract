//
//  LensView.swift
//  tesseract
//
//  The **Lens** card (PRD #612): the take's words, the word the typed one
//  will replace, the field the owner types into, and afterwards what the
//  fix did. One type size and one rhythm (design language §2); the target
//  and fixed words carry the accent, caught words a dotted underline.
//
//  The macOS 27.0 focus freeze: the text field is in the first layout and
//  only ever disabled, never inserted later, and every button is
//  non-focusable (`overlayAffordance()`).
//

import SwiftUI

enum LensStyle {
    static let width: CGFloat = 640
    static let fontSize: CGFloat = 15
    static let rhythm: CGFloat = 10
    static let padding: CGFloat = 18
    static let maxHeight: CGFloat = 340
    static let minHeight: CGFloat = 96
}

/// The few clicks the Lens sends back to its controller.
@MainActor
struct LensActions {
    let undo: @MainActor () -> Void
    let close: @MainActor () -> Void

    static let none = LensActions(undo: {}, close: {})
}

struct LensView: View {
    @Bindable var model: LensModel
    let actions: LensActions
    /// The content's height, for the panel to fit it.
    var onHeightChange: (CGFloat) -> Void = { _ in }

    @FocusState private var fieldFocused: Bool
    @State private var wordsHeight: CGFloat = 0

    /// What is left of the card for the words once the header and the
    /// field have their rows.
    private static let wordsMaxHeight = LensStyle.maxHeight - 2 * LensStyle.padding - 64
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorSchemeContrast) private var contrast

    var body: some View {
        VStack(alignment: .leading, spacing: LensStyle.rhythm) {
            header
            if model.phase == .done, let detail = model.result?.detail {
                Text(detail.prefix(1).uppercased() + detail.dropFirst())
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    words
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .onGeometryChange(for: CGFloat.self) {
                            $0.size.height
                        } action: {
                            wordsHeight = $0
                        }
                }
                .scrollIndicators(.never)
                // As tall as the words, up to the card's limit; longer takes
                // scroll to the word being fixed.
                .frame(height: min(max(wordsHeight, LensStyle.fontSize + 6), Self.wordsMaxHeight))
                .onChange(of: model.target) { _, target in
                    guard let target else { return }
                    proxy.scrollTo(target.start, anchor: .center)
                }
            }
            footer
        }
        .font(.system(size: LensStyle.fontSize))
        .padding(LensStyle.padding)
        .frame(width: LensStyle.width, alignment: .topLeading)
        .onGeometryChange(for: CGFloat.self) {
            $0.size.height
        } action: {
            onHeightChange($0)
        }
        .onChange(of: model.focusRequest) { fieldFocused = true }
        .animation(reduceMotion ? nil : .snappy(duration: 0.18), value: model.text)
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: model.phase == .done ? doneSymbol : "arrow.uturn.backward")
                .foregroundStyle(
                    model.phase == .done ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary)
                )
                .accessibilityHidden(true)
            Text(headerLine)
                .foregroundStyle(model.phase == .done ? .primary : .secondary)
                .lineLimit(1)
            Spacer(minLength: 8)
            if let note = noteLine {
                Text(note)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            if canUndo {
                Button("Undo", action: actions.undo)
                    .overlayAffordance()
                    .foregroundStyle(.tint)
                    .help("Forget what this fix taught")
            }
        }
        .fontWeight(.medium)
    }

    private var doneSymbol: String {
        model.result?.detail == nil ? "checkmark.circle.fill" : "info.circle"
    }

    private var headerLine: String {
        if model.phase == .done, let result = model.result { return result.line }
        guard let take = model.take else { return "" }
        let app = take.app?.name
        switch model.mode {
        case .afterPaste:
            return app.map { take.pasted ? "Last take · in \($0)" : "Last take · for \($0)" }
                ?? "Last take"
        case .fromPage:
            return "Take · \(take.at.formatted(date: .omitted, time: .shortened))"
        }
    }

    private var noteLine: String? {
        if model.note == .undone { return "Undone" }
        if model.phase == .done, let summary = model.learnedSummary { return summary }
        switch model.note {
        case .learned(let heard, let meant): return "Learned \(heard) → \(meant)"
        case .leftAlone(let word, let app): return "\(word) left alone in \(app)"
        case .thisTakeOnly: return "Fixed here only"
        case .undone: return "Undone"
        case nil: return nil
        }
    }

    private var canUndo: Bool {
        !model.receipts.isEmpty && model.note != .undone
    }

    // MARK: Words

    private var words: some View {
        LensFlowLayout(spacing: 4, lineSpacing: 4) {
            ForEach(Array(model.tokens.enumerated()), id: \.offset) { index, token in
                tokenView(index, token)
                    .id(index)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(model.text)
        .accessibilityValue(targetDescription)
    }

    @ViewBuilder
    private func tokenView(_ index: Int, _ token: TakeToken) -> some View {
        let target = model.target
        let isTarget = target?.range.contains(index) ?? false
        if isTarget, let target, index != target.start {
            // The rest of a multi-word target folds into its first word.
            EmptyView()
        } else if isTarget, !model.typed.isEmpty {
            editedTarget
        } else {
            Text(isTarget ? targetText(target) : token.text)
                .fontWeight(model.fixedTokens.contains(index) ? .semibold : .regular)
                .foregroundStyle(tokenStyle(index, isTarget: isTarget))
                .underline(
                    model.catchAt(index) != nil && !model.fixedTokens.contains(index),
                    pattern: .dot, color: .secondary
                )
                .padding(.horizontal, isTarget ? 4 : 0)
                .background(
                    RoundedRectangle(cornerRadius: 5)
                        .fill(isTarget ? AnyShapeStyle(.tint.opacity(0.18)) : AnyShapeStyle(.clear))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 5)
                        .strokeBorder(
                            .tint, lineWidth: isTarget && contrast == .increased ? 1.5 : 0)
                )
                .help(model.catchAt(index).map { "Heard \($0.heard)" } ?? "")
                .onTapGesture { model.pick(index) }
        }
    }

    private func targetText(_ target: LensFix.Span?) -> String {
        guard let target else { return "" }
        return model.tokens[target.range].map(\.text).joined(separator: " ")
    }

    /// The target while typing: the word a commit would write, the part
    /// typed so far in full ink and the rest of its completion as a ghost.
    private var editedTarget: some View {
        let typedCount = model.typed.trimmingCharacters(in: .whitespaces).count
        let shown = model.completion.map { String($0.prefix(typedCount)) } ?? model.typed
        return Text(
            "\(Text(shown).foregroundStyle(.primary))\(Text(model.completionSuffix).foregroundStyle(.tertiary))"
        )
        .fontWeight(.semibold)
        .padding(.horizontal, 4)
        .background(RoundedRectangle(cornerRadius: 5).fill(.tint.opacity(0.18)))
    }

    private func tokenStyle(_ index: Int, isTarget: Bool) -> AnyShapeStyle {
        if isTarget || model.fixedTokens.contains(index) { return AnyShapeStyle(.primary) }
        return AnyShapeStyle(model.phase == .done ? .secondary : .primary)
    }

    private var targetDescription: String {
        guard let target = model.target else { return "" }
        return "Replaces \(targetText(target))"
    }

    // MARK: Footer

    private var footer: some View {
        HStack(spacing: 12) {
            // In the first layout, always: disabled, never removed.
            TextField("Type the word you meant", text: $model.typed)
                .textFieldStyle(.plain)
                .focused($fieldFocused)
                .disabled(model.phase != .fixing)
                .frame(width: 180)
                .opacity(model.phase == .fixing ? 1 : 0)
                .accessibilityLabel("The word you meant")
            Text(hints)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.head)
                .frame(maxWidth: .infinity, alignment: .trailing)
                .opacity(model.phase == .fixing ? 1 : 0)
        }
        // Collapsed, never removed, once the fix is done (the focus freeze).
        .frame(height: model.phase == .fixing ? 22 : 0)
        .clipped()
    }

    private var hints: String {
        let finish = finishLabel
        if model.typed.trimmingCharacters(in: .whitespaces).isEmpty {
            if model.isManualTarget { return "Type the word that goes here · esc back" }
            if model.fixCount > 0 { return "↩ \(finish) · esc close" }
            return "← → pick a word · esc close"
        }
        guard let target = model.target else {
            return "Nothing here sounds like it · ← → pick"
        }
        return "Replaces \(targetText(target)) · ⇥ next · ↩ \(finish)"
    }

    private var finishLabel: String {
        guard let take = model.take else { return "done" }
        switch model.mode {
        case .afterPaste:
            if take.pasted, let app = take.app?.name { return "fix in \(app)" }
            return "fix and learn"
        case .fromPage:
            return "fix and learn"
        }
    }
}

/// Words laid out like text: left to right, wrapping at the width.
struct LensFlowLayout: Layout {
    var spacing: CGFloat
    var lineSpacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        let rows = arrange(subviews, width: width)
        let height = rows.last.map { $0.y + $0.height } ?? 0
        let used = rows.map(\.width).max() ?? 0
        return CGSize(width: proposal.width ?? used, height: height)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        for row in arrange(subviews, width: bounds.width) {
            for item in row.items {
                subviews[item.index].place(
                    at: CGPoint(x: bounds.minX + item.x, y: bounds.minY + row.y),
                    proposal: ProposedViewSize(item.size))
            }
        }
    }

    private struct Row {
        var items: [(index: Int, x: CGFloat, size: CGSize)] = []
        var y: CGFloat = 0
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    private func arrange(_ subviews: Subviews, width: CGFloat) -> [Row] {
        var rows: [Row] = []
        var row = Row()
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            guard size.width > 0 else { continue }
            let x = row.items.isEmpty ? 0 : row.width + spacing
            if !row.items.isEmpty, x + size.width > width {
                rows.append(row)
                row = Row(y: row.y + row.height + lineSpacing)
                row.items.append((index, 0, size))
                row.width = size.width
                row.height = size.height
                continue
            }
            row.items.append((index, x, size))
            row.width = x + size.width
            row.height = max(row.height, size.height)
        }
        if !row.items.isEmpty { rows.append(row) }
        return rows
    }
}
