//
//  LensView.swift
//  tesseract
//
//  The **Lens** card (PRD #612): while the owner talks, the **Live
//  Preview** (confirmed words in full ink, the provisional tail dimmed,
//  learned words flipping to the owner's spelling as they arrive); when the
//  take lands, the words the preview had wrong settle into place; while a
//  take waits or is fixed, the word the typed one will replace, the field
//  the owner types into, and afterwards what the fix did. One type size and
//  one rhythm (design language §2); the target, fixed and caught words carry
//  the accent.
//
//  The macOS 27.0 focus freeze: the text field is in the first layout and
//  only ever disabled, never inserted later, and every button is
//  non-focusable (`overlayAffordance()`). Reduce Motion turns flips and
//  settles into crossfades; Reduce Transparency puts an opaque card behind
//  the words; Increase Contrast outlines the target.
//

import SwiftUI

enum LensStyle {
    static let width: CGFloat = 640
    static let fontSize: CGFloat = 15
    static let rhythm: CGFloat = 10
    static let padding: CGFloat = 18
    static let maxHeight: CGFloat = 340
    static let minHeight: CGFloat = 56
}

/// The few clicks the Lens sends back to its controller.
@MainActor
struct LensActions {
    let undo: @MainActor () -> Void
    let close: @MainActor () -> Void
    var insertRawAnyway: @MainActor () -> Void = {}

    static let none = LensActions(undo: {}, close: {})
}

struct LensView: View {
    @Bindable var model: LensModel
    /// The dictation feed, for the level while listening (nil in tests).
    var feed: DictationFeed?
    let actions: LensActions
    /// The content's height, for the panel to fit it.
    var onHeightChange: (CGFloat) -> Void = { _ in }

    @FocusState private var fieldFocused: Bool
    @State private var wordsHeight: CGFloat = 0

    /// What is left of the card for the words once the header and the
    /// field have their rows.
    private static let wordsMaxHeight = LensStyle.maxHeight - 2 * LensStyle.padding - 64
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast

    private var isLive: Bool { model.phase == .listening || model.phase == .finishing }

    var body: some View {
        // Spacing goes on the rows that are there, so an empty words area or
        // a collapsed field adds no gap.
        VStack(alignment: .leading, spacing: 0) {
            header
            if model.phase == .done, let detail = model.result?.detail {
                Text(detail.prefix(1).uppercased() + detail.dropFirst())
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .padding(.top, LensStyle.rhythm)
            }
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    Group {
                        if isLive { liveWords } else { words }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onGeometryChange(for: CGFloat.self) {
                        $0.size.height
                    } action: {
                        wordsHeight = $0
                    }
                }
                .scrollIndicators(.never)
                // As tall as the words, up to the card's limit; longer takes
                // scroll to the word being fixed, or to the newest words.
                .frame(height: wordsFrameHeight)
                .padding(.top, wordsFrameHeight > 0 ? LensStyle.rhythm : 0)
                .onChange(of: model.target) { _, target in
                    guard let target else { return }
                    proxy.scrollTo(target.start, anchor: .center)
                }
                .onChange(of: model.preview) { _, _ in
                    let last = model.liveTokens.count - 1
                    if last >= 0 { proxy.scrollTo(last, anchor: .bottom) }
                }
            }
            footer
        }
        .font(.system(size: LensStyle.fontSize))
        .padding(LensStyle.padding)
        .frame(width: LensStyle.width, alignment: .topLeading)
        // Reduce Transparency: an opaque card behind the words, whatever the
        // glass does with what is behind it.
        .background(
            reduceTransparency ? Color(nsColor: .windowBackgroundColor) : Color.clear
        )
        .onGeometryChange(for: CGFloat.self) {
            $0.size.height
        } action: {
            onHeightChange($0)
        }
        .onChange(of: model.focusRequest) { fieldFocused = true }
        .animation(reduceMotion ? nil : .snappy(duration: 0.18), value: model.text)
        .animation(reduceMotion ? nil : .snappy(duration: 0.22), value: model.preview)
        .animation(reduceMotion ? nil : .snappy(duration: 0.22), value: model.phase)
    }

    private var wordsFrameHeight: CGFloat {
        let hasWords = isLive ? !model.liveTokens.isEmpty : !model.tokens.isEmpty
        guard hasWords || model.phase == .fixing else { return 0 }
        return min(max(wordsHeight, LensStyle.fontSize + 6), Self.wordsMaxHeight)
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 8) {
            leadingMark
            Text(headerLine)
                .foregroundStyle(headerIsStrong ? .primary : .secondary)
                .lineLimit(1)
            if let count = caughtCount, count > 0 {
                Text("\(count) caught")
                    .foregroundStyle(.tint)
            }
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
            if model.phase == .done, model.canInsertRaw {
                Button("Insert anyway", action: actions.insertRawAnyway)
                    .overlayAffordance()
                    .foregroundStyle(.tint)
            }
        }
        .fontWeight(.medium)
    }

    @ViewBuilder
    private var leadingMark: some View {
        switch model.phase {
        case .listening:
            LensLevelDot(feed: feed, reduceMotion: reduceMotion)
                .accessibilityHidden(true)
        case .finishing:
            ProgressView()
                .controlSize(.small)
                .accessibilityHidden(true)
        case .landed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.tint)
                .accessibilityHidden(true)
        case .done:
            Image(systemName: doneSymbol)
                .foregroundStyle(.tint)
                .accessibilityHidden(true)
        case .fixing:
            Image(systemName: model.mode == .held ? "eye" : "arrow.uturn.backward")
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
        case .hidden:
            EmptyView()
        }
    }

    private var headerIsStrong: Bool {
        model.phase == .done || model.phase == .landed
            || (model.phase == .fixing && model.mode == .held)
    }

    private var doneSymbol: String {
        model.result?.detail == nil ? "checkmark.circle.fill" : "info.circle"
    }

    private var appName: String? { model.liveApp?.name ?? model.take?.app?.name }

    private var headerLine: String {
        switch model.phase {
        case .listening:
            return appName.map { "Listening · \($0)" } ?? "Listening"
        case .finishing:
            return model.isHeld ? "Finishing · it will wait for you" : "Finishing"
        case .landed:
            guard let take = model.landed else { return "" }
            let app = (take.pastedInto ?? take.app)?.name
            if take.pasted { return app.map { "Pasted into \($0)" } ?? "Pasted" }
            // Automatically Insert Text is off: the take is only kept.
            return "Saved in the Dictation history"
        case .done:
            return model.result?.line ?? ""
        case .fixing:
            guard let take = model.take else { return "" }
            let app = take.app?.name
            switch model.mode {
            case .held:
                return app.map { "Waiting for you · not pasted into \($0) yet" }
                    ?? "Waiting for you · not pasted yet"
            case .afterPaste:
                return app.map { take.pasted ? "Last take · in \($0)" : "Last take · for \($0)" }
                    ?? "Last take"
            case .fromPage:
                return "Take · \(take.at.formatted(date: .omitted, time: .shortened))"
            }
        case .hidden:
            return ""
        }
    }

    private var caughtCount: Int? {
        switch model.phase {
        case .listening, .finishing: model.preview?.catches.count
        case .landed: model.landed?.catches.count
        default: nil
        }
    }

    private var noteLine: String? {
        switch model.phase {
        case .listening:
            return model.isHeld ? "Will wait for you" : model.holdHint
        case .finishing:
            return nil
        case .landed:
            return "Missed one? \(model.fixHotkeyLabel)"
        default:
            break
        }
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
        (model.phase == .fixing || model.phase == .done) && !model.receipts.isEmpty
            && model.note != .undone
    }

    // MARK: Live words

    /// The preview: confirmed words in full ink, the tail dimmed, caught
    /// words flipped to the owner's spelling.
    private var liveWords: some View {
        let tokens = model.liveTokens
        let confirmed = model.preview?.confirmedTokens ?? 0
        let caught = Set((model.preview?.catches ?? []).flatMap { Array($0.tokenRange) })
        return Group {
            if tokens.isEmpty {
                Text(model.phase == .finishing ? "" : "Listening…")
                    .foregroundStyle(.tertiary)
            } else {
                LensFlowLayout(spacing: 4, lineSpacing: 4) {
                    ForEach(Array(tokens.enumerated()), id: \.offset) { index, token in
                        // The word's own identity is its text, so a rewritten
                        // word flips in; the slot keeps the index to scroll to.
                        ZStack {
                            Text(token.text)
                                .foregroundStyle(
                                    liveStyle(
                                        index: index, confirmed: confirmed,
                                        caught: caught.contains(index))
                                )
                                .id(token.text)
                                .transition(reduceMotion ? .opacity : .lensFlip)
                        }
                        .id(index)
                    }
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(model.preview?.text ?? "")
            }
        }
    }

    private func liveStyle(index: Int, confirmed: Int, caught: Bool) -> AnyShapeStyle {
        if caught { return AnyShapeStyle(.tint) }
        if model.phase == .finishing || index >= confirmed { return AnyShapeStyle(.secondary) }
        return AnyShapeStyle(.primary)
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
        if model.phase == .landed {
            // The words the preview had wrong settle in the accent; caught
            // words keep it too.
            if model.settled.contains(index) || model.catchAt(index) != nil {
                return AnyShapeStyle(.tint)
            }
            return AnyShapeStyle(.primary)
        }
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
        // Collapsed, never removed, outside fixing (the focus freeze).
        .frame(height: model.phase == .fixing ? 22 : 0)
        .clipped()
        .padding(.top, model.phase == .fixing ? LensStyle.rhythm : 0)
    }

    private var hints: String {
        let finish = finishLabel
        if model.typed.trimmingCharacters(in: .whitespaces).isEmpty {
            if model.isManualTarget { return "Type the word that goes here · esc back" }
            if model.mode == .held { return "↩ paste · esc keep for later" }
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
        case .held:
            return "fix and paste"
        case .afterPaste:
            if take.pasted, let app = (take.pastedInto ?? take.app)?.name { return "fix in \(app)" }
            return "fix and learn"
        case .fromPage:
            return "fix and learn"
        }
    }
}

/// The listening mark: a dot that swells with the voice.
private struct LensLevelDot: View {
    var feed: DictationFeed?
    var reduceMotion: Bool

    var body: some View {
        let level = CGFloat(feed?.level ?? 0)
        Circle()
            .fill(.red)
            .frame(width: 9, height: 9)
            .scaleEffect(reduceMotion ? 1 : 1 + level * 0.7)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.08), value: level)
            .frame(width: 16, height: 16)
    }
}

/// A word flipping into place (a learned word turning into the owner's
/// spelling, a provisional word rewritten by the next decode).
private struct LensFlipModifier: ViewModifier {
    let angle: Double

    func body(content: Content) -> some View {
        content
            .rotation3DEffect(.degrees(angle), axis: (x: 1, y: 0, z: 0))
            .opacity(angle == 0 ? 1 : 0)
    }
}

extension AnyTransition {
    fileprivate static var lensFlip: AnyTransition {
        .asymmetric(
            insertion: .modifier(
                active: LensFlipModifier(angle: -90), identity: LensFlipModifier(angle: 0)),
            removal: .opacity)
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
