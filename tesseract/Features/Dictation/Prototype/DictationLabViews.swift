//
//  DictationLabViews.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  Pieces the five variants share: the heard → meant pair (the signature:
//  what the machine heard in mono, what you meant in the body face), the
//  fix field (a text field holding a take with the suspect word selected),
//  the toast card, and the recording pill the overlay keeps.
//

import AppKit
import SwiftUI

enum LabStyle {
    static let body: CGFloat = 14
    static let rhythm: CGFloat = 12
    static let column: CGFloat = 720
    static let fixShortcut = "⌃⌥Space"
}

// MARK: - Heard → meant

/// The signature element: a correction as a pair. The misheard form is set
/// in mono and struck (it is machine output), the word you meant in the
/// body face with the accent underline every learned word wears.
struct HeardMeant: View {
    let heard: String
    let meant: String
    var size: CGFloat = LabStyle.body

    var body: some View {
        HStack(spacing: 6) {
            Text(heard)
                .font(.system(size: size - 1, design: .monospaced))
                .foregroundStyle(.secondary)
                .strikethrough(true, color: .secondary.opacity(0.6))
            Image(systemName: "arrow.right")
                .font(.system(size: size - 4, weight: .semibold))
                .foregroundStyle(.tertiary)
            LearnedWord(meant, size: size)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(heard) becomes \(meant)")
    }
}

/// A word the app learned: body face, medium weight, accent underline.
struct LearnedWord: View {
    let text: String
    var size: CGFloat = LabStyle.body
    init(_ text: String, size: CGFloat = LabStyle.body) {
        self.text = text
        self.size = size
    }
    var body: some View {
        Text(text)
            .font(.system(size: size, weight: .medium))
            .underline(true, color: .accentColor)
    }
}

/// A take's text with the words the lexicon changed underlined in accent.
func labAttributed(_ text: String, applied: [LabApplied], size: CGFloat = LabStyle.body)
    -> AttributedString
{
    var string = AttributedString(text)
    string.font = .system(size: size)
    for item in applied {
        var searchStart = string.startIndex
        while let range = string[searchStart...].range(of: item.term) {
            string[range].underlineStyle = Text.LineStyle(pattern: .solid, color: .accentColor)
            string[range].font = .system(size: size, weight: .medium)
            searchStart = range.upperBound
        }
    }
    return string
}

// MARK: - Fix field

/// The whole take in one text field with the likely-wrong word selected, so
/// fixing is: type, Return. Suggestions for the selected word sit below and
/// take ⌘1…⌘3. Everything the owner changes is diffed and learned.
struct LabFixField: View {
    let lab: DictationLab
    let take: LabTake
    var title: String = "Fix the last dictation"
    var focusWord: Int?
    var onDone: () -> Void

    @State private var text: String = ""
    @State private var selection: TextSelection?
    @FocusState private var focused: Bool
    @State private var result: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "text.cursor")
                    .foregroundStyle(.secondary)
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("in \(take.appName)")
                    .font(.system(size: 12))
                    .foregroundStyle(.tertiary)
            }
            TextField("", text: $text, selection: $selection, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 17))
                .lineLimit(1...4)
                .focused($focused)
                .onSubmit(apply)
                .onKeyPress(.escape) {
                    onDone()
                    return .handled
                }
            if let result {
                Text(result)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Color.accentColor)
            } else {
                suggestionRow
            }
        }
        .padding(16)
        .frame(width: 640, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 20))
        .padding(10)
        // Escape reaches a text field as cancelOperation, not a key press.
        .onExitCommand(perform: onDone)
        .onAppear {
            text = take.text
            selectSuspect()
            focused = true
        }
    }

    private var selectedWord: String {
        guard case .selection(let range) = selection?.indices, !range.isEmpty else { return "" }
        return String(text[range])
    }

    @ViewBuilder
    private var suggestionRow: some View {
        let options = lab.suggestions(for: selectedWord)
        HStack(spacing: 14) {
            ForEach(Array(options.enumerated()), id: \.offset) { index, option in
                Button {
                    replaceSelection(with: option)
                } label: {
                    HStack(spacing: 4) {
                        Text(option).font(.system(size: 13, weight: .medium))
                        Text("⌘\(index + 1)").font(.system(size: 11)).foregroundStyle(.tertiary)
                    }
                }
                .buttonStyle(.plain)
                .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
            }
            if options.isEmpty {
                Text("Type the right word. Return fixes it and teaches the app.")
                    .font(.system(size: 12))
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            Text("↩ fix · esc close")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
        }
    }

    private func selectSuspect() {
        let words = text.split(separator: " ", omittingEmptySubsequences: false)
        let index = focusWord ?? lab.suspect(in: text)
        guard let index, index < words.count else {
            selection = TextSelection(insertionPoint: text.endIndex)
            return
        }
        var offset = 0
        for word in words.prefix(index) { offset += word.count + 1 }
        let word = words[index].trimmingCharacters(in: .punctuationCharacters)
        guard let start = text.index(text.startIndex, offsetBy: offset, limitedBy: text.endIndex),
            let range = text.range(of: word, range: start..<text.endIndex)
        else { return }
        selection = TextSelection(range: range)
    }

    private func replaceSelection(with option: String) {
        guard case .selection(let range) = selection?.indices else { return }
        text.replaceSubrange(range, with: option)
        let start = range.lowerBound
        if let end = text.index(start, offsetBy: option.count, limitedBy: text.endIndex) {
            selection = TextSelection(range: start..<end)
        }
    }

    private func apply() {
        let corrected = text.replacingOccurrences(of: "\n", with: " ")
        guard corrected != take.text else {
            onDone()
            return
        }
        let before = take.text
        let hunks = LabDiff.hunks(from: before, to: corrected).filter(\.isCorrection)
        result =
            hunks.isEmpty
            ? "Fixed. It looked like a rewrite, so nothing was learned."
            : "Learned " + hunks.map { "\($0.before) → \($0.after)" }.joined(separator: ", ")
        Task {
            try? await Task.sleep(for: .milliseconds(450))
            onDone()
            try? await Task.sleep(for: .milliseconds(120))
            await lab.fix(take.id, to: corrected, source: .fixBar, allowedKeys: 0)
        }
    }
}

// MARK: - Toast card

/// What was learned, with Undo: shown in the card panel over any app.
struct LabToastCard: View {
    let lab: DictationLab
    let toast: LabToast

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: toast.isWarning ? "exclamationmark.triangle" : "sparkle")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(toast.isWarning ? Color.orange : Color.accentColor)
            VStack(alignment: .leading, spacing: 2) {
                Text(toast.title)
                    .font(.system(size: 13, weight: .semibold))
                if let detail = toast.detail {
                    Text(detail)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            Spacer(minLength: 8)
            if !toast.lessonIDs.isEmpty {
                Button("Undo") {
                    lab.undo(toast.lessonIDs)
                }
                .overlayAffordance()
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(width: 440, alignment: .leading)
        .glassEffect(.regular, in: .capsule)
        .padding(8)
    }
}

// MARK: - The pill

/// The recording pill every prototype keeps: the classic pill's live phases
/// without its lingering "Inserted" beat, which the lab's cards replace.
struct LabPillOverlay: View {
    var feed: DictationFeed
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var shown: DictationFeed.Phase = .idle

    var body: some View {
        GlassEffectContainer {
            ZStack(alignment: .bottom) {
                if shown != .idle {
                    content
                        .frame(
                            width: PillMetrics.size(for: shown).width,
                            height: PillMetrics.size(for: shown).height
                        )
                        .glassEffect(.regular, in: .capsule)
                        .transition(
                            reduceMotion
                                ? .opacity
                                : .scale(scale: 0.85, anchor: .bottom).combined(with: .opacity))
                }
            }
            .frame(
                width: PillMetrics.canvasSize.width, height: PillMetrics.canvasSize.height,
                alignment: .bottom)
        }
        .onChange(of: feed.phase, initial: true) { _, phase in
            withAnimation(reduceMotion ? nil : .spring(response: 0.2, dampingFraction: 0.78)) {
                shown = phase
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch shown {
        case .recording:
            AudioBarsView(feed: feed).padding(.horizontal, 10).padding(.vertical, 6)
        case .processing, .proofreading:
            TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { timeline in
                ProcessingDotsView(time: timeline.date.timeIntervalSinceReferenceDate).frame(
                    height: 12)
            }
            .padding(.horizontal, 12)
        case .error(let error):
            Label(
                error.errorDescription ?? "Something went wrong",
                systemImage: "exclamationmark.triangle"
            )
            .font(.system(size: 11, weight: .semibold))
            .lineLimit(1)
            .padding(.horizontal, 12)
        case .idle:
            EmptyView()
        }
    }
}

// MARK: - Flow layout

/// Words that wrap like text but stay individually clickable.
struct WordFlow: Layout {
    var spacing: CGFloat = 4
    var lineSpacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 600
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        var maxX: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width {
                x = 0
                y += lineHeight + lineSpacing
                lineHeight = 0
            }
            x += size.width + spacing
            maxX = max(maxX, x)
            lineHeight = max(lineHeight, size.height)
        }
        return CGSize(width: min(maxX, width), height: y + lineHeight)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        var x = bounds.minX
        var y = bounds.minY
        var lineHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                x = bounds.minX
                y += lineHeight + lineSpacing
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

// MARK: - Status

/// One quiet line: readiness and the two shortcuts.
struct LabStatusLine: View {
    @Environment(DictationCoordinator.self) private var coordinator
    @Environment(TranscriptionEngine.self) private var engine
    @Environment(SettingsManager.self) private var settings
    var extra: String?

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(dotColor)
                .frame(width: 7, height: 7)
            Text(status)
                .foregroundStyle(.secondary)
            Text("·").foregroundStyle(.tertiary)
            Text("\(settings.hotkey.displayString) dictate")
                .foregroundStyle(.tertiary)
            if let extra {
                Text("·").foregroundStyle(.tertiary)
                Text(extra).foregroundStyle(.tertiary)
            }
        }
        .font(.system(size: 12))
    }

    private var status: String {
        if !engine.isModelLoaded { return "Loading the dictation model…" }
        switch coordinator.state {
        case .recording: return "Listening"
        case .processing, .proofreading: return "Transcribing"
        case .error: return "Something went wrong"
        case .idle: return "Ready"
        }
    }

    private var dotColor: Color {
        switch coordinator.state {
        case .recording: return .red
        case .processing, .proofreading: return .orange
        case .error: return .yellow
        case .idle: return engine.isModelLoaded ? .green : .secondary
        }
    }
}
