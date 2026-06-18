//
//  SpeechScrollView.swift
//  tesseract
//
//  Auto-scrolling word-highlighted text view for TTS overlay.
//  Adapted from textream (github.com/f/textream) MarqueeTextView.
//

import SwiftUI
import AppKit

// MARK: - Preference key

struct WordYPreferenceKey: PreferenceKey {
    nonisolated static let defaultValue: [Int: CGFloat] = [:]
    static func reduce(value: inout [Int: CGFloat], nextValue: () -> [Int: CGFloat]) {
        value.merge(nextValue(), uniquingKeysWith: { $1 })
    }
}

// MARK: - SpeechScrollView

struct SpeechScrollView: View {
    let timeline: WordTimeline
    let highlightedCharCount: Int
    var font: NSFont = .systemFont(ofSize: 20, weight: .semibold)
    var highlightColor: Color = .yellow
    var onWordTap: ((Int) -> Void)?

    @State private var scrollOffset: CGFloat = 0
    @State private var manualOffset: CGFloat = 0
    @State private var wordYPositions: [Int: CGFloat] = [:]
    @State private var containerHeight: CGFloat = 0

    var body: some View {
        GeometryReader { geo in
            WordFlowLayout(
                timeline: timeline,
                highlightedCharCount: highlightedCharCount,
                font: font,
                highlightColor: highlightColor,
                containerWidth: geo.size.width,
                onWordTap: { charOffset in
                    manualOffset = 0
                    onWordTap?(charOffset)
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                        recalcCenter(containerHeight: containerHeight)
                    }
                }
            )
            .onPreferenceChange(WordYPreferenceKey.self) { positions in
                wordYPositions = positions
            }
            .offset(y: scrollOffset + manualOffset)
            .animation(.easeOut(duration: 0.5), value: scrollOffset)
            .animation(.easeOut(duration: 0.15), value: manualOffset)
            .onChange(of: geo.size.height) { _, newHeight in
                containerHeight = newHeight
                recalcCenter(containerHeight: newHeight)
            }
            .onChange(of: highlightedCharCount) { _, _ in
                manualOffset = 0
                recalcCenter(containerHeight: containerHeight)
            }
            .onAppear {
                containerHeight = geo.size.height
            }
            .overlay(
                ScrollWheelView(
                    onScroll: { delta in
                        let maxY = wordYPositions.values.max() ?? 0
                        let maxUp = geo.size.height * 0.5
                        let maxDown = max(0, maxY - geo.size.height * 0.5)

                        let newOffset = manualOffset + delta
                        let upperBound = maxUp
                        let lowerBound = -maxDown

                        if newOffset > upperBound {
                            manualOffset = upperBound + (newOffset - upperBound) * 0.2
                        } else if newOffset < lowerBound {
                            manualOffset = lowerBound - (lowerBound - newOffset) * 0.2
                        } else {
                            manualOffset = newOffset
                        }
                    },
                    onScrollEnd: {
                        let maxY = wordYPositions.values.max() ?? 0
                        let upperBound = geo.size.height * 0.5
                        let lowerBound = -max(0, maxY - geo.size.height * 0.5)

                        if manualOffset > upperBound || manualOffset < lowerBound {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                manualOffset = min(upperBound, max(lowerBound, manualOffset))
                            }
                        }
                    }
                )
            )
        }
        .clipped()
        .mask(
            LinearGradient(
                stops: [
                    .init(color: .clear, location: 0),
                    .init(color: .white, location: 0.05),
                    .init(color: .white, location: 0.95),
                    .init(color: .clear, location: 1.0),
                ],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    private func recalcCenter(containerHeight: CGFloat) {
        let wordIdx = activeWordIndex()
        if let wordY = wordYPositions[wordIdx] {
            let center = containerHeight * 0.5
            let target = center - wordY
            if abs(scrollOffset - target) > 1 {
                scrollOffset = target
            }
        }
    }

    private func activeWordIndex() -> Int {
        timeline.activeWordIndex(highlightedCharCount: highlightedCharCount)
    }
}

// MARK: - Word Flow Layout

struct WordFlowLayout: View {
    let timeline: WordTimeline
    let highlightedCharCount: Int
    let font: NSFont
    var highlightColor: Color = .yellow
    let containerWidth: CGFloat
    var onWordTap: ((Int) -> Void)?

    private func nextWordIndex() -> Int {
        for (index, word) in timeline.words.enumerated() where !word.isAnnotation {
            // First non-annotation word that is not yet fully lit (lit < letters).
            if timeline.litFraction(wordIndex: index, charCount: highlightedCharCount) < 1.0 {
                return index
            }
        }
        return -1
    }

    var body: some View {
        let lines = buildLines()
        let nextIdx = nextWordIndex()
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                HStack(spacing: 0) {
                    ForEach(line, id: \.self) { index in
                        wordView(for: index, isNextWord: index == nextIdx)
                            .id(index)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .coordinateSpace(name: "flowLayout")
    }

    private func wordView(for index: Int, isNextWord: Bool) -> some View {
        let word = timeline.words[index]
        let charsIntoWord = highlightedCharCount - word.charOffset
        let isFullyLit =
            timeline.litFraction(wordIndex: index, charCount: highlightedCharCount) >= 1.0
        let isCurrentWord = isNextWord || (charsIntoWord >= 0 && !isFullyLit)

        if word.isAnnotation {
            let annotationColor: Color =
                isFullyLit
                ? Color.white.opacity(0.5)
                : Color.white.opacity(0.2)

            return AnyView(
                Text(word.text + " ")
                    .font(Font(font).italic())
                    .foregroundStyle(annotationColor)
                    .background(
                        GeometryReader { wordGeo in
                            Color.clear.preference(
                                key: WordYPreferenceKey.self,
                                value: [index: wordGeo.frame(in: .named("flowLayout")).midY]
                            )
                        }
                    )
                    .contentShape(Rectangle())
                    .onTapGesture {
                        onWordTap?(word.charOffset)
                    }
            )
        }

        let dimColor: Color =
            isCurrentWord
            ? highlightColor.opacity(0.6)
            : highlightColor

        let wordColor: Color = isFullyLit ? highlightColor.opacity(0.3) : dimColor

        return AnyView(
            Text(word.text + " ")
                .font(Font(font))
                .foregroundStyle(wordColor)
                .underline(isCurrentWord, color: wordColor)
                .background(
                    GeometryReader { wordGeo in
                        Color.clear.preference(
                            key: WordYPreferenceKey.self,
                            value: [index: wordGeo.frame(in: .named("flowLayout")).midY]
                        )
                    }
                )
                .contentShape(Rectangle())
                .onTapGesture {
                    onWordTap?(word.charOffset)
                }
        )
    }

    /// Group word indices into lines that fit `containerWidth`, measuring each word
    /// once. Reads `timeline.words` directly — there is no parallel item model.
    private func buildLines() -> [[Int]] {
        var lines: [[Int]] = [[]]
        var currentLineWidth: CGFloat = 0
        let spaceWidth = (" " as NSString).size(withAttributes: [.font: font]).width

        for (index, word) in timeline.words.enumerated() {
            let wordWidth =
                (word.text as NSString).size(withAttributes: [.font: font]).width + spaceWidth
            if currentLineWidth + wordWidth > containerWidth && !lines[lines.count - 1].isEmpty {
                lines.append([])
                currentLineWidth = 0
            }
            lines[lines.count - 1].append(index)
            currentLineWidth += wordWidth
        }
        return lines
    }
}

// MARK: - Scroll Wheel Handler

struct ScrollWheelView: NSViewRepresentable {
    var onScroll: (CGFloat) -> Void
    var onScrollEnd: (() -> Void)?

    func makeNSView(context: Context) -> ScrollWheelNSView {
        let view = ScrollWheelNSView()
        view.onScroll = onScroll
        view.onScrollEnd = onScrollEnd
        return view
    }

    func updateNSView(_ nsView: ScrollWheelNSView, context: Context) {
        nsView.onScroll = onScroll
        nsView.onScrollEnd = onScrollEnd
    }
}

final class ScrollWheelNSView: NSView {
    var onScroll: ((CGFloat) -> Void)?
    var onScrollEnd: (() -> Void)?
    private var scrollMonitor: Any?

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil && scrollMonitor == nil {
            scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) {
                [weak self] event in
                guard let self, let window = self.window else { return event }
                if event.window == window {
                    let delta = event.scrollingDeltaY
                    let scaled = event.hasPreciseScrollingDeltas ? delta : delta * 10
                    self.onScroll?(scaled)

                    if event.phase == .ended || event.momentumPhase == .ended {
                        self.onScrollEnd?()
                    }
                }
                return event
            }
        }
    }

    override func removeFromSuperview() {
        if let monitor = scrollMonitor {
            NSEvent.removeMonitor(monitor)
            scrollMonitor = nil
        }
        super.removeFromSuperview()
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
        return nil
    }
}
