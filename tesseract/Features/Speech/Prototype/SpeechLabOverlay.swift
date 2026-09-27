//
//  SpeechLabOverlay.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  The new read-along overlay, in two styles the page's settings pick from:
//
//  - Island: a black capsule hanging from the notch with two lines of the
//    text; hover shows pause, stop, progress and "open Speech".
//  - Captions: large two-line captions at the bottom of the screen, the way
//    Live Captions reads.
//
//  Both page through the text two lines at a time (lines are measured once
//  per segment, not per frame) and redraw only when the heard word changes.
//  Today's notch stays available as "Classic" for comparison.
//

import AppKit
import Observation
import SwiftUI

@Observable @MainActor
final class OverlayPresentation {
    var isPresented = false
    var style: OverlayPrefs.Style = .island
    /// Sample words shown when previewing settings with nothing playing.
    var preview: [String]?
    var previewIndex = 0
}

@MainActor
final class SpeechLabOverlayController {
    private unowned let lab: SpeechLab
    private var panel: NSPanel?
    private var panelStyle: OverlayPrefs.Style?
    private let presentation = OverlayPresentation()
    private var hideTask: Task<Void, Never>?
    private var previewTimer: Timer?

    init(lab: SpeechLab) {
        self.lab = lab
    }

    func utteranceBegan(style: OverlayPrefs.Style) {
        stopPreview()
        present(style: style)
    }

    func utteranceEnded() {
        guard previewTimer == nil else { return }
        dismiss()
    }

    /// Show the chosen style for a few seconds with sample words, so a
    /// settings change is visible without speaking anything.
    func preview(style: OverlayPrefs.Style) {
        guard lab.readAlong.mode != .live else {
            // Speaking now: re-present in the new style.
            if style == .island || style == .captions { present(style: style) } else { dismiss() }
            return
        }
        guard style == .island || style == .captions else {
            stopPreview()
            dismiss()
            return
        }
        let words = Self.previewText.split(separator: " ").map(String.init)
        presentation.preview = words
        presentation.previewIndex = 0
        present(style: style)
        previewTimer?.invalidate()
        let timer = Timer(timeInterval: 0.28, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, let words = self.presentation.preview else { return }
                if self.presentation.previewIndex + 1 < words.count {
                    self.presentation.previewIndex += 1
                } else {
                    self.stopPreview()
                    self.dismiss()
                }
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        previewTimer = timer
    }

    private static let previewText =
        "This is how the overlay reads along with you. Each word lights up as it is spoken, two lines at a time."

    private func stopPreview() {
        previewTimer?.invalidate()
        previewTimer = nil
        presentation.preview = nil
    }

    private func present(style: OverlayPrefs.Style) {
        hideTask?.cancel()
        hideTask = nil
        if panel == nil || panelStyle != style {
            panel?.orderOut(nil)
            panel = makePanel(style: style)
            panelStyle = style
            presentation.isPresented = false
        }
        presentation.style = style
        panel?.orderFrontRegardless()
        withAnimation(.spring(response: 0.42, dampingFraction: 0.82)) {
            presentation.isPresented = true
        }
    }

    private func dismiss() {
        guard panel != nil, hideTask == nil else { return }
        withAnimation(.easeIn(duration: 0.22)) {
            presentation.isPresented = false
        }
        hideTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled, let self else { return }
            self.panel?.orderOut(nil)
            self.panel = nil
            self.panelStyle = nil
            self.hideTask = nil
        }
    }

    private func makePanel(style: OverlayPrefs.Style) -> NSPanel? {
        guard let screen = NSScreen.main else { return nil }
        let size = lab.overlay.size
        let frame: NSRect
        let topInset = max(
            screen.safeAreaInsets.top, screen.frame.maxY - screen.visibleFrame.maxY, 24)
        switch style {
        case .captions:
            let width = min(Self.captionsWidth(size), screen.visibleFrame.width - 80)
            let height: CGFloat = 220
            frame = NSRect(
                x: screen.visibleFrame.midX - width / 2, y: screen.visibleFrame.minY + 28,
                width: width, height: height)
        default:
            let width = Self.islandWidth(size) + 40
            let height = topInset + 190
            frame = NSRect(
                x: screen.frame.midX - width / 2, y: screen.frame.maxY - height,
                width: width, height: height)
        }

        let root = SpeechLabOverlayRoot(
            lab: lab, presentation: presentation, topInset: topInset,
            islandWidth: Self.islandWidth(size), captionsWidth: frame.width)
        let hosting = NSHostingView(rootView: root)
        let panel = NSPanel(
            contentRect: frame, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered,
            defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = .screenSaver
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = false
        panel.contentView = hosting
        return panel
    }

    static func islandWidth(_ size: OverlayPrefs.Size) -> CGFloat {
        switch size {
        case .small: 440
        case .medium: 520
        case .large: 640
        }
    }

    static func captionsWidth(_ size: OverlayPrefs.Size) -> CGFloat {
        switch size {
        case .small: 640
        case .medium: 780
        case .large: 940
        }
    }
}

// MARK: - Views

private struct SpeechLabOverlayRoot: View {
    let lab: SpeechLab
    let presentation: OverlayPresentation
    let topInset: CGFloat
    let islandWidth: CGFloat
    let captionsWidth: CGFloat

    var body: some View {
        switch presentation.style {
        case .captions:
            CaptionsOverlay(lab: lab, presentation: presentation, width: captionsWidth)
        default:
            IslandOverlay(
                lab: lab, presentation: presentation, topInset: topInset, width: islandWidth)
        }
    }
}

/// The words on screen: live speech, or preview words while previewing.
private struct OverlayWords {
    let words: [String]
    let active: Int
    let key: String

    @MainActor
    static func current(lab: SpeechLab, presentation: OverlayPresentation) -> OverlayWords {
        if let preview = presentation.preview {
            return OverlayWords(words: preview, active: presentation.previewIndex, key: "preview")
        }
        let readAlong = lab.readAlong
        let words = readAlong.currentWords.map(\.text)
        return OverlayWords(
            words: words, active: readAlong.currentLocalWord,
            key: "\(readAlong.takeID?.uuidString ?? "-")#\(readAlong.segmentIndex)")
    }
}

/// Lines measured once per (segment, size, width); pages are two lines.
@MainActor
private final class CaptionLayoutCache {
    private var key = ""
    private var lines: [Range<Int>] = []

    func lines(for words: [String], key: String, font: NSFont, width: CGFloat) -> [Range<Int>] {
        let fullKey = "\(key)|\(font.pointSize)|\(Int(width))|\(words.count)"
        if fullKey == self.key { return lines }
        let space = (" " as NSString).size(withAttributes: [.font: font]).width
        var result: [Range<Int>] = []
        var start = 0
        var lineWidth: CGFloat = 0
        for (i, word) in words.enumerated() {
            let w = (word as NSString).size(withAttributes: [.font: font]).width
            if lineWidth > 0, lineWidth + w > width {
                result.append(start..<i)
                start = i
                lineWidth = 0
            }
            lineWidth += w + space
        }
        if start < words.count { result.append(start..<words.count) }
        self.key = fullKey
        self.lines = result
        return result
    }
}

private struct CaptionPage: View {
    let words: OverlayWords
    let font: NSFont
    let width: CGFloat
    let tint: Color
    let alignment: HorizontalAlignment
    let cache: CaptionLayoutCache

    var body: some View {
        let lines = cache.lines(for: words.words, key: words.key, font: font, width: width)
        let current = lines.firstIndex { $0.contains(max(words.active, 0)) } ?? 0
        let page = current / 2
        let visible = Array(lines.dropFirst(page * 2).prefix(2))
        VStack(alignment: alignment, spacing: font.pointSize * 0.28) {
            ForEach(Array(visible.enumerated()), id: \.offset) { _, line in
                Text(attributed(line))
                    .font(Font(font))
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
            }
            if visible.count < 2 {
                Text(" ").font(Font(font))
            }
        }
        .id(page)
        .transition(
            .asymmetric(
                insertion: .opacity.combined(with: .offset(y: 6)), removal: .opacity)
        )
        .animation(.easeOut(duration: 0.22), value: page)
    }

    private func attributed(_ line: Range<Int>) -> AttributedString {
        var result = AttributedString()
        for i in line {
            var run = AttributedString(words.words[i] + (i + 1 < line.upperBound ? " " : ""))
            if i < words.active {
                run.foregroundColor = .white
            } else if i == words.active {
                run.foregroundColor = tint
            } else {
                run.foregroundColor = .white.opacity(0.42)
            }
            result += run
        }
        return result
    }
}

private struct OverlayControls: View {
    let lab: SpeechLab
    let compact: Bool

    var body: some View {
        HStack(spacing: 10) {
            button(
                lab.isPaused ? "play.fill" : "pause.fill", help: lab.isPaused ? "Resume" : "Pause"
            ) {
                lab.togglePause()
            }
            button("stop.fill", help: "Stop") { lab.stop() }
            if !compact {
                ProgressView(
                    value: Double(max(lab.readAlong.wordIndex, 0)),
                    total: Double(max(lab.readAlong.totalWords, 1))
                )
                .progressViewStyle(.linear)
                .tint(.white.opacity(0.8))
                .frame(maxWidth: .infinity)
                button("arrow.up.forward.app", help: "Open Speech") {
                    (NSApp.delegate as? AppDelegate)?.navigateToSpeech()
                }
            }
        }
    }

    private func button(_ symbol: String, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 28, height: 28)
                .background(Circle().fill(.white.opacity(0.14)))
        }
        .buttonStyle(.plain)
        .help(help)
    }
}

private struct IslandOverlay: View {
    let lab: SpeechLab
    let presentation: OverlayPresentation
    let topInset: CGFloat
    let width: CGFloat

    @State private var hovering = false
    @State private var cache = CaptionLayoutCache()

    var body: some View {
        let prefs = lab.overlay
        let font = NSFont.systemFont(ofSize: prefs.size.points, weight: .semibold)
        VStack(spacing: 10) {
            Color.clear.frame(height: topInset - 6)
            CaptionPage(
                words: OverlayWords.current(lab: lab, presentation: presentation), font: font,
                width: width - 52, tint: prefs.tint.color, alignment: .leading, cache: cache
            )
            .frame(maxWidth: .infinity, alignment: .leading)
            if hovering, prefs.showsControls {
                OverlayControls(lab: lab, compact: false)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.horizontal, 26)
        .padding(.bottom, 16)
        .frame(width: width)
        .background(
            DynamicIslandShape(topInset: 12, bottomRadius: 26).fill(.black)
        )
        .contentShape(Rectangle())
        .onHover { inside in
            withAnimation(.easeOut(duration: 0.18)) { hovering = inside }
        }
        .scaleEffect(
            x: presentation.isPresented ? 1 : 0.55, y: presentation.isPresented ? 1 : 0.2,
            anchor: .top
        )
        .opacity(presentation.isPresented ? 1 : 0)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .environment(\.colorScheme, .dark)
    }
}

private struct CaptionsOverlay: View {
    let lab: SpeechLab
    let presentation: OverlayPresentation
    let width: CGFloat

    @State private var hovering = false
    @State private var cache = CaptionLayoutCache()

    var body: some View {
        let prefs = lab.overlay
        let font = NSFont.systemFont(ofSize: prefs.size.points + 6, weight: .semibold)
        CaptionPage(
            words: OverlayWords.current(lab: lab, presentation: presentation), font: font,
            width: width - 64, tint: prefs.tint.color, alignment: .center, cache: cache
        )
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 32)
        .padding(.vertical, 18)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous).fill(.black.opacity(0.82))
        )
        .overlay(alignment: .topTrailing) {
            if hovering, prefs.showsControls {
                OverlayControls(lab: lab, compact: true)
                    .padding(8)
                    .transition(.opacity)
            }
        }
        .contentShape(Rectangle())
        .onHover { inside in
            withAnimation(.easeOut(duration: 0.18)) { hovering = inside }
        }
        .offset(y: presentation.isPresented ? 0 : 24)
        .opacity(presentation.isPresented ? 1 : 0)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .environment(\.colorScheme, .dark)
    }
}
