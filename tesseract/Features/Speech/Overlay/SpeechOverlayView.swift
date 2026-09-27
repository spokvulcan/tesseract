//
//  SpeechOverlayView.swift
//  tesseract
//
//  The **Speech Overlay**'s two styles, following the Read-Along:
//
//  - Island: a black capsule hanging from the notch with two lines of the
//    text; hovering shows pause, stop, progress and a way back to Speech.
//  - Captions: two large lines at the bottom of the screen.
//
//  Both show one continuous feed of the reading (ADR-0077): the line being
//  heard on top, the next below. When the voice reaches the next line, the
//  whole feed moves up one line and the finished line fades out; no text is
//  ever swapped in place. Passages are laid out as they arrive, ahead of
//  their audio, and only the lines near the heard one are drawn.
//

import AppKit
import Observation
import SwiftUI

/// What the overlay shows and the actions its controls take. The panel owns
/// it; the views only read it.
@Observable @MainActor
final class SpeechOverlayModel {
    var isPresented = false
    var style: SpeechOverlayStyle = .island
    /// Sample words shown while previewing a settings change with nothing
    /// being read; `previewWord` walks through them.
    var previewWords: [String]?
    var previewWord = 0

    @ObservationIgnored let readAlong: SpeechReadAlong
    @ObservationIgnored let settings: SettingsManager
    @ObservationIgnored private let coordinator: SpeechCoordinator
    @ObservationIgnored private let openSpeechPage: () -> Void

    init(
        readAlong: SpeechReadAlong, settings: SettingsManager, coordinator: SpeechCoordinator,
        openSpeechPage: @escaping () -> Void
    ) {
        self.readAlong = readAlong
        self.settings = settings
        self.coordinator = coordinator
        self.openSpeechPage = openSpeechPage
    }

    /// What the feed lays out: the reading's passages, or the preview's.
    var passages: [ReadAlongPassage] {
        if let previewWords {
            return [
                ReadAlongPassage(index: 0, firstWord: 0, text: previewWords.joined(separator: " "))
            ]
        }
        return readAlong.passages
    }

    /// The word being heard, counted over the whole reading.
    var word: Int {
        if previewWords != nil { return previewWord }
        return (readAlong.segment?.firstWord ?? 0) + readAlong.word
    }

    /// Identifies the reading, so its feed is built once and grown.
    var feedKey: String {
        previewWords != nil ? "preview" : readAlong.utteranceID.uuidString
    }

    var isPaused: Bool {
        if case .paused = coordinator.state { return true }
        return false
    }

    func togglePause() {
        if isPaused { coordinator.resume() } else { coordinator.pause() }
    }

    func stop() { coordinator.stop() }

    func open() { openSpeechPage() }
}

struct SpeechOverlayRoot: View {
    let model: SpeechOverlayModel
    let topInset: CGFloat
    let width: CGFloat

    var body: some View {
        switch model.style {
        case .captions: CaptionsOverlay(model: model, width: width)
        default: IslandOverlay(model: model, topInset: topInset, width: width)
        }
    }
}

// MARK: - Styles

private struct IslandOverlay: View {
    let model: SpeechOverlayModel
    let topInset: CGFloat
    let width: CGFloat

    @State private var hovering = false
    @State private var feed = CaptionFeedCache()

    var body: some View {
        let settings = model.settings
        let font = NSFont.systemFont(ofSize: settings.speechOverlaySize.points, weight: .semibold)
        VStack(spacing: 10) {
            Color.clear.frame(height: max(topInset - 6, 0))
            CaptionFeedView(
                model: model, font: font, width: width - 52,
                tint: settings.speechOverlayTint.color, alignment: .leading, cache: feed
            )
            .frame(maxWidth: .infinity, alignment: .leading)
            if hovering, settings.speechOverlayShowsControls, model.previewWords == nil {
                OverlayControls(model: model, showsProgress: true)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.horizontal, 26)
        .padding(.bottom, 16)
        .frame(width: width)
        .background(DynamicIslandShape(topInset: 12, bottomRadius: 26).fill(.black))
        .contentShape(Rectangle())
        .onHover { inside in
            withAnimation(.easeOut(duration: 0.18)) { hovering = inside }
        }
        .scaleEffect(
            x: model.isPresented ? 1 : 0.55, y: model.isPresented ? 1 : 0.2, anchor: .top
        )
        .opacity(model.isPresented ? 1 : 0)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .environment(\.colorScheme, .dark)
    }
}

private struct CaptionsOverlay: View {
    let model: SpeechOverlayModel
    let width: CGFloat

    @State private var hovering = false
    @State private var feed = CaptionFeedCache()

    var body: some View {
        let settings = model.settings
        let font = NSFont.systemFont(
            ofSize: settings.speechOverlaySize.points + 5, weight: .semibold)
        CaptionFeedView(
            model: model, font: font, width: width - 64, tint: settings.speechOverlayTint.color,
            alignment: .center, cache: feed
        )
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 32)
        .padding(.vertical, 18)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous).fill(.black.opacity(0.82))
        )
        .overlay(alignment: .topTrailing) {
            if hovering, settings.speechOverlayShowsControls, model.previewWords == nil {
                OverlayControls(model: model, showsProgress: false)
                    .padding(8)
                    .transition(.opacity)
            }
        }
        .contentShape(Rectangle())
        .onHover { inside in
            withAnimation(.easeOut(duration: 0.18)) { hovering = inside }
        }
        .offset(y: model.isPresented ? 0 : 24)
        .opacity(model.isPresented ? 1 : 0)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .environment(\.colorScheme, .dark)
    }
}

// MARK: - Pieces

/// One reading's feed, grown as its passages arrive: a passage is measured
/// once, and a new reading, font or width starts a new feed.
@MainActor
private final class CaptionFeedCache {
    private var key = ""
    private var feed = CaptionFeed()

    func feed(
        for passages: [ReadAlongPassage], key: String, font: NSFont, width: CGFloat
    ) -> CaptionFeed {
        let fullKey = "\(key)|\(font.pointSize)|\(Int(width))"
        if fullKey != self.key {
            feed = CaptionFeed()
            self.key = fullKey
        }
        if let first = passages.first { feed.drop(passagesBefore: first.index) }
        let attributes: [NSAttributedString.Key: Any] = [.font: font]
        let space = (" " as NSString).size(withAttributes: attributes).width
        for passage in passages where passage.index > feed.lastPassage {
            feed.append(
                passage,
                wordWidths: passage.words.map {
                    ($0 as NSString).size(withAttributes: attributes).width
                },
                spaceWidth: space, width: width)
        }
        return feed
    }
}

/// The feed: the heard line on top, the next below. Lines sit at their
/// number times the line height in one column; the column moves so the
/// heard line is on top, on a critically damped spring that keeps its speed
/// when the target changes mid-move. Only the lines from one above the
/// heard line to three below exist as views.
private struct CaptionFeedView: View {
    let model: SpeechOverlayModel
    let font: NSFont
    let width: CGFloat
    let tint: Color
    let alignment: HorizontalAlignment
    let cache: CaptionFeedCache

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let feed = cache.feed(for: model.passages, key: model.feedKey, font: font, width: width)
        let word = model.word
        let lineHeight = (font.ascender - font.descender + font.leading + font.pointSize * 0.28)
            .rounded(.up)
        let position = feed.line(holding: max(word, 0))
        let current = position.map { feed.lines[$0].id } ?? 0
        let window =
            position.map { feed.lines[max($0 - 1, 0)..<min($0 + 4, feed.lines.count)] } ?? []
        ZStack(alignment: .topLeading) {
            ForEach(window) { line in
                Text(attributed(line, word: word))
                    .font(Font(font))
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                    .frame(
                        width: width, alignment: Alignment(horizontal: alignment, vertical: .top)
                    )
                    .offset(y: CGFloat(line.id) * lineHeight)
                    .opacity(line.id < current ? 0 : 1)
            }
        }
        .frame(width: width, height: lineHeight, alignment: .topLeading)
        .offset(y: -CGFloat(current) * lineHeight)
        .animation(reduceMotion ? nil : .spring(duration: 0.32, bounce: 0), value: current)
        .frame(width: width, height: 2 * lineHeight, alignment: .topLeading)
        .clipped()
        .mask {
            LinearGradient(
                stops: [
                    .init(color: .clear, location: 0), .init(color: .black, location: 0.05),
                    .init(color: .black, location: 0.95), .init(color: .clear, location: 1),
                ], startPoint: .top, endPoint: .bottom)
        }
    }

    /// Heard words white, the word being heard in the tint on a soft wash,
    /// the rest at 60% (about 8:1 on black).
    private func attributed(_ line: CaptionFeed.Line, word: Int) -> AttributedString {
        var result = AttributedString()
        for (offset, text) in line.words.enumerated() {
            let index = line.firstWord + offset
            var run = AttributedString(text)
            if index < word {
                run.foregroundColor = .white
            } else if index == word {
                run.foregroundColor = tint
                run.backgroundColor = tint.opacity(0.2)
            } else {
                run.foregroundColor = .white.opacity(0.6)
            }
            result += run
            if offset + 1 < line.words.count { result += AttributedString(" ") }
        }
        return result
    }
}

private struct OverlayControls: View {
    let model: SpeechOverlayModel
    let showsProgress: Bool

    var body: some View {
        HStack(spacing: 10) {
            button(
                model.isPaused ? "play.fill" : "pause.fill",
                help: model.isPaused ? "Resume" : "Pause"
            ) {
                model.togglePause()
            }
            button("stop.fill", help: "Stop") { model.stop() }
            if showsProgress {
                let segment = model.readAlong.segment
                ProgressView(
                    value: Double(max(model.readAlong.word, 0)),
                    total: Double(max(segment?.words.words.count ?? 1, 1))
                )
                .progressViewStyle(.linear)
                .tint(.white.opacity(0.8))
                .frame(maxWidth: .infinity)
                button("arrow.up.forward.app", help: "Open Speech") { model.open() }
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
        .accessibilityLabel(help)
    }
}
