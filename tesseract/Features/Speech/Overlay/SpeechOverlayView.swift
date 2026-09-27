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
//  Both show the page of two lines holding the word being heard (lines are
//  measured once per segment) and redraw only when that word changes.
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

    var words: [String] {
        previewWords ?? readAlong.segment?.words.words.map(\.text) ?? []
    }

    var word: Int { previewWords == nil ? readAlong.word : previewWord }

    /// Identifies the text being paged, so lines are measured once per segment.
    var textKey: String {
        previewWords != nil
            ? "preview" : "\(readAlong.utteranceID)#\(readAlong.segment?.index ?? -1)"
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
    @State private var layout = CaptionLineCache()

    var body: some View {
        let settings = model.settings
        let font = NSFont.systemFont(ofSize: settings.speechOverlaySize.points, weight: .semibold)
        VStack(spacing: 10) {
            Color.clear.frame(height: max(topInset - 6, 0))
            CaptionPage(
                model: model, font: font, width: width - 52,
                tint: settings.speechOverlayTint.color, alignment: .leading, layout: layout
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
    @State private var layout = CaptionLineCache()

    var body: some View {
        let settings = model.settings
        let font = NSFont.systemFont(
            ofSize: settings.speechOverlaySize.points + 5, weight: .semibold)
        CaptionPage(
            model: model, font: font, width: width - 64, tint: settings.speechOverlayTint.color,
            alignment: .center, layout: layout
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

/// The last measured line layout, so a segment is measured once.
@MainActor
private final class CaptionLineCache {
    private var key = ""
    private var lines: [Range<Int>] = []

    func lines(for words: [String], key: String, font: NSFont, width: CGFloat) -> [Range<Int>] {
        let fullKey = "\(key)|\(font.pointSize)|\(Int(width))|\(words.count)"
        if fullKey == self.key { return lines }
        let attributes: [NSAttributedString.Key: Any] = [.font: font]
        let widths = words.map { ($0 as NSString).size(withAttributes: attributes).width }
        let space = (" " as NSString).size(withAttributes: attributes).width
        lines = CaptionLayout.lines(wordWidths: widths, spaceWidth: space, width: width)
        self.key = fullKey
        return lines
    }
}

private struct CaptionPage: View {
    let model: SpeechOverlayModel
    let font: NSFont
    let width: CGFloat
    let tint: Color
    let alignment: HorizontalAlignment
    let layout: CaptionLineCache

    var body: some View {
        let words = model.words
        let word = model.word
        let lines = layout.lines(for: words, key: model.textKey, font: font, width: width)
        let page = CaptionLayout.page(for: word, in: lines)
        VStack(alignment: alignment, spacing: font.pointSize * 0.28) {
            ForEach(Array(page), id: \.lowerBound) { line in
                Text(attributed(line, words: words, word: word))
                    .font(Font(font))
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
            }
            if page.count < 2 {
                Text(" ").font(Font(font))
            }
        }
        .id(page.first?.lowerBound ?? 0)
        .transition(
            .asymmetric(insertion: .opacity.combined(with: .offset(y: 6)), removal: .opacity)
        )
        .animation(.easeOut(duration: 0.22), value: page.first?.lowerBound ?? 0)
    }

    /// Heard words white, the word being heard in the tint, the rest dim.
    private func attributed(_ line: Range<Int>, words: [String], word: Int) -> AttributedString {
        var result = AttributedString()
        for index in line {
            var run = AttributedString(words[index] + (index + 1 < line.upperBound ? " " : ""))
            run.foregroundColor =
                index < word ? .white : index == word ? tint : .white.opacity(0.42)
            result += run
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
                    value: Double(max(model.word, 0)),
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
