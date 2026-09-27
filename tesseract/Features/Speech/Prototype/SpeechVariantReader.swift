//
//  SpeechVariantReader.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  B · Reader — the text is the player (Speechify, Apple's Accessibility
//  Reader, Voice Dream). A readable serif column lights up word by word as
//  it is read; click any sentence to start there; a floating controller
//  skips by sentence and changes speed. Editing is a mode, not the page.
//  Signature: the reading itself — nothing on the page but the text and
//  one small glass controller.
//

import AppKit
import SwiftUI

enum ReaderFace: String, CaseIterable, Identifiable {
    case serif, sans, rounded
    var id: String { rawValue }
    var label: String {
        switch self {
        case .serif: "Serif"
        case .sans: "Sans"
        case .rounded: "Rounded"
        }
    }
    var design: Font.Design {
        switch self {
        case .serif: .serif
        case .sans: .default
        case .rounded: .rounded
        }
    }
}

struct ReaderVariant: View {
    @AppStorage("speechPrototype.reader.editing") private var isEditing = false
    @AppStorage("speechPrototype.reader.size") private var textSize = 19.0
    @AppStorage("speechPrototype.reader.face") private var face = ReaderFace.serif
    @AppStorage("speechPrototype.reader.highlight") private var highlight = ReadAlongHighlight.both
    @State private var document = ReadingDocument(text: "")
    @State private var showsDisplay = false
    @State private var showsRecent = false

    var body: some View {
        let lab = SpeechLab.shared
        ZStack(alignment: .bottom) {
            if isEditing {
                ReaderEditor(textSize: textSize, face: face)
            } else if document.words.isEmpty {
                ReaderEmpty(isEditing: $isEditing)
            } else {
                ReaderPage(
                    document: document, textSize: textSize, face: face, highlight: highlight)
            }
            ReaderController(document: document, isEditing: $isEditing)
                .padding(.bottom, 18)
                .padding(.horizontal, 20)
        }
        .safeAreaInset(edge: .top, spacing: 0) { SpeechEngineNotice() }
        // Only this watcher reads the draft, so typing in Edit mode never
        // re-evaluates the page around the editor.
        .background {
            DraftWatcher { draft in
                if !isEditing, document.text != draft { document = ReadingDocument(text: draft) }
            }
        }
        .onChange(of: isEditing) { _, editing in
            if !editing, document.text != lab.draft { document = ReadingDocument(text: lab.draft) }
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Picker("Mode", selection: $isEditing) {
                    Label("Listen", systemImage: "book").tag(false)
                    Label("Edit", systemImage: "pencil").tag(true)
                }
                .pickerStyle(.segmented)
                .help("Listen to the text, or edit it")

                Button {
                    showsRecent.toggle()
                } label: {
                    Label("Recent", systemImage: "clock.arrow.circlepath")
                }
                .help("Recent readings")
                .popover(isPresented: $showsRecent, arrowEdge: .bottom) {
                    ReaderRecent(isEditing: $isEditing, isPresented: $showsRecent)
                }

                Button {
                    showsDisplay.toggle()
                } label: {
                    Label("Display", systemImage: "textformat.size")
                }
                .help("Text size, typeface, highlighting and the overlay")
                .popover(isPresented: $showsDisplay, arrowEdge: .bottom) {
                    ReaderDisplaySettings(textSize: $textSize, face: $face, highlight: $highlight)
                }
            }
        }
    }
}

// MARK: - Reading page

private struct ReaderPage: View {
    let document: ReadingDocument
    let textSize: Double
    let face: ReaderFace
    let highlight: ReadAlongHighlight

    var body: some View {
        let lab = SpeechLab.shared
        let active = Self.activeWord(in: document)
        let activeParagraph = active.map { document.words[$0].paragraph }
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text(
                        "\(SpeechLab.wordCount(document.text)) words · \(SpeechLabFormat.approximateLength(SpeechLab.estimatedSeconds(document.text))) · \(lab.currentVoiceName)"
                    )
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .textCase(.uppercase)
                    .kerning(0.6)

                    LazyVStack(alignment: .leading, spacing: textSize * 1.25) {
                        ForEach(document.paragraphs.indices, id: \.self) { index in
                            let paragraph = document.paragraphs[index]
                            let isActive = activeParagraph == index
                            ReadAlongParagraph(
                                document: document, paragraph: index,
                                activeWord: isActive ? active : nil,
                                heardThrough: Self.heardThrough(
                                    active: active, paragraph: paragraph),
                                font: .system(size: textSize, design: face.design),
                                lineSpacing: textSize * 0.42, highlight: highlight,
                                clickableSentences: true
                            )
                            .equatable()
                            .id(index)
                        }
                    }
                }
                .frame(maxWidth: 680, alignment: .leading)
                .padding(.horizontal, 36)
                .padding(.top, 34)
                .padding(.bottom, 140)
                .frame(maxWidth: .infinity)
            }
            .tint(.primary)
            .environment(
                \.openURL,
                OpenURLAction { url in
                    guard url.scheme == "speech-sentence",
                        let index = Int(url.absoluteString.dropFirst(16)),
                        let start = document.text(fromSentence: index)
                    else { return .systemAction }
                    lab.speak(start.text, wordOffset: start.wordOffset)
                    return .handled
                }
            )
            .onChange(of: activeParagraph) { _, paragraph in
                guard let paragraph else { return }
                withAnimation(.easeInOut(duration: 0.45)) {
                    proxy.scrollTo(paragraph, anchor: UnitPoint(x: 0.5, y: 0.3))
                }
            }
        }
    }

    /// The document word being heard: live speech the page started, or a
    /// replay of a take whose text is the page's text.
    static func activeWord(in document: ReadingDocument) -> Int? {
        let readAlong = SpeechLab.shared.readAlong
        switch readAlong.mode {
        case .live:
            guard readAlong.source == .page, let word = readAlong.documentWordIndex,
                document.words.indices.contains(word)
            else { return nil }
            return word
        case .replay:
            guard readAlong.requestText == document.text, readAlong.wordIndex >= 0,
                document.words.indices.contains(readAlong.wordIndex)
            else { return nil }
            return readAlong.wordIndex
        case .idle:
            return nil
        }
    }

    private static func heardThrough(active: Int?, paragraph: ReadingDocument.Paragraph) -> Int {
        guard let active else { return -1 }
        if active >= paragraph.firstWord + paragraph.wordCount { return Int.max }
        if active < paragraph.firstWord { return paragraph.firstWord }
        return active
    }
}

private struct ReaderEditor: View {
    let textSize: Double
    let face: ReaderFace

    var body: some View {
        @Bindable var lab = SpeechLab.shared
        TextEditor(text: $lab.draft)
            .font(.system(size: textSize, design: face.design))
            .lineSpacing(textSize * 0.42)
            .scrollContentBackground(.hidden)
            .frame(maxWidth: 680)
            .padding(.horizontal, 31)
            .padding(.top, 30)
            .padding(.bottom, 110)
            .frame(maxWidth: .infinity)
            .overlay(alignment: .top) {
                if lab.draft.isEmpty {
                    Text("Write or paste what you want read to you.")
                        .font(.system(size: textSize, design: face.design))
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: 680, alignment: .leading)
                        .padding(.horizontal, 36)
                        .padding(.top, 30)
                        .allowsHitTesting(false)
                }
            }
    }
}

private struct ReaderEmpty: View {
    @Binding var isEditing: Bool

    var body: some View {
        ContentUnavailableView {
            Label("Nothing to read yet", systemImage: "book.closed")
        } description: {
            Text(
                "Paste an article, an email, anything long. It is read to you here, sentence by sentence."
            )
        } actions: {
            HStack {
                Button("Paste") {
                    if let text = NSPasteboard.general.string(forType: .string) {
                        SpeechLab.shared.draft = text
                    }
                }
                .buttonStyle(.borderedProminent)
                Button("Write") { isEditing = true }
            }
        }
    }
}

// MARK: - Floating controller

private struct ReaderController: View {
    let document: ReadingDocument
    @Binding var isEditing: Bool

    var body: some View {
        let lab = SpeechLab.shared
        let status = lab.status
        let active = ReaderPage.activeWord(in: document)
        let sentence = active.flatMap { document.sentence(containingWord: $0) }
        let replaying = lab.readAlong.mode == .replay && lab.player.isPlaying
        let playing =
            replaying
            || {
                if case .speaking = status { return true };
                if case .preparing = status { return true }; return false
            }()

        // Compresses (labels truncate) rather than hiding controls by
        // width, so its minimum never depends on the space it is given.
        GlassEffectContainer(spacing: 10) {
            HStack(spacing: 10) {
                roundButton("backward.fill", help: "Previous sentence") {
                    jump(from: sentence, by: -1)
                }
                .disabled(document.sentences.isEmpty)
                Button {
                    togglePlay(sentence: sentence, status: status, replaying: replaying)
                } label: {
                    Image(systemName: playing ? "pause.fill" : "play.fill")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 34, height: 34)
                        .contentTransition(.symbolEffect(.replace))
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(document.words.isEmpty && !status.isActive)
                .help(playing ? "Pause" : "Read (⌘↩)")
                roundButton("forward.fill", help: "Next sentence") { jump(from: sentence, by: 1) }
                    .disabled(document.sentences.isEmpty)

                VStack(alignment: .leading, spacing: 4) {
                    Text(progressLabel(sentence: sentence, status: status))
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .monospacedDigit()
                    ProgressView(
                        value: Double((sentence ?? -1) + 1),
                        total: Double(max(document.sentences.count, 1))
                    )
                    .progressViewStyle(.linear)
                }
                .frame(minWidth: 50, idealWidth: 170, maxWidth: 170)
                .padding(.horizontal, 6)

                if status.isActive {
                    roundButton("stop.fill", help: "Stop (Esc)") { lab.stop() }
                        .keyboardShortcut(.escape, modifiers: [])
                }
                SpeedMenu()
                    .controlSize(.large)
                Menu {
                    VoiceMenuItems(onEdit: nil)
                } label: {
                    Label(String(lab.currentVoiceName.prefix(16)), systemImage: "person.wave.2")
                        .lineLimit(1)
                }
                .fixedSize()
                .controlSize(.large)
                .help("Voice: \(lab.currentVoiceName)")
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .glassEffect(.regular, in: .capsule)
        }
        .frame(maxWidth: .infinity)
    }

    private func roundButton(_ symbol: String, help: String, action: @escaping () -> Void)
        -> some View
    {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 13, weight: .semibold))
                .frame(width: 26, height: 26)
        }
        .buttonStyle(.glass)
        .buttonBorderShape(.circle)
        .controlSize(.large)
        .help(help)
    }

    private func progressLabel(sentence: Int?, status: SpeechLabStatus) -> String {
        switch status {
        case .loadingModel, .preparing, .capturing: return status.label
        case .error(let message): return message
        default: break
        }
        guard !document.sentences.isEmpty else { return "Nothing to read" }
        if let sentence { return "Sentence \(sentence + 1) of \(document.sentences.count)" }
        return
            "\(document.sentences.count) sentences · \(SpeechLabFormat.approximateLength(SpeechLab.estimatedSeconds(document.text)))"
    }

    private func togglePlay(sentence: Int?, status: SpeechLabStatus, replaying: Bool) {
        let lab = SpeechLab.shared
        if replaying {
            lab.player.pause()
            return
        }
        switch status {
        case .speaking, .preparing, .paused:
            lab.togglePause()
        default:
            isEditing = false
            let start = sentence ?? 0
            guard let from = document.text(fromSentence: start) ?? document.text(fromSentence: 0)
            else {
                lab.speak(lab.draft)
                return
            }
            lab.speak(from.text, wordOffset: from.wordOffset)
        }
    }

    private func jump(from sentence: Int?, by delta: Int) {
        let target = min(max((sentence ?? -1) + delta, 0), document.sentences.count - 1)
        guard let from = document.text(fromSentence: target) else { return }
        SpeechLab.shared.speak(from.text, wordOffset: from.wordOffset)
    }
}

// MARK: - Popovers

private struct ReaderDisplaySettings: View {
    @Binding var textSize: Double
    @Binding var face: ReaderFace
    @Binding var highlight: ReadAlongHighlight

    var body: some View {
        Form {
            Section("Text") {
                HStack(spacing: 10) {
                    Image(systemName: "textformat.size.smaller")
                    Slider(value: $textSize, in: 14...28, step: 1)
                    Image(systemName: "textformat.size.larger")
                }
                Picker("Typeface", selection: $face) {
                    ForEach(ReaderFace.allCases) { face in Text(face.label).tag(face) }
                }
                .pickerStyle(.segmented)
                Picker("Highlight", selection: $highlight) {
                    ForEach(ReadAlongHighlight.allCases) { mode in Text(mode.label).tag(mode) }
                }
            }
            Section("Overlay outside the app") {
                OverlaySettingsControls()
            }
        }
        .formStyle(.grouped)
        .frame(width: 340)
        .frame(minHeight: 420)
    }
}

private struct ReaderRecent: View {
    @Binding var isEditing: Bool
    @Binding var isPresented: Bool

    var body: some View {
        let lab = SpeechLab.shared
        VStack(alignment: .leading, spacing: 0) {
            Text("Recent readings")
                .font(.headline)
                .padding(.horizontal, 16)
                .padding(.top, 14)
                .padding(.bottom, 8)
            if lab.takes.isEmpty {
                Text(
                    "What you listen to is kept here to replay with the text lit up, or to save as audio."
                )
                .foregroundStyle(.secondary)
                .frame(width: 300, alignment: .leading)
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
            } else {
                List {
                    ForEach(lab.takes) { take in
                        HStack(spacing: 10) {
                            Button {
                                isEditing = false
                                lab.draft = take.text
                                lab.player.play(take, from: 0)
                                isPresented = false
                            } label: {
                                Image(systemName: "play.circle.fill")
                                    .font(.system(size: 20))
                                    .foregroundStyle(.tint)
                            }
                            .buttonStyle(.plain)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(take.title).lineLimit(1)
                                Text(
                                    "\(SpeechLabFormat.time(take.duration)) · \(take.voiceName) · \(SpeechLabFormat.relative(take.createdAt))"
                                )
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Menu {
                                TakeMenuItems(take: take) { text in
                                    lab.draft = text
                                    isPresented = false
                                }
                            } label: {
                                Image(systemName: "ellipsis.circle")
                            }
                            .menuIndicator(.hidden)
                            .buttonStyle(.borderless)
                            .fixedSize()
                        }
                        .draggable(TakeFile(take: take))
                    }
                }
                .listStyle(.plain)
                .frame(width: 380, height: 320)
            }
        }
    }
}

/// Calls `onChange` with the draft now and whenever it changes — the one
/// view that depends on the draft, so its dependents stay small.
private struct DraftWatcher: View {
    let onChange: (String) -> Void

    var body: some View {
        Color.clear
            .onChange(of: SpeechLab.shared.draft, initial: true) { _, draft in onChange(draft) }
    }
}
