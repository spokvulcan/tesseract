//
//  SpeechLabComponents.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  Small pieces the variants share — controls, not layouts: a waveform, the
//  hotkey as key caps, voice and speed menus, the human-named delivery
//  sliders, the overlay settings, a take's context menu, and the
//  read-along paragraph. Each variant composes its own page from these.
//

import AppKit
import SwiftUI
import TesseractSpeech

// MARK: - Formatting

enum SpeechLabFormat {
    static func time(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let total = Int(seconds.rounded(.down))
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    static func approximateLength(_ seconds: Int) -> String {
        if seconds < 60 { return "~\(max(seconds, 1)) s" }
        let minutes = Double(seconds) / 60
        return minutes < 10 ? String(format: "~%.1f min", minutes) : "~\(Int(minutes)) min"
    }

    static func relative(_ date: Date) -> String {
        let seconds = Date.now.timeIntervalSince(date)
        if seconds < 60 { return "Just now" }
        return date.formatted(.relative(presentation: .named, unitsStyle: .abbreviated))
    }
}

// MARK: - Waveform

/// Bars from 50 ms peaks. `progress` (0…1) tints what has been heard.
struct WaveformBars: View {
    let peaks: [Float]
    var progress: Double = 0
    var tint: Color = .accentColor
    var rest: Color = .secondary.opacity(0.35)
    var barWidth: CGFloat = 2
    var spacing: CGFloat = 1.5
    var minimumBar: CGFloat = 1.5

    var body: some View {
        Canvas { context, size in
            let step = barWidth + spacing
            let count = max(Int(size.width / step), 1)
            let bars = TakePeaks.resample(peaks, to: count)
            let split = Int(Double(count) * min(max(progress, 0), 1))
            for (i, value) in bars.enumerated() {
                let height = max(minimumBar, CGFloat(value) * size.height)
                let rect = CGRect(
                    x: CGFloat(i) * step, y: (size.height - height) / 2, width: barWidth,
                    height: height)
                context.fill(
                    Path(roundedRect: rect, cornerRadius: barWidth / 2),
                    with: .color(i < split ? tint : rest))
            }
        }
        .accessibilityHidden(true)
    }
}

/// A take's waveform that plays from wherever it is clicked.
struct TakeScrubber: View {
    let take: SpeechTake
    var tint: Color = .accentColor
    var height: CGFloat = 28

    var body: some View {
        let player = SpeechLab.shared.player
        let isCurrent = player.isCurrent(take)
        TimelineView(
            .animation(minimumInterval: 1.0 / 20, paused: !(isCurrent && player.isPlaying))
        ) { _ in
            WaveformBars(
                peaks: take.audio.peaks,
                progress: isCurrent && take.duration > 0 ? player.currentTime / take.duration : 0,
                tint: tint)
        }
        .frame(height: height)
        .overlay {
            GeometryReader { proxy in
                Color.clear
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0).onEnded { value in
                            let fraction = min(
                                max(value.location.x / max(proxy.size.width, 1), 0), 1)
                            SpeechLab.shared.player.play(take, from: fraction * take.duration)
                        })
            }
        }
        .help("Click to play from here")
    }
}

// MARK: - Hotkey as key caps

struct HotkeyCaps: View {
    let combo: KeyCombo

    var body: some View {
        HStack(spacing: 3) {
            ForEach(Array(Self.parts(of: combo.displayString).enumerated()), id: \.offset) {
                _, part in
                Text(part)
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1.5)
                    .background(
                        RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .strokeBorder(.tertiary, lineWidth: 1)
                    )
            }
        }
        .foregroundStyle(.secondary)
        .accessibilityLabel(combo.displayString)
    }

    /// "fnSpace" → ["fn", "Space"]; "⌃⌥S" → ["⌃", "⌥", "S"].
    static func parts(of display: String) -> [String] {
        var rest = Substring(display)
        var parts: [String] = []
        let modifiers = ["⌃", "⌥", "⇧", "⌘", "fn"]
        var matched = true
        while matched, !rest.isEmpty {
            matched = false
            for modifier in modifiers where rest.hasPrefix(modifier) {
                parts.append(modifier)
                rest = rest.dropFirst(modifier.count)
                matched = true
            }
        }
        if !rest.isEmpty { parts.append(String(rest)) }
        return parts
    }
}

// MARK: - Menus

/// Voices as a menu: presets, your designed voices, and "Default".
struct VoiceMenuItems: View {
    let onEdit: (() -> Void)?

    var body: some View {
        let lab = SpeechLab.shared
        let current = lab.currentVoiceDescription
        Section("Built-in") {
            ForEach(SpeechLab.presets) { voice in
                Toggle(
                    isOn: Binding(
                        get: { current == voice.description }, set: { _ in lab.select(voice) })
                ) {
                    Text(voice.name)
                    Text(voice.blurb)
                }
            }
        }
        if !lab.yourVoices.isEmpty {
            Section("Your voices") {
                ForEach(lab.yourVoices) { voice in
                    Toggle(
                        isOn: Binding(
                            get: { current == voice.description }, set: { _ in lab.select(voice) })
                    ) {
                        Text(voice.name)
                    }
                }
            }
        }
        if let onEdit {
            Divider()
            Button("Design a Voice…", action: onEdit)
        }
    }
}

struct SpeedMenu: View {
    var body: some View {
        @Bindable var lab = SpeechLab.shared
        Menu {
            Picker("Speed", selection: $lab.playbackRate) {
                ForEach([Float(0.75), 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0], id: \.self) { rate in
                    Text(Self.label(rate)).tag(rate)
                }
            }
            .pickerStyle(.inline)
        } label: {
            Text(Self.label(lab.playbackRate))
                .monospacedDigit()
        }
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Playback speed — changes pace, not pitch")
    }

    static func label(_ rate: Float) -> String {
        rate == 1 ? "1×" : String(format: "%g×", rate)
    }
}

struct LanguagePicker: View {
    var body: some View {
        let lab = SpeechLab.shared
        Picker(
            "Language",
            selection: Binding(
                get: { lab.currentLanguage },
                set: { lab.settings?.ttsLanguage = $0 })
        ) {
            ForEach(TTSLanguage.allCases) { language in
                Text("\(language.flag) \(language.displayName)").tag(language.rawValue)
            }
        }
    }
}

// MARK: - Delivery (human names over the sampler knobs)

struct DeliveryControls: View {
    var body: some View {
        if let settings = SpeechLab.shared.settings {
            @Bindable var settings = settings
            LabeledSlider(
                title: "Expressiveness", value: $settings.ttsTemperature, range: 0.3...1.5,
                low: "Even", high: "Lively",
                help: "How much pacing, emphasis and intonation vary. The engine's temperature.")
            LabeledSlider(
                title: "Voice steadiness", value: $settings.ttsDetailTemperature, range: 0.1...1.0,
                low: "Steady", high: "Loose",
                help:
                    "How freely the timbre drifts. Lower keeps the voice the same person between passages."
            )
        }
    }
}

struct AdvancedDeliveryControls: View {
    var body: some View {
        if let settings = SpeechLab.shared.settings {
            @Bindable var settings = settings
            LabeledSlider(
                title: "Clarity (top-p)", value: $settings.ttsTopP, range: 0.5...1.0,
                low: "Clear", high: "Free",
                help: "Restricts sampling to the likeliest choices. Lower is clearer and flatter.")
            LabeledSlider(
                title: "Repetition penalty", value: $settings.ttsRepetitionPenalty,
                range: 1.0...1.5, low: "Off", high: "Strong",
                help: "Discourages repeated sounds and audio loops.")
            Picker("Longest passage", selection: $settings.ttsMaxTokens) {
                Text("1 min").tag(1024)
                Text("2 min").tag(2048)
                Text("4 min").tag(4096)
                Text("8 min").tag(8192)
            }
            .help("The most audio one passage may run before it is cut off.")
            LabeledContent("Seed") {
                HStack(spacing: 6) {
                    TextField("Seed", value: $settings.ttsSeed, format: .number.grouping(.never))
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                        .labelsHidden()
                        .frame(width: 72)
                    Button {
                        settings.ttsSeed = Int.random(in: 0...99_999)
                    } label: {
                        Image(systemName: "dice")
                    }
                    .buttonStyle(.borderless)
                    .help("New seed")
                }
            }
            .help("The same seed, text and voice render the same audio.")
            Button("Reset Delivery") {
                let defaults = TTSParameters()
                settings.ttsParameters = defaults
                settings.ttsSeed = 0
            }
        }
    }
}

struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let low: String
    let high: String
    let help: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(value, format: .number.precision(.fractionLength(2)))
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
            }
            // End labels sit under the track: a Slider's own min/max value
            // labels inside a grouped Form loop AppKit's constraint pass.
            Slider(
                value: Binding(get: { value }, set: { value = ($0 / 0.05).rounded() * 0.05 }),
                in: range
            ) {
                Text(title)
            }
            .labelsHidden()
            HStack {
                Text(low)
                Spacer()
                Text(high)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .help(help)
    }
}

// MARK: - Overlay settings

/// The overlay knobs, with a live preview of each change on screen.
struct OverlaySettingsControls: View {
    var showsScope = true

    var body: some View {
        @Bindable var lab = SpeechLab.shared
        Picker("Overlay", selection: $lab.overlay.style) {
            ForEach(OverlayPrefs.Style.allCases) { style in
                Label(style.label, systemImage: style.symbol).tag(style)
            }
        }
        .onChange(of: lab.overlay.style) { _, style in lab.overlayController.preview(style: style) }
        Text(lab.overlay.style.detail)
            .font(.caption)
            .foregroundStyle(.secondary)
        if lab.overlay.style == .island || lab.overlay.style == .captions {
            Picker("Text size", selection: $lab.overlay.size) {
                ForEach(OverlayPrefs.Size.allCases) { size in Text(size.label).tag(size) }
            }
            .pickerStyle(.segmented)
            .onChange(of: lab.overlay.size) { _, _ in
                lab.overlayController.preview(style: lab.overlay.style)
            }
            Picker("Highlight", selection: $lab.overlay.tint) {
                ForEach(OverlayPrefs.Tint.allCases) { tint in
                    Label {
                        Text(tint.label)
                    } icon: {
                        Image(systemName: "circle.fill").foregroundStyle(tint.color)
                    }
                    .tag(tint)
                }
            }
            .onChange(of: lab.overlay.tint) { _, _ in
                lab.overlayController.preview(style: lab.overlay.style)
            }
            Toggle("Pause and stop on hover", isOn: $lab.overlay.showsControls)
        }
        if showsScope, lab.overlay.style != .off {
            Picker("Show for", selection: $lab.overlay.scope) {
                ForEach(OverlayPrefs.Scope.allCases) { scope in Text(scope.label).tag(scope) }
            }
        }
    }
}

// MARK: - Take actions

struct TakeMenuItems: View {
    let take: SpeechTake
    var onUseText: ((String) -> Void)?

    var body: some View {
        let lab = SpeechLab.shared
        Button(lab.player.isCurrent(take) && lab.player.isPlaying ? "Pause" : "Play") {
            lab.player.toggle(take)
        }
        Divider()
        Button("Export as WAV…") { TakeExport.save(take, format: .wav) }
        Button("Export as M4A…") { TakeExport.save(take, format: .m4a) }
        Button("Export Captions (SRT)…") { TakeExport.saveCaptions(take) }
        ShareLink(item: TakeFile(take: take), preview: SharePreview(take.title))
        if take.pinnedVoice != nil, take.source == .voiceTake {
            Button("Use This Take as the Voice") { lab.keepVoice(of: take) }
        }
        Divider()
        Button("Copy Text") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(take.text, forType: .string)
        }
        if let onUseText {
            Button("Put Text in Editor") { onUseText(take.text) }
        }
        Button("Speak Again") { lab.speak(take.text) }
        Divider()
        Button("Delete", role: .destructive) { lab.delete(take) }
    }
}

// MARK: - Engine notice

/// Speech can't start until the Voice Engine is on disk; say so up front.
struct SpeechEngineNotice: View {
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    var body: some View {
        let id = ModelDefinition.defaultTextToSpeechModelID
        switch downloadManager.status(for: id) {
        case .notDownloaded:
            notice {
                Image(systemName: "arrow.down.circle")
                Text(
                    "Speech needs the Voice Engine (\(ModelDefinition.withID(id)?.sizeDescription ?? ""))."
                )
                Spacer(minLength: 8)
                Button("Open Models") { (NSApp.delegate as? AppDelegate)?.navigateToModels() }
                    .buttonStyle(.borderless)
            }
        case .downloading(let progress):
            notice {
                ProgressView(value: progress).controlSize(.small).frame(width: 80)
                Text("The Voice Engine is downloading: \(Int(progress * 100))%")
                Spacer(minLength: 8)
            }
        case .downloaded, .verifying, .error:
            EmptyView()
        }
    }

    private func notice<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 8, content: content)
            .font(.callout)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
    }
}

// MARK: - Status line

struct SpeechStatusLine: View {
    var body: some View {
        let status = SpeechLab.shared.status
        HStack(spacing: 6) {
            switch status {
            case .loadingModel, .capturing, .preparing, .rendering:
                ProgressView().controlSize(.small)
            case .speaking:
                Image(systemName: "waveform")
                    .symbolEffect(.variableColor.iterative, options: .repeating)
                    .foregroundStyle(.tint)
            case .paused:
                Image(systemName: "pause.circle").foregroundStyle(.secondary)
            case .error:
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
            case .idle:
                EmptyView()
            }
            if status != .idle {
                Text(status.label)
                    .foregroundStyle(
                        {
                            if case .error = status { return Color.red }
                            return Color.secondary
                        }()
                    )
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .font(.callout)
        .animation(.default, value: status)
    }
}

// MARK: - Read-along paragraph

/// Which parts of the text light up while it is read (Apple's Read & Speak
/// offers the same four).
enum ReadAlongHighlight: String, CaseIterable, Identifiable {
    case words, sentences, both, none
    var id: String { rawValue }
    var label: String {
        switch self {
        case .words: "Words"
        case .sentences: "Sentences"
        case .both: "Words and sentences"
        case .none: "Nothing"
        }
    }
    var word: Bool { self == .words || self == .both }
    var sentence: Bool { self == .sentences || self == .both }
}

/// One paragraph of the page's text with the heard word lit and the heard
/// sentence tinted. Equatable, so only the paragraph being read redraws.
/// Every sentence is a link (`speech-sentence:N`) so a click can start the
/// reading there.
struct ReadAlongParagraph: View, Equatable {
    let document: ReadingDocument
    let paragraph: Int
    /// Document word being heard, if it falls in this paragraph.
    let activeWord: Int?
    /// Words before this index have been heard (dimming the rest).
    let heardThrough: Int
    let font: Font
    let lineSpacing: CGFloat
    let highlight: ReadAlongHighlight
    var clickableSentences = false

    static func == (lhs: ReadAlongParagraph, rhs: ReadAlongParagraph) -> Bool {
        lhs.paragraph == rhs.paragraph && lhs.activeWord == rhs.activeWord
            && lhs.heardThrough == rhs.heardThrough && lhs.highlight == rhs.highlight
            && lhs.lineSpacing == rhs.lineSpacing && lhs.document == rhs.document
            && lhs.clickableSentences == rhs.clickableSentences
    }

    var body: some View {
        Text(attributed)
            .font(font)
            .lineSpacing(lineSpacing)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var attributed: AttributedString {
        let info = document.paragraphs[paragraph]
        let text = document.text
        // heardThrough < 0: nothing is being read, so nothing dims.
        let reading = heardThrough >= 0
        let activeSentence = activeWord.flatMap { word in
            document.words.indices.contains(word) ? document.words[word].sentence : nil
        }
        var result = AttributedString()
        var cursor = info.range.lowerBound
        for index in info.firstWord..<(info.firstWord + info.wordCount) {
            let word = document.words[index]
            let inActiveSentence = highlight.sentence && word.sentence == activeSentence
            if cursor < word.range.lowerBound {
                var gap = AttributedString(text[cursor..<word.range.lowerBound])
                let sameSentence = index > 0 && document.words[index - 1].sentence == word.sentence
                if inActiveSentence, sameSentence {
                    gap.backgroundColor = Color.accentColor.opacity(0.14)
                }
                if clickableSentences, sameSentence {
                    gap.link = URL(string: "speech-sentence:\(word.sentence)")
                }
                result += gap
            }
            var run = AttributedString(text[word.range])
            if highlight.word, index == activeWord {
                run.foregroundColor = Color.accentColor
                run.backgroundColor = Color.accentColor.opacity(inActiveSentence ? 0.26 : 0.18)
            } else {
                if reading, highlight != .none {
                    run.foregroundColor = index < heardThrough ? Color.primary : Color.secondary
                } else {
                    run.foregroundColor = Color.primary
                }
                if inActiveSentence { run.backgroundColor = Color.accentColor.opacity(0.14) }
            }
            if clickableSentences {
                run.link = URL(string: "speech-sentence:\(word.sentence)")
            }
            result += run
            cursor = word.range.upperBound
        }
        return result
    }
}

// MARK: - Voice takes strip

/// Every "Try another take" of the current voice, newest first. One is in
/// use; any earlier one can be brought back (WellSaid's "Use Take").
struct VoiceTakeStrip: View {
    let description: String
    var limit = 5

    var body: some View {
        let lab = SpeechLab.shared
        let takes = lab.voiceTakes(for: description)
        let kept = lab.keptTakeID(for: description)
        if takes.isEmpty {
            Text("No takes of this voice yet. Each one you try is kept here.")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            VStack(spacing: 6) {
                ForEach(Array(takes.prefix(limit).enumerated()), id: \.element.id) { index, take in
                    HStack(spacing: 8) {
                        Button {
                            lab.player.toggle(take)
                        } label: {
                            Image(
                                systemName: lab.player.isCurrent(take) && lab.player.isPlaying
                                    ? "pause.circle.fill" : "play.circle.fill"
                            )
                            .font(.system(size: 18))
                            .foregroundStyle(.tint)
                        }
                        .buttonStyle(.plain)
                        Text("Take \(takes.count - index)")
                            .monospacedDigit()
                        TakeScrubber(take: take, height: 16)
                        if take.id == kept {
                            Text("In use")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.tint)
                                .frame(width: 44)
                        } else {
                            Button("Use") { lab.keepVoice(of: take) }
                                .controlSize(.small)
                                .frame(width: 44)
                                .help("Make this take the voice from now on")
                        }
                    }
                }
            }
        }
    }
}
