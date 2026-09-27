//
//  SpeechVariantStudio.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  A · Studio — the workspace most TTS products converge on (ElevenLabs'
//  Text to Speech page): the script fills the page, a real player bar sits
//  under it, and an inspector holds Voice, History and Overlay. The player
//  bar is the signature: while speech streams, generated audio fills in
//  ahead of the playhead like a buffer bar; once done, it is a take you can
//  scrub, replay faster, or export.
//

import AppKit
import SwiftUI

struct StudioVariant: View {
    @AppStorage("speechPrototype.studio.inspector") private var showsInspector = true
    @AppStorage("speechPrototype.studio.tab") private var tab = StudioTab.voice
    @State private var selectedTakeID: UUID?

    var body: some View {
        // A plain trailing panel, not `.inspector`: a third split-view column
        // puts the window at its minimum width at typical sizes, and SwiftUI
        // then loops AppKit's constraint pass until it throws.
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                SpeechEngineNotice()
                StudioEditor()
                    .padding(.horizontal, 20)
                    .padding(.top, 12)
                StudioPlayerBar(selectedTakeID: $selectedTakeID)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 14)
            }
            if showsInspector {
                Divider()
                StudioInspector(tab: $tab, selectedTakeID: $selectedTakeID)
                    .frame(width: 310)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .animation(.smooth(duration: 0.25), value: showsInspector)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showsInspector.toggle()
                } label: {
                    Label("Inspector", systemImage: "sidebar.trailing")
                }
                .help("Show or hide Voice, History and Overlay")
            }
        }
    }
}

enum StudioTab: String {
    case voice, history, overlay
}

// MARK: - Editor

private struct StudioEditor: View {
    var body: some View {
        @Bindable var lab = SpeechLab.shared
        VStack(spacing: 0) {
            TextEditor(text: $lab.draft)
                .font(.system(size: 15))
                .lineSpacing(4)
                .scrollContentBackground(.hidden)
                .padding(.horizontal, 11)
                .padding(.top, 12)
                .overlay(alignment: .topLeading) {
                    if lab.draft.isEmpty { StudioEmptyEditor() }
                }
            Divider()
            StudioEditorFooter()
        }
        .background(.background, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.separator, lineWidth: 0.5)
        )
    }
}

private struct StudioEmptyEditor: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Type or paste what you want to hear.")
                .font(.system(size: 15))
                .foregroundStyle(.tertiary)
            HStack(spacing: 8) {
                starter("Paste", symbol: "doc.on.clipboard") {
                    if let text = NSPasteboard.general.string(forType: .string) {
                        SpeechLab.shared.draft = text
                    }
                }
                starter("Sample story", symbol: "book") {
                    SpeechLab.shared.draft = SpeechLab.sampleText
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
    }

    private func starter(_ title: String, symbol: String, action: @escaping () -> Void) -> some View
    {
        Button(action: action) {
            Label(title, systemImage: symbol)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
    }
}

private struct StudioEditorFooter: View {
    // One row whose minimum never depends on its width: texts truncate
    // rather than wrap, and nothing appears or disappears with the width.
    // Inside a split view with an inspector, a width-dependent minimum
    // (wrapping text, ViewThatFits, width-gated controls) either loops
    // AppKit's constraint pass until it throws or pins the window's width.
    var body: some View {
        let lab = SpeechLab.shared
        HStack(spacing: 12) {
            Text(
                "\(lab.draftWordCount) words · \(SpeechLabFormat.approximateLength(lab.draftEstimatedSeconds))"
            )
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .layoutPriority(3)
            EngineReadiness()
                .lineLimit(1)
                .layoutPriority(2)
            Spacer(minLength: 8)
            if let settings = lab.settings {
                HStack(spacing: 4) {
                    HotkeyCaps(combo: settings.ttsHotkey)
                        .fixedSize()
                    Text("reads selected text in any app")
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
                .layoutPriority(1)
            }
            LanguagePicker()
                .labelsHidden()
                .fixedSize()
        }
        .font(.callout)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }
}

// MARK: - Player bar

private struct StudioPlayerBar: View {
    @Binding var selectedTakeID: UUID?

    var body: some View {
        let lab = SpeechLab.shared
        let status = lab.status
        let take = lab.takes.first { $0.id == selectedTakeID } ?? lab.takes.first
        let isLive = lab.liveTake != nil || status.isActive

        // One layout that compresses (title and status truncate) instead of
        // one that swaps by width: its minimum stays low and fixed.
        HStack(spacing: 12) {
            Button {
                transport(take: take, status: status)
            } label: {
                Image(systemName: transportSymbol(take: take, status: status))
                    .font(.system(size: 17, weight: .semibold))
                    .frame(width: 30, height: 30)
                    .contentTransition(.symbolEffect(.replace))
            }
            .buttonStyle(.glass)
            .buttonBorderShape(.circle)
            .controlSize(.large)
            .disabled(take == nil && !isLive)
            .help("Play or pause")

            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 8) {
                    Text(title(take: take, isLive: isLive))
                        .fontWeight(.medium)
                        .lineLimit(1)
                    SpeechStatusLine()
                        .lineLimit(1)
                    Spacer(minLength: 6)
                    StudioTimeLabel(take: take)
                }
                .font(.callout)
                Group {
                    if let live = lab.liveTake {
                        LiveWaveform(live: live)
                    } else if let take {
                        TakeScrubber(take: take, height: 26)
                    } else {
                        Capsule().fill(.quaternary).frame(height: 2).frame(height: 26)
                    }
                }
            }
            .frame(minWidth: 100)

            SpeedMenu()
                .controlSize(.large)

            primaryButton(status: status, take: take)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .glassEffect(.regular, in: .rect(cornerRadius: 22))
    }

    @ViewBuilder
    private func primaryButton(status: SpeechLabStatus, take: SpeechTake?) -> some View {
        let lab = SpeechLab.shared
        if status.isActive {
            Button(role: .destructive) {
                lab.stop()
            } label: {
                Label("Stop", systemImage: "stop.fill")
                    .frame(minWidth: 84)
            }
            .buttonStyle(.glassProminent)
            .tint(.red)
            .controlSize(.large)
            .keyboardShortcut(.escape, modifiers: [])
            .help("Stop (Esc)")
        } else {
            HStack(spacing: 6) {
                Button {
                    lab.speak(lab.draft)
                } label: {
                    Label("Speak", systemImage: "play.fill")
                        .frame(minWidth: 84)
                }
                .buttonStyle(.glassProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(lab.draftIsEmpty)
                .help("Speak the text (⌘↩)")

                Menu {
                    if let take {
                        Section("This take") {
                            Button("Export as WAV…") { TakeExport.save(take, format: .wav) }
                            Button("Export as M4A…") { TakeExport.save(take, format: .m4a) }
                            Button("Export Captions (SRT)…") { TakeExport.saveCaptions(take) }
                        }
                    }
                    Button("Render Without Playing") {
                        Task { await lab.renderOffline(lab.draft) }
                    }
                    Button("Speak Selection in Editor") {
                        if let text = NSApp.keyWindow?.firstResponder as? NSTextView,
                            let range = Range(text.selectedRange(), in: text.string),
                            !range.isEmpty
                        {
                            lab.speak(String(text.string[range]))
                        }
                    }
                } label: {
                    Image(systemName: "chevron.down")
                }
                .menuIndicator(.hidden)
                .fixedSize()
                .controlSize(.large)
                .help("Export, or render without playing")
            }
        }
    }

    private func title(take: SpeechTake?, isLive: Bool) -> String {
        if isLive, let text = SpeechLab.shared.liveTake?.requestText {
            return String(text.prefix(80))
        }
        return take?.title ?? "Nothing generated yet"
    }

    private func transportSymbol(take: SpeechTake?, status: SpeechLabStatus) -> String {
        let lab = SpeechLab.shared
        switch status {
        case .speaking, .preparing: return "pause.fill"
        case .paused: return "play.fill"
        default: break
        }
        if let take, lab.player.isCurrent(take), lab.player.isPlaying { return "pause.fill" }
        return "play.fill"
    }

    private func transport(take: SpeechTake?, status: SpeechLabStatus) {
        let lab = SpeechLab.shared
        switch status {
        case .speaking, .preparing, .paused:
            lab.togglePause()
        default:
            if let take { lab.player.toggle(take) }
        }
    }
}

/// Generated audio fills in ahead of the playhead, on a time scale set by
/// the estimated length, so nothing rescales while it streams.
struct LiveWaveform: View {
    let live: LiveTake

    var body: some View {
        let lab = SpeechLab.shared
        let estimate = Double(SpeechLab.estimatedSeconds(live.requestText ?? ""))
        TimelineView(.animation(minimumInterval: 1.0 / 20)) { _ in
            GeometryReader { proxy in
                let total = max(
                    live.isGenerationFinished ? live.duration : max(estimate, live.duration), 0.1)
                let generatedWidth = proxy.size.width * min(live.duration / total, 1)
                let heard = lab.readAlong.mode == .live ? lab.readAlong.now() : 0
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary).frame(height: 2)
                    WaveformBars(
                        peaks: live.peaks,
                        progress: live.duration > 0 ? heard / live.duration : 0
                    )
                    .frame(width: max(generatedWidth, 2))
                }
                .frame(maxHeight: .infinity)
            }
        }
        .frame(height: 26)
    }
}

private struct StudioTimeLabel: View {
    let take: SpeechTake?

    var body: some View {
        let lab = SpeechLab.shared
        let ticking = lab.liveTake != nil || lab.player.isPlaying
        TimelineView(.animation(minimumInterval: 0.25, paused: !ticking)) { _ in
            Group {
                if let live = lab.liveTake {
                    let estimate = SpeechLab.estimatedSeconds(live.requestText ?? "")
                    Text(
                        "\(SpeechLabFormat.time(lab.readAlong.now())) / \(live.isGenerationFinished ? SpeechLabFormat.time(live.duration) : "~" + SpeechLabFormat.time(Double(max(estimate, Int(live.duration)))))"
                    )
                } else if let take {
                    let current = lab.player.isCurrent(take) ? lab.player.currentTime : 0
                    Text(
                        "\(SpeechLabFormat.time(current)) / \(SpeechLabFormat.time(take.duration))")
                } else {
                    Text("0:00 / 0:00")
                }
            }
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .font(.callout)
        }
        .fixedSize()
    }
}

// MARK: - Inspector

private struct StudioInspector: View {
    @Binding var tab: StudioTab
    @Binding var selectedTakeID: UUID?

    var body: some View {
        VStack(spacing: 0) {
            Picker("Inspector", selection: $tab) {
                Text("Voice").tag(StudioTab.voice)
                Text("History").tag(StudioTab.history)
                Text("Overlay").tag(StudioTab.overlay)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            switch tab {
            case .voice: StudioVoicePane()
            case .history: StudioHistoryPane(selectedTakeID: $selectedTakeID)
            case .overlay:
                Form {
                    Section {
                        OverlaySettingsControls()
                    } footer: {
                        Text(
                            "Changes preview on screen. The overlay follows selected text you speak with the hotkey, in any app."
                        )
                    }
                }
                .formStyle(.grouped)
            }
        }
    }
}

private struct StudioVoicePane: View {
    var body: some View {
        let lab = SpeechLab.shared
        Form {
            if let settings = lab.settings {
                @Bindable var settings = settings
                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(lab.currentVoiceName)
                            .font(.title3.weight(.semibold))
                        TextField(
                            "Description", text: $settings.ttsVoiceDescription,
                            prompt: Text("Describe a voice: who, timbre, pace, mood"),
                            axis: .vertical
                        )
                        .labelsHidden()
                        .lineLimit(2...5)
                        .textFieldStyle(.plain)
                        .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 2)
                    LanguagePicker()
                    HStack {
                        Button("Try Another Take") {
                            lab.tryAnotherTake(sample: lab.draft)
                        }
                        .help(
                            "Render this voice again from your text's opening. The take you hear last is the one kept."
                        )
                        Spacer()
                        Text("The take you hear last is kept")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    VoiceTakeStrip(description: settings.ttsVoiceDescription, limit: 4)
                } header: {
                    Text("Voice")
                }

                Section("Voices") {
                    ForEach(lab.allVoices) { voice in
                        Button {
                            lab.select(voice)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(voice.name)
                                    Text(voice.blurb)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if voice.description == settings.ttsVoiceDescription {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(.tint)
                                }
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }

                Section("Delivery") {
                    DeliveryControls()
                    DisclosureGroup("Advanced") {
                        AdvancedDeliveryControls()
                    }
                }
            }
        }
        .formStyle(.grouped)
    }
}

private struct StudioHistoryPane: View {
    @Binding var selectedTakeID: UUID?

    var body: some View {
        let lab = SpeechLab.shared
        VStack(spacing: 0) {
            if lab.takes.isEmpty {
                // In a scroll view: wrapping text outside one gives the
                // inspector a width-dependent minimum height, which loops
                // the split view's constraint pass at narrow window widths.
                ScrollView {
                    ContentUnavailableView {
                        Label("No takes yet", systemImage: "waveform")
                    } description: {
                        Text(
                            "Everything you hear, from this page or the hotkey, is kept here to replay or export."
                        )
                    }
                    .padding(.top, 80)
                }
            } else {
                List(selection: $selectedTakeID) {
                    ForEach(lab.takes) { take in
                        StudioHistoryRow(take: take, isSelected: selectedTakeID == take.id)
                            .tag(take.id)
                            .contextMenu {
                                TakeMenuItems(take: take) { lab.draft = $0 }
                            }
                            .draggable(TakeFile(take: take))
                    }
                }
                .listStyle(.inset)
                Divider()
                HStack {
                    Text("\(lab.takes.count) takes · drag one out to save it")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Clear") { lab.clearTakes() }
                        .buttonStyle(.borderless)
                }
                .font(.caption)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
        }
    }
}

private struct StudioHistoryRow: View {
    let take: SpeechTake
    let isSelected: Bool

    var body: some View {
        let player = SpeechLab.shared.player
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 10) {
                Button {
                    player.toggle(take)
                } label: {
                    Image(
                        systemName: player.isCurrent(take) && player.isPlaying
                            ? "pause.circle.fill" : "play.circle.fill"
                    )
                    .font(.system(size: 22))
                    .foregroundStyle(.tint)
                }
                .buttonStyle(.plain)
                VStack(alignment: .leading, spacing: 2) {
                    Text(take.title)
                        .lineLimit(1)
                    HStack(spacing: 4) {
                        Image(systemName: take.source.symbol)
                        Text(
                            "\(SpeechLabFormat.time(take.duration)) · \(take.voiceName) · \(SpeechLabFormat.relative(take.createdAt))"
                        )
                        if !take.isComplete {
                            Text("· stopped early")
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            if isSelected || player.isCurrent(take) {
                TakeScrubber(take: take, height: 20)
            }
        }
        .padding(.vertical, 3)
    }
}

/// What gates the first sound here — the local engine's residency — shown
/// where cloud products show credits.
struct EngineReadiness: View {
    var body: some View {
        let engine = SpeechLab.shared.enginePresenter
        let (color, label): (Color, String) =
            engine?.isLoading == true
            ? (.orange, "Engine loading")
            : engine?.isModelLoaded == true ? (.green, "Engine warm") : (.secondary, "Engine idle")
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label)
        }
        .foregroundStyle(.tertiary)
        .help("The voice model runs on this Mac. It loads on first use and stays warm.")
    }
}
