//
//  SpeechVariantFeed.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  E · Feed — speech as a log, in the agent chat's own grammar (the app's
//  reference surface): everything you heard, from this page, from the
//  hotkey in any app, or from the assistant, is a card in one stream,
//  newest at the bottom, with the composer docked below like Messages or
//  Apple's Live Speech. Each card replays with its words lit, drags out as
//  a file, and shares. Signature: the page stops being a tool you open and
//  becomes the memory of everything the Mac has read to you.
//

import AppKit
import SwiftUI

struct FeedVariant: View {
    var body: some View {
        let lab = SpeechLab.shared
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    // Offline renders were never heard; they live in Takes.
                    let heard = lab.takes.filter { $0.source != .render }.reversed() as [SpeechTake]
                    ForEach(Array(heard.enumerated()), id: \.element.id) { index, take in
                        let takes = heard
                        if index == 0
                            || !Calendar.current.isDate(
                                takes[index - 1].createdAt, inSameDayAs: take.createdAt)
                        {
                            FeedDayHeader(date: take.createdAt)
                        }
                        FeedCard(take: take)
                            .id(take.id)
                    }
                    if let live = lab.liveTake {
                        FeedLiveCard(live: live)
                            .id("live")
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .frame(maxWidth: 720)
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity)
            }
            .defaultScrollAnchor(.bottom)
            .onChange(of: lab.liveTake?.id) { _, _ in
                withAnimation(.easeOut(duration: 0.3)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: lab.takes.count) { _, _ in
                withAnimation(.easeOut(duration: 0.3)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
        .overlay {
            if lab.takes.allSatisfy({ $0.source == .render }), lab.liveTake == nil {
                FeedIntro()
                    .padding(.bottom, 40)
                    .allowsHitTesting(false)
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) { SpeechEngineNotice() }
        .safeAreaInset(edge: .bottom, spacing: 0) { FeedComposer() }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                SpeedMenu()
                OverlayToolbarButton()
                Menu {
                    Button("Clear History", role: .destructive) { lab.clearTakes() }
                        .disabled(lab.takes.isEmpty)
                } label: {
                    Label("More", systemImage: "ellipsis.circle")
                }
            }
        }
    }
}

// MARK: - Empty state

private struct FeedIntro: View {
    var body: some View {
        let lab = SpeechLab.shared
        VStack(spacing: 14) {
            Image(systemName: "text.bubble")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(.tertiary)
            Text("Everything read to you, in one place")
                .font(.system(size: 17, weight: .semibold))
            HStack(spacing: 5) {
                Text("Type below, or select text in any app and press")
                if let settings = lab.settings { HotkeyCaps(combo: settings.ttsHotkey) }
            }
            .foregroundStyle(.secondary)
            Text(
                "Each reading stays here to replay with its words lit, save as audio, or drag into another app."
            )
            .foregroundStyle(.tertiary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: 420)
        }
        .frame(maxWidth: .infinity)
    }
}

private struct FeedDayHeader: View {
    let date: Date

    var body: some View {
        Text(
            Calendar.current.isDateInToday(date)
                ? "Today" : date.formatted(date: .abbreviated, time: .omitted)
        )
        .font(.system(size: 11, weight: .semibold))
        .kerning(0.5)
        .textCase(.uppercase)
        .foregroundStyle(.tertiary)
        .frame(maxWidth: .infinity)
        .padding(.top, 6)
    }
}

// MARK: - Cards

private struct FeedCard: View {
    let take: SpeechTake
    @State private var isExpanded = false
    @State private var document: ReadingDocument?

    var body: some View {
        let lab = SpeechLab.shared
        let player = lab.player
        let isReplaying = lab.readAlong.mode == .replay && lab.readAlong.takeID == take.id
        VStack(alignment: .leading, spacing: 10) {
            header
            Group {
                if isReplaying, let document {
                    let active = lab.readAlong.wordIndex >= 0 ? lab.readAlong.wordIndex : nil
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(document.paragraphs.indices, id: \.self) { index in
                            ReadAlongParagraph(
                                document: document, paragraph: index,
                                activeWord: active.flatMap {
                                    document.words.indices.contains($0)
                                        && document.words[$0].paragraph == index ? $0 : nil
                                },
                                heardThrough: active ?? 0,
                                font: .system(size: 14), lineSpacing: 3, highlight: .words
                            )
                            .equatable()
                        }
                    }
                } else {
                    Text(take.text)
                        .font(.system(size: 14))
                        .lineSpacing(3)
                        .lineLimit(isExpanded ? nil : 4)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            if !isReplaying, take.text.count > 280 {
                Button(isExpanded ? "Show Less" : "Show More") { isExpanded.toggle() }
                    .buttonStyle(.link)
                    .font(.caption)
            }
            HStack(spacing: 10) {
                Button {
                    if document == nil { document = ReadingDocument(text: take.text) }
                    player.toggle(take)
                } label: {
                    Image(
                        systemName: player.isCurrent(take) && player.isPlaying
                            ? "pause.circle.fill" : "play.circle.fill"
                    )
                    .font(.system(size: 26))
                    .foregroundStyle(.tint)
                }
                .buttonStyle(.plain)
                .help("Replay")
                TakeScrubber(take: take, height: 24)
                TimelineView(
                    .animation(
                        minimumInterval: 0.25, paused: !(player.isCurrent(take) && player.isPlaying)
                    )
                ) { _ in
                    Text(
                        player.isCurrent(take)
                            ? "\(SpeechLabFormat.time(player.currentTime)) / \(SpeechLabFormat.time(take.duration))"
                            : SpeechLabFormat.time(take.duration)
                    )
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .font(.callout)
                }
                .fixedSize()
                Menu {
                    TakeMenuItems(take: take) { SpeechLab.shared.draft = $0 }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuIndicator(.hidden)
                .buttonStyle(.borderless)
                .fixedSize()
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(.fill.quinary))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(isReplaying ? Color.accentColor.opacity(0.5) : .clear, lineWidth: 1)
        )
        .contextMenu { TakeMenuItems(take: take) { SpeechLab.shared.draft = $0 } }
        .draggable(TakeFile(take: take))
    }

    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: take.source.symbol)
            Text(take.source.label)
            if take.source == .selection, let settings = SpeechLab.shared.settings {
                HotkeyCaps(combo: settings.ttsHotkey)
            }
            Text("·")
            Text(take.voiceName)
            if !take.isComplete {
                Text("· stopped early")
            }
            Spacer()
            Text(take.createdAt.formatted(date: .omitted, time: .shortened))
        }
        .font(.system(size: 11, weight: .medium))
        .foregroundStyle(.secondary)
        .lineLimit(1)
    }
}

/// What is being read right now, lit as it goes.
private struct FeedLiveCard: View {
    let live: LiveTake
    @State private var document = ReadingDocument(text: "")

    var body: some View {
        let lab = SpeechLab.shared
        let active = lab.readAlong.mode == .live ? lab.readAlong.documentWordIndex : nil
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "waveform")
                    .symbolEffect(
                        .variableColor.iterative, options: .repeating, isActive: !lab.isPaused
                    )
                    .foregroundStyle(.tint)
                Text(live.source == .page ? "Reading now" : "\(live.source.label) · reading now")
                Spacer()
                SpeechStatusLine()
            }
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(.secondary)
            .lineLimit(1)

            if document.words.isEmpty {
                Text(live.requestText ?? "…")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(document.paragraphs.indices, id: \.self) { index in
                        ReadAlongParagraph(
                            document: document, paragraph: index,
                            activeWord: active.flatMap {
                                document.words.indices.contains($0)
                                    && document.words[$0].paragraph == index ? $0 : nil
                            },
                            heardThrough: active ?? 0,
                            font: .system(size: 14), lineSpacing: 3, highlight: .words
                        )
                        .equatable()
                    }
                }
            }

            HStack(spacing: 10) {
                Button {
                    lab.togglePause()
                } label: {
                    Image(systemName: lab.isPaused ? "play.circle.fill" : "pause.circle.fill")
                        .font(.system(size: 26))
                        .foregroundStyle(.tint)
                }
                .buttonStyle(.plain)
                LiveWaveform(live: live)
                Button {
                    lab.stop()
                } label: {
                    Image(systemName: "stop.circle")
                        .font(.system(size: 20))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .keyboardShortcut(.escape, modifiers: [])
                .help("Stop (Esc)")
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous).fill(
                Color.accentColor.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.accentColor.opacity(0.4), lineWidth: 1)
        )
        .onAppear { document = ReadingDocument(text: live.requestText ?? "") }
        .onChange(of: live.id) { _, _ in document = ReadingDocument(text: live.requestText ?? "") }
    }
}

// MARK: - Composer

private struct FeedComposer: View {
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        let lab = SpeechLab.shared
        let status = lab.status
        HStack(alignment: .bottom, spacing: 10) {
            Menu {
                VoiceMenuItems(onEdit: nil)
            } label: {
                Label(lab.currentVoiceName, systemImage: "person.wave.2")
                    .lineLimit(1)
            }
            .menuStyle(.button)
            .buttonStyle(.glass)
            .fixedSize()
            .help("Voice")

            TextField(
                "Say", text: $text, prompt: Text("Type something to hear it"), axis: .vertical
            )
            .labelsHidden()
            .textFieldStyle(.plain)
            .font(.system(size: 14))
            .lineLimit(1...6)
            .focused($focused)
            .onSubmit(send)
            .padding(.vertical, 7)

            if status.isActive {
                Button {
                    lab.stop()
                } label: {
                    Image(systemName: "stop.fill")
                        .font(.system(size: 12, weight: .bold))
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .tint(.red)
                .help("Stop (Esc)")
            } else {
                Button(action: send) {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 13, weight: .bold))
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help("Speak (Return). Option-Return adds a line.")
            }
        }
        .controlSize(.large)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .glassEffect(.regular, in: .rect(cornerRadius: 22))
        .frame(maxWidth: 720)
        .padding(.horizontal, 24)
        .padding(.bottom, 16)
        .frame(maxWidth: .infinity)
        .onAppear { focused = true }
    }

    private func send() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        SpeechLab.shared.speak(trimmed)
        text = ""
    }
}
