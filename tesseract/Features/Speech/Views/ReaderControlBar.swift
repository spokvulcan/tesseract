//
//  ReaderControlBar.swift
//  tesseract
//
//  The Reader's floating controls: sentence back, play/pause, sentence
//  forward, where you are (drag to go anywhere), speed, and the voice.
//  Stop appears while reading.
//

import SwiftUI

struct ReaderControlBar: View {
    @Environment(SpeechReader.self) private var reader
    @Environment(SpeechCoordinator.self) private var coordinator
    @Environment(SpeechEnginePresenter.self) private var engine
    @Environment(VoiceLibrary.self) private var library
    let onVoices: () -> Void

    var body: some View {
        let isPlaying = reader.isReading && !reader.isPaused
        let playLabel = isPlaying ? "Pause" : reader.isReading ? "Resume" : "Read"
        GlassEffectContainer(spacing: 10) {
            HStack(spacing: 10) {
                roundButton("backward.fill", label: "Previous sentence") {
                    reader.skip(sentences: -1)
                }
                .disabled(reader.isEmpty)

                Button {
                    reader.isReading ? reader.togglePause() : reader.play()
                } label: {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 34, height: 34)
                        .contentTransition(.symbolEffect(.replace))
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(reader.isEmpty)
                .help("\(playLabel) (⌘↩)")
                .accessibilityLabel(playLabel)

                roundButton("forward.fill", label: "Next sentence") {
                    reader.skip(sentences: 1)
                }
                .disabled(reader.isEmpty)

                VStack(alignment: .leading, spacing: 5) {
                    ReaderStatus()
                    ReaderScrubber(progress: reader.progress) { reader.seek(toFraction: $0) }
                        .disabled(reader.isEmpty)
                }
                .frame(minWidth: 80, idealWidth: 200, maxWidth: 220)
                .padding(.horizontal, 4)

                if reader.isReading {
                    roundButton("stop.fill", label: "Stop") { reader.stop() }
                        .keyboardShortcut(.escape, modifiers: [])
                }

                SpeedMenu()

                Button(action: onVoices) {
                    Label(library.currentName, systemImage: "person.wave.2")
                        .lineLimit(1)
                        .frame(maxWidth: 150)
                }
                .buttonStyle(.glass)
                .controlSize(.large)
                .help("Choose or design a voice")
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .glassEffect(.regular, in: .capsule)
        }
    }

    private func roundButton(_ symbol: String, label: String, action: @escaping () -> Void)
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
        .help(label)
        .accessibilityLabel(label)
    }
}

/// Time left, or what is happening instead: loading, starting, an error.
private struct ReaderStatus: View {
    @Environment(SpeechReader.self) private var reader
    @Environment(SpeechCoordinator.self) private var coordinator
    @Environment(SpeechEnginePresenter.self) private var engine

    var body: some View {
        let (text, isError) = status
        HStack(spacing: 6) {
            if engine.isLoading || isStarting {
                ProgressView().controlSize(.mini)
            }
            Text(text)
                .foregroundStyle(isError ? Color.red : Color.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .font(.system(size: 12, weight: .medium))
        .monospacedDigit()
    }

    private var isStarting: Bool {
        if case .generating = coordinator.state, reader.isReading, reader.highlight == nil {
            return true
        }
        return false
    }

    private var status: (String, Bool) {
        if case .error(let message) = coordinator.state { return (message, true) }
        if engine.isLoading { return ("Loading the voice…", false) }
        if isStarting { return ("Starting…", false) }
        if reader.isEmpty { return ("Nothing to read", false) }
        let left = Self.duration(reader.timeLeft)
        if reader.isPaused { return ("Paused · \(left) left", false) }
        return (reader.isReading ? "\(left) left" : left, false)
    }

    /// "2 h 14 min", "12 min", "under a minute".
    static func duration(_ seconds: TimeInterval) -> String {
        let minutes = Int((seconds / 60).rounded())
        if minutes < 1 { return "under a minute" }
        if minutes < 60 { return "\(minutes) min" }
        return minutes % 60 == 0 ? "\(minutes / 60) h" : "\(minutes / 60) h \(minutes % 60) min"
    }
}

/// Where reading is in the text. Drag or click to go anywhere.
private struct ReaderScrubber: View {
    let progress: Double
    let onSeek: (Double) -> Void
    @State private var dragging: Double?

    var body: some View {
        GeometryReader { proxy in
            let shown = dragging ?? progress
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule()
                    .fill(Color.accentColor)
                    .frame(width: max(4, proxy.size.width * shown))
            }
            .frame(height: 4)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        dragging = min(max(value.location.x / max(proxy.size.width, 1), 0), 1)
                    }
                    .onEnded { value in
                        onSeek(min(max(value.location.x / max(proxy.size.width, 1), 0), 1))
                        dragging = nil
                    })
        }
        .frame(height: 12)
        .accessibilityElement()
        .accessibilityLabel("Position")
        .accessibilityValue("\(Int(progress * 100)) percent")
        .accessibilityAdjustableAction { direction in
            onSeek(min(max(progress + (direction == .increment ? 0.02 : -0.02), 0), 1))
        }
    }
}

/// Read-aloud speed: changes pace, not pitch.
private struct SpeedMenu: View {
    @Environment(SpeechCoordinator.self) private var coordinator
    @Environment(SettingsManager.self) private var settings

    private static let rates: [Double] = [0.75, 0.9, 1.0, 1.15, 1.25, 1.5, 1.75, 2.0]

    var body: some View {
        Menu {
            ForEach(Self.rates, id: \.self) { rate in
                Button {
                    coordinator.setPlaybackRate(rate)
                } label: {
                    if rate == settings.ttsPlaybackRate {
                        Label(Self.label(rate), systemImage: "checkmark")
                    } else {
                        Text(Self.label(rate))
                    }
                }
            }
        } label: {
            Text(Self.label(settings.ttsPlaybackRate))
                .monospacedDigit()
        }
        .menuIndicator(.hidden)
        .fixedSize()
        .buttonStyle(.glass)
        .controlSize(.large)
        .help("Speed: changes pace, not pitch")
        .accessibilityLabel("Speed \(Self.label(settings.ttsPlaybackRate))")
    }

    static func label(_ rate: Double) -> String {
        rate == 1 ? "1×" : String(format: "%g×", rate)
    }
}
