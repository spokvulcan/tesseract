//
//  TransportBar.swift
//  tesseract-ios
//
//  The Reader's controls, the Mac's in one bottom bar: sentence back,
//  play/pause, sentence forward, where you are (drag to go anywhere) with the
//  time left, speed, and the voice.
//

import SwiftUI

struct TransportBar: View {
    let reader: SpeechReader
    @Environment(SpeechCoordinator.self) private var coordinator
    @Environment(SpeechEnginePresenter.self) private var engine
    @Environment(PhoneSettings.self) private var settings
    @Environment(PhoneReading.self) private var reading
    @State private var showsVoices = false

    var body: some View {
        let isPlaying = reader.isReading && !reader.isPaused
        VStack(spacing: 10) {
            VStack(spacing: 4) {
                Scrubber(progress: reader.progress) { reader.seek(toFraction: $0) }
                    .disabled(reader.isEmpty)
                HStack {
                    Text(status.text)
                        .foregroundStyle(status.isError ? Color.red : Color.secondary)
                        .lineLimit(2)
                    Spacer()
                    SpeedMenu()
                }
                .font(.footnote.weight(.medium))
                .monospacedDigit()
            }
            HStack(spacing: 28) {
                Button("Voice", systemImage: "person.wave.2") { showsVoices = true }
                    .labelStyle(.iconOnly)
                    .font(.title3)
                Button("Previous sentence", systemImage: "backward.fill") {
                    reader.skip(sentences: -1)
                }
                .labelStyle(.iconOnly)
                .font(.title2)
                .disabled(reader.isEmpty)
                Button {
                    reader.isReading ? reader.togglePause() : reader.play()
                } label: {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 28, weight: .semibold))
                        .frame(width: 64, height: 64)
                        .contentTransition(.symbolEffect(.replace))
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .disabled(reader.isEmpty)
                .accessibilityLabel(isPlaying ? "Pause" : reader.isReading ? "Resume" : "Read")
                Button("Next sentence", systemImage: "forward.fill") {
                    reader.skip(sentences: 1)
                }
                .labelStyle(.iconOnly)
                .font(.title2)
                .disabled(reader.isEmpty)
                Button("Stop", systemImage: "stop.fill") { reader.stop() }
                    .labelStyle(.iconOnly)
                    .font(.title3)
                    .disabled(!reader.isReading)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 14)
        .padding(.bottom, 8)
        .glassEffect(.regular, in: .rect(cornerRadius: 28))
        .padding(.horizontal, 10)
        .sheet(isPresented: $showsVoices) { VoicePickerSheet() }
    }

    /// Time left, or what is happening instead: loading, starting, an error.
    private var status: (text: String, isError: Bool) {
        if case .error(let message) = coordinator.state { return (message, true) }
        if let notice = reading.notice { return (notice, false) }
        if engine.isLoading { return ("Loading the voice…", false) }
        if case .generating = coordinator.state, reader.isReading, reader.highlight == nil {
            return ("Starting…", false)
        }
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

/// Where reading is in the text. Drag or tap to go anywhere.
private struct Scrubber: View {
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
                    .frame(width: max(6, proxy.size.width * shown))
            }
            .frame(height: dragging == nil ? 5 : 9)
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
                    }
            )
            .animation(.snappy, value: dragging == nil)
        }
        .frame(height: 24)
        .accessibilityElement()
        .accessibilityLabel("Position")
        .accessibilityValue("\(Int(progress * 100)) percent")
        .accessibilityAdjustableAction { direction in
            onSeek(min(max(progress + (direction == .increment ? 0.02 : -0.02), 0), 1))
        }
    }
}

/// Read-aloud speed: changes pace, not pitch. With the neural voice, only
/// the rates its Speed Check found it keeps up with.
private struct SpeedMenu: View {
    @Environment(SpeechCoordinator.self) private var coordinator
    @Environment(PhoneSettings.self) private var settings
    @Environment(PhoneVoice.self) private var voice

    var body: some View {
        Menu {
            Picker(
                "Speed",
                selection: Binding(
                    get: { settings.ttsPlaybackRate },
                    set: { coordinator.setPlaybackRate($0) })
            ) {
                ForEach(voice.playableRates, id: \.self) { rate in
                    Text(Self.label(rate)).tag(rate)
                }
            }
        } label: {
            Text(Self.label(settings.ttsPlaybackRate))
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(.quaternary, in: .capsule)
        }
        .accessibilityLabel("Speed \(Self.label(settings.ttsPlaybackRate))")
    }

    static func label(_ rate: Double) -> String {
        rate == 1 ? "1×" : String(format: "%g×", rate)
    }
}
