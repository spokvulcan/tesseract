//
//  AudioLevelMeter.swift
//  tesseract
//

import SwiftUI

/// Live microphone level bar with a Test button — the Dictation pane's mic
/// check. Metering-only capture: a long test must not accumulate audio or
/// resample it on stop.
struct AudioLevelMeter: View {
    var audioCapture: AudioCaptureEngine
    /// Level source: the Overlay Feed carries the meter for every capture,
    /// the settings test included (the engine's polled level is retired).
    var feed: DictationFeed
    @State private var isTestingMic = false

    var body: some View {
        HStack {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.secondary.opacity(0.2))

                    RoundedRectangle(cornerRadius: 4)
                        .fill(levelColor)
                        .frame(width: geometry.size.width * CGFloat(feed.level))
                        .animation(.linear(duration: 0.1), value: feed.level)
                }
            }
            .accessibilityElement()
            .accessibilityLabel("Audio level meter")
            .accessibilityValue("\(Int(feed.level * 100)) percent")

            Button(isTestingMic ? "Stop" : "Test") {
                toggleMicTest()
            }
            .buttonStyle(.bordered)
            .accessibilityLabel(isTestingMic ? "Stop microphone test" : "Test microphone")
            .accessibilityHint(
                isTestingMic
                    ? "Stops the microphone level test" : "Starts monitoring microphone input level"
            )
        }
    }

    private var levelColor: Color {
        let level = feed.level
        if level > 0.8 {
            return .red
        } else if level > 0.5 {
            return .yellow
        } else {
            return .green
        }
    }

    private func toggleMicTest() {
        if isTestingMic {
            _ = audioCapture.stopCapture()
            isTestingMic = false
        } else {
            do {
                try audioCapture.startLevelMetering()
                isTestingMic = true
            } catch {
                Log.general.error("Failed to start mic test: \(error)")
            }
        }
    }
}
