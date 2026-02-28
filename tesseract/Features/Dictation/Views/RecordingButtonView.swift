//
//  RecordingButtonView.swift
//  tesseract
//

import SwiftUI

struct RecordingButtonView: View {
    let state: DictationState
    let onToggle: () -> Void

    @State private var isPulsing = false

    // Compact button: 64px with pulse ring up to 72 × 1.2 = 86px
    private let buttonSize: CGFloat = 64
    private let containerSize: CGFloat = 96

    var body: some View {
        Button(action: onToggle) {
            ZStack {
                // Pulse ring - only rendered when recording to avoid idle CPU usage
                if state == .recording {
                    Circle()
                        .stroke(Color.red.opacity(0.5), lineWidth: 3)
                        .frame(width: 72, height: 72)
                        .scaleEffect(isPulsing ? 1.2 : 1.0)
                        .opacity(isPulsing ? 0 : 1)
                        .animation(
                            .easeInOut(duration: 1.0).repeatForever(autoreverses: false),
                            value: isPulsing
                        )
                }

                Circle()
                    .fill(buttonBackgroundColor.opacity(0.85))
                    .frame(width: buttonSize, height: buttonSize)
                    .glassEffect(.regular.tint(buttonBackgroundColor))
                    .shadow(color: Color.black.opacity(0.15), radius: 8, y: 3)
                    .animation(.easeInOut(duration: 0.2), value: state)

                Image(systemName: buttonIcon)
                    .font(.system(size: 26))
                    .foregroundStyle(.white)
                    .symbolEffect(.pulse, isActive: state == .processing)
            }
            .frame(width: containerSize, height: containerSize)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityHint(state == .recording ? "Double tap to stop recording" : "Double tap to start recording")
        .onChange(of: state) { _, newState in
            isPulsing = newState == .recording
        }
    }

    private var accessibilityLabel: String {
        switch state {
        case .idle: return "Start Recording"
        case .listening: return "Listening for voice"
        case .recording: return "Recording in progress"
        case .processing: return "Processing transcription"
        case .error: return "Recording error"
        }
    }

    private var buttonBackgroundColor: Color {
        switch state {
        case .idle:
            return .accentColor
        case .listening:
            return .yellow
        case .recording:
            return .red
        case .processing:
            return .orange
        case .error:
            return .red.opacity(0.5)
        }
    }

    private var buttonIcon: String {
        switch state {
        case .idle, .error:
            return "mic.fill"
        case .listening:
            return "ear.fill"
        case .recording:
            return "stop.fill"
        case .processing:
            return "waveform"
        }
    }
}
