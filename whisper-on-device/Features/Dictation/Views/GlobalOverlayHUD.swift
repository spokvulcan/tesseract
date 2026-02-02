//
//  GlobalOverlayHUD.swift
//  whisper-on-device
//

import SwiftUI

/// Global overlay HUD that displays recording waveform or processing indicator.
/// Designed as a floating pill that appears on top of all applications.
struct GlobalOverlayHUD: View {
    let state: DictationState
    let audioLevel: Float

    @State private var smoothedLevel: CGFloat = 0.08
    @State private var isVisible = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    // Updated to smaller, more elegant size
    private let hudWidth: CGFloat = 120
    private let hudHeight: CGFloat = 32
    private let cornerRadius: CGFloat = 16

    var body: some View {
        ZStack {
            if shouldShow {
                hudContent
                    .opacity(isVisible ? 1 : 0)
                    .scaleEffect(isVisible ? 1 : 0.85)
            }
        }
        .onChange(of: state) { _, newState in
            updateVisibility(for: newState)
        }
        .onChange(of: audioLevel) { _, newValue in
            updateAudioLevel(newValue)
        }
        .onAppear {
            updateVisibility(for: state)
        }
    }

    private var shouldShow: Bool {
        switch state {
        case .recording, .processing:
            return true
        default:
            return false
        }
    }

    @ViewBuilder
    private var hudContent: some View {
        Group {
            if state == .recording {
                recordingView
            } else if state == .processing {
                processingView
            }
        }
    }

    private var recordingView: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let phase = reduceMotion ? 0 : CGFloat(time * 2.2)

            pillContainer {
                visualizationContent(level: smoothedLevel, phase: phase)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
            }
        }
    }

    private func visualizationContent(level: CGFloat, phase: CGFloat) -> some View {
        AudioBarsView(level: level, phase: phase)
    }

    private var processingView: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 120.0)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate

            pillContainer {
                ProcessingDotsView(time: time)
            }
        }
    }

    private func pillContainer<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        ZStack {
            // Glass background
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(.ultraThinMaterial)

            // Subtle gradient border
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(
                    LinearGradient(
                        colors: [
                            Color.white.opacity(0.2),
                            Color.white.opacity(0.08)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 0.5
                )

            // Content
            content()
                .clipShape(RoundedRectangle(cornerRadius: cornerRadius - 2, style: .continuous))
        }
        .frame(width: hudWidth, height: hudHeight)
        .shadow(color: Color.black.opacity(0.12), radius: 6, y: 3)
    }

    private func updateVisibility(for newState: DictationState) {
        let show = newState == .recording || newState == .processing

        if show {
            withAnimation(reduceMotion ? nil : .spring(response: 0.3, dampingFraction: 0.7)) {
                isVisible = true
            }
        } else {
            withAnimation(reduceMotion ? nil : .spring(response: 0.25, dampingFraction: 0.8)) {
                isVisible = false
            }
        }
    }

    private func updateAudioLevel(_ newValue: Float) {
        let clamped = max(0.06, min(CGFloat(newValue), 1))
        if reduceMotion {
            smoothedLevel = clamped
        } else {
            withAnimation(.easeOut(duration: 0.1)) {
                smoothedLevel = clamped
            }
        }
    }
}

#Preview("Recording") {
    GlobalOverlayHUD(state: .recording, audioLevel: 0.5)
        .padding(50)
        .background(Color.gray.opacity(0.3))
}

#Preview("Processing") {
    GlobalOverlayHUD(state: .processing, audioLevel: 0)
        .padding(50)
        .background(Color.gray.opacity(0.3))
}
