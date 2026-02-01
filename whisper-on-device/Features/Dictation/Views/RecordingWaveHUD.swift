//
//  RecordingWaveHUD.swift
//  whisper-on-device
//

import SwiftUI

struct RecordingWaveHUD: View {
    @ObservedObject var audioCapture: AudioCaptureEngine
    let state: DictationState

    @State private var isVisible = false
    @State private var smoothedLevel: CGFloat = 0.08

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.animation) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let phase = reduceMotion ? 0 : CGFloat(time * 2.2)

            ZStack {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [
                                        Color.white.opacity(0.45),
                                        Color.white.opacity(0.12),
                                        Color.white.opacity(0.35)
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    )
                    .shadow(color: Color.black.opacity(0.15), radius: 12, y: 6)

                LiquidWaveCanvas(level: smoothedLevel, phase: phase)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 10)

                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.18), lineWidth: 1)
            }
            .frame(width: 240, height: 54)
            .opacity(isVisible ? 1 : 0)
            .scaleEffect(isVisible ? 1 : 0.86)
            .animation(
                reduceMotion ? .linear(duration: 0) : .spring(response: 0.32, dampingFraction: 0.7),
                value: isVisible
            )
        }
        .onAppear {
            isVisible = state == .recording
        }
        .onChange(of: state) { _, newState in
            isVisible = newState == .recording
        }
        .onReceive(audioCapture.$audioLevel) { newValue in
            let clamped = max(0.06, min(CGFloat(newValue), 1))
            if reduceMotion {
                smoothedLevel = clamped
            } else {
                withAnimation(.easeOut(duration: 0.12)) {
                    smoothedLevel = clamped
                }
            }
        }
        .allowsHitTesting(false)
    }
}

private struct LiquidWaveCanvas: View {
    let level: CGFloat
    let phase: CGFloat

    private let baseAmplitude: CGFloat = 4
    private let maxAmplitude: CGFloat = 14

    var body: some View {
        Canvas { context, size in
            let amplitude = baseAmplitude + (maxAmplitude * level)
            let primary = wavePath(size: size, phase: phase, amplitude: amplitude, frequency: 2.0)
            let secondary = wavePath(size: size, phase: phase * 1.4 + 1.2, amplitude: amplitude * 0.6, frequency: 2.6)

            context.addFilter(.blur(radius: 0.2))

            let gradientStart = CGPoint(x: 0, y: size.height * 0.5)
            let gradientEnd = CGPoint(x: size.width, y: size.height * 0.5)

            context.stroke(
                primary,
                with: .linearGradient(
                    Gradient(colors: [
                        Color.white.opacity(0.85),
                        Color.cyan.opacity(0.8),
                        Color.blue.opacity(0.7)
                    ]),
                    startPoint: gradientStart,
                    endPoint: gradientEnd
                ),
                lineWidth: 2
            )

            context.stroke(
                secondary,
                with: .linearGradient(
                    Gradient(colors: [
                        Color.white.opacity(0.6),
                        Color.teal.opacity(0.55),
                        Color.blue.opacity(0.5)
                    ]),
                    startPoint: gradientStart,
                    endPoint: gradientEnd
                ),
                lineWidth: 1.5
            )
        }
        .blendMode(.plusLighter)
    }

    private func wavePath(size: CGSize, phase: CGFloat, amplitude: CGFloat, frequency: CGFloat) -> Path {
        var path = Path()
        let midY = size.height * 0.5
        let width = size.width
        let step: CGFloat = 2

        var x: CGFloat = 0
        while x <= width {
            let progress = x / width
            let angle = progress * frequency * .pi * 2 + phase
            let y = midY + sin(angle) * amplitude
            if x == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
            x += step
        }

        return path
    }
}

#Preview {
    RecordingWaveHUD(audioCapture: AudioCaptureEngine(), state: .recording)
        .padding()
}
