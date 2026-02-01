//
//  LiquidWaveView.swift
//  whisper-on-device
//

import SwiftUI

/// A liquid waveform animation that responds to audio levels.
/// Renders dual sine waves with organic motion and gradient coloring.
struct LiquidWaveView: View {
    let level: CGFloat
    let phase: CGFloat

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let baseAmplitude: CGFloat = 4
    private let maxAmplitude: CGFloat = 16

    var body: some View {
        if reduceMotion {
            // Static bar when reduce motion is enabled
            staticBar
        } else {
            animatedWaves
        }
    }

    private var staticBar: some View {
        RoundedRectangle(cornerRadius: 2)
            .fill(
                LinearGradient(
                    colors: [.white, .cyan.opacity(0.8)],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .frame(height: 4 + level * 8)
            .padding(.horizontal, 20)
    }

    private var animatedWaves: some View {
        Canvas { context, size in
            let amplitude = baseAmplitude + (maxAmplitude * level)

            // Primary wave
            let primary = smoothWavePath(
                size: size,
                phase: phase,
                amplitude: amplitude,
                frequency: 2.0,
                verticalOffset: 0
            )

            // Secondary wave with phase offset
            let secondary = smoothWavePath(
                size: size,
                phase: phase * 1.3 + 1.5,
                amplitude: amplitude * 0.65,
                frequency: 2.4,
                verticalOffset: 0
            )

            context.addFilter(.blur(radius: 0.3))

            let gradientStart = CGPoint(x: 0, y: size.height * 0.5)
            let gradientEnd = CGPoint(x: size.width, y: size.height * 0.5)

            // Draw secondary wave first (behind)
            context.stroke(
                secondary,
                with: .linearGradient(
                    Gradient(colors: [
                        Color.white.opacity(0.55),
                        Color.teal.opacity(0.5),
                        Color.cyan.opacity(0.45)
                    ]),
                    startPoint: gradientStart,
                    endPoint: gradientEnd
                ),
                lineWidth: 1.5
            )

            // Draw primary wave on top
            context.stroke(
                primary,
                with: .linearGradient(
                    Gradient(colors: [
                        Color.white.opacity(0.9),
                        Color.cyan.opacity(0.85),
                        Color.white.opacity(0.8)
                    ]),
                    startPoint: gradientStart,
                    endPoint: gradientEnd
                ),
                lineWidth: 2
            )
        }
        .blendMode(.plusLighter)
    }

    /// Creates a smooth bezier-based wave path for more organic motion
    private func smoothWavePath(
        size: CGSize,
        phase: CGFloat,
        amplitude: CGFloat,
        frequency: CGFloat,
        verticalOffset: CGFloat
    ) -> Path {
        var path = Path()
        let midY = size.height * 0.5 + verticalOffset
        let width = size.width

        // Use more points for smoother curves
        let segments = 32
        let segmentWidth = width / CGFloat(segments)

        var points: [CGPoint] = []

        for i in 0...segments {
            let x = CGFloat(i) * segmentWidth
            let progress = x / width

            // Combine multiple sine waves for organic movement
            let angle1 = progress * frequency * .pi * 2 + phase
            let angle2 = progress * frequency * 0.5 * .pi * 2 + phase * 1.3

            // Window function to taper edges
            let window = sin(progress * .pi)

            let y = midY + (sin(angle1) * amplitude + sin(angle2) * amplitude * 0.3) * window
            points.append(CGPoint(x: x, y: y))
        }

        // Create smooth curve through points using Catmull-Rom to Bezier conversion
        guard points.count > 2 else { return path }

        path.move(to: points[0])

        for i in 0..<(points.count - 1) {
            let p0 = points[max(0, i - 1)]
            let p1 = points[i]
            let p2 = points[i + 1]
            let p3 = points[min(points.count - 1, i + 2)]

            // Catmull-Rom to Bezier control points
            let cp1 = CGPoint(
                x: p1.x + (p2.x - p0.x) / 6,
                y: p1.y + (p2.y - p0.y) / 6
            )
            let cp2 = CGPoint(
                x: p2.x - (p3.x - p1.x) / 6,
                y: p2.y - (p3.y - p1.y) / 6
            )

            path.addCurve(to: p2, control1: cp1, control2: cp2)
        }

        return path
    }
}

#Preview {
    VStack(spacing: 20) {
        LiquidWaveView(level: 0.3, phase: 0)
            .frame(width: 160, height: 24)
            .background(Color.black.opacity(0.3))

        LiquidWaveView(level: 0.7, phase: 1.5)
            .frame(width: 160, height: 24)
            .background(Color.black.opacity(0.3))
    }
    .padding()
    .background(Color.gray.opacity(0.2))
}
