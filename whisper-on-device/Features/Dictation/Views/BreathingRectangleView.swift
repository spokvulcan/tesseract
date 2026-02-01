//
//  BreathingRectangleView.swift
//  whisper-on-device
//
//  iOS 18 Siri-style "breathing rectangle" visualization.
//  A rounded rectangle with morphing edges and animated mesh gradient.

import SwiftUI

/// A Siri-style breathing rectangle visualization that responds to audio levels.
/// Features morphing edges with sine-based displacement and an animated gradient background.
struct BreathingRectangleView: View {
    let level: CGFloat
    let phase: CGFloat

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let cornerRadius: CGFloat = 8

    var body: some View {
        if reduceMotion {
            staticView
        } else {
            animatedView
        }
    }

    private var staticView: some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            .fill(
                LinearGradient(
                    colors: [.cyan.opacity(0.6), .purple.opacity(0.5), .blue.opacity(0.6)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.3), lineWidth: 1)
            )
    }

    private var animatedView: some View {
        Canvas { context, size in
            // Create breathing path with morphing edges
            let path = breathingPath(size: size)

            // Create animated gradient colors based on phase
            let colors = animatedGradientColors

            // Fill with gradient
            context.fill(
                path,
                with: .linearGradient(
                    Gradient(colors: colors),
                    startPoint: CGPoint(x: 0, y: 0),
                    endPoint: CGPoint(x: size.width, y: size.height)
                )
            )

            // Add subtle glow/blur effect
            context.addFilter(.blur(radius: 0.5))

            // Stroke for definition
            context.stroke(
                path,
                with: .color(.white.opacity(0.25)),
                lineWidth: 0.5
            )
        }
    }

    /// Colors that shift based on phase for the animated gradient
    private var animatedGradientColors: [Color] {
        let t = phase * 0.3

        // Shift hue positions over time
        let hue1 = 0.55 + sin(t) * 0.05           // Cyan range
        let hue2 = 0.75 + sin(t + 1.0) * 0.05     // Purple range
        let hue3 = 0.6 + sin(t + 2.0) * 0.05      // Blue range

        // Saturation increases with audio level
        let saturation = 0.5 + level * 0.3
        let brightness = 0.7 + level * 0.2

        return [
            Color(hue: hue1, saturation: saturation, brightness: brightness),
            Color(hue: hue2, saturation: saturation, brightness: brightness),
            Color(hue: hue3, saturation: saturation, brightness: brightness)
        ]
    }

    /// Creates a rounded rectangle path with breathing/morphing edges
    private func breathingPath(size: CGSize) -> Path {
        var path = Path()

        let inset: CGFloat = 2
        let rect = CGRect(x: inset, y: inset,
                          width: size.width - inset * 2,
                          height: size.height - inset * 2)

        // Amplitude of edge displacement based on audio level
        let amplitude = 1.0 + level * 2.5

        // Number of segments per edge for smooth morphing
        let segments = 16

        // Generate points around the rectangle with sine displacement
        var points: [CGPoint] = []

        // Top edge (left to right)
        for i in 0...segments {
            let t = CGFloat(i) / CGFloat(segments)
            let x = rect.minX + cornerRadius + (rect.width - cornerRadius * 2) * t
            let displacement = sin(t * .pi * 4 + phase) * amplitude * edgeWindow(t)
            let y = rect.minY + displacement
            points.append(CGPoint(x: x, y: y))
        }

        // Right edge (top to bottom)
        for i in 1...segments {
            let t = CGFloat(i) / CGFloat(segments)
            let y = rect.minY + cornerRadius + (rect.height - cornerRadius * 2) * t
            let displacement = sin(t * .pi * 4 + phase + 1.5) * amplitude * edgeWindow(t)
            let x = rect.maxX + displacement
            points.append(CGPoint(x: x, y: y))
        }

        // Bottom edge (right to left)
        for i in 1...segments {
            let t = CGFloat(i) / CGFloat(segments)
            let x = rect.maxX - cornerRadius - (rect.width - cornerRadius * 2) * t
            let displacement = sin(t * .pi * 4 + phase + 3.0) * amplitude * edgeWindow(t)
            let y = rect.maxY + displacement
            points.append(CGPoint(x: x, y: y))
        }

        // Left edge (bottom to top)
        for i in 1..<segments {
            let t = CGFloat(i) / CGFloat(segments)
            let y = rect.maxY - cornerRadius - (rect.height - cornerRadius * 2) * t
            let displacement = sin(t * .pi * 4 + phase + 4.5) * amplitude * edgeWindow(t)
            let x = rect.minX + displacement
            points.append(CGPoint(x: x, y: y))
        }

        // Build smooth path through points
        guard points.count > 2 else { return path }

        path.move(to: points[0])

        for i in 0..<points.count {
            let p0 = points[(i - 1 + points.count) % points.count]
            let p1 = points[i]
            let p2 = points[(i + 1) % points.count]
            let p3 = points[(i + 2) % points.count]

            // Catmull-Rom to Bezier conversion
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

        path.closeSubpath()
        return path
    }

    /// Window function to reduce displacement near corners
    private func edgeWindow(_ t: CGFloat) -> CGFloat {
        sin(t * .pi)
    }
}

#Preview("Static") {
    BreathingRectangleView(level: 0.5, phase: 0)
        .frame(width: 100, height: 28)
        .padding()
        .background(Color.black.opacity(0.8))
}

#Preview("Animated") {
    TimelineView(.animation) { timeline in
        let phase = CGFloat(timeline.date.timeIntervalSinceReferenceDate * 2.0)
        VStack(spacing: 16) {
            Text("Low Level").foregroundStyle(.secondary)
            BreathingRectangleView(level: 0.2, phase: phase)
                .frame(width: 100, height: 28)

            Text("Medium Level").foregroundStyle(.secondary)
            BreathingRectangleView(level: 0.5, phase: phase)
                .frame(width: 100, height: 28)

            Text("High Level").foregroundStyle(.secondary)
            BreathingRectangleView(level: 0.9, phase: phase)
                .frame(width: 100, height: 28)
        }
        .padding()
        .background(Color.black.opacity(0.8))
    }
}
