//
//  OrganicBlobView.swift
//  whisper-on-device
//
//  Organic blob/orb visualization that morphs based on audio.
//  Creates a liquid droplet effect with smooth deformations.

import SwiftUI

/// An organic blob visualization that deforms smoothly based on audio levels.
/// Uses simplex-like noise for organic motion with gradient fills.
struct OrganicBlobView: View {
    let level: CGFloat
    let phase: CGFloat

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        if reduceMotion {
            staticView
        } else {
            animatedView
        }
    }

    private var staticView: some View {
        GeometryReader { geometry in
            let size = min(geometry.size.width, geometry.size.height)

            Ellipse()
                .fill(
                    RadialGradient(
                        colors: [.white, .cyan.opacity(0.7), .purple.opacity(0.5)],
                        center: .center,
                        startRadius: 0,
                        endRadius: size * 0.4
                    )
                )
                .frame(width: size * 0.6, height: size * 0.5)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var animatedView: some View {
        GeometryReader { geometry in
            let size = min(geometry.size.width, geometry.size.height)

            Canvas { context, canvasSize in
                let center = CGPoint(x: canvasSize.width / 2, y: canvasSize.height / 2)
                let baseRadius = size * 0.35

                // Create the organic blob path
                let blobPath = organicBlobPath(
                    center: center,
                    baseRadius: baseRadius,
                    level: level,
                    phase: phase
                )

                // Draw glow layer
                context.addFilter(.blur(radius: 8 + level * 8))
                context.fill(
                    blobPath,
                    with: .color(.cyan.opacity(0.3 + level * 0.2))
                )

                // Reset filter for main blob
                context.addFilter(.blur(radius: 0))

                // Main blob with gradient
                let gradient = Gradient(colors: gradientColors)
                context.fill(
                    blobPath,
                    with: .radialGradient(
                        gradient,
                        center: center,
                        startRadius: 0,
                        endRadius: baseRadius * 1.2
                    )
                )

                // Inner highlight
                let highlightPath = organicBlobPath(
                    center: CGPoint(x: center.x - baseRadius * 0.15,
                                    y: center.y - baseRadius * 0.15),
                    baseRadius: baseRadius * 0.3,
                    level: level * 0.5,
                    phase: phase * 1.3
                )

                context.addFilter(.blur(radius: 2))
                context.fill(
                    highlightPath,
                    with: .color(.white.opacity(0.4))
                )
            }
        }
    }

    private var gradientColors: [Color] {
        let intensity = 0.6 + level * 0.4
        return [
            .white,
            Color.cyan.opacity(intensity),
            Color.blue.opacity(intensity * 0.8),
            Color.purple.opacity(intensity * 0.6)
        ]
    }

    /// Creates an organic blob path using noise-based displacement
    private func organicBlobPath(
        center: CGPoint,
        baseRadius: CGFloat,
        level: CGFloat,
        phase: CGFloat
    ) -> Path {
        var path = Path()

        let segments = 64
        var points: [CGPoint] = []

        // Amount of deformation based on audio level
        let deformAmount = 0.08 + level * 0.15

        for i in 0..<segments {
            let angle = (CGFloat(i) / CGFloat(segments)) * 2 * .pi

            // Multiple noise frequencies for organic feel
            let noise1 = sin(angle * 3 + phase * 1.2) * deformAmount
            let noise2 = sin(angle * 5 + phase * 0.8 + 1.5) * deformAmount * 0.5
            let noise3 = sin(angle * 7 + phase * 1.5 + 3.0) * deformAmount * 0.25

            // Breathing effect - overall size oscillation
            let breathe = 1 + sin(phase) * 0.03 * (1 + level)

            // Combined displacement
            let displacement = 1 + noise1 + noise2 + noise3
            let radius = baseRadius * displacement * breathe

            let x = center.x + cos(angle) * radius
            let y = center.y + sin(angle) * radius
            points.append(CGPoint(x: x, y: y))
        }

        // Build smooth closed path
        guard points.count > 2 else { return path }

        path.move(to: points[0])

        for i in 0..<points.count {
            let p0 = points[(i - 1 + points.count) % points.count]
            let p1 = points[i]
            let p2 = points[(i + 1) % points.count]
            let p3 = points[(i + 2) % points.count]

            // Catmull-Rom spline for smooth curves
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
}

#Preview("Static") {
    OrganicBlobView(level: 0.5, phase: 0)
        .frame(width: 80, height: 32)
        .padding()
        .background(Color.black.opacity(0.8))
}

#Preview("Animated") {
    TimelineView(.animation) { timeline in
        let phase = CGFloat(timeline.date.timeIntervalSinceReferenceDate * 1.8)
        VStack(spacing: 20) {
            Text("Low Level").foregroundStyle(.secondary)
            OrganicBlobView(level: 0.2, phase: phase)
                .frame(width: 80, height: 32)

            Text("Medium Level").foregroundStyle(.secondary)
            OrganicBlobView(level: 0.5, phase: phase)
                .frame(width: 80, height: 32)

            Text("High Level").foregroundStyle(.secondary)
            OrganicBlobView(level: 0.9, phase: phase)
                .frame(width: 80, height: 32)
        }
        .padding()
        .background(Color.black.opacity(0.8))
    }
}

#Preview("Large Orb") {
    TimelineView(.animation) { timeline in
        let phase = CGFloat(timeline.date.timeIntervalSinceReferenceDate * 1.8)
        OrganicBlobView(level: 0.6, phase: phase)
            .frame(width: 120, height: 120)
            .padding()
            .background(Color.black.opacity(0.9))
    }
}
