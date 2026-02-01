//
//  PulsingRingsView.swift
//  whisper-on-device
//
//  Minimal pulsing concentric rings visualization.
//  Creates a ripple/broadcast effect when speaking.

import SwiftUI

/// A minimal pulsing rings visualization that creates ripple effects based on audio level.
/// Features concentric circles that pulse outward with staggered animations.
struct PulsingRingsView: View {
    let level: CGFloat
    let phase: CGFloat

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let ringCount = 3

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
            let centerSize = size * 0.4

            ZStack {
                // Single static ring
                Circle()
                    .strokeBorder(
                        LinearGradient(
                            colors: [.white.opacity(0.6), .cyan.opacity(0.4)],
                            startPoint: .top,
                            endPoint: .bottom
                        ),
                        lineWidth: 2
                    )
                    .frame(width: centerSize * (1 + level * 0.5),
                           height: centerSize * (1 + level * 0.5))

                // Center dot
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [.white, .cyan.opacity(0.8)],
                            center: .center,
                            startRadius: 0,
                            endRadius: centerSize * 0.3
                        )
                    )
                    .frame(width: centerSize * 0.4, height: centerSize * 0.4)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var animatedView: some View {
        GeometryReader { geometry in
            let size = min(geometry.size.width, geometry.size.height)

            ZStack {
                // Pulsing rings
                ForEach(0..<ringCount, id: \.self) { index in
                    PulsingRing(
                        index: index,
                        totalRings: ringCount,
                        level: level,
                        phase: phase,
                        baseSize: size
                    )
                }

                // Center orb
                CenterOrb(level: level, phase: phase, baseSize: size)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

/// Individual pulsing ring with staggered animation
private struct PulsingRing: View {
    let index: Int
    let totalRings: Int
    let level: CGFloat
    let phase: CGFloat
    let baseSize: CGFloat

    var body: some View {
        let phaseOffset = CGFloat(index) * (2.0 * .pi / CGFloat(totalRings))
        let ringPhase = phase + phaseOffset

        // Ring expands and fades based on phase
        let expansion = (1 + sin(ringPhase)) * 0.5  // 0 to 1
        let scale = 0.3 + expansion * 0.7           // 0.3 to 1.0
        let opacity = (1 - expansion) * (0.3 + level * 0.5)  // Fade as it expands

        // Ring size increases with audio level
        let levelBoost = 1 + level * 0.3

        Circle()
            .strokeBorder(
                LinearGradient(
                    colors: [
                        Color.white.opacity(opacity),
                        Color.cyan.opacity(opacity * 0.7)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                ),
                lineWidth: 1.5 - expansion * 0.5  // Thinner as it expands
            )
            .frame(width: baseSize * scale * levelBoost,
                   height: baseSize * scale * levelBoost)
    }
}

/// Central orb that pulses with audio
private struct CenterOrb: View {
    let level: CGFloat
    let phase: CGFloat
    let baseSize: CGFloat

    var body: some View {
        let pulse = 1 + sin(phase * 2) * 0.1 * level
        let orbSize = baseSize * 0.25 * pulse

        Circle()
            .fill(
                RadialGradient(
                    colors: [
                        .white,
                        .cyan.opacity(0.8 + level * 0.2),
                        .blue.opacity(0.4)
                    ],
                    center: .center,
                    startRadius: 0,
                    endRadius: orbSize * 0.6
                )
            )
            .frame(width: orbSize, height: orbSize)
            .shadow(color: .cyan.opacity(0.4 + level * 0.3), radius: 4 + level * 4)
    }
}

#Preview("Static") {
    PulsingRingsView(level: 0.5, phase: 0)
        .frame(width: 80, height: 32)
        .padding()
        .background(Color.black.opacity(0.8))
}

#Preview("Animated") {
    TimelineView(.animation) { timeline in
        let phase = CGFloat(timeline.date.timeIntervalSinceReferenceDate * 1.5)
        VStack(spacing: 20) {
            Text("Low Level").foregroundStyle(.secondary)
            PulsingRingsView(level: 0.2, phase: phase)
                .frame(width: 80, height: 32)

            Text("Medium Level").foregroundStyle(.secondary)
            PulsingRingsView(level: 0.5, phase: phase)
                .frame(width: 80, height: 32)

            Text("High Level").foregroundStyle(.secondary)
            PulsingRingsView(level: 0.9, phase: phase)
                .frame(width: 80, height: 32)
        }
        .padding()
        .background(Color.black.opacity(0.8))
    }
}
