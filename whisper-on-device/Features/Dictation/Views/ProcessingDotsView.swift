//
//  ProcessingDotsView.swift
//  whisper-on-device
//

import SwiftUI

/// Animated processing indicator with pulsing white dots.
/// Uses external time input with onChange - same pattern as AudioBarsView.
/// Animation flows left-to-right with gentle size variation.
struct ProcessingDotsView: View {
    let time: Double

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var phase: Double = 0

    private let dotCount = 5
    private let dotSize: CGFloat = 8
    private let maxScale: CGFloat = 1.5
    private let dotSpacing: CGFloat = 4

    // Container size to accommodate max scaled dot
    private var containerSize: CGFloat { dotSize * maxScale }

    // Soft off-white color for elegance
    private let dotColor = Color(white: 0.85)

    var body: some View {
        HStack(spacing: dotSpacing) {
            ForEach(0..<dotCount, id: \.self) { index in
                dotView(index: index)
                    .frame(width: containerSize, height: containerSize)
            }
        }
        .drawingGroup()  // GPU-accelerated rendering
        .onChange(of: time) { _, newTime in
            if !reduceMotion {
                phase = newTime * 3.5
            }
        }
    }

    private func dotView(index: Int) -> some View {
        // Left-to-right wave with subtle offset
        let dotPhase = phase - Double(index) * 0.35

        // Primary wave for size
        let wave = (sin(dotPhase) + 1.0) / 2.0

        // Secondary gentle vibration
        let vibration = sin(dotPhase * 1.8) * 0.15

        let scale: CGFloat = reduceMotion ? 1.0 : (0.4 + wave * 0.9 + vibration)
        let opacity = reduceMotion ? 0.9 : (0.6 + wave * 0.4)

        return Circle()
            .fill(dotColor.opacity(opacity))
            .frame(width: dotSize, height: dotSize)
            .scaleEffect(scale)
    }
}

#Preview {
    ZStack {
        Color.black.opacity(0.6)

        TimelineView(.animation) { timeline in
            ProcessingDotsView(time: timeline.date.timeIntervalSinceReferenceDate)
                .frame(height: 32)
                .padding(.horizontal, 20)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 16))
        }
    }
    .frame(width: 200, height: 100)
}
