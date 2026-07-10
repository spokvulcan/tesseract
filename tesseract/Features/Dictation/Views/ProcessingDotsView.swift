//
//  ProcessingDotsView.swift
//  tesseract
//

import SwiftUI

/// Animated processing indicator with pulsing dots.
/// Animation flows left-to-right with gentle size variation. `phase` is derived
/// straight from the driving `time` — mirroring it into `@State` (the old
/// shape) doubled every 60 fps frame into two body evaluations and added a
/// frame of latency for nothing: unlike AudioBarsView, no history is kept.
struct ProcessingDotsView: View {
    let time: Double

    private var phase: Double { time * 3.5 }

    private let dotCount = 5
    private let dotSize: CGFloat = 8
    private let maxScale: CGFloat = 1.5
    private let dotSpacing: CGFloat = 4

    // Container size to accommodate max scaled dot
    private var containerSize: CGFloat { dotSize * maxScale }

    // Follows the effective appearance: the pill tracks the system appearance
    // (DependencyContainer wires no `contentAppearance` override), so a fixed
    // white would vanish on the light-mode glass.
    private let dotColor = Color.primary

    var body: some View {
        HStack(spacing: dotSpacing) {
            ForEach(0..<dotCount, id: \.self) { index in
                dotView(index: index)
                    .frame(width: containerSize, height: containerSize)
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

        let scale: CGFloat = 0.4 + wave * 0.9 + vibration
        let opacity = 0.6 + wave * 0.4

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
