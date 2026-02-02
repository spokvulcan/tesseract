//
//  AudioBarsView.swift
//  whisper-on-device
//

import SwiftUI

/// Timeline-style audio visualization with bars scrolling from right to left.
/// Each bar represents a captured audio level sample, creating a waveform history.
struct AudioBarsView: View {
    let level: CGFloat
    let phase: CGFloat  // unused, kept for API compatibility

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var samples: [CGFloat] = Array(repeating: 0.1, count: 20)

    private let barCount = 20
    private let barWidth: CGFloat = 3
    private let barSpacing: CGFloat = 2
    private let cornerRadius: CGFloat = 1.5
    private let minHeight: CGFloat = 0.15

    var body: some View {
        GeometryReader { geometry in
            HStack(alignment: .center, spacing: barSpacing) {
                ForEach(0..<samples.count, id: \.self) { index in
                    barView(for: samples[index], containerHeight: geometry.size.height)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .onChange(of: level) { _, newLevel in
            pushSample(newLevel)
        }
        .onAppear {
            samples = Array(repeating: 0.1, count: barCount)
        }
    }

    private func barView(for sampleLevel: CGFloat, containerHeight: CGFloat) -> some View {
        let heightRatio = minHeight + (1 - minHeight) * sampleLevel
        let barHeight = max(containerHeight * heightRatio, 2)

        return RoundedRectangle(cornerRadius: cornerRadius)
            .fill(barColor(for: sampleLevel))
            .frame(width: barWidth, height: barHeight)
    }

    private func barColor(for sampleLevel: CGFloat) -> Color {
        // Red-orange gradient based on intensity
        let intensity = min(sampleLevel * 1.2, 1.0)
        return Color(
            red: 0.9 + intensity * 0.1,
            green: 0.25 + (1 - intensity) * 0.15,
            blue: 0.2
        )
    }

    private func pushSample(_ newLevel: CGFloat) {
        if reduceMotion {
            samples.removeFirst()
            samples.append(newLevel)
        } else {
            withAnimation(.linear(duration: 0.05)) {
                samples.removeFirst()
                samples.append(newLevel)
            }
        }
    }
}

#Preview("Audio Bars") {
    ZStack {
        Color.black.opacity(0.8)

        AudioBarsView(level: 0.5, phase: 0)
            .frame(width: 120, height: 24)
            .padding()
    }
}
