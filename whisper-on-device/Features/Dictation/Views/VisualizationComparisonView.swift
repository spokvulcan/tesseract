//
//  VisualizationComparisonView.swift
//  whisper-on-device
//
//  Side-by-side comparison of all visualization options.
//  Use this view in Xcode Previews to evaluate each style.

import SwiftUI

/// Visualization style options for the recording overlay
enum VisualizationType: String, CaseIterable, Identifiable {
    case liquidWave
    case breathingRectangle
    case pulsingRings
    case organicBlob

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .liquidWave:
            "Liquid Wave"
        case .breathingRectangle:
            "Breathing Rectangle"
        case .pulsingRings:
            "Pulsing Rings"
        case .organicBlob:
            "Organic Blob"
        }
    }

    var description: String {
        switch self {
        case .liquidWave:
            "Classic dual sine waves"
        case .breathingRectangle:
            "iOS 18 Siri-style morphing rectangle"
        case .pulsingRings:
            "Minimal concentric ripples"
        case .organicBlob:
            "Modern liquid droplet orb"
        }
    }
}

/// Comparison view showing all visualization options side-by-side
struct VisualizationComparisonView: View {
    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let phase = CGFloat(time * 2.0)

            // Simulated audio level that cycles up and down
            let simulatedLevel = CGFloat((sin(time * 1.5) + 1) / 2 * 0.7 + 0.15)

            ScrollView {
                VStack(spacing: 32) {
                    headerSection(level: simulatedLevel)

                    comparisonGrid(level: simulatedLevel, phase: phase)

                    containerComparisonSection(level: simulatedLevel, phase: phase)
                }
                .padding(24)
            }
            .background(Color(white: 0.1))
        }
    }

    private func headerSection(level: CGFloat) -> some View {
        VStack(spacing: 8) {
            Text("Visualization Comparison")
                .font(.title2.bold())
                .foregroundStyle(.white)

            Text("Simulated audio level: \(Int(level * 100))%")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func comparisonGrid(level: CGFloat, phase: CGFloat) -> some View {
        LazyVGrid(columns: [
            GridItem(.flexible()),
            GridItem(.flexible())
        ], spacing: 24) {
            ForEach(VisualizationType.allCases) { type in
                visualizationCard(type: type, level: level, phase: phase)
            }
        }
    }

    private func visualizationCard(
        type: VisualizationType,
        level: CGFloat,
        phase: CGFloat
    ) -> some View {
        VStack(spacing: 12) {
            Text(type.displayName)
                .font(.headline)
                .foregroundStyle(.white)

            // Visualization in a mock container
            ZStack {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.ultraThinMaterial)

                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.15), lineWidth: 0.5)

                visualizationView(for: type, level: level, phase: phase)
                    .padding(8)
            }
            .frame(width: 120, height: 32)
            .shadow(color: .black.opacity(0.2), radius: 6, y: 2)

            Text(type.description)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(height: 32)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.white.opacity(0.05))
        )
    }

    @ViewBuilder
    private func visualizationView(
        for type: VisualizationType,
        level: CGFloat,
        phase: CGFloat
    ) -> some View {
        switch type {
        case .liquidWave:
            LiquidWaveView(level: level, phase: phase)

        case .breathingRectangle:
            BreathingRectangleView(level: level, phase: phase)

        case .pulsingRings:
            PulsingRingsView(level: level, phase: phase)

        case .organicBlob:
            OrganicBlobView(level: level, phase: phase)
        }
    }

    private func containerComparisonSection(level: CGFloat, phase: CGFloat) -> some View {
        VStack(spacing: 16) {
            Text("Container Size Comparison")
                .font(.headline)
                .foregroundStyle(.white)

            HStack(spacing: 32) {
                VStack(spacing: 8) {
                    Text("Current (180×44)")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    mockContainer(width: 180, height: 44, level: level, phase: phase)
                }

                VStack(spacing: 8) {
                    Text("New (120×32)")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    mockContainer(width: 120, height: 32, level: level, phase: phase)
                }
            }
        }
        .padding(.top, 16)
    }

    private func mockContainer(
        width: CGFloat,
        height: CGFloat,
        level: CGFloat,
        phase: CGFloat
    ) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: height / 2, style: .continuous)
                .fill(.ultraThinMaterial)

            RoundedRectangle(cornerRadius: height / 2, style: .continuous)
                .strokeBorder(Color.white.opacity(0.15), lineWidth: 0.5)

            OrganicBlobView(level: level, phase: phase)
                .padding(.horizontal, width == 180 ? 20 : 10)
                .padding(.vertical, 6)
        }
        .frame(width: width, height: height)
        .shadow(color: .black.opacity(0.15), radius: width == 180 ? 12 : 6, y: 4)
    }
}

#Preview {
    VisualizationComparisonView()
        .frame(width: 500, height: 700)
}

#Preview("Individual Visualizations") {
    TimelineView(.animation) { timeline in
        let phase = CGFloat(timeline.date.timeIntervalSinceReferenceDate * 2.0)

        HStack(spacing: 20) {
            ForEach(VisualizationType.allCases) { type in
                VStack {
                    Text(type.displayName)
                        .font(.caption)

                    ZStack {
                        RoundedRectangle(cornerRadius: 16)
                            .fill(.ultraThinMaterial)

                        switch type {
                        case .liquidWave:
                            LiquidWaveView(level: 0.5, phase: phase)
                        case .breathingRectangle:
                            BreathingRectangleView(level: 0.5, phase: phase)
                        case .pulsingRings:
                            PulsingRingsView(level: 0.5, phase: phase)
                        case .organicBlob:
                            OrganicBlobView(level: 0.5, phase: phase)
                        }
                    }
                    .frame(width: 100, height: 28)
                }
            }
        }
        .padding()
        .background(Color.black.opacity(0.8))
    }
}
