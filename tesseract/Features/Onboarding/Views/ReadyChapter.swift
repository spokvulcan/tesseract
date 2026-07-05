//
//  ReadyChapter.swift
//  tesseract
//
//  Chapter 6 — the honest finish. All-green earns the tour's single
//  celebratory beat; anything still in flight is shown as live progress, never
//  a fake success. The hotkey reminder lives here.
//

import SwiftUI

struct ReadyChapter: View {
    let controller: OnboardingTourController

    @EnvironmentObject private var permissionsManager: PermissionsManager
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        ChapterScaffold(
            kicker: "Chapter 6 · Ready",
            title: controller.isSetupComplete
                ? "Your intelligence is home"
                : "Almost there",
            subtitle: controller.isSetupComplete
                ? "Everything is on this Mac and ready. It stays yours."
                : "You can start now — the rest keeps arriving in the background."
        ) {
            VStack(spacing: 14) {
                StagePanel(maxWidth: 480) {
                    VStack(spacing: 10) {
                        modelRow(
                            label: "Speech to text",
                            id: controller.speechToTextModelID)
                        modelRow(label: "Voice", id: controller.voiceModelID)
                        modelRow(
                            label: "Agent",
                            id: controller.chosenAgentModelID)

                        Divider()

                        permissionRow(
                            label: "Microphone",
                            state: permissionsManager.microphonePermission)
                        permissionRow(
                            label: "Accessibility",
                            state: permissionsManager.accessibilityPermission)
                    }
                }

                HStack(spacing: 6) {
                    Image(systemName: "keyboard")
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                    Text("Hold ")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        + Text(settings.hotkey.displayString)
                        .font(.system(size: 12, weight: .semibold))
                        + Text(" in any app to dictate.")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .overlay {
            if controller.isSetupComplete {
                CelebrationBurst()
                    .allowsHitTesting(false)
            }
        }
    }

    private func modelRow(label: String, id: String) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .frame(width: 110, alignment: .leading)
            Text(ModelDefinition.withID(id)?.displayName ?? id)
                .font(.system(size: 12, weight: .medium))
            Spacer(minLength: 0)
            modelStatusChip(for: id)
        }
    }

    @ViewBuilder
    private func modelStatusChip(for id: String) -> some View {
        switch controller.status(for: id) {
        case .downloaded:
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 12))
                .foregroundStyle(.green)
        case .downloading(let progress):
            HStack(spacing: 5) {
                ProgressView(value: progress)
                    .controlSize(.small)
                    .frame(width: 56)
                Text(progress.formatted(.wholePercent))
                    .font(.system(size: 10.5).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        case .verifying:
            HStack(spacing: 5) {
                ProgressView().controlSize(.mini)
                Text("Verifying")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.secondary)
            }
        case .notDownloaded:
            Text("Queued")
                .font(.system(size: 10.5))
                .foregroundStyle(.tertiary)
        case .error:
            Label("Retry from Models", systemImage: "exclamationmark.circle")
                .font(.system(size: 10.5))
                .foregroundStyle(.orange)
        }
    }

    private func permissionRow(label: String, state: PermissionState) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .frame(width: 110, alignment: .leading)
            Spacer(minLength: 0)
            if state == .granted {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(.green)
            } else {
                Text("Optional — grant any time in Settings")
                    .font(.system(size: 10.5))
                    .foregroundStyle(.tertiary)
            }
        }
    }
}

// MARK: - Celebration

/// The tour's one particle moment: a single burst, fired once per appearance
/// of the all-green Ready chapter, then over — the timeline is torn down when
/// the burst ends, so no frames tick while the user sits on the final screen.
/// Skipped under Reduce Motion.
private struct CelebrationBurst: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var startDate: Date?
    @State private var isOver = false

    private static let duration = 1.6

    private struct Particle {
        let angle: Double
        let speed: Double
        let size: Double
        let hue: Double
        let spin: Double
    }

    private static let particles: [Particle] = (0..<46).map { index in
        // Deterministic pseudo-randomness: golden-angle spread, varied radii.
        let golden = Double(index) * 2.39996
        return Particle(
            angle: golden,
            speed: 130 + (Double((index * 37) % 100) / 100) * 170,
            size: 3 + Double((index * 53) % 100) / 100 * 4,
            hue: Double((index * 71) % 100) / 100,
            spin: Double((index * 29) % 100) / 100 * 6
        )
    }

    var body: some View {
        if reduceMotion || isOver {
            EmptyView()
        } else {
            TimelineView(.animation(minimumInterval: 1 / 60)) { timeline in
                Canvas { context, size in
                    guard let start = startDate else { return }
                    let elapsed = timeline.date.timeIntervalSince(start)
                    guard elapsed >= 0, elapsed < Self.duration else { return }
                    let t = elapsed / Self.duration

                    let origin = CGPoint(x: size.width / 2, y: size.height * 0.34)
                    for particle in Self.particles {
                        let distance = particle.speed * t * (2 - t) / 2 * 2
                        let x = origin.x + cos(particle.angle) * distance
                        let y =
                            origin.y + sin(particle.angle) * distance * 0.85
                            + 120 * t * t
                        let fade = 1 - t
                        let rect = CGRect(
                            x: x, y: y,
                            width: particle.size, height: particle.size * 1.8)
                        var cell = context
                        cell.translateBy(x: rect.midX, y: rect.midY)
                        cell.rotate(by: .radians(particle.spin * t * 2))
                        cell.opacity = fade
                        cell.fill(
                            Path(
                                roundedRect: CGRect(
                                    x: -rect.width / 2, y: -rect.height / 2,
                                    width: rect.width, height: rect.height),
                                cornerRadius: 1),
                            with: .color(
                                Color(
                                    hue: 0.55 + particle.hue * 0.25,
                                    saturation: 0.65, brightness: 0.95)))
                    }
                }
            }
            .onAppear { startDate = Date() }
            .task {
                try? await Task.sleep(for: .seconds(Self.duration + 0.1))
                isOver = true
            }
        }
    }
}
