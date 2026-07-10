//
//  ServerChapter.swift
//  tesseract
//
//  Chapter 6 — the reveal: under everything sits an OpenAI-compatible local
//  server whose tiered RAM + SSD radix prefix cache is the hero. Full
//  production value, aimed at the power user (ADR-0021 deliberately kept it
//  late — the foundation lands harder after the experience).
//

import SwiftUI

struct ServerChapter: View {
    @Environment(SettingsManager.self) private var settings
    @State private var copiedEndpoint = false

    private var endpoint: String {
        "http://127.0.0.1:\(HTTPServer.clampedPort(settings.serverPort))/v1"
    }

    var body: some View {
        ChapterScaffold(
            kicker: "Chapter 6 · The Server",
            title: "Everything you just saw is a server",
            subtitle: "An OpenAI-compatible endpoint lives inside Tesseract — point any "
                + "client at it and it answers from this Mac, never the cloud."
        ) {
            VStack(spacing: OnboardingType.rhythm) {
                StagePanel(maxWidth: 560) {
                    VStack(alignment: .leading, spacing: OnboardingType.rhythm) {
                        endpointRow

                        RadixTreeDiagram()
                            .frame(height: 112)

                        HStack(alignment: .firstTextBaseline, spacing: 6) {
                            Text("one shared prefix · RAM + SSD")
                                .font(OnboardingType.body.monospaced())
                                .foregroundStyle(.tint)
                            Spacer(minLength: 0)
                        }

                        Text(
                            "Three requests, one shared prefix. The radix tree keeps the "
                                + "shared run warm in RAM, spills cold runs to SSD, and "
                                + "restores them on demand — every follow-up starts from "
                                + "where you left off."
                        )
                        .font(OnboardingType.body)
                        .lineSpacing(3)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                }

                Text("Works with OpenCode and any OpenAI-compatible client.")
                    .font(OnboardingType.body)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    /// Status dot, selectable endpoint, one-tap copy — the power-user handshake.
    private var endpointRow: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(settings.isServerEnabled ? .green : .secondary)
                .frame(width: 7, height: 7)
            Text(endpoint)
                .font(OnboardingType.body.monospaced())
                .textSelection(.enabled)
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(endpoint, forType: .string)
                copiedEndpoint = true
                Task {
                    try? await Task.sleep(for: .seconds(1.6))
                    copiedEndpoint = false
                }
            } label: {
                Image(systemName: copiedEndpoint ? "checkmark" : "doc.on.doc")
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(
                        copiedEndpoint ? AnyShapeStyle(.green) : AnyShapeStyle(.secondary)
                    )
                    .contentTransition(.symbolEffect(.replace))
            }
            .buttonStyle(.plain)
            .help("Copy endpoint")
            .accessibilityLabel(copiedEndpoint ? "Copied" : "Copy endpoint")
            Spacer(minLength: 0)
            Text(
                settings.isServerEnabled
                    ? "Serving" : "Enable in Settings → Server"
            )
            .font(OnboardingType.body)
            .foregroundStyle(.secondary)
        }
    }
}

/// The radix idea drawn, not explained: one trunk of cached token cells
/// branches into three request tails. A pulse rides the trunk and takes a
/// different tail each cycle — three requests, one shared prefix. Static
/// (trunk lit, tails quiet) under Reduce Motion.
private struct RadixTreeDiagram: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private struct Tail {
        let cells: Int
        let lane: CGFloat  // -1 above trunk, 0 level, +1 below
        let label: String
    }

    private let trunkCells = 7
    private let tails: [Tail] = [
        Tail(cells: 3, lane: -1, label: "req 1"),
        Tail(cells: 5, lane: 0, label: "req 2"),
        Tail(cells: 2, lane: 1, label: "req 3"),
    ]

    /// Seconds per request ride (trunk → one tail).
    private let cycle: Double = 2.6

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30, paused: reduceMotion)) { timeline in
            Canvas { context, size in
                let time = timeline.date.timeIntervalSinceReferenceDate
                let phase = (time / cycle).truncatingRemainder(dividingBy: Double(tails.count))
                let activeTail = reduceMotion ? -1 : Int(phase)
                // Pulse position along the ride: trunk occupies 0..0.6, tail 0.6..1.
                let rideU = reduceMotion ? 0.0 : phase.truncatingRemainder(dividingBy: 1)

                let gap: CGFloat = 4
                let connectorWidth: CGFloat = 26
                let labelWidth: CGFloat = 34
                let maxTailCells = tails.map(\.cells).max() ?? 0
                let cellWidth =
                    (size.width - connectorWidth - labelWidth
                        - gap * CGFloat(trunkCells + maxTailCells))
                    / CGFloat(trunkCells + maxTailCells)
                let cellHeight: CGFloat = 10
                let laneGap: CGFloat = 32
                let centerY = size.height / 2

                // Trunk: accent cells, brightened where the pulse passes.
                for cell in 0..<trunkCells {
                    let x = CGFloat(cell) * (cellWidth + gap)
                    let cellU = (Double(cell) + 0.5) / Double(trunkCells) * 0.6
                    let pulse = reduceMotion ? 0.35 : max(0, 1 - abs(rideU - cellU) * 5)
                    let rect = CGRect(
                        x: x, y: centerY - cellHeight / 2, width: cellWidth, height: cellHeight)
                    context.fill(
                        Path(roundedRect: rect, cornerRadius: 3),
                        with: .color(Color.accentColor.opacity(0.4 + 0.5 * pulse)))
                }

                let trunkEnd = CGFloat(trunkCells) * (cellWidth + gap) - gap
                let tailStart = trunkEnd + connectorWidth

                for (index, tail) in tails.enumerated() {
                    let tailY = centerY + tail.lane * laneGap
                    let isActive = index == activeTail

                    // Connector: a quiet curve from trunk end to the tail's lane;
                    // the active one brightens while the pulse crosses it.
                    var connector = Path()
                    connector.move(to: CGPoint(x: trunkEnd, y: centerY))
                    connector.addCurve(
                        to: CGPoint(x: tailStart, y: tailY),
                        control1: CGPoint(x: trunkEnd + connectorWidth * 0.7, y: centerY),
                        control2: CGPoint(x: trunkEnd + connectorWidth * 0.3, y: tailY))
                    let crossing = isActive ? max(0, 1 - abs(rideU - 0.62) * 8) : 0
                    context.stroke(
                        connector,
                        with: .color(Color.accentColor.opacity(0.18 + 0.55 * crossing)),
                        lineWidth: 1.5)

                    // Tail cells: quiet by default; the active tail lights as the
                    // pulse arrives.
                    for cell in 0..<tail.cells {
                        let x = tailStart + CGFloat(cell) * (cellWidth + gap)
                        let cellU = 0.64 + (Double(cell) + 0.5) / Double(tail.cells) * 0.36
                        let pulse = isActive ? max(0, 1 - abs(rideU - cellU) * 5) : 0
                        let rect = CGRect(
                            x: x, y: tailY - cellHeight / 2, width: cellWidth, height: cellHeight)
                        context.fill(
                            Path(roundedRect: rect, cornerRadius: 3),
                            with: .color(
                                pulse > 0.01
                                    ? Color.accentColor.opacity(0.15 + 0.6 * pulse)
                                    : Color.secondary.opacity(0.18)))
                    }

                    // Tail label, seated after its row — structure, not decoration.
                    let labelX = tailStart + CGFloat(tail.cells) * (cellWidth + gap) + 4
                    context.draw(
                        Text(tail.label)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.tertiary),
                        at: CGPoint(x: labelX, y: tailY),
                        anchor: .leading)
                }
            }
        }
        .accessibilityLabel(
            "Diagram: one cached prefix trunk branching into three request tails — "
                + "each request reuses the shared prefix")
    }
}
