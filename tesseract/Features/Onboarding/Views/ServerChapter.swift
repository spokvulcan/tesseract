//
//  ServerChapter.swift
//  tesseract
//
//  Chapter 5 — the reveal: under everything sits an OpenAI-compatible local
//  server whose tiered RAM + SSD radix prefix cache is the hero. Full
//  production value, aimed at the power user (ADR-0021 kept it at position
//  five deliberately — the foundation lands harder after the experience).
//

import SwiftUI

struct ServerChapter: View {
    @Environment(SettingsManager.self) private var settings

    private var endpoint: String {
        "http://127.0.0.1:\(HTTPServer.clampedPort(settings.serverPort))/v1"
    }

    var body: some View {
        ChapterScaffold(
            kicker: "Chapter 5 · The Server",
            title: "Everything you just saw is a server",
            subtitle: "An OpenAI-compatible endpoint lives inside Tesseract — point any "
                + "client at it and it answers from this Mac, never the cloud."
        ) {
            VStack(spacing: 12) {
                StagePanel(maxWidth: 560) {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack(spacing: 8) {
                            Circle()
                                .fill(settings.isServerEnabled ? .green : .secondary)
                                .frame(width: 7, height: 7)
                            Text(endpoint)
                                .font(.system(size: 13, design: .monospaced))
                                .textSelection(.enabled)
                            Spacer(minLength: 0)
                            Text(
                                settings.isServerEnabled
                                    ? "Serving" : "Enable in Settings → Server"
                            )
                            .font(.system(size: 10.5))
                            .foregroundStyle(.secondary)
                        }

                        PrefixCacheDiagram()
                            .frame(height: 96)

                        Text(
                            "The tiered RAM + SSD prefix cache remembers your context "
                                + "between requests, so every follow-up starts warm — "
                                + "hit rates no other on-device stack matches."
                        )
                        .font(.system(size: 11))
                        .lineSpacing(2.5)
                        .foregroundStyle(.secondary)
                    }
                }

                Text("Works with OpenCode and any OpenAI-compatible client.")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }
        }
    }
}

/// Three requests sharing a warm prefix: the shared cells glow accent, the
/// per-request tails stay quiet, and a slow sweep re-lights the shared run —
/// the radix idea drawn, not explained. Static under Reduce Motion.
private struct PrefixCacheDiagram: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let rows = 3
    private let sharedCells = 7
    private let tailCells = [3, 5, 2]

    private static let bracketLabel = Text("cached prefix — served from RAM or SSD")
        .font(.system(size: 9, design: .monospaced))
        .foregroundStyle(.secondary)

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30, paused: reduceMotion)) { timeline in
            Canvas { context, size in
                let t = reduceMotion ? 0.5 : timeline.date.timeIntervalSinceReferenceDate
                let totalCells = sharedCells + (tailCells.max() ?? 0)
                let cellWidth = size.width / CGFloat(totalCells) - 4
                let rowHeight = size.height / CGFloat(rows)
                let sweep = (t * 0.35).truncatingRemainder(dividingBy: 1.4)

                for row in 0..<rows {
                    let y = CGFloat(row) * rowHeight + rowHeight / 2
                    let cells = sharedCells + tailCells[row]
                    for cell in 0..<cells {
                        let x = CGFloat(cell) * (cellWidth + 4)
                        let rect = CGRect(
                            x: x, y: y - 5, width: cellWidth, height: 10)
                        let path = Path(roundedRect: rect, cornerRadius: 3)

                        if cell < sharedCells {
                            let phase = Double(cell) / Double(sharedCells)
                            let pulse = max(0, 1 - abs(sweep - phase * 0.9) * 3.2)
                            let base = 0.32 + 0.55 * pulse
                            context.fill(
                                path,
                                with: .linearGradient(
                                    Gradient(colors: [
                                        OnboardingPalette.accentCyan.opacity(base),
                                        OnboardingPalette.accentViolet.opacity(base),
                                    ]),
                                    startPoint: CGPoint(x: rect.minX, y: rect.midY),
                                    endPoint: CGPoint(x: rect.maxX, y: rect.midY)))
                        } else {
                            context.fill(
                                path, with: .color(.secondary.opacity(0.18)))
                        }
                    }
                }

                // The bracket under the shared run, labelled once.
                let sharedWidth = CGFloat(sharedCells) * (cellWidth + 4) - 4
                context.draw(
                    Self.bracketLabel,
                    at: CGPoint(x: sharedWidth / 2, y: size.height - 2),
                    anchor: .bottom)
            }
        }
        .accessibilityLabel(
            "Diagram: three requests sharing one cached prefix served from RAM or SSD")
    }
}
