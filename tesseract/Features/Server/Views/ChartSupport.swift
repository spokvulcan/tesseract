//
//  ChartSupport.swift
//  tesseract
//

import Charts
import SwiftUI
import Textual

// MARK: - Chart palette

/// The app's categorical chart palette (design language §5): four fixed
/// slots, assigned in fixed order, never cycled. Validated with the dataviz
/// palette validator 2026-07-10, light on `#fcfcfb` and dark on `#1a1a19`:
/// all checks pass (worst adjacent CVD ΔE ≈ 38–40). Light-mode aqua/orange
/// sit below 3:1 surface contrast — the relief rule is satisfied by legends,
/// axis labels, and tooltips (never color-alone identity).
enum ChartPalette {
    /// Slot 1 — blue. TTFT lookup · hit-rate line · SSD deleted.
    static let slot1 = DynamicColor(light: rgb(0x2A78D6), dark: rgb(0x3987E5))
    /// Slot 2 — aqua. TTFT restore · token-reuse line.
    static let slot2 = DynamicColor(light: rgb(0x1BAF7A), dark: rgb(0x199E70))
    /// Slot 3 — the brand warm orange, dark step lowered into the dark
    /// lightness band. TTFT prefill (the cost the cache exists to avoid) ·
    /// SSD written. Prefill wears this slot on every chart, app-wide.
    static let slot3 = DynamicColor(light: rgb(0xD68C27), dark: rgb(0xC67F16))
    /// Slot 4 — violet. TTFT residual.
    static let slot4 = DynamicColor(light: rgb(0x4A3AA7), dark: rgb(0x9085E9))

    private static func rgb(_ hex: UInt32) -> Color {
        Color(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}

// MARK: - Chart hover (cursor + tooltip)

/// Full-plot hover catcher for a `chartOverlay`: reports the pointer's
/// position translated into plot-area coordinates, and exits when the
/// pointer leaves the plot. Each chart resolves the position to its own
/// nearest data point via `ChartProxy.value(atX:)` and draws the hairline
/// `RuleMark` cursor + `ChartTooltipChrome` annotation itself.
struct ChartHoverOverlay: View {
    let proxy: ChartProxy
    let onMove: (CGPoint) -> Void
    let onExit: () -> Void

    var body: some View {
        GeometryReader { geo in
            Rectangle()
                .fill(Color.clear)
                .contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let location):
                        guard let anchor = proxy.plotFrame else { return }
                        let frame = geo[anchor]
                        guard frame.contains(location) else {
                            onExit()
                            return
                        }
                        onMove(
                            CGPoint(
                                x: location.x - frame.minX,
                                y: location.y - frame.minY
                            ))
                    case .ended:
                        onExit()
                    }
                }
        }
    }
}

/// The one tooltip surface: a quiet thin-material chip with a hairline
/// ring — standard materials only (content layer), values in text tokens,
/// identity carried by the color dot beside the text, never by coloring
/// the text itself.
struct ChartTooltipChrome<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            content
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 6))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(.quaternary, lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.10), radius: 5, y: 2)
    }
}

/// One tooltip line: optional series dot · label · value.
struct ChartTooltipRow: View {
    var dot: DynamicColor?
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 5) {
            if let dot {
                Circle()
                    .fill(dot)
                    .frame(width: 6, height: 6)
            }
            Text(label)
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value)
                .fontWeight(.semibold)
                .monospacedDigit()
                .foregroundStyle(.primary)
        }
        .font(.caption2)
    }
}
