//
//  TesseractMarkView.swift
//  tesseract
//
//  The tour's signature: a live 4D hypercube (tesseract) wireframe, double-
//  rotating in the XW and YZ planes and projected 4D→3D→2D. Its edges fill
//  with the accent gradient as setup bytes land — the mark *is* the download
//  indicator (ADR-0021). Under Reduce Motion the rotation freezes at a
//  composed pose; the progress fill still updates.
//

import SwiftUI

struct TesseractMarkView: View {
    /// Setup completion in [0, 1] — filled edges out of 32.
    var progress: Double

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 60, paused: reduceMotion)) { timeline in
            Canvas { context, size in
                let angle =
                    reduceMotion
                    ? 0.9 : timeline.date.timeIntervalSinceReferenceDate * 0.35
                draw(in: &context, size: size, angle: angle)
            }
        }
        .accessibilityLabel(accessibilityDescription)
    }

    private var accessibilityDescription: String {
        let percent = Int((progress.clamped01) * 100)
        return percent >= 100
            ? "Tesseract mark, setup complete"
            : "Tesseract mark, setup \(percent) percent complete"
    }

    // MARK: - Geometry

    /// The 16 vertices of the unit tesseract: every corner of (±1)⁴.
    private static let vertices: [SIMD4<Double>] = (0..<16).map { index in
        SIMD4(
            index & 1 == 0 ? -1 : 1,
            index & 2 == 0 ? -1 : 1,
            index & 4 == 0 ? -1 : 1,
            index & 8 == 0 ? -1 : 1
        )
    }

    /// The 32 edges: vertex pairs at Hamming distance 1, in a stable order so
    /// the progress fill grows deterministically.
    private static let edges: [(Int, Int)] = {
        var result: [(Int, Int)] = []
        for a in 0..<16 {
            for bit in [1, 2, 4, 8] {
                let b = a | bit
                if b != a && a & bit == 0 {
                    result.append((a, b))
                }
            }
        }
        return result
    }()

    private func draw(in context: inout GraphicsContext, size: CGSize, angle: Double) {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let scale = min(size.width, size.height) * 0.34

        let projected = Self.vertices.map { project($0, angle: angle, scale: scale) }
        let filledCount = Int((progress.clamped01 * Double(Self.edges.count)).rounded())

        let hairline = GraphicsContext.Shading.color(
            colorScheme == .dark
                ? Color.white.opacity(0.22) : Color.black.opacity(0.25))

        for (index, edge) in Self.edges.enumerated() {
            let from = offsetPoint(projected[edge.0], by: center)
            let to = offsetPoint(projected[edge.1], by: center)
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)

            if index < filledCount {
                let gradient = OnboardingPalette.accentGradient
                var glow = context
                glow.addFilter(.blur(radius: 2.4))
                glow.stroke(
                    path,
                    with: .linearGradient(gradient, startPoint: from, endPoint: to),
                    style: StrokeStyle(lineWidth: 2.4, lineCap: .round))
                context.stroke(
                    path,
                    with: .linearGradient(gradient, startPoint: from, endPoint: to),
                    style: StrokeStyle(lineWidth: 1.3, lineCap: .round))
            } else {
                context.stroke(
                    path, with: hairline,
                    style: StrokeStyle(lineWidth: 0.7, lineCap: .round))
            }
        }

        // Vertices as faint points, so the form reads even fully unfilled.
        for point in projected {
            let dot = CGRect(
                x: center.x + point.x - 1.2, y: center.y + point.y - 1.2,
                width: 2.4, height: 2.4)
            context.fill(Path(ellipseIn: dot), with: hairline)
        }
    }

    /// Double rotation (XW and YZ planes), then perspective 4D→3D→2D.
    private func project(_ vertex: SIMD4<Double>, angle: Double, scale: Double) -> CGPoint {
        var v = vertex

        let cosA = cos(angle)
        let sinA = sin(angle)
        // XW plane
        let x = v.x * cosA - v.w * sinA
        let w = v.x * sinA + v.w * cosA
        v.x = x
        v.w = w
        // YZ plane (slightly detuned so the motion never loops visibly)
        let cosB = cos(angle * 0.62)
        let sinB = sin(angle * 0.62)
        let y = v.y * cosB - v.z * sinB
        let z = v.y * sinB + v.z * cosB
        v.y = y
        v.z = z

        let wDistance = 3.2
        let wFactor = wDistance / (wDistance - v.w)
        let x3 = v.x * wFactor
        let y3 = v.y * wFactor
        let z3 = v.z * wFactor

        let zDistance = 4.4
        let zFactor = zDistance / (zDistance - z3)
        return CGPoint(x: x3 * zFactor * scale, y: y3 * zFactor * scale)
    }

    private func offsetPoint(_ point: CGPoint, by center: CGPoint) -> CGPoint {
        CGPoint(x: center.x + point.x, y: center.y + point.y)
    }
}

extension Double {
    fileprivate var clamped01: Double { Swift.min(1, Swift.max(0, self)) }
}
