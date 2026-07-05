//
//  TesseractMarkView.swift
//  tesseract
//
//  The tour's signature: a 4D hypercube (tesseract) wireframe turning slowly
//  in the XW plane — the classic inside-out tesseract motion — under a fixed
//  studio tilt, drawn as monochrome ink with depth cues: near edges bright and
//  weighty, far edges faint. Edges brighten as setup bytes land — the mark
//  *is* the download indicator (ADR-0021). The projection is re-fit to the
//  canvas every frame, so it can never clip. Under Reduce Motion the rotation
//  freezes at a composed pose; the progress fill still updates.
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
                    ? 0.55 : timeline.date.timeIntervalSinceReferenceDate * 0.22
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

    private struct ProjectedVertex {
        let point: CGPoint  // unscaled projection-space coordinates
        let depth: Double  // combined perspective factor; larger = nearer
    }

    private func draw(in context: inout GraphicsContext, size: CGSize, angle: Double) {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let projected = Self.vertices.map { project($0, angle: angle) }

        // Fit the whole figure inside the canvas with a fixed margin, every
        // frame — the stacked perspective divisions make any analytic bound
        // loose enough to clip.
        let extent = projected.reduce(0.001) {
            max($0, max(abs($1.point.x), abs($1.point.y)))
        }
        let fit = (min(size.width, size.height) / 2 - 4) / extent

        let depths = projected.map(\.depth)
        let farthest = depths.min() ?? 0
        let depthSpan = max((depths.max() ?? 1) - farthest, 0.001)

        let filledCount = Int((progress.clamped01 * Double(Self.edges.count)).rounded())
        let ink = colorScheme == .dark ? Color.white : Color.black

        // Far edges draw first so near ones read on top.
        let ordered = Self.edges.enumerated().sorted { lhs, rhs in
            edgeDepth(lhs.element, in: projected) < edgeDepth(rhs.element, in: projected)
        }

        for (index, edge) in ordered {
            let a = projected[edge.0]
            let b = projected[edge.1]
            var path = Path()
            path.move(
                to: CGPoint(x: center.x + a.point.x * fit, y: center.y + a.point.y * fit))
            path.addLine(
                to: CGPoint(x: center.x + b.point.x * fit, y: center.y + b.point.y * fit))

            let nearness = (edgeDepth(edge, in: projected) - farthest) / depthSpan

            if index < filledCount {
                context.stroke(
                    path,
                    with: .color(ink.opacity(0.45 + 0.5 * nearness)),
                    style: StrokeStyle(lineWidth: 1.0 + 0.6 * nearness, lineCap: .round))
            } else {
                context.stroke(
                    path,
                    with: .color(ink.opacity(0.08 + 0.14 * nearness)),
                    style: StrokeStyle(lineWidth: 0.7, lineCap: .round))
            }
        }
    }

    private func edgeDepth(_ edge: (Int, Int), in projected: [ProjectedVertex]) -> Double {
        (projected[edge.0].depth + projected[edge.1].depth) / 2
    }

    // Fixed studio tilt, so the nested-cube form reads as an object.
    private static let cosTiltY = cos(0.55), sinTiltY = sin(0.55)
    private static let cosTiltX = cos(-0.32), sinTiltX = sin(-0.32)

    /// One slow XW-plane rotation, perspective 4D→3D, the fixed tilt, then
    /// perspective 3D→2D.
    private func project(_ vertex: SIMD4<Double>, angle: Double) -> ProjectedVertex {
        let cosA = cos(angle)
        let sinA = sin(angle)
        var x = vertex.x * cosA - vertex.w * sinA
        let w = vertex.x * sinA + vertex.w * cosA
        var y = vertex.y
        var z = vertex.z

        // 4D → 3D: the inner/outer cube separation. |w| ≤ √2, so the
        // denominator stays comfortably positive.
        let wFactor = 3.0 / (3.0 - w)
        x *= wFactor
        y *= wFactor
        z *= wFactor

        (x, z) = (x * Self.cosTiltY + z * Self.sinTiltY, -x * Self.sinTiltY + z * Self.cosTiltY)
        (y, z) = (y * Self.cosTiltX - z * Self.sinTiltX, y * Self.sinTiltX + z * Self.cosTiltX)

        // 3D → 2D: gentle depth. |z| ≤ wFactor·√3 < 3.3, again safely bounded.
        let zFactor = 7.0 / (7.0 - z)
        return ProjectedVertex(
            point: CGPoint(x: x * zFactor, y: y * zFactor),
            depth: wFactor * zFactor)
    }
}

extension Double {
    fileprivate var clamped01: Double { Swift.min(1, Swift.max(0, self)) }
}
