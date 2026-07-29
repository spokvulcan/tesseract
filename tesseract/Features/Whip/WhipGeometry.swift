//
//  WhipGeometry.swift
//  tesseract
//
//  Turning the chain into something drawable. Pure and `nonisolated`, so the
//  outline can be checked in tests without a window.
//
//  A tapered whip cannot be a stroked path: strokes have one width for their
//  whole length. So the body is built as a *filled polygon* — walk the chain
//  offsetting by the local half-width on one side, walk back offsetting on the
//  other, close. The per-point radii the physics already keeps for mass and
//  stiffness are exactly the offsets needed, so the taper costs nothing extra.
//

import CoreGraphics
import Foundation

nonisolated enum WhipGeometry {

    /// Builds the closed outline of the thong, from the handle's far end to the
    /// tip and back. Returns an empty path for degenerate input rather than
    /// trapping — a whip with one point is not worth a crash.
    static func outline(points: [CGPoint], radii: [CGFloat]) -> CGPath {
        let path = CGMutablePath()
        guard points.count >= 2, radii.count == points.count else { return path }

        // Smooth before offsetting. Bending stiffness goes to nearly zero at the
        // tip — correctly, that is what makes the crack work — so a thong lying
        // coiled genuinely kinks at sharp angles, and offsetting a sharp kink
        // makes the two sides of the outline cross and read as a spike. Cutting
        // the corners first costs nothing and is honest: it smooths how the whip
        // is *drawn*, never where the simulation thinks it is.
        let (points, radii) = smoothed(points: points, radii: radii, passes: 2)
        let normals = self.normals(for: points)

        // Down one side…
        for index in points.indices {
            let offset = CGPoint(
                x: points[index].x + normals[index].dx * radii[index],
                y: points[index].y + normals[index].dy * radii[index])
            if index == 0 {
                path.move(to: offset)
            } else {
                path.addLine(to: offset)
            }
        }

        // …round the tip, so the whip ends in a point rather than a chopped-off
        // rectangle. The last radius is sub-pixel, so a short arc reads as a
        // taper to nothing.
        if let last = points.last, let lastRadius = radii.last {
            path.addArc(
                center: last, radius: max(lastRadius, 0.2),
                startAngle: atan2(normals[points.count - 1].dy, normals[points.count - 1].dx),
                endAngle: atan2(-normals[points.count - 1].dy, -normals[points.count - 1].dx),
                clockwise: true)
        }

        // …and back up the other.
        for index in points.indices.reversed() {
            let offset = CGPoint(
                x: points[index].x - normals[index].dx * radii[index],
                y: points[index].y - normals[index].dy * radii[index])
            path.addLine(to: offset)
        }

        path.closeSubpath()
        return path
    }

    /// Chaikin corner-cutting: replaces each interior corner with two points a
    /// quarter and three quarters along its segments, which converges on a
    /// quadratic B-spline. Endpoints are preserved so the whip still starts at
    /// the handle and ends at the tip. Radii are carried through the same
    /// interpolation, so the taper survives the smoothing.
    static func smoothed(
        points: [CGPoint], radii: [CGFloat], passes: Int
    ) -> (points: [CGPoint], radii: [CGFloat]) {
        var points = points
        var radii = radii

        for _ in 0..<max(0, passes) {
            guard points.count >= 3 else { break }
            var nextPoints: [CGPoint] = [points[0]]
            var nextRadii: [CGFloat] = [radii[0]]
            nextPoints.reserveCapacity(points.count * 2)
            nextRadii.reserveCapacity(points.count * 2)

            for index in 0..<(points.count - 1) {
                let a = points[index]
                let b = points[index + 1]
                nextPoints.append(
                    CGPoint(x: a.x * 0.75 + b.x * 0.25, y: a.y * 0.75 + b.y * 0.25))
                nextPoints.append(
                    CGPoint(x: a.x * 0.25 + b.x * 0.75, y: a.y * 0.25 + b.y * 0.75))
                nextRadii.append(radii[index] * 0.75 + radii[index + 1] * 0.25)
                nextRadii.append(radii[index] * 0.25 + radii[index + 1] * 0.75)
            }

            nextPoints.append(points[points.count - 1])
            nextRadii.append(radii[radii.count - 1])
            points = nextPoints
            radii = nextRadii
        }

        return (points, radii)
    }

    /// Unit normals at each point: perpendicular to the average of the incoming
    /// and outgoing segment directions, so the outline does not pinch at bends.
    static func normals(for points: [CGPoint]) -> [CGVector] {
        guard points.count >= 2 else {
            return Array(repeating: CGVector(dx: 0, dy: 1), count: points.count)
        }

        var normals: [CGVector] = []
        normals.reserveCapacity(points.count)

        for index in points.indices {
            let previous = index > 0 ? points[index - 1] : points[index]
            let next = index < points.count - 1 ? points[index + 1] : points[index]

            var direction = CGVector(dx: next.x - previous.x, dy: next.y - previous.y)
            let length = hypot(direction.dx, direction.dy)
            if length < 0.0001 {
                direction = CGVector(dx: 0, dy: 1)
            } else {
                direction.dx /= length
                direction.dy /= length
            }
            // Rotate 90°.
            normals.append(CGVector(dx: -direction.dy, dy: direction.dx))
        }
        return normals
    }

    /// The handle drawn as a capsule between its two points. Returned separately
    /// from the thong so it can take a different fill — it is a different
    /// material, and a continuous taper from grip to tip would hide where the
    /// rigid part ends.
    static func handlePath(from grip: CGPoint, to tip: CGPoint, radius: CGFloat) -> CGPath {
        let path = CGMutablePath()
        let direction = CGVector(dx: tip.x - grip.x, dy: tip.y - grip.y)
        let length = hypot(direction.dx, direction.dy)
        guard length > 0.0001 else {
            path.addEllipse(
                in: CGRect(
                    x: grip.x - radius, y: grip.y - radius,
                    width: radius * 2, height: radius * 2))
            return path
        }

        let unit = CGVector(dx: direction.dx / length, dy: direction.dy / length)
        let normal = CGVector(dx: -unit.dy, dy: unit.dx)

        // The butt end is slightly fatter than the neck — a real grip flares so
        // it does not slide out of your hand, and the flare is what makes the
        // shape read as a handle rather than a stick.
        let buttRadius = radius * 1.18
        let neckRadius = radius * 0.86

        path.move(
            to: CGPoint(x: grip.x + normal.dx * buttRadius, y: grip.y + normal.dy * buttRadius))
        path.addLine(
            to: CGPoint(x: tip.x + normal.dx * neckRadius, y: tip.y + normal.dy * neckRadius))
        path.addArc(
            center: tip, radius: neckRadius,
            startAngle: atan2(normal.dy, normal.dx),
            endAngle: atan2(-normal.dy, -normal.dx),
            clockwise: true)
        path.addLine(
            to: CGPoint(x: grip.x - normal.dx * buttRadius, y: grip.y - normal.dy * buttRadius))
        path.addArc(
            center: grip, radius: buttRadius,
            startAngle: atan2(-normal.dy, -normal.dx),
            endAngle: atan2(normal.dy, normal.dx),
            clockwise: true)
        path.closeSubpath()
        return path
    }

    /// Bounding box of the whole whip, padded for the rim and shadow, so the
    /// renderer can invalidate just the region that changed instead of the
    /// whole multi-display canvas.
    static func bounds(points: [CGPoint], radii: [CGFloat], padding: CGFloat) -> CGRect {
        guard let first = points.first else { return .zero }
        var minX = first.x, maxX = first.x, minY = first.y, maxY = first.y
        for index in points.indices {
            let radius = index < radii.count ? radii[index] : 0
            minX = min(minX, points[index].x - radius)
            maxX = max(maxX, points[index].x + radius)
            minY = min(minY, points[index].y - radius)
            maxY = max(maxY, points[index].y + radius)
        }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
            .insetBy(dx: -padding, dy: -padding)
    }
}
