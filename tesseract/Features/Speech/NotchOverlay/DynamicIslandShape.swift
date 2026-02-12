//
//  DynamicIslandShape.swift
//  tesseract
//
//  Adapted from textream (github.com/f/textream)
//

import SwiftUI

struct DynamicIslandShape: Shape {
    var topInset: CGFloat = 16
    var bottomRadius: CGFloat = 18

    var animatableData: AnimatablePair<CGFloat, CGFloat> {
        get { AnimatablePair(topInset, bottomRadius) }
        set {
            topInset = newValue.first
            bottomRadius = newValue.second
        }
    }

    func path(in rect: CGRect) -> Path {
        let w = rect.width
        let h = rect.height
        let t = topInset
        let br = bottomRadius
        var p = Path()

        p.move(to: CGPoint(x: 0, y: 0))

        // Top-left concave curve
        p.addQuadCurve(
            to: CGPoint(x: t, y: t),
            control: CGPoint(x: t, y: 0)
        )

        p.addLine(to: CGPoint(x: t, y: h - br))

        // Bottom-left convex corner
        p.addQuadCurve(
            to: CGPoint(x: t + br, y: h),
            control: CGPoint(x: t, y: h)
        )

        p.addLine(to: CGPoint(x: w - t - br, y: h))

        // Bottom-right convex corner
        p.addQuadCurve(
            to: CGPoint(x: w - t, y: h - br),
            control: CGPoint(x: w - t, y: h)
        )

        p.addLine(to: CGPoint(x: w - t, y: t))

        // Top-right concave curve
        p.addQuadCurve(
            to: CGPoint(x: w, y: 0),
            control: CGPoint(x: w - t, y: 0)
        )

        p.closeSubpath()
        return p
    }
}
