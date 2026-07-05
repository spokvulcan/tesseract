//
//  OnboardingBackdrop.swift
//  tesseract
//
//  The tour's slow-breathing MeshGradient backdrop, tuned per Chapter —
//  dark-first, with a quiet high-luminance treatment in light mode. Under
//  Reduce Motion the mesh holds still.
//

import SwiftUI

struct OnboardingBackdrop: View {
    var chapter: OnboardingTourController.Chapter

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 20, paused: reduceMotion)) { timeline in
            let t = reduceMotion ? 0 : timeline.date.timeIntervalSinceReferenceDate
            let sway = Float(0.10 * sin(t * 0.23))
            let lift = Float(0.08 * cos(t * 0.17))

            MeshGradient(
                width: 3,
                height: 3,
                points: [
                    [0, 0], [0.5, 0], [1, 0],
                    [0, 0.5], [0.5 + sway, 0.5 + lift], [1, 0.5],
                    [0, 1], [0.5, 1], [1, 1],
                ],
                colors: meshColors
            )
        }
        .overlay {
            // A quiet vignette keeps the content region legible at any mesh pose.
            RadialGradient(
                colors: [
                    .clear,
                    (colorScheme == .dark ? Color.black : Color.white).opacity(0.25),
                ],
                center: .center, startRadius: 220, endRadius: 560
            )
        }
        .ignoresSafeArea()
        .animation(.easeInOut(duration: 1.4), value: chapter)
    }

    /// Nine mesh colors: tinted corners around a chapter-specific heart.
    private var meshColors: [Color] {
        let p = palette
        return [
            p.base, p.base, p.edge,
            p.edge, p.heart, p.base,
            p.base, p.edge, p.base,
        ]
    }

    private struct Palette {
        let base: Color
        let edge: Color
        let heart: Color
    }

    private var palette: Palette {
        if colorScheme == .dark { return darkPalette }
        return lightPalette
    }

    private var darkPalette: Palette {
        switch chapter {
        case .welcome:
            Palette(
                base: Color(red: 0.02, green: 0.016, blue: 0.045),
                edge: Color(red: 0.10, green: 0.09, blue: 0.30),
                heart: Color(red: 0.24, green: 0.15, blue: 0.46))
        case .agent:
            Palette(
                base: Color(red: 0.016, green: 0.025, blue: 0.055),
                edge: Color(red: 0.06, green: 0.12, blue: 0.24),
                heart: Color(red: 0.04, green: 0.24, blue: 0.33))
        case .dictation:
            Palette(
                base: Color(red: 0.045, green: 0.025, blue: 0.03),
                edge: Color(red: 0.20, green: 0.11, blue: 0.07),
                heart: Color(red: 0.32, green: 0.15, blue: 0.14))
        case .voice:
            Palette(
                base: Color(red: 0.016, green: 0.035, blue: 0.04),
                edge: Color(red: 0.05, green: 0.19, blue: 0.17),
                heart: Color(red: 0.07, green: 0.28, blue: 0.19))
        case .server:
            Palette(
                base: Color(red: 0.008, green: 0.016, blue: 0.024),
                edge: Color(red: 0.03, green: 0.14, blue: 0.11),
                heart: Color(red: 0.03, green: 0.16, blue: 0.22))
        case .ready:
            Palette(
                base: Color(red: 0.027, green: 0.024, blue: 0.055),
                edge: Color(red: 0.13, green: 0.12, blue: 0.33),
                heart: Color(red: 0.30, green: 0.24, blue: 0.12))
        }
    }

    private var lightPalette: Palette {
        switch chapter {
        case .welcome:
            Palette(
                base: Color(red: 0.97, green: 0.97, blue: 0.99),
                edge: Color(red: 0.90, green: 0.89, blue: 0.98),
                heart: Color(red: 0.84, green: 0.80, blue: 0.96))
        case .agent:
            Palette(
                base: Color(red: 0.96, green: 0.97, blue: 0.99),
                edge: Color(red: 0.87, green: 0.92, blue: 0.97),
                heart: Color(red: 0.80, green: 0.90, blue: 0.95))
        case .dictation:
            Palette(
                base: Color(red: 0.99, green: 0.97, blue: 0.96),
                edge: Color(red: 0.98, green: 0.92, blue: 0.87),
                heart: Color(red: 0.97, green: 0.88, blue: 0.82))
        case .voice:
            Palette(
                base: Color(red: 0.96, green: 0.99, blue: 0.98),
                edge: Color(red: 0.88, green: 0.96, blue: 0.93),
                heart: Color(red: 0.82, green: 0.94, blue: 0.88))
        case .server:
            Palette(
                base: Color(red: 0.96, green: 0.97, blue: 0.98),
                edge: Color(red: 0.88, green: 0.93, blue: 0.92),
                heart: Color(red: 0.84, green: 0.92, blue: 0.94))
        case .ready:
            Palette(
                base: Color(red: 0.97, green: 0.97, blue: 0.99),
                edge: Color(red: 0.91, green: 0.90, blue: 0.98),
                heart: Color(red: 0.97, green: 0.93, blue: 0.80))
        }
    }
}
