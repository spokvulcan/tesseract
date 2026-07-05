//
//  OnboardingBackdrop.swift
//  tesseract
//
//  The tour's backdrop: a quiet neutral graphite wash (soft paper-white in
//  light mode) with a faint lift behind the hero. Static by design — the
//  motion in the tour belongs to the mark and the chapters, not the walls.
//

import SwiftUI

struct OnboardingBackdrop: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        ZStack {
            colorScheme == .dark ? Color(white: 0.09) : Color(white: 0.97)

            LinearGradient(
                colors: [
                    Color.white.opacity(colorScheme == .dark ? 0.05 : 0.6),
                    .clear,
                ],
                startPoint: .top, endPoint: .center)
        }
        .ignoresSafeArea()
    }
}
