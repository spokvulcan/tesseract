//
//  ProcessingDotsView.swift
//  whisper-on-device
//

import SwiftUI

/// Animated processing indicator with 3 pulsing dots.
/// Uses staggered scale and opacity animation with gradient fills.
struct ProcessingDotsView: View {
    @State private var isAnimating = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let dotSize: CGFloat = 6
    private let dotSpacing: CGFloat = 8
    private let animationDuration: Double = 0.6

    var body: some View {
        HStack(spacing: dotSpacing) {
            ForEach(0..<3) { index in
                dot(index: index)
            }
        }
        .onAppear {
            if !reduceMotion {
                isAnimating = true
            }
        }
        .onDisappear {
            isAnimating = false
        }
    }

    private func dot(index: Int) -> some View {
        Circle()
            .fill(dotGradient)
            .frame(width: dotSize, height: dotSize)
            .scaleEffect(scaleForDot(index: index))
            .opacity(opacityForDot(index: index))
            .animation(
                reduceMotion ? nil : Animation
                    .easeInOut(duration: animationDuration)
                    .repeatForever(autoreverses: true)
                    .delay(Double(index) * 0.15),
                value: isAnimating
            )
    }

    private var dotGradient: LinearGradient {
        LinearGradient(
            colors: [.white, .cyan.opacity(0.9)],
            startPoint: .top,
            endPoint: .bottom
        )
    }

    private func scaleForDot(index: Int) -> CGFloat {
        if reduceMotion {
            return 1.0
        }
        return isAnimating ? 1.0 : 0.7
    }

    private func opacityForDot(index: Int) -> Double {
        if reduceMotion {
            return 1.0
        }
        return isAnimating ? 1.0 : 0.5
    }
}

#Preview {
    VStack(spacing: 20) {
        ProcessingDotsView()
            .padding()
            .background(Color.black.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }
    .padding()
}
