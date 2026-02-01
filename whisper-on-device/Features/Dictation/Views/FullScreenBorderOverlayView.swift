//
//  FullScreenBorderOverlayView.swift
//  whisper-on-device
//
//  Apple Intelligence Glow Effect - Optimized for 120fps
//  Based on: https://github.com/jacobamobin/AppleIntelligenceGlowEffect

import SwiftUI

// MARK: - Main Glow Effect View (Optimized)

struct GlowEffect: View {
    let audioLevel: CGFloat
    let theme: GlowTheme

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 120.0)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate

            Canvas { context, size in
                drawGlowBorder(context: context, size: size, time: time)
            }
        }
        .ignoresSafeArea()
        .drawingGroup() // Rasterize to GPU texture for better performance
    }

    private func drawGlowBorder(context: GraphicsContext, size: CGSize, time: Double) {
        let rect = CGRect(origin: .zero, size: size)

        // Audio-reactive values
        let level = max(0.05, min(audioLevel, 1.0))
        let widthMultiplier = 1.0 + (level * 1.2)
        let brightnessBoost = level * 0.2

        // Create angular gradient that animates over time
        let gradient = createAnimatedGradient(time: time, brightnessBoost: brightnessBoost)

        // Draw multiple stroke layers (from outer glow to sharp edge)
        // Layer 4: Outer glow (widest, most blur)
        drawStrokeLayer(
            context: context,
            rect: rect,
            gradient: gradient,
            width: 15 * widthMultiplier,
            blur: 12,
            opacity: 0.6
        )

        // Layer 3: Medium glow
        drawStrokeLayer(
            context: context,
            rect: rect,
            gradient: gradient,
            width: 11 * widthMultiplier,
            blur: 8,
            opacity: 0.7
        )

        // Layer 2: Inner glow
        drawStrokeLayer(
            context: context,
            rect: rect,
            gradient: gradient,
            width: 8 * widthMultiplier,
            blur: 3,
            opacity: 0.85
        )

        // Layer 1: Sharp edge (no blur)
        drawStrokeLayer(
            context: context,
            rect: rect,
            gradient: gradient,
            width: 5 * widthMultiplier,
            blur: 0,
            opacity: 1.0
        )
    }

    private func drawStrokeLayer(
        context: GraphicsContext,
        rect: CGRect,
        gradient: Gradient,
        width: CGFloat,
        blur: CGFloat,
        opacity: Double
    ) {
        var ctx = context

        if blur > 0 {
            ctx.addFilter(.blur(radius: blur))
        }
        ctx.opacity = opacity

        let path = Rectangle().path(in: rect.insetBy(dx: width / 2, dy: width / 2))

        ctx.stroke(
            path,
            with: .conicGradient(
                gradient,
                center: CGPoint(x: rect.midX, y: rect.midY)
            ),
            lineWidth: width
        )
    }

    private func createAnimatedGradient(time: Double, brightnessBoost: CGFloat) -> Gradient {
        let colors = theme.colors

        // Animate color positions smoothly over time
        let speed = 0.15
        let colorCount = colors.count
        let basePositions: [Double] = (0..<colorCount).map { Double($0) / Double(colorCount) }

        var stops: [Gradient.Stop] = []
        for (index, color) in colors.enumerated() {
            // Each color position oscillates with different phase
            let phase = Double(index) * 0.5
            let oscillation = sin(time * speed + phase) * 0.08
            let position = (basePositions[index] + oscillation).truncatingRemainder(dividingBy: 1.0)

            // Apply brightness boost
            let adjustedColor = brightnessBoost > 0 ? color.brightness(brightnessBoost) : color
            stops.append(Gradient.Stop(color: adjustedColor, location: max(0, min(1, position))))
        }

        // Sort by position for proper gradient rendering
        stops.sort { $0.location < $1.location }

        return Gradient(stops: stops)
    }
}

// MARK: - Color Extension

extension Color {
    init(hex: String) {
        let scanner = Scanner(string: hex)
        _ = scanner.scanString("#")

        var hexNumber: UInt64 = 0
        scanner.scanHexInt64(&hexNumber)

        let r = Double((hexNumber & 0xff0000) >> 16) / 255
        let g = Double((hexNumber & 0x00ff00) >> 8) / 255
        let b = Double(hexNumber & 0x0000ff) / 255

        self.init(red: r, green: g, blue: b)
    }

    func brightness(_ amount: CGFloat) -> Color {
        // Approximate brightness adjustment
        return self.opacity(1.0 + Double(amount) * 0.5)
    }
}

// MARK: - Wrapper for App Integration

struct FullScreenBorderOverlayView: View {
    let state: DictationState
    let audioLevel: Float
    let theme: GlowTheme

    @State private var isVisible = false

    var body: some View {
        ZStack {
            if shouldShow {
                GlowEffect(audioLevel: CGFloat(audioLevel), theme: theme)
                    .opacity(isVisible ? 1 : 0)
            }
        }
        .onChange(of: state) { _, newState in
            handleStateChange(newState)
        }
        .onAppear {
            handleStateChange(state)
        }
    }

    private var shouldShow: Bool {
        switch state {
        case .recording, .processing, .error:
            return true
        default:
            return false
        }
    }

    private func handleStateChange(_ newState: DictationState) {
        switch newState {
        case .recording, .processing, .error:
            withAnimation(.easeInOut(duration: 0.3)) {
                isVisible = true
            }
        default:
            withAnimation(.easeInOut(duration: 0.25)) {
                isVisible = false
            }
        }
    }
}

// MARK: - Preview

#Preview {
    GlowEffect(audioLevel: 0.5, theme: .appleIntelligence)
        .background(Color.black)
}

#Preview("Matrix") {
    GlowEffect(audioLevel: 0.5, theme: .matrix)
        .background(Color.black)
}

#Preview("Fire") {
    GlowEffect(audioLevel: 0.5, theme: .fire)
        .background(Color.black)
}
