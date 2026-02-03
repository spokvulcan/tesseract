//
//  GlobalOverlayHUD.swift
//  tesseract
//

import SwiftUI

/// Global overlay HUD that displays recording waveform or processing indicator.
/// Designed as a floating pill that appears on top of all applications.
struct GlobalOverlayHUD: View {
    /// Observable state shared with the panel controller (not replaced on updates)
    var overlayState: OverlayState

    @State private var smoothedLevel: CGFloat = 0.08
    @State private var isVisible = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    enum Metrics {
        static let recordingSize = CGSize(width: 120, height: 32)
        static let processingSize = CGSize(width: 112, height: 34)
        static let errorSize = CGSize(width: 260, height: 44)
    }

    var body: some View {
        ZStack {
            if shouldShow {
                hudContent
                    .opacity(isVisible ? 1 : 0)
                    .scaleEffect(isVisible ? 1 : 0.85)
            }
        }
        .onChange(of: overlayState.dictationState) { _, newState in
            updateVisibility(for: newState)
        }
        .onChange(of: overlayState.audioLevel) { _, newValue in
            updateAudioLevel(newValue)
        }
        .onAppear {
            updateVisibility(for: overlayState.dictationState)
        }
    }

    private var shouldShow: Bool {
        switch overlayState.dictationState {
        case .recording, .processing, .error:
            return true
        default:
            return false
        }
    }

    @ViewBuilder
    private var hudContent: some View {
        Group {
            if overlayState.dictationState == .recording {
                recordingView
            } else if overlayState.dictationState == .processing {
                processingView
            } else if case .error = overlayState.dictationState {
                errorView
            }
        }
    }

    private var recordingView: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let phase = CGFloat(time * 2.2)

            pillContainer(style: .recording, time: time) {
                visualizationContent(level: smoothedLevel, phase: phase)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
            }
        }
    }

    private func visualizationContent(level: CGFloat, phase: CGFloat) -> some View {
        AudioBarsView(level: level, phase: phase)
    }

    private var processingView: some View {
        Group {
            if reduceMotion {
                pillContainer(style: .processing, time: nil) {
                    processingContent(time: nil)
                }
            } else {
                TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
                    let time = timeline.date.timeIntervalSinceReferenceDate

                    pillContainer(style: .processing, time: time) {
                        processingContent(time: time)
                    }
                }
            }
        }
    }

    private func processingContent(time: TimeInterval?) -> some View {
        HStack(spacing: 8) {
            if let time {
                ProcessingDotsView(time: time)
                    .frame(height: 12)
            } else {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private var errorView: some View {
        pillContainer(style: .error, time: nil) {
            HStack(alignment: .center, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 12, weight: .semibold))
                Text(errorMessage)
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                    .minimumScaleFactor(0.9)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .accessibilityLabel(errorMessage)
        }
    }

    private func pillContainer<Content: View>(
        style: PillStyle,
        time: TimeInterval?,
        @ViewBuilder content: () -> Content
    ) -> some View {
        let size = pillSize(for: style)
        let cornerRadius = size.height / 2

        return ZStack {
            // Glass background
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(backgroundMaterial(for: style))

            if let time, !reduceMotion {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(sheenGradient(for: style))
                    .opacity(0.18)
                    .offset(x: sheenOffset(for: time))
                    .blendMode(.screen)
                    .mask(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            }

            // Subtle gradient border
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(borderGradient(for: style), lineWidth: borderWidth(for: style))

            // Content
            content()
                .clipShape(RoundedRectangle(cornerRadius: cornerRadius - 2, style: .continuous))
        }
        .frame(width: size.width, height: size.height)
        .shadow(color: shadowColor(for: style), radius: shadowRadius(for: style), y: 3)
    }

    private enum PillStyle {
        case recording
        case processing
        case error
    }

    private func pillSize(for style: PillStyle) -> CGSize {
        switch style {
        case .error:
            return Metrics.errorSize
        case .recording:
            return Metrics.recordingSize
        case .processing:
            return Metrics.processingSize
        }
    }

    private func backgroundMaterial(for style: PillStyle) -> Material {
        switch style {
        case .error:
            return .thickMaterial
        case .recording, .processing:
            return .thickMaterial
        }
    }

    private func borderGradient(for style: PillStyle) -> LinearGradient {
        switch style {
        case .error:
            return LinearGradient(
                colors: [
                    Color.white.opacity(0.2),
                    Color.white.opacity(0.08)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        case .recording, .processing:
            return LinearGradient(
                colors: [
                    Color.white.opacity(0.2),
                    Color.white.opacity(0.08)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }
    }

    private func borderWidth(for style: PillStyle) -> CGFloat {
        switch style {
        case .error:
            return 0
        case .recording, .processing:
            return 0.5
        }
    }

    private func shadowColor(for style: PillStyle) -> Color {
        switch style {
        case .error:
            return Color.black.opacity(0.12)
        case .recording, .processing:
            return Color.black.opacity(0.12)
        }
    }

    private func shadowRadius(for style: PillStyle) -> CGFloat {
        switch style {
        case .error:
            return 8
        case .recording, .processing:
            return 6
        }
    }

    private func sheenGradient(for style: PillStyle) -> LinearGradient {
        switch style {
        case .error:
            return LinearGradient(
                colors: [
                    Color.white.opacity(0.0),
                    Color.white.opacity(0.25),
                    Color.white.opacity(0.0)
                ],
                startPoint: .leading,
                endPoint: .trailing
            )
        case .recording, .processing:
            return LinearGradient(
                colors: [
                    Color.white.opacity(0.0),
                    Color.white.opacity(0.18),
                    Color.white.opacity(0.0)
                ],
                startPoint: .leading,
                endPoint: .trailing
            )
        }
    }

    private func sheenOffset(for time: TimeInterval) -> CGFloat {
        let progress = CGFloat((time * 0.12).truncatingRemainder(dividingBy: 1.0))
        return (progress * 2 - 1) * 50
    }

    private var errorMessage: String {
        if case .error(let message) = overlayState.dictationState {
            return message
        }
        return "Something went wrong."
    }

    private func updateVisibility(for newState: DictationState) {
        let show: Bool
        switch newState {
        case .recording, .processing, .error:
            show = true
        default:
            show = false
        }

        if show {
            animateVisibility(show: true, response: 0.3, dampingFraction: 0.7)
        } else {
            animateVisibility(show: false, response: 0.25, dampingFraction: 0.8)
        }
    }

    private func animateVisibility(show: Bool, response: Double, dampingFraction: Double) {
        if reduceMotion {
            isVisible = show
            return
        }
        withAnimation(.spring(response: response, dampingFraction: dampingFraction)) {
            isVisible = show
        }
    }

    private func updateAudioLevel(_ newValue: Float) {
        let clamped = max(0.06, min(CGFloat(newValue), 1))
        withAnimation(.easeOut(duration: 0.1)) {
            smoothedLevel = clamped
        }
    }
}

#Preview("Recording") {
    let state = OverlayState()
    state.dictationState = .recording
    state.audioLevel = 0.5
    return GlobalOverlayHUD(overlayState: state)
        .padding(50)
        .background(Color.gray.opacity(0.3))
}

#Preview("Processing") {
    let state = OverlayState()
    state.dictationState = .processing
    state.audioLevel = 0
    return GlobalOverlayHUD(overlayState: state)
        .padding(50)
        .background(Color.gray.opacity(0.3))
}
