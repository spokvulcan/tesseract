//
//  GlobalOverlayHUD.swift
//  tesseract
//

import SwiftUI

/// Global overlay HUD that displays recording waveform or processing indicator.
/// A Liquid Glass pill floating on top of all applications: one `.glassEffect`
/// capsule whose tint carries the state (recording red, error amber, processing
/// neutral) with vibrant content on top — a floating control, exactly the layer
/// the HIG sanctions glass for. The old hand-rolled treatment (material fill,
/// moving sheen, gradient border, per-style shadow) is fully replaced by the
/// system material, which also adapts to light/dark, the Clear/Tinted
/// appearance setting, and Reduce Transparency on its own.
struct GlobalOverlayHUD: View {
    /// Observable state shared with the panel controller (not replaced on updates)
    var overlayState: OverlayState

    @State private var smoothedLevel: CGFloat = 0.08
    @State private var isVisible = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GlassEffectContainer {
            ZStack {
                if shouldShow {
                    hudContent
                        .opacity(isVisible ? 1 : 0)
                        .scaleEffect(isVisible ? 1 : 0.85)
                }
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
        overlayState.dictationState.showsOverlay
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
        pillContainer(style: .recording) {
            AudioBarsView(level: smoothedLevel)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
        }
    }

    private var processingView: some View {
        Group {
            if reduceMotion {
                pillContainer(style: .processing) {
                    processingContent(time: nil)
                }
            } else {
                TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
                    let time = timeline.date.timeIntervalSinceReferenceDate

                    pillContainer(style: .processing) {
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
        pillContainer(style: .error) {
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
        @ViewBuilder content: () -> Content
    ) -> some View {
        // Size from the shared PillMetrics keyed on the live dictation state, so the
        // hosted pill always matches the NSPanel frame the placement sized (which
        // reads the same source). `hudContent` only renders for overlay-showing
        // states, so the state→size lookup is always one of the three pill sizes.
        let size = PillMetrics.size(for: overlayState.dictationState)

        return content()
            .frame(width: size.width, height: size.height)
            .glassEffect(style.glass, in: .capsule)
    }

    private enum PillStyle {
        case recording
        case processing
        case error

        /// Semantic tint only (HIG rule): red = the microphone is live, amber =
        /// needs attention; processing stays neutral. The regular variant keeps
        /// the content legible over whatever app the pill floats above.
        var glass: Glass {
            switch self {
            case .recording:
                return .regular.tint(.red.opacity(0.16))
            case .processing:
                return .regular
            case .error:
                return .regular.tint(.orange.opacity(0.16))
            }
        }
    }

    private var errorMessage: String {
        if case .error(let message) = overlayState.dictationState {
            return message
        }
        return "Something went wrong."
    }

    private func updateVisibility(for newState: DictationState) {
        if newState.showsOverlay {
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
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Processing") {
    let state = OverlayState()
    state.dictationState = .processing
    state.audioLevel = 0
    return GlobalOverlayHUD(overlayState: state)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Error") {
    let state = OverlayState()
    state.dictationState = .error("No speech was detected.")
    return GlobalOverlayHUD(overlayState: state)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}
