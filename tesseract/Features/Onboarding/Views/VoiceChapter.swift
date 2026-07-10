//
//  VoiceChapter.swift
//  tesseract
//
//  Chapter 4 — text-to-speech. Its Try-it speaks the chapter's own headline
//  out loud through the real speech coordinator once the voice model is on
//  disk; until then, an honest locked state under the scripted waveform.
//

import SwiftUI

struct VoiceChapter: View {
    let controller: OnboardingTourController

    @Environment(SpeechCoordinator.self) private var speechCoordinator
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private static let sampleLine =
        "Hello. I'm Tesseract — a voice made entirely on this Mac."

    private var isLive: Bool {
        OnboardingTryIt.voiceIsLive(voiceModelDownloaded: controller.voiceModelReady)
    }

    private var isSpeaking: Bool {
        speechCoordinator.state != .idle
    }

    var body: some View {
        ChapterScaffold(
            kicker: "Chapter 4 · Voice",
            title: "It speaks for itself",
            subtitle: "Natural long-form speech, synthesized on device — "
                + "read a page, a draft, or an answer aloud."
        ) {
            StagePanel(maxWidth: 520) {
                VStack(spacing: OnboardingType.rhythm) {
                    WaveformView(isActive: isSpeaking && !reduceMotion)
                        .frame(height: 56)

                    if isLive {
                        HStack(spacing: 10) {
                            Button {
                                if isSpeaking {
                                    speechCoordinator.stop()
                                } else {
                                    speechCoordinator.speakText(Self.sampleLine)
                                }
                            } label: {
                                Label(
                                    isSpeaking ? "Stop" : "Hear it speak",
                                    systemImage: isSpeaking ? "stop.fill" : "play.fill"
                                )
                                .frame(minWidth: 110)
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.regular)

                            Text(speakStatusLine)
                                .font(OnboardingType.body)
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        TryItLockedSlot(
                            icon: "speaker.badge.exclamationmark",
                            status: controller.status(for: controller.voiceModelID),
                            modelNoun: "voice"
                        )
                    }
                }
            }
        }
        .onDisappear {
            if isSpeaking { speechCoordinator.stop() }
        }
        .animation(.spring(response: 0.45, dampingFraction: 0.85), value: isLive)
    }

    private var speakStatusLine: String {
        switch speechCoordinator.state {
        case .generating: "Generating on your GPU\u{2026}"
        case .playing, .streaming, .streamingLongForm: "Playing"
        case .error(let message): message
        default: "\u{201C}\(Self.sampleLine)\u{201D}"
        }
    }

}

/// The scripted waveform: a quiet field of bars that comes alive while speech
/// actually plays. Static under Reduce Motion.
private struct WaveformView: View {
    var isActive: Bool

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30, paused: reduceMotion)) { timeline in
            Canvas { context, size in
                let t = reduceMotion ? 1.7 : timeline.date.timeIntervalSinceReferenceDate
                let barCount = 42
                let spacing = size.width / CGFloat(barCount)
                let energy: Double = isActive ? 1.0 : 0.28
                // One shading per frame, shared by all 42 bars: the system
                // accent while speaking, quiet neutral ink while idle.
                let shading: GraphicsContext.Shading = .color(
                    isActive
                        ? Color.accentColor.opacity(0.85)
                        : Color.secondary.opacity(0.4))

                for index in 0..<barCount {
                    let x = CGFloat(index) * spacing + spacing / 2
                    let normalized = Double(index) / Double(barCount - 1)
                    let envelope = sin(normalized * .pi)
                    let wobble =
                        sin(t * 4.1 + Double(index) * 0.9)
                        * sin(t * 2.3 + Double(index) * 0.35)
                    let height = max(
                        2.5,
                        size.height * 0.5 * envelope * energy * (0.45 + 0.55 * abs(wobble)))

                    let rect = CGRect(
                        x: x - 1.5, y: size.height / 2 - height / 2,
                        width: 3, height: height)
                    let path = Path(roundedRect: rect, cornerRadius: 1.5)
                    context.fill(path, with: shading)
                }
            }
        }
        .accessibilityHidden(true)
    }
}
