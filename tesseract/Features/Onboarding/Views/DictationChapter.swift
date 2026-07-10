//
//  DictationChapter.swift
//  tesseract
//
//  Chapter 3 — push-to-talk dictation, and the chapter that carries both
//  permissions (asked in context, user-initiated, never gating Continue). Its
//  Try-it is the tour's signature satisfaction beat: hold the button and
//  dictate a real sentence into the tour itself.
//

import SwiftUI

struct DictationChapter: View {
    let controller: OnboardingTourController

    @EnvironmentObject private var permissionsManager: PermissionsManager
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        ChapterScaffold(
            kicker: "Chapter 3 · Dictation",
            title: "Hold a key. Speak. It types.",
            subtitle: "Push-to-talk dictation into any app on your Mac — "
                + "transcribed on device, in your language."
        ) {
            VStack(spacing: OnboardingType.rhythm) {
                HStack(alignment: .top, spacing: OnboardingType.rhythm) {
                    PermissionCard(
                        icon: "mic.fill",
                        title: "Microphone",
                        benefit: "Hears you while you hold the key — for dictation "
                            + "and for talking to the agent. Audio never leaves this Mac.",
                        state: permissionsManager.microphonePermission,
                        grantAction: {
                            Task { _ = await permissionsManager.requestMicrophonePermission() }
                        },
                        recoverAction: {
                            permissionsManager.openSystemPreferences(for: "microphone")
                        }
                    )

                    PermissionCard(
                        icon: "keyboard",
                        title: "Accessibility",
                        benefit: "Lets the push-to-talk hotkey work everywhere without "
                            + "typing stray characters while you speak.",
                        state: permissionsManager.accessibilityPermission,
                        grantAction: {
                            permissionsManager.requestAccessibilityPermission()
                        },
                        recoverAction: {
                            permissionsManager.requestAccessibilityPermission()
                        }
                    )
                }
                .frame(maxWidth: 560)

                DictationTryItStrip(controller: controller)

                languageLine
            }
        }
        .onAppear {
            permissionsManager.checkMicrophonePermission()
            permissionsManager.checkAccessibilityPermission()
        }
    }

    private var languageLine: some View {
        @Bindable var settings = settings
        return HStack(spacing: 6) {
            Image(systemName: "globe")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
            Text("Listening in \(settings.selectedLanguage.displayName)")
                .font(OnboardingType.body)
                .foregroundStyle(.secondary)
            LanguageChangePopover(selectedLanguage: $settings.language)
        }
    }
}

private struct LanguageChangePopover: View {
    @Binding var selectedLanguage: String
    @State private var isPresented = false

    var body: some View {
        Button("Change") {
            isPresented = true
        }
        .buttonStyle(.plain)
        .font(OnboardingType.body.weight(.medium))
        .foregroundStyle(.tint)
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            CompactLanguagePickerView(selectedLanguage: $selectedLanguage)
                .frame(width: 260, height: 320)
                .padding(8)
        }
    }
}

// MARK: - Try-it

/// The live dictation slot: a dedicated voice-input controller over the shared
/// capture engine — real capture, real transcription, right in the tour. When
/// preconditions are missing it degrades to an honest locked state; the tour
/// never blocks (`OnboardingTryIt.dictationIsLive`).
private struct DictationTryItStrip: View {
    let controller: OnboardingTourController

    @EnvironmentObject private var permissionsManager: PermissionsManager
    @Environment(SettingsManager.self) private var settings
    @Environment(AudioCaptureEngine.self) private var audioCapture
    @Environment(TranscriptionEngine.self) private var transcriptionEngine

    @State private var voice: AgentVoiceInputController?
    @State private var isHolding = false
    @State private var transcribed = ""

    private var isLive: Bool {
        OnboardingTryIt.dictationIsLive(
            microphone: permissionsManager.microphonePermission,
            speechModelDownloaded: controller.speechModelReady)
    }

    var body: some View {
        StagePanel(maxWidth: 560) {
            if isLive {
                liveSlot
            } else {
                lockedSlot
            }
        }
        .animation(.spring(response: 0.45, dampingFraction: 0.85), value: isLive)
    }

    @ViewBuilder
    private var liveSlot: some View {
        HStack(spacing: 14) {
            holdButton

            VStack(alignment: .leading, spacing: 3) {
                if transcribed.isEmpty {
                    Text(statusLine)
                        .font(OnboardingType.body)
                        .foregroundStyle(.secondary)
                } else {
                    Text(transcribed)
                        .font(OnboardingType.body.weight(.medium))
                        .lineLimit(2)
                        .transition(.blurReplace)
                }
                if case .error(let message) = voice?.voiceState {
                    Text(message)
                        .font(OnboardingType.body)
                        .foregroundStyle(.orange)
                }
            }
            Spacer(minLength: 0)
        }
        .onAppear { makeVoiceControllerIfNeeded() }
    }

    private var statusLine: String {
        guard transcriptionEngine.isModelLoaded else {
            return "Warming the model up\u{2026} one moment."
        }
        switch voice?.voiceState {
        case .recording: return "Listening\u{2026}"
        case .transcribing: return "Transcribing\u{2026}"
        default: return "Hold, say something, release."
        }
    }

    private var holdButton: some View {
        let recording = voice?.voiceState == .recording
        return Circle()
            .fill(recording ? AnyShapeStyle(.tint) : AnyShapeStyle(.quaternary))
            .frame(width: 44, height: 44)
            .overlay {
                Image(systemName: "mic.fill")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(recording ? AnyShapeStyle(.white) : AnyShapeStyle(.secondary))
                    .symbolEffect(.pulse, options: .repeating, isActive: recording)
            }
            .scaleEffect(recording ? 1.12 : 1)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: recording)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !isHolding, transcriptionEngine.isModelLoaded else { return }
                        isHolding = true
                        transcribed = ""
                        voice?.start()
                    }
                    .onEnded { _ in
                        isHolding = false
                        voice?.finishCapture()
                    }
            )
            .accessibilityLabel("Hold to dictate")
    }

    private var lockedSlot: some View {
        TryItLockedSlot(
            icon: "mic.badge.xmark",
            status: controller.status(for: controller.speechToTextModelID),
            modelNoun: "speech",
            overrideReason: permissionsManager.microphonePermission == .granted
                ? nil : "Grant the microphone above and this slot goes live."
        )
    }

    private func makeVoiceControllerIfNeeded() {
        guard voice == nil else { return }
        let controller = AgentVoiceInputController(
            audioCapture: audioCapture,
            transcriptionEngine: transcriptionEngine,
            settings: settings
        )
        controller.onVoiceTranscription = { text in
            withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                transcribed = text
            }
        }
        voice = controller
    }
}
