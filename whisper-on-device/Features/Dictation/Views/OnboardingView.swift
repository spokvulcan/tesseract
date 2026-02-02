//
//  OnboardingView.swift
//  whisper-on-device
//

import SwiftUI

struct OnboardingView: View {
    @ObservedObject var permissionsManager: PermissionsManager
    @Binding var isPresented: Bool

    @State private var currentStep = 0

    private let totalSteps = 5

    private var canGoBack: Bool {
        currentStep > 0
    }

    private var isLastStep: Bool {
        currentStep == totalSteps - 1
    }

    var body: some View {
        VStack(spacing: 0) {
            // Progress indicator
            ProgressView(value: Double(currentStep + 1), total: Double(totalSteps))
                .padding(.horizontal)
                .padding(.top)

            // Step indicator
            Text("Step \(currentStep + 1) of \(totalSteps)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.top, 8)

            // Scrollable content area
            ScrollView {
                VStack {
                    stepContent
                        .frame(maxWidth: .infinity)
                }
                .padding()
            }

            Divider()

            // Navigation buttons
            HStack {
                if canGoBack {
                    Button("Back") {
                        previousStep()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                }

                Spacer()

                if isLastStep {
                    Button("Get Started") {
                        complete()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                } else {
                    Button("Next") {
                        nextStep()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(!canProceed)
                }
            }
            .padding()
        }
        .frame(width: 550, height: 500)
        .background(.ultraThinMaterial)
    }

    @ViewBuilder
    private var stepContent: some View {
        switch currentStep {
        case 0:
            WelcomeStepContent()
        case 1:
            LanguageSelectionStepContent()
        case 2:
            MicrophonePermissionStepContent(permissionsManager: permissionsManager)
        case 3:
            AccessibilityPermissionStepContent(permissionsManager: permissionsManager)
        case 4:
            ReadyStepContent()
        default:
            EmptyView()
        }
    }

    private var canProceed: Bool {
        switch currentStep {
        case 2:
            // Microphone step - can proceed if granted or denied (user can fix later)
            return permissionsManager.microphonePermission != .requesting
        default:
            return true
        }
    }

    private func nextStep() {
        withAnimation(.easeInOut(duration: 0.2)) {
            currentStep = min(currentStep + 1, totalSteps - 1)
        }
    }

    private func previousStep() {
        withAnimation(.easeInOut(duration: 0.2)) {
            currentStep = max(currentStep - 1, 0)
        }
    }

    private func complete() {
        SettingsManager.shared.hasCompletedOnboarding = true
        isPresented = false
    }
}

// MARK: - Welcome Step Content

struct WelcomeStepContent: View {
    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 20)

            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 80))
                .foregroundStyle(.tint)

            Text("Welcome to WhisperOnDevice")
                .font(.largeTitle)
                .fontWeight(.bold)
                .multilineTextAlignment(.center)

            Text("A privacy-focused voice-to-text app that runs entirely on your device. Your voice never leaves your Mac.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 20)

            VStack(alignment: .leading, spacing: 16) {
                FeatureRow(icon: "lock.shield", title: "100% Private", description: "All processing happens locally")
                FeatureRow(icon: "bolt", title: "Fast & Accurate", description: "Powered by Whisper Large V3 Turbo")
                FeatureRow(icon: "keyboard", title: "Push-to-Talk", description: "Simple hotkey to start dictating")
            }
            .padding(.top, 12)

            Spacer(minLength: 20)
        }
    }
}

struct FeatureRow: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundStyle(.tint)
                .frame(width: 32)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .fontWeight(.medium)
                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Language Selection Step Content

struct LanguageSelectionStepContent: View {
    @ObservedObject private var settings = SettingsManager.shared

    var body: some View {
        VStack(spacing: 20) {
            Spacer(minLength: 20)

            Image(systemName: "globe")
                .font(.system(size: 60))
                .foregroundStyle(.tint)

            Text("Choose Your Language")
                .font(.title)
                .fontWeight(.bold)

            Text("Select the language you'll primarily speak. This helps improve transcription accuracy.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 20)

            CompactLanguagePickerView(selectedLanguage: $settings.language)
                .padding(.horizontal, 20)

            Spacer(minLength: 20)
        }
    }
}

// MARK: - Microphone Permission Step Content

struct MicrophonePermissionStepContent: View {
    @ObservedObject var permissionsManager: PermissionsManager

    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 20)

            Image(systemName: "mic.circle.fill")
                .font(.system(size: 60))
                .foregroundStyle(permissionColor)

            Text("Microphone Access")
                .font(.title)
                .fontWeight(.bold)

            Text("WhisperOnDevice needs microphone access to hear your voice for transcription.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 20)

            Spacer(minLength: 20)

            permissionContent

            Spacer(minLength: 20)
        }
    }

    @ViewBuilder
    private var permissionContent: some View {
        switch permissionsManager.microphonePermission {
        case .granted:
            Label("Microphone access granted", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.headline)

        case .denied:
            VStack(spacing: 12) {
                Label("Microphone access denied", systemImage: "xmark.circle.fill")
                    .foregroundStyle(.red)

                Button("Open System Settings") {
                    permissionsManager.openSystemPreferences(for: "microphone")
                }
                .buttonStyle(.bordered)

                Text("You can continue, but dictation won't work until you grant access.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

        case .requesting:
            ProgressView("Requesting permission...")

        case .unknown, .restricted:
            Button("Request Microphone Access") {
                Task {
                    _ = await permissionsManager.requestMicrophonePermission()
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
    }

    private var permissionColor: Color {
        switch permissionsManager.microphonePermission {
        case .granted:
            return .green
        case .denied:
            return .red
        default:
            return .blue
        }
    }
}

// MARK: - Accessibility Permission Step Content

struct AccessibilityPermissionStepContent: View {
    @ObservedObject var permissionsManager: PermissionsManager

    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 20)

            Image(systemName: "hand.raised.circle.fill")
                .font(.system(size: 60))
                .foregroundStyle(permissionColor)

            Text("Accessibility Access")
                .font(.title)
                .fontWeight(.bold)

            Text("WhisperOnDevice needs Accessibility access to capture your hotkey without typing unwanted characters.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 20)

            Spacer(minLength: 20)

            permissionContent

            Spacer(minLength: 20)
        }
        .onAppear {
            permissionsManager.checkAccessibilityPermission()
        }
    }

    @ViewBuilder
    private var permissionContent: some View {
        switch permissionsManager.accessibilityPermission {
        case .granted:
            Label("Accessibility access granted", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.headline)

        case .denied, .unknown:
            VStack(spacing: 12) {
                Button("Grant Accessibility Access") {
                    permissionsManager.requestAccessibilityPermission()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                Text("Without Accessibility access, the hotkey may type characters while recording. You can skip this and configure it later.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 20)
            }

        case .requesting:
            VStack(spacing: 8) {
                ProgressView()
                Text("Waiting for permission...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

        case .restricted:
            VStack(spacing: 8) {
                Label("Accessibility access restricted", systemImage: "xmark.circle.fill")
                    .foregroundStyle(.red)

                Text("Your system restricts accessibility access. You can continue without it.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
    }

    private var permissionColor: Color {
        switch permissionsManager.accessibilityPermission {
        case .granted:
            return .green
        case .denied:
            return .orange
        default:
            return .blue
        }
    }
}

// MARK: - Ready Step Content

struct ReadyStepContent: View {
    @ObservedObject private var settings = SettingsManager.shared

    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 20)

            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 80))
                .foregroundStyle(.green)

            Text("You're All Set!")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("Here's how to use WhisperOnDevice:")
                .font(.body)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 16) {
                HotkeyHint(
                    icon: "keyboard",
                    title: "Push-to-Talk",
                    description: "Press and hold \(settings.hotkey.displayString) to record, release to transcribe"
                )

                HotkeyHint(
                    icon: "globe",
                    title: "Language",
                    description: "Speaking in \(settings.selectedLanguage.displayName)"
                )

                HotkeyHint(
                    icon: "menubar.rectangle",
                    title: "Menu Bar",
                    description: "Click the menu bar icon for quick access"
                )

                HotkeyHint(
                    icon: "gear",
                    title: "Settings",
                    description: "Customize hotkeys, language, and more"
                )
            }
            .padding(.horizontal, 20)

            Spacer(minLength: 20)
        }
    }
}

struct HotkeyHint: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(.tint)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .fontWeight(.medium)
                Text(description)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

#Preview {
    OnboardingView(
        permissionsManager: PermissionsManager(),
        isPresented: .constant(true)
    )
}
