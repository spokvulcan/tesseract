//
//  OnboardingView.swift
//  whisper-on-device
//

import SwiftUI

struct OnboardingView: View {
    @ObservedObject var permissionsManager: PermissionsManager
    @ObservedObject var modelManager: ModelManager
    @Binding var isPresented: Bool

    @State private var currentStep = 0

    private let totalSteps = 5

    var body: some View {
        VStack(spacing: 0) {
            // Progress indicator
            ProgressView(value: Double(currentStep + 1), total: Double(totalSteps))
                .padding()

            // Content
            TabView(selection: $currentStep) {
                WelcomeStep(onContinue: nextStep)
                    .tag(0)

                MicrophonePermissionStep(
                    permissionsManager: permissionsManager,
                    onContinue: nextStep
                )
                .tag(1)

                AccessibilityPermissionStep(
                    permissionsManager: permissionsManager,
                    onContinue: nextStep
                )
                .tag(2)

                ModelDownloadStep(
                    modelManager: modelManager,
                    onContinue: nextStep,
                    onSkip: nextStep
                )
                .tag(3)

                ReadyStep(onComplete: complete)
                    .tag(4)
            }
            .tabViewStyle(.automatic)
        }
        .frame(width: 500, height: 400)
    }

    private func nextStep() {
        withAnimation {
            currentStep = min(currentStep + 1, totalSteps - 1)
        }
    }

    private func complete() {
        SettingsManager.shared.hasCompletedOnboarding = true
        isPresented = false
    }
}

// MARK: - Welcome Step

struct WelcomeStep: View {
    let onContinue: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 80))
                .foregroundStyle(.tint)

            Text("Welcome to WhisperOnDevice")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("A privacy-focused voice-to-text app that runs entirely on your device.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Spacer()

            Button("Get Started") {
                onContinue()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Spacer()
        }
        .padding()
    }
}

// MARK: - Microphone Permission Step

struct MicrophonePermissionStep: View {
    @ObservedObject var permissionsManager: PermissionsManager
    let onContinue: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

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
                .padding(.horizontal, 40)

            Spacer()

            switch permissionsManager.microphonePermission {
            case .granted:
                Label("Microphone access granted", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)

                Button("Continue") {
                    onContinue()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

            case .denied:
                Label("Microphone access denied", systemImage: "xmark.circle.fill")
                    .foregroundStyle(.red)

                Button("Open System Settings") {
                    permissionsManager.openSystemPreferences(for: "microphone")
                }
                .buttonStyle(.bordered)

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

            Spacer()
        }
        .padding()
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

// MARK: - Accessibility Permission Step

struct AccessibilityPermissionStep: View {
    @ObservedObject var permissionsManager: PermissionsManager
    let onContinue: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

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
                .padding(.horizontal, 40)

            Spacer()

            switch permissionsManager.accessibilityPermission {
            case .granted:
                Label("Accessibility access granted", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)

                Button("Continue") {
                    onContinue()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

            case .denied, .unknown:
                VStack(spacing: 12) {
                    Button("Grant Accessibility Access") {
                        permissionsManager.requestAccessibilityPermission()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)

                    Button("Skip for now") {
                        onContinue()
                    }
                    .buttonStyle(.bordered)
                    .foregroundStyle(.secondary)

                    Text("Without Accessibility access, the hotkey may type characters while recording.")
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
                Label("Accessibility access restricted", systemImage: "xmark.circle.fill")
                    .foregroundStyle(.red)

                Button("Skip for now") {
                    onContinue()
                }
                .buttonStyle(.bordered)
            }

            Spacer()
        }
        .padding()
        .onAppear {
            permissionsManager.checkAccessibilityPermission()
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

// MARK: - Model Download Step

struct ModelDownloadStep: View {
    @EnvironmentObject private var container: DependencyContainer
    @ObservedObject var modelManager: ModelManager
    let onContinue: () -> Void
    let onSkip: () -> Void

    @State private var selectedModel: WhisperModel = .base
    @State private var isLoadingModel = false

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "brain.head.profile")
                .font(.system(size: 60))
                .foregroundStyle(.tint)

            Text("Download Model")
                .font(.title)
                .fontWeight(.bold)

            Text("Choose a transcription model. Larger models are more accurate but use more memory.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            // Model picker
            Picker("Model", selection: $selectedModel) {
                ForEach(WhisperModel.allCases) { model in
                    Text("\(model.displayName) (\(String(format: "%.1f", model.sizeGB)) GB)")
                        .tag(model)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            Text(selectedModel.description)
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            if modelManager.isDownloading[selectedModel] == true {
                VStack(spacing: 8) {
                    ProgressView(value: modelManager.downloadProgress[selectedModel] ?? 0)
                        .frame(width: 200)

                    Text("Downloading... \(Int((modelManager.downloadProgress[selectedModel] ?? 0) * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Button("Cancel") {
                        modelManager.cancelDownload(selectedModel)
                    }
                    .buttonStyle(.bordered)
                }
            } else if isLoadingModel {
                VStack(spacing: 8) {
                    ProgressView()
                    Text("Loading model...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if modelManager.isModelDownloaded(selectedModel) {
                Label("Model downloaded", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)

                Button("Continue") {
                    // Save selection and load model before continuing
                    SettingsManager.shared.whisperModel = selectedModel
                    loadModelAndContinue()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            } else {
                HStack(spacing: 16) {
                    Button("Skip for now") {
                        onSkip()
                    }
                    .buttonStyle(.bordered)

                    Button("Download") {
                        Task {
                            try? await modelManager.downloadModel(selectedModel)
                            // Save selection after successful download
                            SettingsManager.shared.whisperModel = selectedModel
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
            }

            Spacer()
        }
        .padding()
    }

    private func loadModelAndContinue() {
        isLoadingModel = true
        Task {
            let modelPath = container.modelManager.getLocalModelPath(selectedModel)
            try? await container.transcriptionEngine.loadModel(selectedModel, modelPath: modelPath)
            isLoadingModel = false
            onContinue()
        }
    }
}

// MARK: - Ready Step

struct ReadyStep: View {
    let onComplete: () -> Void
    @ObservedObject private var settings = SettingsManager.shared

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 80))
                .foregroundStyle(.green)

            Text("You're All Set!")
                .font(.largeTitle)
                .fontWeight(.bold)

            VStack(alignment: .leading, spacing: 12) {
                HotkeyHint(
                    icon: "keyboard",
                    title: "Push-to-Talk",
                    description: "Press and hold \(settings.hotkey.displayString) to record, release to transcribe"
                )

                HotkeyHint(
                    icon: "menubar.rectangle",
                    title: "Menu Bar",
                    description: "Click the menu bar icon for quick access"
                )

                HotkeyHint(
                    icon: "gear",
                    title: "Settings",
                    description: "Customize hotkeys, models, and more"
                )
            }
            .padding(.horizontal, 40)

            Spacer()

            Button("Start Using WhisperOnDevice") {
                onComplete()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Spacer()
        }
        .padding()
    }
}

struct HotkeyHint: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .frame(width: 24)
                .foregroundStyle(.secondary)

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

#Preview {
    OnboardingView(
        permissionsManager: PermissionsManager(),
        modelManager: ModelManager(),
        isPresented: .constant(true)
    )
}
