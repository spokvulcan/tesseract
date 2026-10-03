//
//  ViewModifiers.swift
//  tesseract
//

import SwiftUI

// MARK: - Card Background

struct CardBackgroundModifier: ViewModifier {
    var cornerRadius: CGFloat = Theme.Radius.medium
    var material: Material = .thickMaterial

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(material)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(.white.opacity(0.1), lineWidth: 0.5)
            )
    }
}

extension View {
    func cardBackground(
        cornerRadius: CGFloat = Theme.Radius.medium,
        material: Material = .thickMaterial
    ) -> some View {
        modifier(CardBackgroundModifier(cornerRadius: cornerRadius, material: material))
    }
}

// MARK: - Bubble Background

struct BubbleBackgroundModifier: ViewModifier {
    var style: AnyShapeStyle
    var cornerRadius: CGFloat = Theme.Radius.medium

    func body(content: Content) -> some View {
        content
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
            .background(style)
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
    }
}

extension View {
    func bubbleBackground(
        _ style: AnyShapeStyle = AnyShapeStyle(.fill.quaternary),
        cornerRadius: CGFloat = Theme.Radius.medium
    ) -> some View {
        modifier(BubbleBackgroundModifier(style: style, cornerRadius: cornerRadius))
    }
}

// MARK: - Dependency Injection

extension View {
    /// Injects all dependencies from the container, organized by feature scope.
    @MainActor
    func injectDependencies(from container: DependencyContainer) -> some View {
        self
            .environmentObject(container)
            .injectCoreDependencies(from: container)
            .injectDictationDependencies(from: container)
            .injectSpeechDependencies(from: container)
            .injectAgentDependencies(from: container)
            .injectModelDependencies(from: container)
            .injectServerDependencies(from: container)
    }

    // MARK: - Scoped Injection

    /// Core services used across multiple features: settings and permissions.
    @MainActor
    func injectCoreDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.settingsManager)
            .environment(container.agentBrowser)
            .environment(container.mcpClientManager)
            .environmentObject(container.permissionsManager)
    }

    /// Dictation feature: coordinator, transcription engine/history, audio
    /// capture, and the **Lens** for fixing a take from the Dictation page.
    @MainActor
    func injectDictationDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.dictationCoordinator)
            .environment(container.transcriptionEngine)
            .environment(container.transcriptionHistory)
            .environment(container.correctionPairStore)
            .environment(container.learnedWordStore)
            .environment(container.audioCaptureEngine)
            .environment(
                \.fixInLens,
                FixInLensAction { [weak container] take in
                    container?.dictationLens.openFromPage(take) ?? false
                })
    }

    /// Speech/TTS feature: coordinator and engine presenter.
    @MainActor
    func injectSpeechDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.speechCoordinator)
            .environment(container.speechEnginePresenter)
            .environment(container.speechReader)
            .environment(container.voiceLibrary)
    }

    /// Agent feature: the Chat Session, its leaf controllers, engine, store.
    @MainActor
    func injectAgentDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.chatSession)
            .environment(container.composerDraft)
            .environment(container.visionAvailability)
            .environment(container.commandPalette)
            .environment(container.skillPills)
            .environment(container.agentVoiceInput)
            .environment(container.companionVoiceSession)
            .environment(container.companionPresence)
            .environment(container.agentSystemPromptInspector)
            .environment(container.agentEngine)
            .environment(container.appshotController)
            .environmentObject(container.agentConversationStore)
    }

    /// The Companion's day: the Today page and its Day Thread chat (whose
    /// transcript rows need the agent engine, settings and the composer draft).
    @MainActor
    func injectCompanionDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.agenda)
            .environment(container.companionRuntime)
            .environment(container.dayThread)
            .environment(container.captureService)
            .environment(container.profileStore)
            .environment(container.settingsManager)
            .environment(container.agentEngine)
            .environment(container.composerDraft)
    }

    /// Model management and inference arbitration.
    @MainActor
    func injectModelDependencies(from container: DependencyContainer) -> some View {
        self
            .environmentObject(container.modelDownloadManager)
            .environment(container.inferenceArbiter)
    }

    /// Server API: activity log + observability state + HTTP listener lifecycle.
    @MainActor
    func injectServerDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.serverGenerationLog)
            .environment(container.promptCacheTelemetryStore)
            .environment(container.httpServer)
    }
}
