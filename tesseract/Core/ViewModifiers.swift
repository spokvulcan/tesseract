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
    }

    // MARK: - Scoped Injection

    /// Core services used across multiple features: settings and permissions.
    @MainActor
    func injectCoreDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.settingsManager)
            .environmentObject(container.permissionsManager)
    }

    /// Dictation feature: coordinator, transcription engine/history, audio capture.
    @MainActor
    func injectDictationDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.dictationCoordinator)
            .environment(container.transcriptionEngine)
            .environment(container.transcriptionHistory)
            .environment(container.audioCaptureEngine)
    }

    /// Speech/TTS feature: coordinator and engine.
    @MainActor
    func injectSpeechDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.speechCoordinator)
            .environment(container.speechEngine)
    }

    /// Agent feature: coordinator, engine, conversation store, scheduling.
    @MainActor
    func injectAgentDependencies(from container: DependencyContainer) -> some View {
        self
            .environment(container.agentCoordinator)
            .environment(container.agentEngine)
            .environment(container.schedulingService)
            .environmentObject(container.agentConversationStore)
            .environmentObject(container.scheduledTaskStore)
    }

    /// Model management, image generation, and inference arbitration.
    @MainActor
    func injectModelDependencies(from container: DependencyContainer) -> some View {
        self
            .environmentObject(container.modelDownloadManager)
            .environmentObject(container.imageGenEngine)
            .environmentObject(container.zimageGenEngine)
            .environment(container.inferenceArbiter)
    }
}
