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
    /// Injects all required environment objects from the DependencyContainer.
    /// Use this single modifier instead of chaining multiple .environmentObject() calls.
    @MainActor
    func injectDependencies(from container: DependencyContainer) -> some View {
        self
            .environmentObject(container)
            .environmentObject(container.dictationCoordinator)
            .environmentObject(container.transcriptionEngine)
            .environmentObject(container.transcriptionHistory)
            .environmentObject(container.permissionsManager)
            .environmentObject(container.audioCaptureEngine)
            .environmentObject(container.speechCoordinator)
            .environmentObject(container.speechEngine)
            .environment(container.agentCoordinator)
            .environmentObject(container.agentEngine)
            .environmentObject(container.agentConversationStore)
            .environmentObject(container.imageGenEngine)
            .environmentObject(container.zimageGenEngine)
            .environmentObject(container.modelDownloadManager)
    }
}
