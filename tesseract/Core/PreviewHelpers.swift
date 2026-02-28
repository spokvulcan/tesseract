//
//  PreviewHelpers.swift
//  tesseract
//

#if DEBUG
import SwiftUI

extension View {
    /// Injects all required environment objects for previewing any view in the app.
    func previewEnvironment() -> some View {
        let container = DependencyContainer()
        return self
            .environmentObject(container)
            .environmentObject(container.dictationCoordinator)
            .environmentObject(container.transcriptionEngine)
            .environmentObject(container.transcriptionHistory)
            .environmentObject(container.permissionsManager)
            .environmentObject(container.audioCaptureEngine)
            .environmentObject(container.speechCoordinator)
            .environmentObject(container.speechEngine)
            .environmentObject(container.agentCoordinator)
            .environmentObject(container.agentEngine)
            .environmentObject(container.agentConversationStore)
            .environmentObject(container.imageGenEngine)
            .environmentObject(container.zimageGenEngine)
            .environmentObject(container.modelDownloadManager)
    }
}
#endif
