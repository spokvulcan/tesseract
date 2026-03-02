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
        return self.injectDependencies(from: container)
    }
}
#endif
