//
//  ServerPromptCacheView.swift
//  tesseract
//

import SwiftUI

struct ServerPromptCacheView: View {
    var body: some View {
        ContentUnavailableView(
            "Prompt Cache",
            systemImage: "tray.2",
            description: Text("Prefix cache observability coming soon.")
        )
        .navigationTitle("Prompt Cache")
    }
}
