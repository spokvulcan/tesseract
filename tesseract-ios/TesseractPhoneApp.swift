//
//  TesseractPhoneApp.swift
//  tesseract-ios
//
//  The iPhone app's entry. Release 1 reads text aloud (ADR-0084); until its
//  Reader lands (#515, slice 5) the app shows a placeholder, and the target
//  exists to keep the shared speech code building for iOS.
//

import SwiftUI

@main
struct TesseractPhoneApp: App {
    var body: some Scene {
        WindowGroup {
            PlaceholderView()
        }
    }
}

/// The screen the app shows until the Reader exists.
struct PlaceholderView: View {
    var body: some View {
        ContentUnavailableView(
            "Tesseract",
            systemImage: "speaker.wave.2",
            description: Text("Read-aloud is on its way.")
        )
    }
}
