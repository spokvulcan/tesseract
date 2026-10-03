//
//  TesseractPhoneApp.swift
//  tesseract-ios
//
//  The iPhone app's entry. Release 1 reads text aloud (ADR-0084): the
//  Library, and a Reader for each of its texts.
//

import SwiftUI

@main
struct TesseractPhoneApp: App {
    @State private var container = PhoneContainer()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            LibraryScreen()
                .environment(container.settings)
                .environment(container.library)
                .environment(container.reading)
                .environment(container.intake)
                .environment(container.coordinator)
                .environment(container.engine)
                .onOpenURL { container.intake.open(file: $0) }
        }
        .onChange(of: scenePhase, initial: true) { _, phase in
            // Texts shared while the app was away open on its return.
            if phase == .active { container.intake.takeInShared() }
        }
    }
}
