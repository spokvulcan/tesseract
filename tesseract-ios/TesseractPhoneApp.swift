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

    var body: some Scene {
        WindowGroup {
            LibraryScreen()
                .environment(container.settings)
                .environment(container.library)
                .environment(container.reading)
                .environment(container.coordinator)
                .environment(container.engine)
        }
    }
}
