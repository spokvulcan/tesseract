//
//  TesseractPhoneApp.swift
//  tesseract-ios
//
//  The iPhone app's entry. Release 1 reads text aloud (ADR-0084): the
//  Library, and a Reader for each of its texts.
//

import SwiftUI
import UIKit

@main
struct TesseractPhoneApp: App {
    @UIApplicationDelegateAdaptor(PhoneAppDelegate.self) private var delegate
    @Environment(\.scenePhase) private var scenePhase

    private var container: PhoneContainer { delegate.container }

    var body: some Scene {
        WindowGroup {
            LibraryScreen()
                .environment(container.settings)
                .environment(container.library)
                .environment(container.reading)
                .environment(container.intake)
                .environment(container.coordinator)
                .environment(container.engine)
                .environment(container.voice)
                .environment(\.readingMeter, container.meter)
                .onOpenURL { container.intake.open(file: $0) }
        }
        .onChange(of: scenePhase, initial: true) { _, phase in
            guard phase == .active else { return }
            // Texts shared while the app was away open on its return.
            container.intake.takeInShared()
            // The neural voice prepares (or its download goes on) once the
            // app is in front, never on a background relaunch.
            container.voice.start()
        }
    }
}

/// Owns the composition root, so a relaunch in the background to finish the
/// voice's download reaches its URLSession before any scene exists.
@MainActor
final class PhoneAppDelegate: NSObject, UIApplicationDelegate {
    let container = PhoneContainer()

    func application(
        _ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        guard identifier == PhoneModelFetching.sessionIdentifier else {
            completionHandler()
            return
        }
        nonisolated(unsafe) let completion = completionHandler
        container.voice.fetching.handleEvents { completion() }
    }
}
