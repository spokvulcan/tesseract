//
//  ContentView.swift
//  tesseract
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject var coordinator: DictationCoordinator
    @ObservedObject var transcriptionEngine: TranscriptionEngine
    @ObservedObject var history: TranscriptionHistory
    @ObservedObject var permissionsManager: PermissionsManager
    @ObservedObject var audioCapture: AudioCaptureEngine
    @ObservedObject var speechCoordinator: SpeechCoordinator
    @ObservedObject var speechEngine: SpeechEngine
    @ObservedObject var imageGenEngine: ImageGenEngine
    @ObservedObject var zimageGenEngine: ZImageGenEngine

    @Binding var selectedNavigation: NavigationItem?

    // Sidebar closed by default
    @State private var columnVisibility: NavigationSplitViewVisibility = .detailOnly

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            SidebarView(selection: $selectedNavigation)
                .navigationDestination(for: NavigationItem.self) { page in
                    page.viewForPage(
                        coordinator: coordinator,
                        transcriptionEngine: transcriptionEngine,
                        history: history,
                        permissionsManager: permissionsManager,
                        audioCapture: audioCapture,
                        speechCoordinator: speechCoordinator,
                        speechEngine: speechEngine,
                        imageGenEngine: imageGenEngine,
                        zimageGenEngine: zimageGenEngine
                    )
                }
        } detail: {
            if let selected = selectedNavigation {
                selected.viewForPage(
                    coordinator: coordinator,
                    transcriptionEngine: transcriptionEngine,
                    history: history,
                    permissionsManager: permissionsManager,
                    audioCapture: audioCapture,
                    speechCoordinator: speechCoordinator,
                    speechEngine: speechEngine,
                    imageGenEngine: imageGenEngine,
                    zimageGenEngine: zimageGenEngine
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .scrollEdgeEffectStyle(.soft, for: .top)
            } else {
                NavigationItem.dictation.viewForPage(
                    coordinator: coordinator,
                    transcriptionEngine: transcriptionEngine,
                    history: history,
                    permissionsManager: permissionsManager,
                    audioCapture: audioCapture,
                    speechCoordinator: speechCoordinator,
                    speechEngine: speechEngine,
                    imageGenEngine: imageGenEngine,
                    zimageGenEngine: zimageGenEngine
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .scrollEdgeEffectStyle(.soft, for: .top)
            }
        }
        .navigationSplitViewStyle(.balanced)
        .scrollEdgeEffectStyle(.soft, for: .top)
        // Silently consume paste commands to prevent system alert sound
        // when text is injected while the app window is focused
        .onPasteCommand(of: [.plainText]) { _ in }
    }
}
