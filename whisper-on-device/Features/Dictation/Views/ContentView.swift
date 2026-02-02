//
//  ContentView.swift
//  whisper-on-device
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject var coordinator: DictationCoordinator
    @ObservedObject var transcriptionEngine: TranscriptionEngine
    @ObservedObject var history: TranscriptionHistory
    @ObservedObject var permissionsManager: PermissionsManager
    @ObservedObject var audioCapture: AudioCaptureEngine

    @Binding var selectedNavigation: NavigationItem?

    var body: some View {
        NavigationSplitView {
            SidebarView(selection: $selectedNavigation)
                .navigationDestination(for: NavigationItem.self) { page in
                    page.viewForPage(
                        coordinator: coordinator,
                        transcriptionEngine: transcriptionEngine,
                        history: history,
                        permissionsManager: permissionsManager,
                        audioCapture: audioCapture
                    )
                }
        } detail: {
            if let selected = selectedNavigation {
                selected.viewForPage(
                    coordinator: coordinator,
                    transcriptionEngine: transcriptionEngine,
                    history: history,
                    permissionsManager: permissionsManager,
                    audioCapture: audioCapture
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(.clear)
            } else {
                NavigationItem.dictation.viewForPage(
                    coordinator: coordinator,
                    transcriptionEngine: transcriptionEngine,
                    history: history,
                    permissionsManager: permissionsManager,
                    audioCapture: audioCapture
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(.clear)
            }
        }
        .navigationSplitViewStyle(.balanced)
        // Silently consume paste commands to prevent system alert sound
        // when text is injected while the app window is focused
        .onPasteCommand(of: [.plainText]) { _ in }
    }
}

#Preview {
    @Previewable @State var selection: NavigationItem? = .dictation
    let audioCapture = AudioCaptureEngine()
    return ContentView(
        coordinator: DictationCoordinator(
            audioCapture: audioCapture,
            transcriptionEngine: TranscriptionEngine(),
            textInjector: TextInjector(),
            history: TranscriptionHistory()
        ),
        transcriptionEngine: TranscriptionEngine(),
        history: TranscriptionHistory(),
        permissionsManager: PermissionsManager(),
        audioCapture: audioCapture,
        selectedNavigation: $selection
    )
}
