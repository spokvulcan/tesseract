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
        .overlay(alignment: .top) {
            VisualEffectView(material: .hudWindow, blendingMode: .withinWindow, state: .active)
                .frame(height: 52)
                .mask(alignment: .top) {
                    LinearGradient(
                        stops: [
                            .init(color: .black, location: 0.5),
                            .init(color: .black.opacity(0.95), location: 0.7),
                            .init(color: .black.opacity(0.5), location: 0.85),
                            .init(color: .clear, location: 1.0),
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
        }
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
