//
//  ContentView.swift
//  tesseract
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    let container: DependencyContainer
    @Binding var selectedNavigation: NavigationItem?

    // Sidebar closed by default
    @State private var columnVisibility: NavigationSplitViewVisibility = .detailOnly

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            SidebarView(selection: $selectedNavigation)
                .navigationDestination(for: NavigationItem.self) { page in
                    injectedDestinationView(for: page)
                }
        } detail: {
            if let selected = selectedNavigation {
                injectedDestinationView(for: selected)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .scrollEdgeEffectStyle(.soft, for: .top)
            } else {
                injectedDestinationView(for: .dictation)
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

    @MainActor
    @ViewBuilder
    private func injectedDestinationView(for page: NavigationItem) -> some View {
        switch page {
        case .dictation:
            DictationContentView()
                .injectDictationDependencies(from: container)
        case .speech:
            SpeechContentView()
                .injectSpeechDependencies(from: container)
        case .agent:
            AgentContentView()
                .injectAgentDependencies(from: container)
                .environment(container.speechCoordinator)
                .environment(container.transcriptionEngine)
                .environmentObject(container.modelDownloadManager)
        case .scheduled:
            ScheduledTasksView()
                .injectAgentDependencies(from: container)
        case .image:
            ImageGenContentView()
                .injectModelDependencies(from: container)
        case .zimage:
            ZImageGenContentView()
                .injectModelDependencies(from: container)
        case .general:
            GeneralSettingsSection()
        case .server:
            ServerSettingsView()
                .injectCoreDependencies(from: container)
        case .model:
            ModelsPageView()
                .environmentObject(container)
                .environmentObject(container.modelDownloadManager)
        case .recording:
            RecordingSettingsSection()
                .environmentObject(container)
        }
    }
}
