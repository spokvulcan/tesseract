//
//  ContentView.swift
//  tesseract
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Binding var selectedNavigation: NavigationItem?

    // Sidebar closed by default
    @State private var columnVisibility: NavigationSplitViewVisibility = .detailOnly

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            SidebarView(selection: $selectedNavigation)
                .navigationDestination(for: NavigationItem.self) { page in
                    page.destinationView
                }
        } detail: {
            if let selected = selectedNavigation {
                selected.destinationView
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .scrollEdgeEffectStyle(.soft, for: .top)
            } else {
                NavigationItem.dictation.destinationView
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
