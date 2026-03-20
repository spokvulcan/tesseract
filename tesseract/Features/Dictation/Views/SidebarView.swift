//
//  SidebarView.swift
//  tesseract
//

import SwiftUI

struct SidebarView: View {
    @Binding var selection: NavigationItem?
    @Environment(SchedulingService.self) private var schedulingService

    var body: some View {
        List(selection: $selection) {
            Section {
                ForEach(NavigationItem.mainPages) { page in
                    NavigationLink(value: page) {
                        Label(page.name, systemImage: page.symbolName)
                    }
                    .badge(page == .scheduled ? schedulingService.unreadResultCount : 0)
                }
            }

            Section("Settings") {
                ForEach(NavigationItem.settingsPages) { page in
                    NavigationLink(value: page) {
                        Label(page.name, systemImage: page.symbolName)
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .frame(minWidth: 180)
    }
}

#Preview {
    @Previewable @State var selection: NavigationItem? = .dictation
    // Need to provide environment in preview, but easier to just mock it or not preview
    // In actual project, usually you'd mock it, but for now we just wrap it:
    return SidebarView(selection: $selection)
}
