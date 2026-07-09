//
//  SidebarView.swift
//  tesseract
//

import SwiftUI

struct SidebarView: View {
    @Binding var selection: NavigationItem?

    var body: some View {
        List(selection: $selection) {
            Section {
                ForEach(NavigationItem.mainPages) { page in
                    NavigationLink(value: page) {
                        Label(page.name, systemImage: page.symbolName)
                    }
                }
            }

            Section("Server") {
                ForEach(NavigationItem.serverPages) { page in
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
