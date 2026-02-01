//
//  SidebarView.swift
//  whisper-on-device
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
    return SidebarView(selection: $selection)
}
