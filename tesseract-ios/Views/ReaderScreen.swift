//
//  ReaderScreen.swift
//  tesseract-ios
//
//  One text of the Library in the **Reader**: the text, following the voice,
//  over the transport bar.
//

import SwiftUI

struct ReaderScreen: View {
    let id: UUID
    @Environment(PhoneReading.self) private var reading
    @Environment(ReaderLibrary.self) private var library
    @Environment(PhoneSettings.self) private var settings
    @State private var reader: SpeechReader?
    @State private var missing = false

    var body: some View {
        Group {
            if let reader {
                PhoneReaderTextView(
                    reader: reader, font: font, highlightStyle: settings.readerHighlight,
                    bottomInset: 0
                )
                .ignoresSafeArea(edges: .bottom)
                .safeAreaInset(edge: .bottom) {
                    TransportBar(reader: reader)
                }
            } else if missing {
                ContentUnavailableView("This text was removed", systemImage: "text.page.slash")
            } else {
                ProgressView()
            }
        }
        .navigationTitle(library.entry(id)?.title ?? "")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: id) {
            reader = reading.open(id)
            missing = reader == nil
        }
    }

    private var font: UIFont {
        let size = CGFloat(settings.readerTextSize)
        let base = UIFont.systemFont(ofSize: size)
        guard let serif = base.fontDescriptor.withDesign(.serif) else { return base }
        return UIFont(descriptor: serif, size: size)
    }
}
