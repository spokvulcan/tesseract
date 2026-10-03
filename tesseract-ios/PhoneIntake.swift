//
//  PhoneIntake.swift
//  tesseract-ios
//
//  Texts arriving from outside the app (#515): a file opened from Files or
//  another app, and what the share extension left in the inbox. Each becomes
//  a Library text, and the newest opens in the Reader.
//

import Foundation
import Observation

@Observable @MainActor
final class PhoneIntake {
    /// The Library's navigation: the texts open on the stack.
    var path: [UUID] = []
    /// Why the last file couldn't be read, while it shows.
    var failure: String?

    @ObservationIgnored private let library: ReaderLibrary

    init(library: ReaderLibrary) {
        self.library = library
    }

    /// Adds the file at `url` and opens it.
    func open(file url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let incoming = try TextIntake.text(fromFile: url)
            show(library.add(incoming.text, title: incoming.title).id)
        } catch TextIntake.Failure.noText {
            failure =
                "“\(url.lastPathComponent)” has no text to read. A scanned PDF holds only pictures of its pages."
        } catch {
            failure =
                "Tesseract can't read “\(url.lastPathComponent)”. It reads text, Markdown and PDF files."
        }
    }

    /// Adds what the share extension left, and opens the newest of it.
    func takeInShared() {
        guard let inbox = LibraryInbox.shared(), let newest = library.takeIn(from: inbox).last
        else { return }
        show(newest.id)
    }

    private func show(_ id: UUID) {
        path = [id]
    }
}
