//
//  ReaderDocumentStore.swift
//  tesseract
//
//  The Reader's text and its **Bookmark** on disk, so a long text survives a
//  relaunch and reading resumes where it stopped. The text is a plain file
//  (a whole book is too big for defaults); the bookmark sits beside it.
//

import Foundation

nonisolated struct ReaderDocumentStore: Sendable {
    let directory: URL

    /// `directory`: defaults to the app-support home the other small stores
    /// use (a scratch folder under a test runner, ADR-0073).
    init(directory: URL? = nil) {
        self.directory =
            directory
            ?? StorageEnvironment.applicationSupport
            .appendingPathComponent("Tesseract Agent", isDirectory: true)
            .appendingPathComponent("Speech", isDirectory: true)
    }

    private var textURL: URL { directory.appendingPathComponent("reader.txt") }
    private var bookmarkURL: URL { directory.appendingPathComponent("reader-bookmark.json") }

    private struct Bookmark: Codable {
        let offset: Int
        /// The text's length when saved: a bookmark past it is stale.
        let length: Int
    }

    /// The saved text and bookmark; empty text and 0 when there is none.
    func load() -> (text: String, bookmark: Int) {
        let text = (try? String(contentsOf: textURL, encoding: .utf8)) ?? ""
        guard let data = try? Data(contentsOf: bookmarkURL),
            let bookmark = try? JSONDecoder().decode(Bookmark.self, from: data),
            bookmark.length == (text as NSString).length
        else { return (text, 0) }
        return (text, min(max(bookmark.offset, 0), bookmark.length))
    }

    func save(text: String) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? text.write(to: textURL, atomically: true, encoding: .utf8)
    }

    func save(bookmark: Int, length: Int) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        guard let data = try? JSONEncoder().encode(Bookmark(offset: bookmark, length: length))
        else {
            return
        }
        try? data.write(to: bookmarkURL, options: .atomic)
    }
}
