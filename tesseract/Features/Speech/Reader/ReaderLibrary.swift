//
//  ReaderLibrary.swift
//  tesseract
//
//  The **Library** (#515): every text the owner has added on the phone,
//  newest first, each with its own **Bookmark**. The Reader opens one text of
//  it at a time. Each text keeps the Mac Reader's own files, a plain text
//  file and its bookmark beside it, in a folder of its own; one small index
//  lists them, so the list never reads a text to show it.
//

import Foundation
import Observation
import TesseractSpeech

@Observable @MainActor
final class ReaderLibrary {
    /// One text of the Library, as its list shows it.
    struct Entry: Codable, Identifiable, Equatable, Sendable {
        let id: UUID
        var title: String
        let added: Date
        /// UTF-16 length of the text.
        let length: Int
        /// The voice's language for it (a `TTSLanguage` raw value), found
        /// when it was added; nil when there was nothing to judge by.
        let language: String?
    }

    /// Newest first.
    private(set) var entries: [Entry] = []
    /// How far each text has been read, 0 to 1, as last loaded.
    private(set) var progress: [UUID: Double] = [:]
    /// Nothing was on disk when this opened: a first launch.
    let isNew: Bool

    @ObservationIgnored private let directory: URL
    @ObservationIgnored private var indexURL: URL {
        directory.appendingPathComponent("library.json")
    }

    /// `directory`: defaults to the app-support home the other small stores
    /// use (a scratch folder under a test runner, ADR-0073).
    init(directory: URL? = nil) {
        self.directory =
            directory
            ?? StorageEnvironment.applicationSupport
            .appendingPathComponent("Tesseract", isDirectory: true)
            .appendingPathComponent("Library", isDirectory: true)
        let data = try? Data(contentsOf: self.directory.appendingPathComponent("library.json"))
        isNew = data == nil
        if let data, let saved = try? JSONDecoder().decode([Entry].self, from: data) {
            entries = saved.sorted { $0.added > $1.added }
        }
        refreshProgress()
    }

    /// Adds `text` as the newest entry and returns it. Without a title, its
    /// first line names it.
    @discardableResult
    func add(_ text: String, title: String? = nil, added: Date = .now) -> Entry {
        let entry = Entry(
            id: UUID(), title: title.flatMap(Self.cleanTitle) ?? Self.title(for: text),
            added: added, length: (text as NSString).length,
            language: TTSLanguage.detected(in: text)?.rawValue)
        store(for: entry.id).save(text: text)
        entries.insert(entry, at: 0)
        entries.sort { $0.added > $1.added }
        progress[entry.id] = 0
        saveIndex()
        return entry
    }

    func remove(_ id: UUID) {
        entries.removeAll { $0.id == id }
        progress[id] = nil
        try? FileManager.default.removeItem(at: folder(for: id))
        saveIndex()
    }

    func rename(_ id: UUID, to title: String) {
        guard let index = entries.firstIndex(where: { $0.id == id }),
            let clean = Self.cleanTitle(title)
        else { return }
        entries[index].title = clean
        saveIndex()
    }

    func entry(_ id: UUID) -> Entry? {
        entries.first { $0.id == id }
    }

    /// The text's own files: what a Reader opens it from.
    func store(for id: UUID) -> ReaderDocumentStore {
        ReaderDocumentStore(directory: folder(for: id))
    }

    /// Re-reads every text's bookmark, after a reading moved one.
    func refreshProgress() {
        var next: [UUID: Double] = [:]
        for entry in entries {
            guard let bookmark = store(for: entry.id).loadBookmark(), bookmark.length > 0,
                bookmark.length == entry.length
            else {
                next[entry.id] = 0
                continue
            }
            next[entry.id] = Double(bookmark.offset) / Double(bookmark.length)
        }
        if next != progress { progress = next }
    }

    // MARK: - Titles

    /// The first line with words in it, cut to a phrase's length.
    static func title(for text: String) -> String {
        let line =
            text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty } ?? ""
        return cleanTitle(line) ?? "Untitled"
    }

    private static let titleLimit = 80

    /// Trimmed, on one line, and at most `titleLimit` characters, cut at a
    /// word. Nil when nothing is left.
    private static func cleanTitle(_ raw: String) -> String? {
        let words = raw.split(whereSeparator: \.separatesWords)
        guard !words.isEmpty else { return nil }
        var title = ""
        for word in words {
            let next = title.isEmpty ? String(word) : title + " " + word
            if next.count > titleLimit {
                if title.isEmpty { return String(next.prefix(titleLimit)) + "…" }
                return title + "…"
            }
            title = next
        }
        return title
    }

    // MARK: - Saving

    private func folder(for id: UUID) -> URL {
        directory.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    private func saveIndex() {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        guard let data = try? JSONEncoder().encode(entries) else { return }
        try? data.write(to: indexURL, options: .atomic)
    }
}
