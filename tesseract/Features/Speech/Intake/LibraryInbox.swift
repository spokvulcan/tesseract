//
//  LibraryInbox.swift
//  tesseract
//
//  Where the share extension leaves texts for the Library (#515). An
//  extension can't reach the app's own files, so it drops each text, as one
//  JSON file, in a folder of the app group both can reach; the app takes
//  them into the Library when it next comes to the front. The extension never
//  writes the Library itself, so the two never write one index at once.
//

import Foundation

nonisolated struct LibraryInbox: Sendable {
    /// The app group the app and its share extension share.
    static let appGroup = "group.app.tesseract.agent"

    let directory: URL

    /// The app group's inbox; nil when the app group isn't there (an
    /// unsigned build).
    static func shared() -> LibraryInbox? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroup)
            .map { LibraryInbox(directory: $0.appendingPathComponent("Inbox", isDirectory: true)) }
    }

    private struct Dropped: Codable {
        let text: IncomingText
        let dropped: Date
    }

    /// Leaves `text` for the app.
    func drop(_ text: IncomingText, at date: Date = .now) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(Dropped(text: text, dropped: date))
        try data.write(
            to: directory.appendingPathComponent("\(UUID().uuidString).json"), options: .atomic)
    }

    /// The texts waiting, oldest first, with when each was dropped and the
    /// file to remove once it is taken.
    func pending() -> [(text: IncomingText, dropped: Date, file: URL)] {
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "json" }
            .compactMap { file in
                guard let data = try? Data(contentsOf: file),
                    let dropped = try? JSONDecoder().decode(Dropped.self, from: data)
                else { return nil }
                return (dropped.text, dropped.dropped, file)
            }
            .sorted { $0.dropped < $1.dropped }
    }

    func remove(_ file: URL) {
        try? FileManager.default.removeItem(at: file)
    }
}
