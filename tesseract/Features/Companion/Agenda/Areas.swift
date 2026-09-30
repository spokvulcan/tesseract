//
//  Areas.swift
//  tesseract
//
//  Areas are the parts of the owner's life a day spans — Work, a side
//  project, Health, Life. Each Area is a Reminders list, mapped once in
//  Settings; one list is the Inbox, where captures without a home land.
//  Until the owner maps anything, every list is an Area under its own name
//  and the default list is the Inbox.
//

import Foundation

nonisolated struct Area: Sendable, Equatable, Hashable, Identifiable {
    /// The Reminders list that stores this Area.
    let id: String
    var name: String
    var colorHex: String?
}

/// The owner's Area mapping, persisted as JSON in Settings.
nonisolated struct AreaMap: Sendable, Equatable, Codable {

    struct Entry: Sendable, Equatable, Codable {
        var listID: String
        var name: String
    }

    /// Lists that count as Areas, with the owner's names for them. Empty means
    /// "not configured": every list is an Area under its own title.
    var entries: [Entry] = []
    /// The Inbox list. nil means the default Reminders list.
    var inboxListID: String?

    static let empty = AreaMap()

    init(entries: [Entry] = [], inboxListID: String? = nil) {
        self.entries = entries
        self.inboxListID = inboxListID
    }

    init(json: String) {
        self = (try? JSONDecoder().decode(AreaMap.self, from: Data(json.utf8))) ?? .empty
    }

    var json: String {
        (try? JSONEncoder().encode(self)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }

    /// The Areas among the current lists, in the lists' order. Entries whose
    /// list is gone are skipped.
    func areas(in lists: [AgendaList]) -> [Area] {
        if entries.isEmpty {
            return lists.map { Area(id: $0.id, name: $0.title, colorHex: $0.colorHex) }
        }
        return lists.compactMap { list in
            entries.first { $0.listID == list.id }.map {
                Area(
                    id: list.id, name: $0.name.isEmpty ? list.title : $0.name,
                    colorHex: list.colorHex)
            }
        }
    }

    /// The Inbox list, falling back to the default list.
    func inbox(in lists: [AgendaList]) -> AgendaList? {
        if let inboxListID, let list = lists.first(where: { $0.id == inboxListID }) { return list }
        return lists.first(where: \.isDefault) ?? lists.first
    }

    /// The Area a name refers to: exact (case-insensitive) first, then a
    /// unique prefix ("tess" → "Tesseract"). nil when nothing matches or the
    /// prefix is ambiguous.
    func area(named name: String, in lists: [AgendaList]) -> Area? {
        let wanted = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !wanted.isEmpty else { return nil }
        let all = areas(in: lists)
        if let exact = all.first(where: { $0.name.lowercased() == wanted }) { return exact }
        let prefixed = all.filter { $0.name.lowercased().hasPrefix(wanted) }
        return prefixed.count == 1 ? prefixed[0] : nil
    }

    /// The Area a reminder's list belongs to, if that list is an Area.
    func area(forListID listID: String, in lists: [AgendaList]) -> Area? {
        areas(in: lists).first { $0.id == listID }
    }
}
