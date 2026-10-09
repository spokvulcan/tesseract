//
//  ProfileStore.swift
//  tesseract
//
//  The owner's Profile: a small set of facts they approved, and Jarvis's
//  "Should I remember this?" proposals. Memory is small and the owner's:
//  facts come in only when the owner asks (`remember`) or approves a
//  proposal; they can read, edit and delete every one; and they ride only
//  the Day Opening — ordinary chats get none of them.
//
//  A readable JSON file under the agent root, written atomically.
//

import Foundation
import Observation

nonisolated struct ProfileFact: Sendable, Equatable, Hashable, Codable, Identifiable {
    enum Source: String, Sendable, Codable {
        /// The owner said "remember that…".
        case chat
        /// The owner kept a proposal.
        case proposal
        /// The owner wrote it on the Profile page.
        case owner
    }

    let id: String
    var text: String
    var area: String?
    var source: Source
    var createdAt: Date
    var updatedAt: Date
}

nonisolated struct FactProposal: Sendable, Equatable, Hashable, Codable, Identifiable {
    enum Status: String, Sendable, Codable {
        case open, kept, edited, dropped
    }

    let id: String
    var text: String
    var reason: String
    var source: String
    var status: Status
    var createdAt: Date
    var decidedAt: Date?
}

@Observable @MainActor
final class ProfileStore {

    private(set) var facts: [ProfileFact] = []
    private(set) var proposals: [FactProposal] = []

    @ObservationIgnored private let url: URL?
    @ObservationIgnored private let trace: CompanionTrace?
    @ObservationIgnored private let now: @MainActor () -> Date

    /// `url` nil keeps the Profile in memory (tests).
    init(url: URL?, trace: CompanionTrace? = nil, now: @escaping @MainActor () -> Date = Date.init)
    {
        self.url = url
        self.trace = trace
        self.now = now
        load()
    }

    static var productionURL: URL {
        PathSandbox.defaultRoot
            .appendingPathComponent("profile", isDirectory: true)
            .appendingPathComponent("profile.json")
    }

    var openProposals: [FactProposal] { proposals.filter { $0.status == .open } }

    // MARK: Facts

    @discardableResult
    func add(_ text: String, area: String? = nil, source: ProfileFact.Source) -> ProfileFact? {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        if let existing = facts.first(where: { $0.text.lowercased() == text.lowercased() }) {
            return existing
        }
        let fact = ProfileFact(
            id: UUID().uuidString, text: text, area: area, source: source, createdAt: now(),
            updatedAt: now())
        facts.append(fact)
        save()
        trace?.record(
            .profileChanged, fields: ["action": "added", "source": .string(source.rawValue)])
        return fact
    }

    func update(_ id: String, text: String, area: String? = nil) {
        guard let index = facts.firstIndex(where: { $0.id == id }) else { return }
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        facts[index].text = text
        facts[index].area = area
        facts[index].updatedAt = now()
        save()
        trace?.record(.profileChanged, fields: ["action": "edited"])
    }

    @discardableResult
    func delete(_ id: String) -> ProfileFact? {
        guard let index = facts.firstIndex(where: { $0.id == id }) else { return nil }
        let removed = facts.remove(at: index)
        save()
        trace?.record(.profileChanged, fields: ["action": "deleted"])
        return removed
    }

    /// The one fact a "forget" means: by its id, by its exact words, or the
    /// only fact holding every word the owner gave. Nil when that is not one
    /// fact — a shared word ("allergic to peanuts" against "allergic to
    /// penicillin") must never delete the wrong one; there is no undo.
    func factToForget(_ query: String) -> ProfileFact? {
        if let byID = facts.first(where: { $0.id == query }) { return byID }
        let words = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let exact = facts.first(where: { $0.text.lowercased() == words }) { return exact }
        let terms = RecallQuery.terms(query)
        guard !terms.isEmpty else { return nil }
        let holding = facts.filter { fact in
            let text = fact.text.lowercased()
            return terms.allSatisfy { text.contains($0) }
        }
        return holding.count == 1 ? holding[0] : nil
    }

    /// Facts whose words overlap the query, best first.
    func search(_ query: String) -> [ProfileFact] {
        let terms = RecallQuery.terms(query)
        guard !terms.isEmpty else { return facts }
        return
            facts
            .map { fact in (fact, terms.filter { fact.text.lowercased().contains($0) }.count) }
            .filter { $0.1 > 0 }
            .sorted { $0.1 > $1.1 }
            .map(\.0)
    }

    // MARK: Proposals

    /// Night Reflection's proposals, minus any the Profile or an open
    /// proposal already says.
    func propose(_ drafts: [ProposalDraft], source: String) {
        var added = 0
        for draft in drafts.prefix(3) {
            let text = draft.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            // Known, open, or already answered "Not true": never asked twice.
            let known =
                facts.contains { $0.text.lowercased() == text.lowercased() }
                || proposals.contains { $0.text.lowercased() == text.lowercased() }
            guard !known else { continue }
            proposals.append(
                FactProposal(
                    id: UUID().uuidString, text: text, reason: draft.reason, source: source,
                    status: .open, createdAt: now()))
            added += 1
            trace?.record(.factProposed, fields: ["source": .string(source)])
        }
        if added > 0 { save() }
    }

    /// Remember it, remember an edited version, or drop it ("Not true").
    func decide(_ id: String, keep: Bool, editedText: String? = nil) {
        guard let index = proposals.firstIndex(where: { $0.id == id }),
            proposals[index].status == .open
        else { return }
        let proposal = proposals[index]
        proposals[index].decidedAt = now()
        if keep {
            let text = editedText?.trimmingCharacters(in: .whitespacesAndNewlines)
            let edited = text.map { !$0.isEmpty && $0 != proposal.text } ?? false
            proposals[index].status = edited ? .edited : .kept
            add(edited ? text! : proposal.text, source: .proposal)
        } else {
            proposals[index].status = .dropped
        }
        save()
        trace?.record(
            .factDecided,
            fields: [
                "status": .string(proposals[index].status.rawValue),
                "secondsToDecide": .double(now().timeIntervalSince(proposal.createdAt)),
            ])
    }

    // MARK: Persistence

    private struct File: Codable {
        var facts: [ProfileFact]
        var proposals: [FactProposal]
    }

    private func load() {
        guard let url, let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let file = try? decoder.decode(File.self, from: data) else {
            // Kept aside, not overwritten by the next save: the facts may be
            // recoverable by hand.
            let aside = url.deletingPathExtension()
                .appendingPathExtension("unreadable-\(Int(Date().timeIntervalSince1970)).json")
            try? FileManager.default.moveItem(at: url, to: aside)
            Log.companion.error(
                "Profile file unreadable at \(url.path); moved to \(aside.lastPathComponent), starting empty"
            )
            return
        }
        facts = file.facts
        proposals = file.proposals
    }

    private func save() {
        guard let url else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(File(facts: facts, proposals: proposals)) else {
            return
        }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }
}
