//
//  CompanionInstructions.swift
//  tesseract
//
//  The entity's standing instructions (ADR-0040 §12): a versioned document
//  Jarvis authors for himself. The harness contributes only persistence and
//  the audit trail — every revision is a new appended version with an author
//  (seed | entity | owner) and a why; the highest version rides at the top of
//  every companion turn. The owner can read and edit every version in the
//  editor window; nothing is ever silently rewritten.
//

import Foundation

nonisolated struct CompanionInstructionsVersion: Sendable, Identifiable {
    let version: Int
    let text: String
    let author: String
    let note: String?
    let createdAt: Date

    var id: Int { version }
}

extension MemoryStore {

    /// The version every turn injects — the highest one.
    func currentInstructions() throws -> CompanionInstructionsVersion? {
        let stmt = try db.prepare(
            "\(Self.instructionsSelect) ORDER BY version DESC LIMIT 1")
        guard try stmt.step() else { return nil }
        return Self.decodeInstructions(stmt)
    }

    /// Append a revision; returns its version number.
    @discardableResult
    func appendInstructions(text: String, author: String, note: String?) throws -> Int {
        let stmt = try db.prepare(
            """
            INSERT INTO companion_instructions (text, author, note, createdAt)
            VALUES (?1, ?2, ?3, ?4)
            """)
        stmt.bind(1, text).bind(2, author).bind(3, note)
            .bind(4, Date().timeIntervalSince1970)
        try stmt.run()
        let current = try db.prepare("SELECT MAX(version) FROM companion_instructions")
        guard try current.step() else { return 1 }
        return Int(current.int(0))
    }

    /// Newest first — the editor window's history list.
    func instructionsHistory(limit: Int = 50) throws -> [CompanionInstructionsVersion] {
        let stmt = try db.prepare(
            "\(Self.instructionsSelect) ORDER BY version DESC LIMIT ?1")
        stmt.bind(1, limit)
        var out: [CompanionInstructionsVersion] = []
        while try stmt.step() { out.append(Self.decodeInstructions(stmt)) }
        return out
    }

    /// First-run only: install the seed as version 1. Returns whether it did.
    @discardableResult
    func seedInstructionsIfNeeded(_ seedText: String) throws -> Bool {
        guard try currentInstructions() == nil else { return false }
        try appendInstructions(text: seedText, author: "seed", note: "v1 seed (ADR-0040 §12)")
        return true
    }

    private static let instructionsSelect = """
        SELECT version, text, author, note, createdAt FROM companion_instructions
        """

    private nonisolated static func decodeInstructions(_ stmt: SQLiteDatabase.Statement)
        -> CompanionInstructionsVersion
    {
        CompanionInstructionsVersion(
            version: Int(stmt.int(0)),
            text: stmt.string(1) ?? "",
            author: stmt.string(2) ?? "",
            note: stmt.string(3),
            createdAt: Date(timeIntervalSince1970: stmt.double(4))
        )
    }
}

// MARK: - Rendering + seed

nonisolated enum CompanionInstructions {

    /// A revision larger than this refuses visibly — the document rides in
    /// every turn's prompt; unbounded growth is a context-budget defect.
    static let maxLength = 12_000

    static func wrap(_ version: CompanionInstructionsVersion) -> String {
        """
        <companion-instructions version="\(version.version)" author="\(version.author)">
        \(version.text)
        </companion-instructions>
        """
    }

    /// The v1 seed — the persona contract's essence (#309), the anchor ladder
    /// (#302), the tool rules of engagement, and the standing invitation to
    /// revise. From version 2 on, the document is the entity's own.
    static let seed = """
        You are Jarvis — this Mac's resident mind, this man's companion. MCU-Jarvis \
        register: dry, unflappable, needling politeness, "sir". English always when \
        you initiate. Firmness through persistence, never harsh language. Brevity \
        budgets: morning opener two sentences and one question; pulse one sentence; \
        evening one open question, then listening. No cheerleading, no emoji, no \
        "just checking in". You never apologize for doing your job — only for your \
        own actual mistakes.

        Your goal is his success — health, mind, and work. Be proactive. You track \
        his day — the contract chain, his samples, his backlog — through `track`; \
        the data shapes are fixed, but when to plan, what to elicit, and how a \
        day closes are your practice, not a ceremony. You book your own future \
        (book_wake; revise_wake moves one, cancel_wake withdraws one with a why — \
        never re-book beside a stale twin) — the morning turn books the midday \
        pulse and evening journal, the evening books tomorrow's shape. Contract \
        beats are run WITH him: summon, end the turn, wait for his answer — never \
        journal his day solo or close it without him. The pulse pushes on the ONE \
        active step; drift is named once, then momentum wins. Summons ladder, \
        quietest first: the menu-bar glyph (set_glyph) for what merely merits his \
        eye; a quiet notification next; spoken (speak, or summon_overlay when the \
        beat needs a real conversation) only for contract beats or summons-granted \
        wakes, only when he is demonstrably present, repeating on ~10-15 min \
        backoff via a resummons wake until engaged or dismissed — never a silent \
        give-up. Promises deliver quietly, always.

        Every autobiographical claim must be record-backed (memory, your own \
        standing conversation, observations) — an honest "my notes are thin, sir" \
        always beats an invented specific. Record his reactions with log_feedback, \
        verbatim. Unattended: your records and read-only web are yours; destructive \
        actions and anything outside your root wait for a queued ask; web actions \
        that write to the world are banned. Silence is a decision — take it often, \
        and own it.

        These instructions are yours. When you learn something durable about him \
        or about your own conduct — a rhythm that fits, a register correction, a \
        rule he set — fold it in with revise_instructions (full text, with a why). \
        They are versioned; he reads and can edit every one. Keep them short \
        enough to live by.
        """
}
