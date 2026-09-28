//
//  DictationLabLexicon.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  What the lab learned: terms the owner fixed, typed or accepted, each with
//  the misheard forms that should become it. One fix teaches two things
//  (docs/research/2026-09-28-dictation-correction-ux.md §4): an exact
//  "heard → meant" replacement, applied to every later take, and a
//  vocabulary term the recognizer is biased toward. Pure values, no I/O.
//

import Foundation

nonisolated enum LabSource: String, Codable, Sendable, CaseIterable {
    case fixBar = "Fix bar"
    case overlay = "Overlay"
    case teach = "Teach"
    case watched = "Your edit"
    case voice = "Voice"
    case page = "Page"
    case typed = "Added"
    case memory = "Memory"
}

nonisolated struct LabTerm: Identifiable, Equatable, Sendable {
    let id: UUID
    /// The spelling to write.
    var term: String
    /// Misheard forms that become `term` (lowercased), learned from fixes.
    var heardAs: [String]
    var source: LabSource
    var addedAt: Date
    /// Fixes that taught this term.
    var fixes: Int
    /// Takes the term's replacements changed since it was learned.
    var applied: Int
    var lastUsed: Date?

    init(term: String, heardAs: [String] = [], source: LabSource, at date: Date = Date()) {
        self.id = UUID()
        self.term = term
        self.heardAs = heardAs
        self.source = source
        self.addedAt = date
        self.fixes = heardAs.isEmpty ? 0 : 1
        self.applied = 0
    }
}

/// One replacement the lexicon made in a take.
nonisolated struct LabApplied: Equatable, Sendable {
    let heard: String
    let term: String
}

/// One word-level change between what was inserted and what it should be.
nonisolated struct LabHunk: Equatable, Sendable {
    let before: String
    let after: String

    /// A small change that sounds like the original, or that writes a word
    /// no dictionary knows: a recognition error, worth learning. Anything
    /// else is the owner rewriting, and is not learned.
    var isCorrection: Bool {
        let b = before.split(separator: " ")
        let a = after.split(separator: " ")
        guard !b.isEmpty, !a.isEmpty, b.count <= 3, a.count <= 4 else { return false }
        let bare = { (s: String) in s.trimmingCharacters(in: .punctuationCharacters) }
        if bare(before).isEmpty || bare(after).isEmpty { return false }
        if bare(before) == bare(after) { return false }  // punctuation only
        if bare(before).lowercased() == bare(after).lowercased() { return true }  // casing
        let similarity = Phonetic.similarity(bare(before), bare(after))
        return similarity >= 0.45 || Phonetic.looksLikeName(bare(after))
    }
}

nonisolated struct LabLexicon: Equatable, Sendable {
    private(set) var terms: [LabTerm] = []
    /// Forgotten terms and forms, so an Undo or a Forget is never re-learned.
    private(set) var tombstones: Set<String> = []

    // MARK: Apply

    /// Rewrites every learned misheard form in `text`, longest first,
    /// case-insensitive, on word boundaries.
    func apply(to text: String) -> (text: String, applied: [LabApplied]) {
        var result = text
        var applied: [LabApplied] = []
        let rules = terms.flatMap { term in term.heardAs.map { (heard: $0, term: term.term) } }
            .sorted { $0.heard.count > $1.heard.count }
        for rule in rules {
            let pattern =
                "(?<![\\w-])" + NSRegularExpression.escapedPattern(for: rule.heard) + "(?![\\w-])"
            guard
                let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
            else {
                continue
            }
            let range = NSRange(result.startIndex..., in: result)
            let matches = regex.matches(in: result, range: range)
            guard !matches.isEmpty else { continue }
            for match in matches.reversed() {
                guard let r = Range(match.range, in: result) else { continue }
                let found = String(result[r])
                if found == rule.term { continue }
                result.replaceSubrange(r, with: rule.term)
                applied.append(LabApplied(heard: found, term: rule.term))
            }
        }
        return (result, applied)
    }

    /// The terms the recognizer should lean toward, most useful first.
    var biasTerms: [String] {
        terms.sorted { ($0.applied + $0.fixes * 2) > ($1.applied + $1.fixes * 2) }
            .prefix(50).map(\.term)
    }

    // MARK: Learn

    /// Learns a fix: adds `after` as a term (or finds the one it spells) and
    /// `before` as a form that becomes it. Returns false when the fix was
    /// forgotten earlier and must not come back.
    @discardableResult
    mutating func learn(_ hunk: LabHunk, source: LabSource, at date: Date = Date()) -> Bool {
        let heard = hunk.before.trimmingCharacters(in: .punctuationCharacters).lowercased()
        let meant = hunk.after.trimmingCharacters(in: .punctuationCharacters)
        guard !heard.isEmpty, !meant.isEmpty else { return false }
        if tombstones.contains(Self.tombstone(heard, meant)) { return false }
        if let index = terms.firstIndex(where: { $0.term == meant }) {
            if !terms[index].heardAs.contains(heard) {
                terms[index].heardAs.append(heard)
            }
            terms[index].fixes += 1
            terms[index].lastUsed = date
        } else {
            var term = LabTerm(term: meant, heardAs: [heard], source: source, at: date)
            term.lastUsed = date
            terms.insert(term, at: 0)
        }
        return true
    }

    /// Adds a term with no misheard form yet (typed by hand, or accepted
    /// from memory); it still biases the recognizer.
    mutating func add(term raw: String, source: LabSource) {
        let term = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !term.isEmpty, !terms.contains(where: { $0.term == term }) else { return }
        terms.insert(LabTerm(term: term, source: source), at: 0)
    }

    /// Takes back one lesson: drops the form (and the term, if nothing else
    /// taught it) and remembers not to learn it again.
    mutating func unlearn(_ hunk: LabHunk) {
        let heard = hunk.before.trimmingCharacters(in: .punctuationCharacters).lowercased()
        let meant = hunk.after.trimmingCharacters(in: .punctuationCharacters)
        tombstones.insert(Self.tombstone(heard, meant))
        guard let index = terms.firstIndex(where: { $0.term == meant }) else { return }
        terms[index].heardAs.removeAll { $0 == heard }
        terms[index].fixes = max(0, terms[index].fixes - 1)
        if terms[index].heardAs.isEmpty, terms[index].fixes == 0,
            terms[index].source != .typed, terms[index].source != .memory
        {
            terms.remove(at: index)
        }
    }

    mutating func forget(_ id: UUID) {
        guard let index = terms.firstIndex(where: { $0.id == id }) else { return }
        for heard in terms[index].heardAs {
            tombstones.insert(Self.tombstone(heard, terms[index].term))
        }
        terms.remove(at: index)
    }

    mutating func removeForm(_ heard: String, from id: UUID) {
        guard let index = terms.firstIndex(where: { $0.id == id }) else { return }
        tombstones.insert(Self.tombstone(heard, terms[index].term))
        terms[index].heardAs.removeAll { $0 == heard }
    }

    mutating func noteApplied(_ applied: [LabApplied], at date: Date = Date()) {
        for item in applied {
            guard let index = terms.firstIndex(where: { $0.term == item.term }) else { continue }
            terms[index].applied += 1
            terms[index].lastUsed = date
        }
    }

    // MARK: Suggest

    /// Known terms that sound like `word`, best first: what the fix surfaces
    /// offer. Suggestions only; phonetic matches are never applied on their
    /// own (errors note §1.8: too many false alarms).
    func suggestions(for word: String, limit: Int = 3) -> [String] {
        let bare = word.trimmingCharacters(in: .punctuationCharacters)
        // Short words have one- or two-letter keys that match anything.
        guard bare.count >= 3, Phonetic.key(bare).count >= 2 else { return [] }
        var scored: [(String, Double)] = []
        for term in terms where term.term.lowercased() != bare.lowercased() {
            if term.heardAs.contains(bare.lowercased()) {
                scored.append((term.term, 2))
                continue
            }
            let s = max(
                Phonetic.similarity(bare, term.term),
                term.heardAs.map { Phonetic.similarity(bare, $0) }.max() ?? 0)
            let shortest = min(Phonetic.key(bare).count, Phonetic.key(term.term).count)
            if s >= (shortest <= 3 ? 0.67 : 0.5) { scored.append((term.term, s)) }
        }
        // A casing fix is always a candidate for an all-caps or all-lowercase name.
        var result = scored.sorted { $0.1 > $1.1 }.map(\.0)
        if bare == bare.uppercased(), bare.count > 2 {
            result.append(bare.prefix(1).uppercased() + bare.dropFirst().lowercased())
        }
        return Array(NSOrderedSet(array: result).array.compactMap { $0 as? String }.prefix(limit))
    }

    static func tombstone(_ heard: String, _ meant: String) -> String {
        heard.lowercased() + "\u{1F}" + meant
    }
}

// MARK: - Diff

nonisolated enum LabDiff {
    /// Word-level changes from `before` to `after` (an LCS over words; runs
    /// of deletions and insertions pair up into one hunk).
    static func hunks(from before: String, to after: String) -> [LabHunk] {
        let old = before.split(separator: " ").map(String.init)
        let new = after.split(separator: " ").map(String.init)
        guard old != new else { return [] }
        var lcs = [[Int]](
            repeating: [Int](repeating: 0, count: new.count + 1), count: old.count + 1)
        for i in stride(from: old.count - 1, through: 0, by: -1) {
            for j in stride(from: new.count - 1, through: 0, by: -1) {
                lcs[i][j] =
                    Self.same(old[i], new[j])
                    ? lcs[i + 1][j + 1] + 1 : max(lcs[i + 1][j], lcs[i][j + 1])
            }
        }
        var hunks: [LabHunk] = []
        var deleted: [String] = []
        var inserted: [String] = []
        func flush() {
            if !deleted.isEmpty || !inserted.isEmpty {
                hunks.append(
                    LabHunk(
                        before: deleted.joined(separator: " "),
                        after: inserted.joined(separator: " ")))
            }
            deleted = []
            inserted = []
        }
        var i = 0
        var j = 0
        while i < old.count, j < new.count {
            if Self.same(old[i], new[j]) {
                flush()
                i += 1
                j += 1
            } else if lcs[i + 1][j] >= lcs[i][j + 1] {
                deleted.append(old[i])
                i += 1
            } else {
                inserted.append(new[j])
                j += 1
            }
        }
        deleted.append(contentsOf: old[i...])
        inserted.append(contentsOf: new[j...])
        flush()
        return hunks
    }

    /// Words match exactly, punctuation included, so a casing or a spelling
    /// fix is a hunk of its own.
    private static func same(_ a: String, _ b: String) -> Bool { a == b }
}

// MARK: - Phonetic key

/// A small sound-alike key: consonant skeleton with common spellings folded
/// (ph→f, ck/c/q→k, x→ks, z→s, w/h/y dropped inside a word), vowels kept only
/// at the start. "cloud" and "Claude" both become "kld"; "SRACT" and
/// "Tesseract" differ by one letter.
nonisolated enum Phonetic {
    static func key(_ text: String) -> String {
        let lower = text.lowercased().filter { $0.isLetter || $0.isNumber || $0 == " " }
        var out = ""
        for word in lower.split(separator: " ") {
            var s = String(word)
            for (a, b) in [
                ("ph", "f"), ("ck", "k"), ("sch", "sk"), ("tch", "ch"), ("x", "ks"), ("qu", "k"),
            ] {
                s = s.replacingOccurrences(of: a, with: b)
            }
            var key = ""
            for (index, ch) in s.enumerated() {
                var c = ch
                if "cq".contains(c) { c = "k" }
                if c == "z" { c = "s" }
                if "aeiou".contains(c) {
                    if index == 0 { key.append("a") }
                    continue
                }
                if index > 0, "why".contains(c) { continue }
                if key.last == c { continue }
                key.append(c)
            }
            out += key
        }
        return out
    }

    /// 0…1, from the edit distance between the two keys.
    static func similarity(_ a: String, _ b: String) -> Double {
        let ka = key(a)
        let kb = key(b)
        guard !ka.isEmpty || !kb.isEmpty else { return 1 }
        let d = levenshtein(Array(ka), Array(kb))
        return 1 - Double(d) / Double(max(ka.count, kb.count))
    }

    /// Mixed case inside the word, a digit, or a dot: CamelCase names,
    /// versions and file names, which no misrecognition produces by itself.
    static func looksLikeName(_ word: String) -> Bool {
        let inner = word.dropFirst()
        return inner.contains(where: \.isUppercase) || word.contains(where: \.isNumber)
            || word.contains(".")
    }

    static func levenshtein(_ a: [Character], _ b: [Character]) -> Int {
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        var previous = Array(0...b.count)
        for i in 1...a.count {
            var current = [i] + [Int](repeating: 0, count: b.count)
            for j in 1...b.count {
                current[j] = min(
                    previous[j] + 1, current[j - 1] + 1,
                    previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
            }
            previous = current
        }
        return previous[b.count]
    }
}
