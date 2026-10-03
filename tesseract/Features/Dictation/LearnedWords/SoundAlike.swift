//
//  SoundAlike.swift
//  tesseract
//
//  "Sounds like" (PRD #612): the small phonetic key the Lens uses to find
//  the word a fix replaces, and the learning gate uses to tell a
//  mishearing from a rewrite. A consonant skeleton with common spellings
//  folded and number words written as digits, compared by edit distance:
//  "Claude" and "cloud" share a key, "SRACT" is one letter from
//  "Tesseract", "work tree" is "worktree", "D flash two" is "DFlash2".
//
//  A key match only ever *picks* a word the owner is fixing or lets a fix
//  be learned. It never changes text on its own: the 2026-09-28 replay
//  found sound-alike application raised three false alarms per catch.
//

import Foundation

nonisolated enum SoundAlike {

    /// The score at and above which two words sound alike.
    static let threshold = 0.5

    private static let numberWords: [String: String] = [
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    ]

    /// The phonetic key: lowercased letters and digits with spaces removed
    /// (so a split word keys like the whole one), number words as digits,
    /// common spellings folded (ph→f, wh→w, ck→k, soft c→s, hard c and q→k,
    /// x→ks, z→s), vowels dropped after the first letter, doubled letters
    /// collapsed.
    static func key(_ text: String) -> String {
        // Other scripts and accents fold to Latin first ("Клод" keys like
        // "Claude"), so a take in another language still finds its target.
        let latin =
            text.unicodeScalars.allSatisfy(\.isASCII)
            ? text
            : (text.applyingTransform(.toLatin, reverse: false) ?? text)
                .folding(options: [.diacriticInsensitive, .caseInsensitive], locale: nil)
        let words = latin.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map { numberWords[String($0)] ?? String($0) }
        var s = Array(words.joined().unicodeScalars.filter { $0.isASCII }.map(Character.init))
        guard !s.isEmpty else { return "" }

        var folded: [Character] = []
        var i = 0
        while i < s.count {
            let c = s[i]
            let next: Character? = i + 1 < s.count ? s[i + 1] : nil
            switch (c, next) {
            case ("p", "h"):
                folded.append("f")
                i += 2
                continue
            case ("w", "h"):
                folded.append("w")
                i += 2
                continue
            case ("c", "k"):
                folded.append("k")
                i += 2
                continue
            default:
                break
            }
            switch c {
            case "c":
                if let next, "eiy".contains(next) {
                    folded.append("s")
                } else {
                    folded.append("k")
                }
            case "q": folded.append("k")
            case "x": folded.append(contentsOf: ["k", "s"])
            case "z": folded.append("s")
            default: folded.append(c)
            }
            i += 1
        }
        s = folded

        var key: [Character] = [s[0]]
        for c in s.dropFirst() where !"aeiouy".contains(c) {
            if key.last != c { key.append(c) }
        }
        return String(key)
    }

    /// How alike two words sound, 0…1: one minus the edit distance between
    /// their keys over the longer key. Zero when either has no key.
    static func similarity(_ a: String, _ b: String) -> Double {
        let ka = Array(key(a))
        let kb = Array(key(b))
        guard !ka.isEmpty, !kb.isEmpty else { return 0 }
        return 1 - Double(levenshtein(ka, kb)) / Double(max(ka.count, kb.count))
    }

    /// Whether two words sound alike enough for a fix to be a mishearing.
    static func soundsAlike(_ a: String, _ b: String) -> Bool {
        similarity(a, b) >= threshold
    }

    /// How alike two words are spelled, 0…1, on lowercased letters and
    /// digits: the tie-break between words that sound the same.
    static func spelling(_ a: String, _ b: String) -> Double {
        let sa = Array(a.lowercased().filter { $0.isLetter || $0.isNumber })
        let sb = Array(b.lowercased().filter { $0.isLetter || $0.isNumber })
        guard !sa.isEmpty || !sb.isEmpty else { return 1 }
        return 1 - Double(levenshtein(sa, sb)) / Double(max(sa.count, sb.count))
    }

    static func levenshtein<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        var previous = Array(0...b.count)
        var current = [Int](repeating: 0, count: b.count + 1)
        for i in 1...a.count {
            current[0] = i
            for j in 1...b.count {
                current[j] = min(
                    previous[j] + 1, current[j - 1] + 1,
                    previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
            }
            swap(&previous, &current)
        }
        return previous[b.count]
    }
}
