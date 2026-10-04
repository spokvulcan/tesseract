//
//  CatchRecord.swift
//  tesseract
//
//  The **catch record** (PRD #612): what the Dictation page shows. This
//  week's catches and fixes per day, one tile per **Learned Word**, and
//  today's takes, each one click from a fix in the **Lens**. Pure: built
//  from the stores' values, so the page and its tests read the same thing.
//

import Foundation

nonisolated struct CatchRecord: Equatable, Sendable {

    /// One day of the week chart.
    struct Day: Equatable, Identifiable, Sendable {
        /// The start of the local day.
        let date: Date
        /// Catches by the Learned Words still known (a forgotten word's
        /// catches leave the chart and the sentence with it; today's takes
        /// still show what it caught).
        let caught: Int
        /// Words the owner fixed in the Lens.
        let fixed: Int

        var id: Date { date }
    }

    /// One Learned Word.
    struct Tile: Equatable, Identifiable, Sendable {
        let id: UUID
        let meant: String
        /// Every way it was heard, first learned first.
        let heard: [String]
        let learnedAt: Date
        /// The fixes that taught it.
        let fixes: Int
        let caught: Int
        let caughtThisWeek: Int
        let lastCaughtAt: Date?
        /// The names of the apps it is left alone in.
        let leftAloneIn: [String]
        /// The fix that taught it, in context.
        let example: LearnedWord.Example?
    }

    /// One of today's takes.
    struct Take: Equatable, Identifiable, Sendable {
        /// The history entry's id.
        let id: UUID
        let at: Date
        let text: String
        let catches: [LearnedWordCatch]
        /// Words fixed in the Lens.
        let fixes: Int
        let pairID: UUID?
        let app: TranscriptionEntry.App?

        /// The take as the Lens opens it from the page.
        var lensTake: DictatedTake {
            DictatedTake.fromPage(
                pairID: pairID, text: text, catches: catches, app: app, at: at)
        }
    }

    /// A stretch of text, marked when it is a caught word or the word a fix
    /// changed.
    struct Run: Equatable, Sendable {
        let text: String
        let marked: Bool
    }

    /// The last seven days, oldest first, today last.
    let days: [Day]
    /// The Learned Words still known, most recently taught first.
    let tiles: [Tile]
    /// Today's takes, newest first.
    let today: [Take]

    var caughtThisWeek: Int { days.reduce(0) { $0 + $1.caught } }
    var fixesThisWeek: Int { days.reduce(0) { $0 + $1.fixed } }
    /// The fixes that taught the words still known.
    var teachingFixes: Int { tiles.reduce(0) { $0 + $1.fixes } }

    static let empty = CatchRecord(days: [], tiles: [], today: [])

    init(days: [Day], tiles: [Tile], today: [Take]) {
        self.days = days
        self.tiles = tiles
        self.today = today
    }

    /// - Parameters:
    ///   - words: the Learned Words, in the store's order (most recently
    ///     taught first).
    ///   - pairs: the Correction Pairs, whose fixes are counted per day.
    ///   - entries: the transcription history, newest first.
    init(
        words: [LearnedWord], pairs: [CorrectionPair], entries: [TranscriptionEntry],
        now: Date, calendar: Calendar = .current
    ) {
        let today = calendar.startOfDay(for: now)
        let dates = (0..<7).reversed().compactMap {
            calendar.date(byAdding: .day, value: -$0, to: today)
        }
        let keys = dates.map { LearnedWord.dayKey(for: $0, calendar: calendar) }
        let known = words.filter { !$0.isForgotten }

        var fixedByDay: [String: Int] = [:]
        if let first = dates.first {
            for pair in pairs {
                for fix in pair.fixes where fix.at >= first {
                    fixedByDay[LearnedWord.dayKey(for: fix.at, calendar: calendar), default: 0] += 1
                }
            }
        }

        days = zip(dates, keys).map { date, key in
            Day(
                date: date,
                caught: known.reduce(0) { $0 + ($1.catchesByDay[key] ?? 0) },
                fixed: fixedByDay[key] ?? 0)
        }

        tiles = known.map { word in
            Tile(
                id: word.id, meant: word.meant, heard: word.heard, learnedAt: word.learnedAt,
                fixes: word.fixes, caught: word.totalCatches,
                caughtThisWeek: keys.reduce(0) { $0 + (word.catchesByDay[$1] ?? 0) },
                lastCaughtAt: word.lastCaughtAt, leftAloneIn: word.leftAloneIn.map(\.name),
                example: word.example)
        }

        let fixesByPair = Dictionary(
            pairs.map { ($0.id, $0.fixes.count) }, uniquingKeysWith: { first, _ in first })
        self.today =
            entries
            .filter { calendar.isDate($0.timestamp, inSameDayAs: now) }
            .sorted { $0.timestamp > $1.timestamp }
            .map { entry in
                Take(
                    id: entry.id, at: entry.timestamp, text: entry.text, catches: entry.catches,
                    fixes: entry.pairID.flatMap { fixesByPair[$0] } ?? 0, pairID: entry.pairID,
                    app: entry.app)
            }
    }

    // MARK: - Words

    /// The page's one sentence: this week's catches and the owner's fixes.
    var headline: String {
        let caught = caughtThisWeek
        let fixed = fixesThisWeek
        let fixedWords = "\(fixed) \(fixed == 1 ? "word" : "words")"
        if caught > 0 {
            let catches =
                "This week I caught \(caught) \(caught == 1 ? "mistake" : "mistakes") "
                + "before \(caught == 1 ? "it" : "they") reached an app"
            return fixed > 0 ? "\(catches), and you fixed \(fixedWords)." : "\(catches)."
        }
        if fixed > 0 {
            return "This week you fixed \(fixedWords); nothing caught yet."
        }
        return tiles.isEmpty ? "Nothing learned yet." : "Nothing to catch yet this week."
    }

    /// The line under it: what the owner taught, or how to start.
    func detail(fixHotkey: String) -> String {
        guard !tiles.isEmpty else {
            return "Fix a word in the Lens (\(fixHotkey) after a paste) and I learn it "
                + "for every take after."
        }
        let words = tiles.count
        let fixes = teachingFixes
        return "You taught me \(words) \(words == 1 ? "word" : "words") "
            + "with \(fixes) \(fixes == 1 ? "fix" : "fixes")."
    }

    // MARK: - Marking

    /// `text` in runs, the tokens in `tokenRanges` marked (a take's catches).
    static func runs(_ text: String, marking tokenRanges: [Range<Int>]) -> [Run] {
        let tokens = TakeText.tokens(text)
        let ranges = tokenRanges.compactMap { range -> Range<String.Index>? in
            guard range.lowerBound >= 0, range.upperBound <= tokens.count, !range.isEmpty
            else { return nil }
            return TakeText.coreRange(tokens[range])
        }
        return runs(text, marking: ranges)
    }

    /// `text` in runs, every whole-word occurrence of a phrase marked,
    /// ignoring case (a word's heard forms in its example's before half,
    /// its spelling in the after half).
    static func runs(_ text: String, marking phrases: [String]) -> [Run] {
        let tokens = TakeText.tokens(text)
        let forms = phrases.map { TakeText.normalized($0).split(separator: " ").map(String.init) }
            .filter { !$0.isEmpty }
            .sorted { $0.count > $1.count }
        var ranges: [Range<String.Index>] = []
        var i = 0
        while i < tokens.count {
            let match = forms.first { form in
                i + form.count <= tokens.count
                    && zip(tokens[i..<(i + form.count)], form).allSatisfy { $0.bare == $1 }
            }
            if let match, let range = TakeText.coreRange(tokens[i..<(i + match.count)]) {
                ranges.append(range)
                i += match.count
            } else {
                i += 1
            }
        }
        return runs(text, marking: ranges)
    }

    private static func runs(_ text: String, marking ranges: [Range<String.Index>]) -> [Run] {
        var runs: [Run] = []
        var cursor = text.startIndex
        for range in ranges.sorted(by: { $0.lowerBound < $1.lowerBound })
        where range.lowerBound >= cursor {
            if cursor < range.lowerBound {
                runs.append(Run(text: String(text[cursor..<range.lowerBound]), marked: false))
            }
            runs.append(Run(text: String(text[range]), marked: true))
            cursor = range.upperBound
        }
        if cursor < text.endIndex {
            runs.append(Run(text: String(text[cursor...]), marked: false))
        }
        return runs
    }
}

extension TranscriptionEntry {
    /// The entry as the Lens opens it from the history.
    var lensTake: DictatedTake {
        DictatedTake.fromPage(
            pairID: pairID, text: text, catches: catches, app: app, at: timestamp)
    }
}

extension DictatedTake {
    /// A take opened from the Dictation page: nothing is pasted back, so the
    /// app only names it and scopes a word left alone (its pid is long
    /// stale).
    nonisolated static func fromPage(
        pairID: UUID?, text: String, catches: [LearnedWordCatch],
        app: TranscriptionEntry.App?, at: Date
    ) -> DictatedTake {
        DictatedTake(
            pairID: pairID, text: text, catches: catches,
            app: app.map { TargetApp(bundleID: $0.bundleID, name: $0.name, pid: 0) },
            pasted: false, at: at)
    }
}
