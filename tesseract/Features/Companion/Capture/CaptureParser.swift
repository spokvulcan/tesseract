//
//  CaptureParser.swift
//  tesseract
//
//  A captured thought, turned into a reminder without a model: "remind me to
//  call the dentist tomorrow at 10" becomes "Call the dentist", due tomorrow
//  10:00. Relative timing like "after the 1:1" lands at the end of that
//  calendar event; "#health" (or a leading "Health:") files it in an Area;
//  anything without a time lands undated, in the Inbox. Deterministic and
//  instant, so capture works while the model is busy, unloaded or failing.
//

import Foundation

nonisolated struct CaptureIntent: Equatable, Sendable {
    var title: String
    var due: Date?
    var dueHasTime: Bool
    /// The Area it was filed under, when the words named one.
    var area: Area?
    /// The event a relative time was anchored to.
    var anchorEventID: String?
}

nonisolated enum CaptureParser {

    static func parse(
        _ raw: String, now: Date, events: [AgendaEvent], areas: [Area],
        calendar: Calendar = .current
    ) -> CaptureIntent? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        text = stripLeadIn(text)

        var intent = CaptureIntent(title: "", due: nil, dueHasTime: false)

        // Area: "#health" anywhere, or a leading "Health:".
        if let (area, rest) = takeArea(from: text, areas: areas) {
            intent.area = area
            text = rest
        }

        // "after the 1:1" → the end of that event.
        if let (event, rest) = takeAnchor(from: text, events: events, now: now) {
            intent.due = event.end
            intent.dueHasTime = true
            intent.anchorEventID = event.id
            text = rest
        }

        if intent.due == nil,
            let (date, hasTime, rest) = takeRelative(from: text, now: now, calendar: calendar)
        {
            intent.due = date
            intent.dueHasTime = hasTime
            text = rest
        }

        if intent.due == nil,
            let (date, rest) = takeDayPart(from: text, now: now, calendar: calendar)
        {
            intent.due = date
            intent.dueHasTime = true
            text = rest
        }

        if intent.due == nil,
            let (date, hasTime, rest) = takeDetectedDate(from: text, now: now, calendar: calendar)
        {
            intent.due = date
            intent.dueHasTime = hasTime
            text = rest
        }

        if intent.due == nil, let (date, rest) = takeClock(from: text, now: now, calendar: calendar)
        {
            intent.due = date
            intent.dueHasTime = true
            text = rest
        }

        intent.title = tidyTitle(text)
        guard !intent.title.isEmpty else { return nil }
        return intent
    }

    // MARK: - Pieces

    private static let leadIns = [
        "remind me to ", "remind me that ", "remind me about ", "remind me ",
        "remember to ", "don't forget to ", "dont forget to ", "i need to ", "i have to ",
        "i must ", "need to ", "todo: ", "todo ", "to do: ", "task: ", "add a reminder to ",
        "add reminder to ", "reminder: ", "reminder to ",
    ]

    static func stripLeadIn(_ text: String) -> String {
        // A trailing space lets a bare "remind me to" strip to nothing.
        var text = text + " "
        var changed = true
        while changed {
            changed = false
            for lead in leadIns where text.lowercased().hasPrefix(lead) {
                text = String(text.dropFirst(lead.count))
                changed = true
            }
        }
        return text.trimmingCharacters(in: .whitespaces)
    }

    private static func takeArea(from text: String, areas: [Area]) -> (Area, String)? {
        func match(_ word: String) -> Area? {
            let wanted = word.lowercased()
            if let exact = areas.first(where: { $0.name.lowercased() == wanted }) { return exact }
            let prefixed = areas.filter { $0.name.lowercased().hasPrefix(wanted) }
            return prefixed.count == 1 ? prefixed[0] : nil
        }
        if let tag = text.firstMatch(of: /(?:^|\s)#([\p{L}\p{N}_-]+)/),
            let area = match(String(tag.1))
        {
            var rest = text
            rest.removeSubrange(tag.range)
            return (area, rest.trimmingCharacters(in: .whitespaces))
        }
        if let lead = text.prefixMatch(of: /([\p{L}\p{N} _-]{2,30}):\s*/),
            let area = areas.first(where: { $0.name.lowercased() == String(lead.1).lowercased() })
        {
            return (area, String(text[lead.range.upperBound...]))
        }
        return nil
    }

    private static func takeAnchor(
        from text: String, events: [AgendaEvent], now: Date
    ) -> (AgendaEvent, String)? {
        guard let match = text.firstMatch(of: /(?i)\s*\bafter\s+(?:the\s+|my\s+|our\s+)?(.+)$/)
        else {
            return nil
        }
        // Try the longest phrase first, then shorter ones: "after the 1:1 tomorrow".
        let words = String(match.1).split(separator: " ").map(String.init)
        for count in stride(from: min(words.count, 5), through: 1, by: -1) {
            let phrase = words.prefix(count).joined(separator: " ")
                .trimmingCharacters(in: .punctuationCharacters)
            guard !phrase.isEmpty else { continue }
            if let event = AgendaTime.anchorEvent(phrase, in: events, now: now) {
                var rest = text
                let tail = words.dropFirst(count).joined(separator: " ")
                rest.replaceSubrange(match.range, with: tail.isEmpty ? "" : " " + tail)
                return (event, rest.trimmingCharacters(in: .whitespaces))
            }
        }
        return nil
    }

    private static let numberWords: [String: Int] = [
        "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty-five": 45, "half an": 30,
    ]

    private static func takeRelative(
        from text: String, now: Date, calendar: Calendar
    ) -> (Date, Bool, String)? {
        guard
            let match = text.firstMatch(
                of:
                    /(?i)\bin\s+(\d+|a|an|one|two|three|four|five|six|ten|fifteen|twenty|thirty|forty-five|half an)\s+(minutes?|mins?|hours?|hrs?|days?|weeks?)\b/
            )
        else { return nil }
        let amountWord = String(match.1).lowercased()
        guard let amount = Int(amountWord) ?? numberWords[amountWord] else { return nil }
        let unit = String(match.2).lowercased()
        var rest = text
        rest.removeSubrange(match.range)
        rest = rest.trimmingCharacters(in: .whitespaces)
        if amountWord == "half an" {
            return (now.addingTimeInterval(30 * 60), true, rest)
        }
        if unit.hasPrefix("m") {
            return (now.addingTimeInterval(TimeInterval(amount * 60)), true, rest)
        }
        if unit.hasPrefix("h") {
            return (now.addingTimeInterval(TimeInterval(amount * 3600)), true, rest)
        }
        let days = unit.hasPrefix("w") ? amount * 7 : amount
        guard
            let date = calendar.date(byAdding: .day, value: days, to: calendar.startOfDay(for: now))
        else { return nil }
        return (date, false, rest)
    }

    private static func takeDayPart(
        from text: String, now: Date, calendar: Calendar
    ) -> (Date, String)? {
        let parts: [(Regex<Substring>, Int, Int)] = [
            (/(?i)\btomorrow morning\b/, 1, 9),
            (/(?i)\btomorrow afternoon\b/, 1, 15),
            (/(?i)\btomorrow (?:evening|night)\b/, 1, 19),
            (/(?i)\bthis morning\b/, 0, 9),
            (/(?i)\bthis afternoon\b/, 0, 15),
            (/(?i)\bthis evening\b/, 0, 19),
            (/(?i)\btonight\b/, 0, 20),
        ]
        for (regex, dayOffset, hour) in parts {
            guard let match = text.firstMatch(of: regex) else { continue }
            let day =
                calendar.date(byAdding: .day, value: dayOffset, to: calendar.startOfDay(for: now))
                ?? now
            guard var date = calendar.date(bySettingHour: hour, minute: 0, second: 0, of: day)
            else {
                continue
            }
            // Already past this part of today: an hour from now instead.
            if date <= now { date = now.addingTimeInterval(3600) }
            var rest = text
            rest.removeSubrange(match.range)
            return (date, rest.trimmingCharacters(in: .whitespaces))
        }
        return nil
    }

    private static func takeDetectedDate(
        from text: String, now: Date, calendar: Calendar
    ) -> (Date, Bool, String)? {
        guard
            let detector = try? NSDataDetector(
                types: NSTextCheckingResult.CheckingType.date.rawValue)
        else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        // "the Today page" names a thing, not a day: skip a lone day word
        // that follows an article.
        let result = detector.matches(in: text, options: [], range: range).first { match in
            guard let found = Range(match.range, in: text) else { return false }
            let before = text[..<found.lowerBound].lowercased()
            let followsArticle = ["the ", "a ", "an "].contains { before.hasSuffix($0) }
            return !(followsArticle && !text[found].contains(" "))
        }
        guard let result, var date = result.date, let matchRange = Range(result.range, in: text)
        else { return nil }
        let phrase = text[matchRange].lowercased()
        let hasTime =
            phrase.contains(":") || phrase.contains("am") || phrase.contains("pm")
            || phrase.contains("noon") || phrase.contains("midnight")
            || phrase.firstMatch(of: /\bat\s+\d/) != nil
        if hasTime {
            let saidMeridiem = phrase.contains("am") || phrase.contains("pm")
            // "at 9" said at 14:00 means 21:00 today, not this morning.
            if date <= now, !saidMeridiem, calendar.isDate(date, inSameDayAs: now) {
                let evening = date.addingTimeInterval(12 * 3600)
                date =
                    evening > now && calendar.component(.hour, from: date) < 12
                    ? evening : calendar.date(byAdding: .day, value: 1, to: date) ?? date
            }
        } else {
            date = calendar.startOfDay(for: date)
        }
        var rest = text
        rest.removeSubrange(matchRange)
        return (date, hasTime, rest.trimmingCharacters(in: .whitespaces))
    }

    /// "at 9", "at 9:30", "at 9pm", "at 21:00" — a clock NSDataDetector
    /// leaves alone. A bare hour already past this morning means tonight;
    /// otherwise a past time means tomorrow.
    private static func takeClock(
        from text: String, now: Date, calendar: Calendar
    ) -> (Date, String)? {
        guard
            let match = text.firstMatch(of: /(?i)\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b/),
            var hour = Int(match.1), hour < 24
        else { return nil }
        let minute = match.2.flatMap { Int($0) } ?? 0
        guard minute < 60 else { return nil }
        let meridiem = match.3?.lowercased()
        if meridiem == "pm", hour < 12 { hour += 12 }
        if meridiem == "am", hour == 12 { hour = 0 }
        guard var date = calendar.date(bySettingHour: hour, minute: minute, second: 0, of: now)
        else { return nil }
        if date <= now {
            let evening = date.addingTimeInterval(12 * 3600)
            date =
                meridiem == nil && hour < 12 && evening > now
                ? evening : calendar.date(byAdding: .day, value: 1, to: date) ?? date
        }
        var rest = text
        rest.removeSubrange(match.range)
        return (date, rest.trimmingCharacters(in: .whitespaces))
    }

    /// Trailing prepositions and punctuation the time phrase left behind go;
    /// the first letter is capitalised.
    static func tidyTitle(_ text: String) -> String {
        var title = text.trimmingCharacters(
            in: .whitespacesAndNewlines.union(.punctuationCharacters))
        var changed = true
        while changed {
            changed = false
            for dangling in [" at", " on", " by", " for", " in", " from", ","] {
                if title.lowercased().hasSuffix(dangling) {
                    title = String(title.dropLast(dangling.count))
                        .trimmingCharacters(in: .whitespaces)
                    changed = true
                }
            }
        }
        title = title.replacingOccurrences(of: "  ", with: " ")
        guard let first = title.first else { return "" }
        return first.uppercased() + title.dropFirst()
    }
}
