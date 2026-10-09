//
//  CaptureParser.swift
//  tesseract
//
//  A captured thought, turned into a reminder without a model: "remind me to
//  call the dentist tomorrow at 10" becomes "Call the dentist", due tomorrow
//  10:00. Relative timing like "after the 1:1" lands at the end of that
//  calendar event; "#health" (or a leading "Health:") files it in an Area;
//  anything without a time lands undated, in the Inbox. Days and weekdays
//  count from `now`: the system date reader counts from the real clock, so
//  only what it alone knows (a date like "15 October") is left to it.
//  Deterministic and instant, so capture works while the model is busy,
//  unloaded or failing.
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
            let (date, hasTime, rest) = takeDayWord(from: text, now: now, calendar: calendar)
        {
            intent.due = date
            intent.dueHasTime = hasTime
            text = rest
        }

        if intent.due == nil,
            let (date, hasTime, rest) = takeWeekday(from: text, now: now, calendar: calendar)
        {
            intent.due = date
            intent.dueHasTime = hasTime
            text = rest
        }

        if intent.due == nil, let (date, rest) = takeWeek(from: text, now: now, calendar: calendar)
        {
            intent.due = date
            intent.dueHasTime = false
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
        // "after I get home", "after we talk": a clause, not an event — its
        // one-letter words would match any title holding that letter.
        let opener = words.first?.lowercased().trimmingCharacters(in: .punctuationCharacters) ?? ""
        guard !clauseOpeners.contains(opener) else { return nil }
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

    /// Words that open a clause after "after", never an event's name.
    private static let clauseOpeners: Set<String> = [
        "i", "i'm", "im", "i've", "a", "an", "we", "you", "he", "she", "it", "they", "that",
        "this",
    ]

    /// An hour said without am/pm: 1 to 7 is the afternoon or evening ("call
    /// Anna tomorrow at 3" is 15:00, not an alarm before dawn); 8 to 12 as said.
    static func hour(_ hour: Int, meridiem: String?) -> Int {
        switch meridiem {
        case "pm": return hour < 12 ? hour + 12 : hour
        case "am": return hour == 12 ? 0 : hour
        default: return (1...7).contains(hour) ? hour + 12 : hour
        }
    }

    /// "at 9", "at 9:30", "at 9pm" anywhere in the text (a Regex is not
    /// Sendable, so one is made per use).
    private static var clockPattern: Regex<(Substring, Substring, Substring?, Substring?)> {
        /(?i)\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b/
    }

    /// The words before end in an article — "the Today page" names a thing,
    /// not a day. A whole word: "call Anna tomorrow" ends in "a", not "a ".
    private static func followsArticle(_ before: Substring) -> Bool {
        guard let last = before.split(whereSeparator: \.isWhitespace).last else { return false }
        return ["the", "a", "an"].contains(last.lowercased())
    }

    /// The owner's day (until 04:00 still yesterday's), for "tomorrow": at
    /// 01:00 on Saturday, tomorrow is Saturday.
    private static func ownersDay(_ now: Date, _ calendar: Calendar) -> Date {
        DayKey(for: now, calendar: calendar).date(calendar: calendar)
            ?? calendar.startOfDay(for: now)
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
            let base = dayOffset > 0 ? ownersDay(now, calendar) : calendar.startOfDay(for: now)
            let day = calendar.date(byAdding: .day, value: dayOffset, to: base) ?? now
            var rest = text
            rest.removeSubrange(match.range)
            rest = rest.trimmingCharacters(in: .whitespaces)
            // A clock said with it ("tonight at 9") sets the time, read in
            // that part of the day.
            var hour = hour
            var minute = 0
            if let clock = rest.firstMatch(of: clockPattern), let said = Int(clock.1), said < 24 {
                let meridiem = clock.3?.lowercased()
                hour =
                    meridiem != nil
                    ? Self.hour(said, meridiem: meridiem)
                    : (hour >= 12 && said < 12 ? said + 12 : said)
                minute = clock.2.flatMap { Int($0) }.flatMap { $0 < 60 ? $0 : nil } ?? 0
                rest.removeSubrange(clock.range)
                rest = rest.trimmingCharacters(in: .whitespaces)
            }
            guard var date = calendar.date(bySettingHour: hour, minute: minute, second: 0, of: day)
            else {
                continue
            }
            // Already past this part of today: an hour from now instead.
            if date <= now { date = now.addingTimeInterval(3600) }
            return (date, rest)
        }
        return nil
    }

    /// "today", "tomorrow", "the day after tomorrow", with an optional clock
    /// ("tomorrow at 10"), counted from `now`. NSDataDetector resolves these
    /// against the real clock and can't be given another, so the parser keeps
    /// the common ones to itself.
    private static func takeDayWord(
        from text: String, now: Date, calendar: Calendar
    ) -> (Date, Bool, String)? {
        // "the Today page" names a thing, not a day: skip a day word that
        // follows an article, as the detector's path does.
        let pattern =
            /(?i)\b((?:the )?day after tomorrow|tomorrow|today)(?:'s|’s)?\b(?:,?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b)?/
        guard
            let match = text.matches(of: pattern).first(where: { match in
                let word = String(match.1).lowercased()
                return word.hasPrefix("the ")
                    || !followsArticle(text[..<match.range.lowerBound])
            })
        else { return nil }
        let word = String(match.1).lowercased()
        let offset = word.hasSuffix("after tomorrow") ? 2 : word == "tomorrow" ? 1 : 0
        let base = offset > 0 ? ownersDay(now, calendar) : calendar.startOfDay(for: now)
        guard let day = calendar.date(byAdding: .day, value: offset, to: base) else { return nil }
        var rest = text
        rest.removeSubrange(match.range)
        rest = rest.trimmingCharacters(in: .whitespaces)
        // The clock right after the day word, or one said elsewhere ("at 3pm
        // tomorrow").
        var hourText = match.2.map(String.init)
        var minuteText = match.3.map(String.init)
        var meridiem = match.4?.lowercased()
        if hourText == nil, let clock = rest.firstMatch(of: clockPattern) {
            hourText = String(clock.1)
            minuteText = clock.2.map(String.init)
            meridiem = clock.3?.lowercased()
            rest.removeSubrange(clock.range)
            rest = rest.trimmingCharacters(in: .whitespaces)
        } else if hourText == nil, takeNoon(&rest) {
            // "at noon tomorrow": the clock pattern reads digits only.
            hourText = "12"
            meridiem = "pm"
        }
        guard let hourText, let said = Int(hourText), said < 24 else {
            return (day, false, rest)
        }
        let minute = minuteText.flatMap { Int($0) } ?? 0
        guard minute < 60 else { return nil }
        let hour = Self.hour(said, meridiem: meridiem)
        guard var date = calendar.date(bySettingHour: hour, minute: minute, second: 0, of: day)
        else { return nil }
        // "today at 9" said at 14:00 means tonight, as a bare clock does.
        if offset == 0, date <= now, meridiem == nil, hour < 12 {
            date = date.addingTimeInterval(12 * 3600)
        }
        return (date, true, rest)
    }

    /// "noon" or "midday", with or without "at": taken from `rest`.
    private static func takeNoon(_ rest: inout String) -> Bool {
        guard let noon = rest.firstMatch(of: /(?i)\b(?:at\s+)?(?:noon|midday)\b/) else {
            return false
        }
        rest.removeSubrange(noon.range)
        rest = rest.trimmingCharacters(in: .whitespaces)
        return true
    }

    /// "Monday", "on Friday", "this Friday", "by Friday", "next Friday", with
    /// a part of the day or a clock as a day word takes one ("Friday
    /// morning", "Monday at 10", "at 3pm on Friday") — counted from `now`,
    /// as NSDataDetector can't be: it counted from the real clock, so the
    /// same words gave another date in a test, and "next Monday" said on a
    /// Friday was ten days on. A weekday alone, or with "on", is its next
    /// one (said on a Friday, "Friday" is a week on); "this" and "by" count
    /// today; "next" is next week's.
    private static func takeWeekday(
        from text: String, now: Date, calendar: Calendar
    ) -> (Date, Bool, String)? {
        let pattern =
            /(?i)\b(?:(on|this|by|next)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:'s|’s)?\b(?:\s+(morning|afternoon|evening|night))?/
        guard
            let match = text.matches(of: pattern).first(where: { match in
                match.1 != nil || !followsArticle(text[..<match.range.lowerBound])
            })
        else { return nil }
        let names = [
            "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        ]
        guard let index = names.firstIndex(of: String(match.2).lowercased()) else { return nil }
        let today = ownersDay(now, calendar)
        var ahead = (index + 1 - calendar.component(.weekday, from: today) + 7) % 7
        switch match.1.map({ String($0).lowercased() }) {
        case "this", "by":
            break
        case "next":
            // Next week's, by the calendar's own first day of the week.
            guard let week = calendar.dateInterval(of: .weekOfYear, for: today),
                let start = calendar.date(byAdding: .day, value: 7, to: week.start),
                let day = (0..<7).lazy.compactMap({
                    calendar.date(byAdding: .day, value: $0, to: start)
                }).first(where: { calendar.component(.weekday, from: $0) == index + 1 }),
                let days = calendar.dateComponents([.day], from: today, to: day).day
            else { return nil }
            ahead = days
        default:
            if ahead == 0 { ahead = 7 }
        }
        guard let day = calendar.date(byAdding: .day, value: ahead, to: today) else { return nil }
        var rest = text
        rest.removeSubrange(match.range)
        rest = rest.trimmingCharacters(in: .whitespaces)
        var hour: Int?
        var minute = 0
        var meridiem: String?
        if let part = match.3.map({ String($0).lowercased() }) {
            hour = ["morning": 9, "afternoon": 15, "evening": 19, "night": 20][part]
        }
        if let clock = rest.firstMatch(of: clockPattern), let said = Int(clock.1), said < 24 {
            meridiem = clock.3?.lowercased()
            minute = clock.2.flatMap { Int($0) }.flatMap { $0 < 60 ? $0 : nil } ?? 0
            // Read in the part of the day said with it ("Friday evening at 7",
            // "Friday morning at 7"), as a day part does.
            if meridiem == nil, let part = hour {
                hour = part >= 12 && said < 12 ? said + 12 : said
            } else {
                hour = Self.hour(said, meridiem: meridiem)
            }
            rest.removeSubrange(clock.range)
            rest = rest.trimmingCharacters(in: .whitespaces)
        } else if takeNoon(&rest) {
            hour = 12
        }
        guard let hour else { return (day, false, rest) }
        guard var date = calendar.date(bySettingHour: hour, minute: minute, second: 0, of: day)
        else { return nil }
        // "this Friday at 9" said on Friday at 14:00 means tonight.
        if ahead == 0, date <= now, meridiem == nil, hour < 12 {
            date = date.addingTimeInterval(12 * 3600)
        }
        return (date, true, rest)
    }

    /// "this weekend" or "at the weekend" (its Saturday, or today if it is
    /// already the weekend) and "next week" (its first day), undated in
    /// time: what the owner can only say roughly still gets a day.
    private static func takeWeek(
        from text: String, now: Date, calendar: Calendar
    ) -> (Date, String)? {
        let today = ownersDay(now, calendar)
        var day: Date?
        var found: Range<String.Index>?
        if let match = text.firstMatch(of: /(?i)\b(?:this|at the|on the|over the)\s+weekend\b/) {
            let weekday = calendar.component(.weekday, from: today)
            let ahead = weekday == 1 || weekday == 7 ? 0 : 7 - weekday
            day = calendar.date(byAdding: .day, value: ahead, to: today)
            found = match.range
        } else if let match = text.firstMatch(of: /(?i)\bnext\s+week\b/),
            let week = calendar.dateInterval(of: .weekOfYear, for: today)
        {
            day = calendar.date(byAdding: .day, value: 7, to: week.start)
            found = match.range
        }
        guard let day, let found else { return nil }
        var rest = text
        rest.removeSubrange(found)
        return (day, rest.trimmingCharacters(in: .whitespaces))
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
            return !(followsArticle(text[..<found.lowerBound]) && !text[found].contains(" "))
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
            // A bare 1 to 7 o'clock is the afternoon or evening.
            let detectedHour = calendar.component(.hour, from: date)
            if !saidMeridiem, (1...7).contains(detectedHour) {
                date = date.addingTimeInterval(12 * 3600)
            }
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
            let match = text.firstMatch(of: clockPattern),
            let said = Int(match.1), said < 24
        else { return nil }
        let minute = match.2.flatMap { Int($0) } ?? 0
        guard minute < 60 else { return nil }
        let meridiem = match.3?.lowercased()
        let hour = Self.hour(said, meridiem: meridiem)
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
            // Only what a removed time leaves ("call mom at"): "log in" and
            // "pay for" keep their words.
            for dangling in [" at", " on", " by", ","] {
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
