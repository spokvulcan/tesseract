import Foundation

/// Multi-strategy date parser supporting ISO8601, relative times, and natural language.
enum DateParsingUtility {

    /// Attempts to parse a date string using multiple strategies.
    /// Returns `nil` if no strategy succeeds.
    static func parse(_ input: String) -> Date? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        // Strategy 1: ISO8601 full (2026-02-16T14:30:00Z)
        if let date = parseISO8601Full(trimmed) { return date }

        // Strategy 2: ISO8601 date-only (2026-02-16)
        if let date = parseISO8601DateOnly(trimmed) { return date }

        // Strategy 3: Relative ("in 30 minutes", "in 2 hours", "in 3 days")
        if let date = parseRelative(trimmed) { return date }

        // Strategy 4: "tomorrow" with optional time
        if let date = parseTomorrow(trimmed) { return date }

        // Strategy 5: "at 3pm", "at 15:00"
        if let date = parseAtTime(trimmed) { return date }

        return nil
    }

    // MARK: - Strategies

    private static func parseISO8601Full(_ input: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        if let date = formatter.date(from: input) { return date }
        // Try without timezone
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: input)
    }

    private static func parseISO8601DateOnly(_ input: String) -> Date? {
        let pattern = #"^\d{4}-\d{2}-\d{2}$"#
        guard input.range(of: pattern, options: .regularExpression) != nil else { return nil }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        guard let date = formatter.date(from: input) else { return nil }
        // Set to noon to avoid timezone edge cases
        return Calendar.current.date(bySettingHour: 12, minute: 0, second: 0, of: date)
    }

    private static func parseRelative(_ input: String) -> Date? {
        let pattern = #"^in\s+(\d+)\s+(minute|minutes|min|mins|hour|hours|hr|hrs|day|days|week|weeks)$"#
        guard let match = input.range(of: pattern, options: .regularExpression) else { return nil }
        let text = String(input[match])

        // Extract number and unit
        let numberPattern = #"(\d+)"#
        guard let numRange = text.range(of: numberPattern, options: .regularExpression) else { return nil }
        guard let amount = Int(text[numRange]) else { return nil }

        let now = Date()
        if text.contains("minute") || text.contains("min") {
            return Calendar.current.date(byAdding: .minute, value: amount, to: now)
        } else if text.contains("hour") || text.contains("hr") {
            return Calendar.current.date(byAdding: .hour, value: amount, to: now)
        } else if text.contains("day") {
            return Calendar.current.date(byAdding: .day, value: amount, to: now)
        } else if text.contains("week") {
            return Calendar.current.date(byAdding: .weekOfYear, value: amount, to: now)
        }
        return nil
    }

    private static func parseTomorrow(_ input: String) -> Date? {
        guard input.hasPrefix("tomorrow") else { return nil }
        guard let tomorrow = Calendar.current.date(byAdding: .day, value: 1, to: Date()) else { return nil }

        // "tomorrow at 3pm" or just "tomorrow"
        let afterTomorrow = input.dropFirst("tomorrow".count).trimmingCharacters(in: .whitespaces)
        if afterTomorrow.isEmpty {
            return Calendar.current.date(bySettingHour: 9, minute: 0, second: 0, of: tomorrow)
        }
        let timeStr = afterTomorrow.hasPrefix("at ") ? String(afterTomorrow.dropFirst(3)) : afterTomorrow
        if let (hour, minute) = parseTimeComponents(timeStr) {
            return Calendar.current.date(bySettingHour: hour, minute: minute, second: 0, of: tomorrow)
        }
        return Calendar.current.date(bySettingHour: 9, minute: 0, second: 0, of: tomorrow)
    }

    private static func parseAtTime(_ input: String) -> Date? {
        let timeStr: String
        if input.hasPrefix("at ") {
            timeStr = String(input.dropFirst(3))
        } else {
            return nil
        }

        guard let (hour, minute) = parseTimeComponents(timeStr) else { return nil }

        let now = Date()
        var target = Calendar.current.date(bySettingHour: hour, minute: minute, second: 0, of: now)!

        // If the time has already passed today, advance to tomorrow
        if target <= now {
            target = Calendar.current.date(byAdding: .day, value: 1, to: target)!
        }
        return target
    }

    // MARK: - Helpers

    /// Parses time strings like "3pm", "3:30pm", "15:00", "3:30 pm"
    private static func parseTimeComponents(_ input: String) -> (hour: Int, minute: Int)? {
        let clean = input.trimmingCharacters(in: .whitespaces)

        // 24-hour format: "15:00", "9:30"
        let h24 = #"^(\d{1,2}):(\d{2})$"#
        if clean.range(of: h24, options: .regularExpression) != nil {
            let parts = clean.split(separator: ":")
            if let h = Int(parts[0]), let m = Int(parts[1]), h >= 0, h < 24, m >= 0, m < 60 {
                return (h, m)
            }
        }

        // 12-hour format: "3pm", "3:30pm", "3 pm", "12:30 am"
        let h12 = #"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$"#
        if let match = clean.wholeMatch(of: /^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$/) {
            guard let h = Int(match.1) else { return nil }
            let m = match.2.flatMap { Int($0) } ?? 0
            let isPM = match.3 == "pm"
            var hour = h
            if isPM && hour != 12 { hour += 12 }
            if !isPM && hour == 12 { hour = 0 }
            if hour >= 0, hour < 24, m >= 0, m < 60 {
                return (hour, m)
            }
        }

        return nil
    }
}
