//
//  CronExpression.swift
//  tesseract
//

import Foundation

// MARK: - CronError

nonisolated enum CronError: LocalizedError, Sendable {
    case invalidFieldCount(Int)
    case invalidField(String, position: Int, reason: String)
    case valueOutOfRange(Int, validRange: ClosedRange<Int>, position: Int)

    var errorDescription: String? {
        switch self {
        case .invalidFieldCount(let count):
            "Expected 5 fields in cron expression, got \(count)"
        case .invalidField(let field, let position, let reason):
            "Invalid cron field '\(field)' at position \(position): \(reason)"
        case .valueOutOfRange(let value, let validRange, let position):
            "Value \(value) out of range \(validRange) at position \(position)"
        }
    }
}

// MARK: - CronField

nonisolated indirect enum CronField: Sendable, Equatable {
    case any
    case value(Int)
    case range(Int, Int)
    case step(base: CronField, Int)
    case list([CronField])

    static func parse(_ string: String, validRange: ClosedRange<Int>, position: Int) throws -> CronField {
        // List: contains comma at top level
        if string.contains(",") {
            let parts = string.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            guard parts.count >= 2 else {
                throw CronError.invalidField(string, position: position, reason: "empty list element")
            }
            let fields = try parts.map { try parse($0, validRange: validRange, position: position) }
            return .list(fields)
        }

        // Step: contains /
        if string.contains("/") {
            let parts = string.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
            guard parts.count == 2 else {
                throw CronError.invalidField(string, position: position, reason: "invalid step expression")
            }
            guard let stepValue = Int(parts[1]), stepValue > 0 else {
                throw CronError.invalidField(string, position: position, reason: "step must be a positive integer")
            }
            let base = try parse(parts[0], validRange: validRange, position: position)
            return .step(base: base, stepValue)
        }

        // Range: contains -
        if string.contains("-") {
            let parts = string.split(separator: "-", omittingEmptySubsequences: false).map(String.init)
            guard parts.count == 2 else {
                throw CronError.invalidField(string, position: position, reason: "invalid range expression")
            }
            guard let low = Int(parts[0]) else {
                throw CronError.invalidField(string, position: position, reason: "non-numeric range start")
            }
            guard let high = Int(parts[1]) else {
                throw CronError.invalidField(string, position: position, reason: "non-numeric range end")
            }
            let normalizedLow = position == 4 && low == 7 ? 0 : low
            let normalizedHigh = position == 4 && high == 7 ? 0 : high
            guard validRange.contains(normalizedLow) else {
                throw CronError.valueOutOfRange(low, validRange: validRange, position: position)
            }
            guard validRange.contains(normalizedHigh) else {
                throw CronError.valueOutOfRange(high, validRange: validRange, position: position)
            }
            // Wrap-around ranges (low > high) are only valid for day-of-week
            // where 7→0 normalization can produce them (e.g., 5-7 → 5-0).
            if normalizedLow > normalizedHigh && position != 4 {
                throw CronError.invalidField(string, position: position, reason: "range start must not exceed range end")
            }
            return .range(normalizedLow, normalizedHigh)
        }

        // Any: *
        if string == "*" {
            return .any
        }

        // Value: integer
        guard let intValue = Int(string) else {
            throw CronError.invalidField(string, position: position, reason: "non-numeric value")
        }
        let normalized = position == 4 && intValue == 7 ? 0 : intValue
        guard validRange.contains(normalized) else {
            throw CronError.valueOutOfRange(intValue, validRange: validRange, position: position)
        }
        return .value(normalized)
    }

    func matches(_ value: Int, in validRange: ClosedRange<Int>) -> Bool {
        switch self {
        case .any:
            return true
        case .value(let v):
            return value == v
        case .range(let low, let high):
            if low <= high {
                return value >= low && value <= high
            } else {
                // Wrap-around range (e.g., 5-2 in dow means Fri-Tue)
                return value >= low || value <= high
            }
        case .step(let base, let step):
            switch base {
            case .any:
                return (value - validRange.lowerBound) % step == 0
            case .range(let low, let high):
                if low <= high {
                    if value < low || value > high { return false }
                    return (value - low) % step == 0
                } else {
                    // Wrap-around range (e.g., 5-0 for Fri-Sun after 7→0 normalization)
                    let expanded = CronField.range(low, high).expandedValues(in: validRange)
                    var current = 0
                    while current < expanded.count {
                        if expanded[current] == value { return true }
                        current += step
                    }
                    return false
                }
            case .value(let start):
                if value < start { return false }
                return (value - start) % step == 0
            default:
                return false
            }
        case .list(let fields):
            return fields.contains { $0.matches(value, in: validRange) }
        }
    }

    func expandedValues(in validRange: ClosedRange<Int>) -> [Int] {
        switch self {
        case .any:
            return Array(validRange)
        case .value(let v):
            return [v]
        case .range(let low, let high):
            if low <= high {
                return Array(low...high)
            } else {
                return Array(low...validRange.upperBound) + Array(validRange.lowerBound...high)
            }
        case .step(let base, let step):
            switch base {
            case .any:
                var values: [Int] = []
                var current = validRange.lowerBound
                while current <= validRange.upperBound {
                    values.append(current)
                    current += step
                }
                return values
            case .range(let low, let high):
                if low <= high {
                    var values: [Int] = []
                    var current = low
                    while current <= high {
                        values.append(current)
                        current += step
                    }
                    return values
                } else {
                    // Wrap-around: expand the full range, then take every Nth element
                    let expanded = CronField.range(low, high).expandedValues(in: validRange)
                    return stride(from: 0, to: expanded.count, by: step).map { expanded[$0] }
                }
            case .value(let v):
                var values: [Int] = []
                var current = v
                while current <= validRange.upperBound {
                    values.append(current)
                    current += step
                }
                return values
            default:
                return []
            }
        case .list(let fields):
            return fields.flatMap { $0.expandedValues(in: validRange) }.sorted()
        }
    }

    var expressionString: String {
        switch self {
        case .any:
            return "*"
        case .value(let v):
            return "\(v)"
        case .range(let low, let high):
            return "\(low)-\(high)"
        case .step(let base, let step):
            return "\(base.expressionString)/\(step)"
        case .list(let fields):
            return fields.map(\.expressionString).joined(separator: ",")
        }
    }
}

// MARK: - CronField + Codable

nonisolated extension CronField: Codable {
    private enum CodingKeys: String, CodingKey {
        case type, value, low, high, base, step, fields
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "any":
            self = .any
        case "value":
            self = .value(try container.decode(Int.self, forKey: .value))
        case "range":
            self = .range(
                try container.decode(Int.self, forKey: .low),
                try container.decode(Int.self, forKey: .high)
            )
        case "step":
            self = .step(
                base: try container.decode(CronField.self, forKey: .base),
                try container.decode(Int.self, forKey: .step)
            )
        case "list":
            self = .list(try container.decode([CronField].self, forKey: .fields))
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Unknown CronField type: \(type)")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .any:
            try container.encode("any", forKey: .type)
        case .value(let v):
            try container.encode("value", forKey: .type)
            try container.encode(v, forKey: .value)
        case .range(let low, let high):
            try container.encode("range", forKey: .type)
            try container.encode(low, forKey: .low)
            try container.encode(high, forKey: .high)
        case .step(let base, let step):
            try container.encode("step", forKey: .type)
            try container.encode(base, forKey: .base)
            try container.encode(step, forKey: .step)
        case .list(let fields):
            try container.encode("list", forKey: .type)
            try container.encode(fields, forKey: .fields)
        }
    }
}

// MARK: - CronExpression

nonisolated struct CronExpression: Codable, Sendable, Equatable {
    let minute: CronField
    let hour: CronField
    let dayOfMonth: CronField
    let month: CronField
    let dayOfWeek: CronField

    private static let fieldRanges: [ClosedRange<Int>] = [
        0...59,  // minute
        0...23,  // hour
        1...31,  // day of month
        1...12,  // month
        0...6,   // day of week (0=Sun, 7 normalized to 0)
    ]

    init(minute: CronField, hour: CronField, dayOfMonth: CronField, month: CronField, dayOfWeek: CronField) {
        self.minute = minute
        self.hour = hour
        self.dayOfMonth = dayOfMonth
        self.month = month
        self.dayOfWeek = dayOfWeek
    }

    init(parsing expression: String) throws {
        let fields = expression.split(whereSeparator: \.isWhitespace).map(String.init)
        guard fields.count == 5 else {
            throw CronError.invalidFieldCount(fields.count)
        }

        minute = try CronField.parse(fields[0], validRange: Self.fieldRanges[0], position: 0)
        hour = try CronField.parse(fields[1], validRange: Self.fieldRanges[1], position: 1)
        dayOfMonth = try CronField.parse(fields[2], validRange: Self.fieldRanges[2], position: 2)
        month = try CronField.parse(fields[3], validRange: Self.fieldRanges[3], position: 3)
        dayOfWeek = try CronField.parse(fields[4], validRange: Self.fieldRanges[4], position: 4)
    }

    func matches(_ date: Date, in timeZone: TimeZone = .current) -> Bool {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        let components = calendar.dateComponents([.minute, .hour, .day, .month, .weekday], from: date)

        guard let min = components.minute,
              let hr = components.hour,
              let dom = components.day,
              let mon = components.month,
              let wd = components.weekday else { return false }

        // Foundation weekday: 1=Sun..7=Sat → cron: 0=Sun..6=Sat
        let cronDow = wd - 1

        guard minute.matches(min, in: Self.fieldRanges[0]) else { return false }
        guard hour.matches(hr, in: Self.fieldRanges[1]) else { return false }
        guard month.matches(mon, in: Self.fieldRanges[3]) else { return false }

        // Vixie-cron OR semantics: if both dom and dow are constrained, either can match
        let domConstrained = dayOfMonth != .any
        let dowConstrained = dayOfWeek != .any

        if domConstrained && dowConstrained {
            return dayOfMonth.matches(dom, in: Self.fieldRanges[2]) ||
                   dayOfWeek.matches(cronDow, in: Self.fieldRanges[4])
        } else {
            return dayOfMonth.matches(dom, in: Self.fieldRanges[2]) &&
                   dayOfWeek.matches(cronDow, in: Self.fieldRanges[4])
        }
    }

    func nextOccurrence(after date: Date, in timeZone: TimeZone = .current) -> Date? {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone

        // Start one minute after the given date, seconds floored to 0
        guard var candidate = calendar.date(byAdding: .minute, value: 1, to: date) else { return nil }
        var components = calendar.dateComponents([.year, .month, .day, .hour, .minute], from: candidate)
        components.second = 0
        guard let floored = calendar.date(from: components) else { return nil }
        candidate = floored

        let minuteValues = minute.expandedValues(in: Self.fieldRanges[0])
        let hourValues = hour.expandedValues(in: Self.fieldRanges[1])
        let monthValues = month.expandedValues(in: Self.fieldRanges[3])
        let domValues = dayOfMonth.expandedValues(in: Self.fieldRanges[2])
        let dowValues = dayOfWeek.expandedValues(in: Self.fieldRanges[4])

        let domConstrained = dayOfMonth != .any
        let dowConstrained = dayOfWeek != .any

        func dc(_ year: Int, _ month: Int, _ day: Int, _ hour: Int, _ minute: Int) -> DateComponents {
            var nc = DateComponents()
            nc.year = year; nc.month = month; nc.day = day
            nc.hour = hour; nc.minute = minute; nc.second = 0
            return nc
        }

        // Construct a date from components. Returns (date, dstGap) where dstGap is true
        // if Calendar silently adjusted the time (spring-forward DST gap).
        func makeDate(from nc: DateComponents) -> (Date, Bool)? {
            guard let result = calendar.date(from: nc) else { return nil }
            let actual = calendar.dateComponents([.hour, .minute], from: result)
            let gapped = actual.hour != nc.hour || actual.minute != nc.minute
            return (result, gapped)
        }

        // 4-year safety cap
        let maxIterations = 366 * 4
        var iterations = 0

        while iterations < maxIterations {
            iterations += 1
            let comps = calendar.dateComponents([.year, .month, .day, .hour, .minute, .weekday], from: candidate)
            guard let cYear = comps.year,
                  let cMonth = comps.month,
                  let cDay = comps.day,
                  let cHour = comps.hour,
                  let cMinute = comps.minute,
                  let cWeekday = comps.weekday else { return nil }

            // Check month
            if !monthValues.contains(cMonth) {
                guard let nextMonth = monthValues.first(where: { $0 > cMonth }) else {
                    guard let (next, _) = makeDate(from: dc(cYear + 1, monthValues[0], 1, hourValues[0], minuteValues[0])) else { return nil }
                    candidate = next
                    continue
                }
                guard let (next, _) = makeDate(from: dc(cYear, nextMonth, 1, hourValues[0], minuteValues[0])) else { return nil }
                candidate = next
                continue
            }

            // Check day (with OR semantics)
            let cronDow = cWeekday - 1
            let dayMatches: Bool
            if domConstrained && dowConstrained {
                dayMatches = domValues.contains(cDay) || dowValues.contains(cronDow)
            } else {
                dayMatches = domValues.contains(cDay) && dowValues.contains(cronDow)
            }

            if !dayMatches {
                if let (next, _) = makeDate(from: dc(cYear, cMonth, cDay + 1, hourValues[0], minuteValues[0])) {
                    let nextComps = calendar.dateComponents([.month], from: next)
                    if nextComps.month != cMonth {
                        candidate = next
                        continue
                    }
                    candidate = next
                } else {
                    guard let (next, _) = makeDate(from: dc(cYear, cMonth + 1, 1, hourValues[0], minuteValues[0])) else { return nil }
                    candidate = next
                }
                continue
            }

            // Check hour
            if !hourValues.contains(cHour) {
                // Check if a requested hour falls in a DST gap on the current day.
                // If so, return the first valid wall-clock time after the transition
                // (i.e., the start of the post-gap hour, minute :00).
                for targetHour in hourValues where targetHour < cHour {
                    if let (adjusted, gapped) = makeDate(from: dc(cYear, cMonth, cDay, targetHour, 0)), gapped {
                        let adjComps = calendar.dateComponents([.hour], from: adjusted)
                        if adjComps.hour == cHour {
                            // The gap pushed targetHour into cHour — return first moment of post-gap hour
                            return adjusted
                        }
                    }
                }

                guard let nextHour = hourValues.first(where: { $0 > cHour }) else {
                    guard let (next, _) = makeDate(from: dc(cYear, cMonth, cDay + 1, hourValues[0], minuteValues[0])) else { return nil }
                    candidate = next
                    continue
                }
                guard let (next, _) = makeDate(from: dc(cYear, cMonth, cDay, nextHour, minuteValues[0])) else {
                    guard let next = calendar.date(byAdding: .minute, value: 1, to: candidate) else { return nil }
                    candidate = next
                    continue
                }
                candidate = next
                continue
            }

            // Check minute
            if !minuteValues.contains(cMinute) {
                guard let nextMinute = minuteValues.first(where: { $0 > cMinute }) else {
                    guard let (next, _) = makeDate(from: dc(cYear, cMonth, cDay, cHour + 1, minuteValues[0])) else { return nil }
                    candidate = next
                    continue
                }
                guard let (next, _) = makeDate(from: dc(cYear, cMonth, cDay, cHour, nextMinute)) else {
                    guard let next = calendar.date(byAdding: .minute, value: 1, to: candidate) else { return nil }
                    candidate = next
                    continue
                }
                candidate = next
                continue
            }

            // All fields match
            return candidate
        }

        return nil
    }

    var expression: String {
        [minute, hour, dayOfMonth, month, dayOfWeek]
            .map(\.expressionString)
            .joined(separator: " ")
    }

    var humanReadable: String {
        // Common patterns
        if case .any = minute, case .any = hour, case .any = dayOfMonth, case .any = month, case .any = dayOfWeek {
            return "Every minute"
        }

        if case .step(let base, let step) = minute, base == .any,
           case .any = hour, case .any = dayOfMonth, case .any = month, case .any = dayOfWeek {
            return "Every \(step) minutes"
        }

        if case .value(0) = minute, case .value(0) = hour,
           case .any = dayOfMonth, case .any = month, case .any = dayOfWeek {
            return "Every day at midnight"
        }

        if case .value(let m) = minute, case .value(let h) = hour,
           case .any = dayOfMonth, case .any = month, case .any = dayOfWeek {
            return "Every day at \(formatTime(hour: h, minute: m))"
        }

        if case .value(let m) = minute, case .value(let h) = hour,
           case .any = dayOfMonth, case .any = month, case .range(1, 5) = dayOfWeek {
            return "Weekdays at \(formatTime(hour: h, minute: m))"
        }

        if case .value(let m) = minute, case .value(let h) = hour,
           case .any = dayOfMonth, case .any = month, case .range(0, 0) = dayOfWeek {
            return "Sundays at \(formatTime(hour: h, minute: m))"
        }

        if case .value(0) = minute, case .any = hour, case .any = dayOfMonth, case .any = month, case .any = dayOfWeek {
            return "Every hour"
        }

        // Compositional fallback
        return buildDescription()
    }

    private func formatTime(hour: Int, minute: Int) -> String {
        let period = hour >= 12 ? "PM" : "AM"
        let displayHour = hour == 0 ? 12 : (hour > 12 ? hour - 12 : hour)
        return minute == 0 ? "\(displayHour):00 \(period)" : String(format: "%d:%02d \(period)", displayHour, minute)
    }

    private func buildDescription() -> String {
        var parts: [String] = []

        switch minute {
        case .any: break
        case .value(let v): parts.append("at minute \(v)")
        case .step(_, let s): parts.append("every \(s) minutes")
        default: parts.append("minutes \(minute.expressionString)")
        }

        switch hour {
        case .any: break
        case .value(let v): parts.append("at \(formatTime(hour: v, minute: 0))")
        case .step(_, let s): parts.append("every \(s) hours")
        default: parts.append("hours \(hour.expressionString)")
        }

        switch dayOfMonth {
        case .any: break
        case .value(let v): parts.append("on day \(v)")
        default: parts.append("days \(dayOfMonth.expressionString)")
        }

        switch month {
        case .any: break
        case .value(let v): parts.append("in \(monthName(v))")
        default: parts.append("months \(month.expressionString)")
        }

        switch dayOfWeek {
        case .any: break
        case .value(let v): parts.append("on \(dayName(v))")
        case .range(let lo, let hi): parts.append("\(dayName(lo))-\(dayName(hi))")
        default: parts.append("days of week \(dayOfWeek.expressionString)")
        }

        return parts.isEmpty ? expression : parts.joined(separator: ", ")
    }

    private func monthName(_ m: Int) -> String {
        let names = ["", "January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"]
        return m >= 1 && m <= 12 ? names[m] : "\(m)"
    }

    private func dayName(_ d: Int) -> String {
        let names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return d >= 0 && d <= 6 ? names[d] : "\(d)"
    }
}
