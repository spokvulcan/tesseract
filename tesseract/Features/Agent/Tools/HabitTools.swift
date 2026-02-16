import Foundation
import MLXLMCommon

// MARK: - Models

struct Habit: Codable, Sendable {
    let id: UUID
    let name: String
    let frequency: HabitFrequency
    var timeOfDay: String?
    let createdAt: Date
    var archived: Bool

    init(name: String, frequency: HabitFrequency, timeOfDay: String? = nil) {
        self.id = UUID()
        self.name = name
        self.frequency = frequency
        self.timeOfDay = timeOfDay
        self.createdAt = Date()
        self.archived = false
    }
}

enum HabitFrequency: String, Codable, Sendable {
    case daily, weekdays, weekly
}

struct HabitLogEntry: Codable, Sendable {
    let id: UUID
    let habitId: UUID
    let date: String  // yyyy-MM-dd
    var note: String?
    let loggedAt: Date

    init(habitId: UUID, date: String, note: String? = nil) {
        self.id = UUID()
        self.habitId = habitId
        self.date = date
        self.note = note
        self.loggedAt = Date()
    }
}

// MARK: - Create Habit Tool

struct CreateHabitTool: AgentTool {
    let name = "create_habit"
    let description = "Create a new habit to track"
    let parameters: [ToolParameter] = [
        .required("name", type: .string, description: "Name of the habit"),
        .required("frequency", type: .string, description: "Frequency: daily, weekdays, or weekly"),
        .optional("time_of_day", type: .string, description: "Preferred time (e.g. morning, evening, 7am)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let name = arguments.string(for: "name") else {
            throw AgentToolError.missingArgument("name")
        }
        guard let freqStr = arguments.string(for: "frequency"),
              let frequency = HabitFrequency(rawValue: freqStr.lowercased())
        else {
            throw AgentToolError.invalidArgument("frequency", "must be daily, weekdays, or weekly")
        }

        // Check for duplicate name
        let habits: [Habit] = await store.loadArray(Habit.self, from: "habits.json")
        if habits.contains(where: { $0.name.lowercased() == name.lowercased() && !$0.archived }) {
            return "A habit named \"\(name)\" already exists."
        }

        let timeOfDay = arguments.string(for: "time_of_day")
        let habit = Habit(name: name, frequency: frequency, timeOfDay: timeOfDay)
        await store.append(habit, to: "habits.json")

        var result = "Created habit: \"\(name)\" (\(frequency.rawValue))"
        if let time = timeOfDay { result += " — \(time)" }
        return result
    }
}

// MARK: - Log Habit Tool

struct LogHabitTool: AgentTool {
    let name = "log_habit"
    let description = "Log completion of a habit for today (or a specific date)"
    let parameters: [ToolParameter] = [
        .required("habit_name", type: .string, description: "Name of the habit (case-insensitive)"),
        .optional("date", type: .string, description: "Date to log for (default: today, format: YYYY-MM-DD)"),
        .optional("note", type: .string, description: "Optional note about the session"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let habitName = arguments.string(for: "habit_name") else {
            throw AgentToolError.missingArgument("habit_name")
        }

        let habits: [Habit] = await store.loadArray(Habit.self, from: "habits.json")
        guard let habit = habits.first(where: { $0.name.lowercased() == habitName.lowercased() && !$0.archived }) else {
            return "No active habit named \"\(habitName)\". Use create_habit first."
        }

        let dateStr: String
        if let providedDate = arguments.string(for: "date") {
            dateStr = providedDate
        } else {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd"
            dateStr = formatter.string(from: Date())
        }

        // Check for duplicate log on same day
        let logs: [HabitLogEntry] = await store.loadArray(HabitLogEntry.self, from: "habit_logs.json")
        if logs.contains(where: { $0.habitId == habit.id && $0.date == dateStr }) {
            return "\"\(habit.name)\" already logged for \(dateStr)."
        }

        let note = arguments.string(for: "note")
        let entry = HabitLogEntry(habitId: habit.id, date: dateStr, note: note)
        await store.append(entry, to: "habit_logs.json")

        return "Logged \"\(habit.name)\" for \(dateStr)."
    }
}

// MARK: - Habit Status Tool

struct HabitStatusTool: AgentTool {
    let name = "habit_status"
    let description = "Get habit tracking statistics including streaks and completion rates"
    let parameters: [ToolParameter] = [
        .optional("habit_name", type: .string, description: "Specific habit name (shows all if omitted)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let habits: [Habit] = await store.loadArray(Habit.self, from: "habits.json")
            .filter { !$0.archived }
        let allLogs: [HabitLogEntry] = await store.loadArray(HabitLogEntry.self, from: "habit_logs.json")

        let targetHabits: [Habit]
        if let name = arguments.string(for: "habit_name") {
            targetHabits = habits.filter { $0.name.lowercased() == name.lowercased() }
            if targetHabits.isEmpty {
                return "No active habit named \"\(name)\"."
            }
        } else {
            targetHabits = habits
        }

        guard !targetHabits.isEmpty else {
            return "No habits being tracked yet."
        }

        var lines: [String] = []
        for habit in targetHabits {
            let logs = allLogs.filter { $0.habitId == habit.id }
            let logDates = Set(logs.map(\.date))
            let stats = calculateStreak(frequency: habit.frequency, logDates: logDates, since: habit.createdAt)

            var line = "**\(habit.name)** (\(habit.frequency.rawValue))"
            line += "\n  Current streak: \(stats.currentStreak) | Longest: \(stats.longestStreak)"
            line += "\n  Completion rate: \(stats.completionRate)% | Total: \(logDates.count)"
            lines.append(line)
        }
        return lines.joined(separator: "\n\n")
    }

    // MARK: - Streak Algorithm

    private struct HabitStats {
        let currentStreak: Int
        let longestStreak: Int
        let completionRate: Int
    }

    private func calculateStreak(frequency: HabitFrequency, logDates: Set<String>, since createdAt: Date) -> HabitStats {
        let calendar = Calendar.current
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        let today = calendar.startOfDay(for: Date())

        // Build expected dates from creation date to today
        var expectedDates: [String] = []
        var cursor = calendar.startOfDay(for: createdAt)
        while cursor <= today {
            let weekday = calendar.component(.weekday, from: cursor)
            let include: Bool
            switch frequency {
            case .daily:
                include = true
            case .weekdays:
                include = weekday >= 2 && weekday <= 6  // Mon-Fri
            case .weekly:
                include = weekday == 2  // Just use Monday as the check day
            }
            if include {
                expectedDates.append(formatter.string(from: cursor))
            }
            cursor = calendar.date(byAdding: .day, value: 1, to: cursor)!
        }

        guard !expectedDates.isEmpty else {
            return HabitStats(currentStreak: 0, longestStreak: 0, completionRate: 0)
        }

        // For weekly frequency, check if any day in each ISO week has a log
        let completedDates: Set<String>
        if frequency == .weekly {
            var weeklyCompleted = Set<String>()
            for dateStr in expectedDates {
                if let date = formatter.date(from: dateStr) {
                    let weekOfYear = calendar.component(.weekOfYear, from: date)
                    let year = calendar.component(.yearForWeekOfYear, from: date)
                    // Check if any log exists in this week
                    let hasLog = logDates.contains { logDateStr in
                        guard let logDate = formatter.date(from: logDateStr) else { return false }
                        return calendar.component(.weekOfYear, from: logDate) == weekOfYear
                            && calendar.component(.yearForWeekOfYear, from: logDate) == year
                    }
                    if hasLog { weeklyCompleted.insert(dateStr) }
                }
            }
            completedDates = weeklyCompleted
        } else {
            completedDates = logDates
        }

        // Walk backwards for current streak (grace: today doesn't need to be logged)
        var currentStreak = 0
        var startIdx = expectedDates.count - 1
        // If today isn't logged, give grace and start from yesterday
        if !completedDates.contains(expectedDates[startIdx]) {
            startIdx -= 1
        }
        for i in stride(from: startIdx, through: 0, by: -1) {
            if completedDates.contains(expectedDates[i]) {
                currentStreak += 1
            } else {
                break
            }
        }

        // Longest streak
        var longestStreak = 0
        var streak = 0
        for dateStr in expectedDates {
            if completedDates.contains(dateStr) {
                streak += 1
                longestStreak = max(longestStreak, streak)
            } else {
                streak = 0
            }
        }

        // Completion rate
        let completed = expectedDates.filter { completedDates.contains($0) }.count
        let rate = expectedDates.isEmpty ? 0 : (completed * 100) / expectedDates.count

        return HabitStats(currentStreak: currentStreak, longestStreak: longestStreak, completionRate: rate)
    }
}
