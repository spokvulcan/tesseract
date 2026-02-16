import Foundation
import MLXLMCommon

// MARK: - Model

struct MoodEntry: Codable, Sendable {
    let id: UUID
    let score: Int
    let note: String?
    let createdAt: Date

    init(score: Int, note: String? = nil) {
        self.id = UUID()
        self.score = score
        self.note = note
        self.createdAt = Date()
    }
}

// MARK: - Mood Log Tool

struct MoodLogTool: AgentTool {
    let name = "mood_log"
    let description = "Log the user's current mood on a 1-10 scale"
    let parameters: [ToolParameter] = [
        .required("score", type: .int, description: "Mood score from 1 (very low) to 10 (excellent)"),
        .optional("note", type: .string, description: "Optional note about what's affecting mood"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let score = arguments.int(for: "score") else {
            throw AgentToolError.missingArgument("score")
        }
        guard score >= 1, score <= 10 else {
            throw AgentToolError.invalidArgument("score", "must be between 1 and 10, got \(score)")
        }

        let note = arguments.string(for: "note")
        let entry = MoodEntry(score: score, note: note)
        await store.append(entry, to: "moods.json")

        let emoji = moodEmoji(score)
        var result = "Logged mood: \(score)/10 \(emoji)"
        if let note { result += " — \(note)" }
        return result
    }

    private func moodEmoji(_ score: Int) -> String {
        switch score {
        case 1...3: return "(low)"
        case 4...6: return "(moderate)"
        case 7...8: return "(good)"
        case 9...10: return "(great)"
        default: return ""
        }
    }
}

// MARK: - List Moods Tool

struct ListMoodsTool: AgentTool {
    let name = "list_moods"
    let description = "Show recent mood entries and average"
    let parameters: [ToolParameter] = [
        .optional("days", type: .int, description: "Number of days to look back (default: 7)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let days = arguments.int(for: "days") ?? 7
        let cutoff = Calendar.current.date(byAdding: .day, value: -days, to: Date())!

        let allMoods: [MoodEntry] = await store.loadArray(MoodEntry.self, from: "moods.json")
        let recent = allMoods.filter { $0.createdAt >= cutoff }
            .sorted { $0.createdAt > $1.createdAt }

        guard !recent.isEmpty else {
            return "No mood entries in the last \(days) days."
        }

        let avg = Double(recent.reduce(0) { $0 + $1.score }) / Double(recent.count)

        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short

        var lines: [String] = ["Mood over last \(days) days (avg: \(String(format: "%.1f", avg))/10):"]
        for entry in recent {
            var line = "- \(entry.score)/10 — \(formatter.string(from: entry.createdAt))"
            if let note = entry.note { line += " (\(note))" }
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }
}
