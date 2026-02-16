import Foundation
import MLXLMCommon

// MARK: - Models

struct Goal: Codable, Sendable {
    let id: UUID
    var name: String
    var description: String?
    var category: String?
    var targetDate: Date?
    var status: GoalStatus
    var progressNotes: [ProgressNote]
    let createdAt: Date

    init(name: String, description: String? = nil, category: String? = nil, targetDate: Date? = nil) {
        self.id = UUID()
        self.name = name
        self.description = description
        self.category = category
        self.targetDate = targetDate
        self.status = .active
        self.progressNotes = []
        self.createdAt = Date()
    }
}

enum GoalStatus: String, Codable, Sendable {
    case active, completed, archived
}

struct ProgressNote: Codable, Sendable {
    let note: String
    let date: Date

    init(note: String) {
        self.note = note
        self.date = Date()
    }
}

// MARK: - Create Goal Tool

struct CreateGoalTool: AgentTool {
    let name = "create_goal"
    let description = "Create a new goal to track"
    let parameters: [ToolParameter] = [
        .required("name", type: .string, description: "Name of the goal"),
        .optional("description", type: .string, description: "Detailed description"),
        .optional("target_date", type: .string, description: "Target completion date (ISO8601 or natural language)"),
        .optional("category", type: .string, description: "Category (e.g. health, career, personal)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let name = arguments.string(for: "name") else {
            throw AgentToolError.missingArgument("name")
        }
        let description = arguments.string(for: "description")
        let category = arguments.string(for: "category")
        let targetDate = arguments.string(for: "target_date").flatMap { DateParsingUtility.parse($0) }

        let goal = Goal(name: name, description: description, category: category, targetDate: targetDate)
        await store.append(goal, to: "goals.json")

        var result = "Created goal: \"\(name)\" (id: \(goal.id.uuidString.prefix(8)))"
        if let targetDate {
            let formatter = DateFormatter()
            formatter.dateStyle = .medium
            result += " — target: \(formatter.string(from: targetDate))"
        }
        return result
    }
}

// MARK: - List Goals Tool

struct ListGoalsTool: AgentTool {
    let name = "list_goals"
    let description = "List goals, optionally filtered by category or status"
    let parameters: [ToolParameter] = [
        .optional("category", type: .string, description: "Filter by category"),
        .optional("status", type: .string, description: "Filter by status: active, completed, or archived"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        var goals: [Goal] = await store.loadArray(Goal.self, from: "goals.json")

        if let category = arguments.string(for: "category") {
            goals = goals.filter { ($0.category ?? "").lowercased() == category.lowercased() }
        }
        if let statusStr = arguments.string(for: "status"), let status = GoalStatus(rawValue: statusStr.lowercased()) {
            goals = goals.filter { $0.status == status }
        }

        guard !goals.isEmpty else {
            return "No goals found."
        }

        let formatter = DateFormatter()
        formatter.dateStyle = .medium

        var lines: [String] = ["\(goals.count) goal(s):"]
        for goal in goals {
            var line = "- [\(goal.status.rawValue.uppercased())] \(goal.name) (id: \(goal.id.uuidString.prefix(8)))"
            if let cat = goal.category { line += " [\(cat)]" }
            if let target = goal.targetDate { line += " — due: \(formatter.string(from: target))" }
            if !goal.progressNotes.isEmpty {
                line += " — \(goal.progressNotes.count) update(s)"
            }
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Update Goal Tool

struct UpdateGoalTool: AgentTool {
    let name = "update_goal"
    let description = "Add a progress note or change the status of a goal"
    let parameters: [ToolParameter] = [
        .required("goal_id", type: .string, description: "Goal ID (first 8 characters suffice)"),
        .optional("progress_note", type: .string, description: "Progress update to add"),
        .optional("status", type: .string, description: "New status: active, completed, or archived"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let goalId = arguments.string(for: "goal_id") else {
            throw AgentToolError.missingArgument("goal_id")
        }

        var goals: [Goal] = await store.loadArray(Goal.self, from: "goals.json")
        let prefix = goalId.lowercased()
        guard let idx = goals.firstIndex(where: { $0.id.uuidString.lowercased().hasPrefix(prefix) }) else {
            return "No goal found with ID starting with \"\(goalId)\"."
        }

        var updates: [String] = []

        if let note = arguments.string(for: "progress_note") {
            goals[idx].progressNotes.append(ProgressNote(note: note))
            updates.append("added progress note")
        }
        if let statusStr = arguments.string(for: "status"), let status = GoalStatus(rawValue: statusStr.lowercased()) {
            goals[idx].status = status
            updates.append("status → \(status.rawValue)")
        }

        guard !updates.isEmpty else {
            return "No updates provided. Use progress_note or status."
        }

        await store.save(goals, to: "goals.json")
        return "Updated goal \"\(goals[idx].name)\": \(updates.joined(separator: ", "))"
    }
}
