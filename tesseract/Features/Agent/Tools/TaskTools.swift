import Foundation
import MLXLMCommon

// MARK: - Model

struct AgentTask: Codable, Sendable {
    let id: UUID
    let title: String
    var status: AgentTaskStatus
    var dueDate: Date?
    var goalId: UUID?
    var priority: TaskPriority
    let createdAt: Date

    init(title: String, dueDate: Date? = nil, goalId: UUID? = nil, priority: TaskPriority = .medium) {
        self.id = UUID()
        self.title = title
        self.status = .pending
        self.dueDate = dueDate
        self.goalId = goalId
        self.priority = priority
        self.createdAt = Date()
    }
}

enum AgentTaskStatus: String, Codable, Sendable {
    case pending, completed
}

enum TaskPriority: String, Codable, Sendable {
    case low, medium, high
}

// MARK: - Create Task Tool

struct CreateTaskTool: AgentTool {
    let name = "create_task"
    let description = "Create a new task"
    let parameters: [ToolParameter] = [
        .required("title", type: .string, description: "Task title"),
        .optional("due_date", type: .string, description: "Due date (ISO8601 or natural language like 'tomorrow', 'in 3 days')"),
        .optional("goal_id", type: .string, description: "Link to a goal by ID"),
        .optional("priority", type: .string, description: "Priority: low, medium, or high (default: medium)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let title = arguments.string(for: "title") else {
            throw AgentToolError.missingArgument("title")
        }
        let dueDate = arguments.string(for: "due_date").flatMap { DateParsingUtility.parse($0) }
        let goalId = arguments.string(for: "goal_id").flatMap { UUID(uuidString: $0) }
        let priority = arguments.string(for: "priority")
            .flatMap { TaskPriority(rawValue: $0.lowercased()) } ?? .medium

        let task = AgentTask(title: title, dueDate: dueDate, goalId: goalId, priority: priority)
        await store.append(task, to: "tasks.json")

        var result = "Created task: \"\(title)\" [\(priority.rawValue)] (id: \(task.id.uuidString.prefix(8)))"
        if let dueDate {
            let formatter = DateFormatter()
            formatter.dateStyle = .medium
            result += " — due: \(formatter.string(from: dueDate))"
        }
        return result
    }
}

// MARK: - List Tasks Tool

struct ListTasksTool: AgentTool {
    let name = "list_tasks"
    let description = "List tasks, optionally filtered by status or date"
    let parameters: [ToolParameter] = [
        .optional("filter", type: .string, description: "Filter: today, upcoming, overdue, or all (default: all)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let tasks: [AgentTask] = await store.loadArray(AgentTask.self, from: "tasks.json")
        let filter = arguments.string(for: "filter")?.lowercased() ?? "all"

        let calendar = Calendar.current
        let now = Date()
        let startOfToday = calendar.startOfDay(for: now)
        let endOfToday = calendar.date(byAdding: .day, value: 1, to: startOfToday)!

        let filtered: [AgentTask]
        switch filter {
        case "today":
            filtered = tasks.filter { task in
                task.status == .pending && task.dueDate.map { $0 >= startOfToday && $0 < endOfToday } ?? false
            }
        case "upcoming":
            filtered = tasks.filter { task in
                task.status == .pending && task.dueDate.map { $0 >= endOfToday } ?? false
            }
        case "overdue":
            filtered = tasks.filter { task in
                task.status == .pending && task.dueDate.map { $0 < startOfToday } ?? false
            }
        default:
            filtered = tasks.filter { $0.status == .pending }
        }

        guard !filtered.isEmpty else {
            return filter == "all" ? "No pending tasks." : "No \(filter) tasks."
        }

        let formatter = DateFormatter()
        formatter.dateStyle = .medium

        let sorted = filtered.sorted { a, b in
            // High priority first, then by due date
            if a.priority != b.priority {
                return priorityOrder(a.priority) > priorityOrder(b.priority)
            }
            if let ad = a.dueDate, let bd = b.dueDate { return ad < bd }
            if a.dueDate != nil { return true }
            return false
        }

        var lines: [String] = ["\(sorted.count) task(s):"]
        for task in sorted {
            var line = "- [\(task.priority.rawValue.uppercased())] \(task.title) (id: \(task.id.uuidString.prefix(8)))"
            if let due = task.dueDate {
                let label = due < startOfToday ? "OVERDUE" : formatter.string(from: due)
                line += " — due: \(label)"
            }
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }

    private func priorityOrder(_ p: TaskPriority) -> Int {
        switch p {
        case .low: 0
        case .medium: 1
        case .high: 2
        }
    }
}

// MARK: - Complete Task Tool

struct CompleteTaskTool: AgentTool {
    let name = "complete_task"
    let description = "Mark a task as completed"
    let parameters: [ToolParameter] = [
        .required("task_id", type: .string, description: "Task ID (first 8 characters suffice)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let taskId = arguments.string(for: "task_id") else {
            throw AgentToolError.missingArgument("task_id")
        }

        var tasks: [AgentTask] = await store.loadArray(AgentTask.self, from: "tasks.json")
        let prefix = taskId.lowercased()
        guard let idx = tasks.firstIndex(where: { $0.id.uuidString.lowercased().hasPrefix(prefix) }) else {
            return "No task found with ID starting with \"\(taskId)\"."
        }

        if tasks[idx].status == .completed {
            return "Task \"\(tasks[idx].title)\" is already completed."
        }

        tasks[idx].status = .completed
        await store.save(tasks, to: "tasks.json")
        return "Completed task: \"\(tasks[idx].title)\""
    }
}
