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

struct TaskCreateTool: AgentTool {
    let name = "task_create"
    let description = "Create a new task. Do not call task_complete in the same response."
    let parameters: [ToolParameter] = [
        .required("title", type: .string, description: "Task title"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let title = arguments.string(for: "title") else {
            throw AgentToolError.missingArgument("title")
        }

        let task = AgentTask(title: title)
        await store.append(task, to: "tasks.json")

        return "Done. Created task \"\(title)\" [id: \(task.id.uuidString.prefix(8))]."
    }
}

// MARK: - List Tasks Tool

struct TaskListTool: AgentTool {
    let name = "task_list"
    let description = "List all pending tasks. Only call once per response."
    let parameters: [ToolParameter] = []

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let tasks: [AgentTask] = await store.loadArray(AgentTask.self, from: "tasks.json")
        let pending = tasks.filter { $0.status == .pending }

        guard !pending.isEmpty else {
            return "No pending tasks."
        }

        var lines: [String] = ["\(pending.count) task(s):"]
        for task in pending {
            lines.append("- \(task.title) [id: \(task.id.uuidString.prefix(8))]")
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Complete Task Tool

struct TaskCompleteTool: AgentTool {
    let name = "task_complete"
    let description = "Mark one task as completed by ID. Do not call task_create in the same response."
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
        return "Done. Completed task \"\(tasks[idx].title)\"."
    }
}
