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
    let description = "Create a new task."
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

        return "Done. Created task \"\(title)\"."
    }
}

// MARK: - List Tasks Tool

struct TaskListTool: AgentTool {
    let name = "task_list"
    let description = "List all pending tasks."
    let parameters: [ToolParameter] = []

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let tasks: [AgentTask] = await store.loadArray(AgentTask.self, from: "tasks.json")
        let pending = tasks.filter { $0.status == .pending }

        guard !pending.isEmpty else {
            return "No pending tasks."
        }

        var lines: [String] = ["\(pending.count) pending task(s):"]
        for (i, task) in pending.enumerated() {
            lines.append("\(i + 1). \(task.title)")
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Complete Task Tool

struct TaskCompleteTool: AgentTool {
    let name = "task_complete"
    let description = "Mark a pending task as completed by its number (as shown by task_list)."
    let parameters: [ToolParameter] = [
        .required("index", type: .int, description: "The 1-based task number from task_list"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let index = arguments.int(for: "index") else {
            throw AgentToolError.missingArgument("index")
        }

        var tasks: [AgentTask] = await store.loadArray(AgentTask.self, from: "tasks.json")
        let pending = tasks.enumerated().filter { $0.element.status == .pending }

        guard index >= 1 && index <= pending.count else {
            return "Invalid task number \(index). You have \(pending.count) pending tasks."
        }

        let targetIndex = pending[index - 1].offset
        tasks[targetIndex].status = .completed
        await store.save(tasks, to: "tasks.json")
        return "Done. Completed task \"\(tasks[targetIndex].title)\"."
    }
}
