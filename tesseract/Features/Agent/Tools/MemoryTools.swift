import Foundation
import MLXLMCommon

// MARK: - Model

struct AgentMemory: Codable, Sendable {
    let id: UUID
    let fact: String
    let category: String?
    let createdAt: Date

    init(fact: String, category: String? = nil) {
        self.id = UUID()
        self.fact = fact
        self.category = category
        self.createdAt = Date()
    }
}

// MARK: - Remember Tool

struct RememberTool: AgentTool {
    let name = "remember"
    let description = "Store a fact or preference about the user for future recall"
    let parameters: [ToolParameter] = [
        .required("fact", type: .string, description: "The fact or preference to remember"),
        .optional("category", type: .string, description: "Category for organization (e.g. health, work, preferences)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let fact = arguments.string(for: "fact") else {
            throw AgentToolError.missingArgument("fact")
        }
        let category = arguments.string(for: "category")
        let memory = AgentMemory(fact: fact, category: category)
        await store.append(memory, to: "memories.json")
        let categoryNote = category.map { " (category: \($0))" } ?? ""
        return "Remembered: \"\(fact)\"\(categoryNote)"
    }
}

// MARK: - Recall Tool

struct RecallTool: AgentTool {
    let name = "recall"
    let description = "Search stored memories by keyword"
    let parameters: [ToolParameter] = [
        .required("query", type: .string, description: "Keywords to search for in memories"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let query = arguments.string(for: "query") else {
            throw AgentToolError.missingArgument("query")
        }

        let memories: [AgentMemory] = await store.loadArray(AgentMemory.self, from: "memories.json")
        guard !memories.isEmpty else {
            return "No memories stored yet."
        }

        let queryWords = query.lowercased().split(separator: " ").map(String.init)

        // Score each memory by keyword overlap
        var scored: [(AgentMemory, Int)] = []
        for memory in memories {
            let searchText = "\(memory.fact) \(memory.category ?? "")".lowercased()
            let score = queryWords.reduce(0) { acc, word in
                acc + (searchText.contains(word) ? 1 : 0)
            }
            if score > 0 {
                scored.append((memory, score))
            }
        }

        guard !scored.isEmpty else {
            return "No memories matching \"\(query)\"."
        }

        // Sort by score descending, take top 10
        scored.sort { $0.1 > $1.1 }
        let top = scored.prefix(10)

        let formatter = DateFormatter()
        formatter.dateStyle = .medium

        var lines: [String] = ["Found \(top.count) matching memories:"]
        for (memory, _) in top {
            let date = formatter.string(from: memory.createdAt)
            let cat = memory.category.map { " [\($0)]" } ?? ""
            lines.append("- \(memory.fact)\(cat) (saved \(date))")
        }
        return lines.joined(separator: "\n")
    }
}
