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

// MARK: - Memory Save Tool

struct MemorySaveTool: AgentTool {
    let name = "memory_save"
    let description = "Save a fact, preference, or important information about the user to long-term memory. Reports if a similar memory already exists."
    let parameters: [ToolParameter] = [
        .required("fact", type: .string, description: "The fact or preference to save"),
        .optional("category", type: .string, description: "Category for organization (e.g. preference, health, work, personal)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let fact = arguments.string(for: "fact") else {
            throw AgentToolError.missingArgument("fact")
        }
        let category = arguments.string(for: "category")

        // Check for duplicate/similar fact
        let memories: [AgentMemory] = await store.loadArray(AgentMemory.self, from: "memories.json")
        let factLower = fact.lowercased()
        if let existing = memories.first(where: { $0.fact.lowercased() == factLower }) {
            return "Already saved: \"\(existing.fact)\""
        }

        let memory = AgentMemory(fact: fact, category: category)
        await store.append(memory, to: "memories.json")
        let catLabel = category.map { " [\($0)]" } ?? ""
        return "Saved to memory\(catLabel): \"\(fact)\""
    }
}

// MARK: - Memory Search Tool

struct MemorySearchTool: AgentTool {
    let name = "memory_search"
    let description = "Search saved memories by keyword. Returns matching facts and preferences."
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
