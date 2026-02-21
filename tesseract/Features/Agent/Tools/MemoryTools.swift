import Foundation
import MLXLMCommon

// MARK: - Model

struct AgentMemory: Codable, Sendable {
    let text: String
    let createdAt: Date

    init(text: String) {
        self.text = text
        self.createdAt = Date()
    }

    // Backward-compatible decoding: supports both new `text` and legacy `fact` key.
    // Extra fields (id, category) in old files are silently ignored.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.text = try container.decodeIfPresent(String.self, forKey: .text)
            ?? container.decode(String.self, forKey: .fact)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
    }

    private enum CodingKeys: String, CodingKey {
        case text, fact, createdAt
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(text, forKey: .text)
        try container.encode(createdAt, forKey: .createdAt)
    }
}

// MARK: - Memory Save Tool

struct MemorySaveTool: AgentTool {
    let name = "memory_save"
    let description = "Save a fact, preference, or important information about the user to long-term memory."
    let parameters: [ToolParameter] = [
        .required("text", type: .string, description: "The fact or preference to save"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let text = arguments.string(for: "text") else {
            throw AgentToolError.missingArgument("text")
        }

        let memory = AgentMemory(text: text)
        await store.append(memory, to: "memories.json")
        return "Saved to memory: \"\(text)\""
    }
}

// MARK: - Memory Update Tool

struct MemoryUpdateTool: AgentTool {
    let name = "memory_update"
    let description = "Update an existing memory by its number (as shown in 'What I Know About You'). Use this to consolidate or correct memories."
    let parameters: [ToolParameter] = [
        .required("index", type: .int, description: "The 1-based memory number to update"),
        .required("text", type: .string, description: "The new text for this memory"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let index = arguments.int(for: "index") else {
            throw AgentToolError.missingArgument("index")
        }
        guard let text = arguments.string(for: "text") else {
            throw AgentToolError.missingArgument("text")
        }

        var memories: [AgentMemory] = await store.loadArray(AgentMemory.self, from: "memories.json")
        guard index >= 1 && index <= memories.count else {
            return "Invalid memory number \(index). You have \(memories.count) memories."
        }

        let old = memories[index - 1].text
        memories[index - 1] = AgentMemory(text: text)
        await store.save(memories, to: "memories.json")
        return "Updated memory #\(index): \"\(old)\" → \"\(text)\""
    }
}

// MARK: - Memory Delete Tool

struct MemoryDeleteTool: AgentTool {
    let name = "memory_delete"
    let description = "Delete one or more memories by their numbers (as shown in 'What I Know About You'). Pass all numbers at once to avoid index shifting."
    let parameters: [ToolParameter] = [
        .required("indices", type: .array(elementType: .int), description: "The 1-based memory numbers to delete, e.g. [3, 4, 5]"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let indices = arguments.intArray(for: "indices"), !indices.isEmpty else {
            throw AgentToolError.missingArgument("indices")
        }

        var memories: [AgentMemory] = await store.loadArray(AgentMemory.self, from: "memories.json")
        let count = memories.count

        // Validate all indices first
        let invalid = indices.filter { $0 < 1 || $0 > count }
        if !invalid.isEmpty {
            return "Invalid memory numbers: \(invalid). You have \(count) memories."
        }

        // Remove in descending order so earlier indices stay valid
        let sorted = Set(indices).sorted(by: >)
        var deleted: [String] = []
        for idx in sorted {
            deleted.append(memories[idx - 1].text)
            memories.remove(at: idx - 1)
        }

        await store.save(memories, to: "memories.json")

        if deleted.count == 1 {
            return "Deleted memory: \"\(deleted[0])\""
        }
        let list = deleted.reversed().map { "\"\($0)\"" }.joined(separator: ", ")
        return "Deleted \(deleted.count) memories: \(list)"
    }
}
