import Foundation
import MLXLMCommon

// MARK: - WebSearchTool Factory

nonisolated func createWebSearchTool() -> AgentToolDefinition {
    AgentToolDefinition(
        name: "web_search",
        label: "Web Search",
        description: "Search the web using DuckDuckGo. Returns titles, URLs, and brief snippets for selecting pages to fetch with web_fetch. Use this when you need current information, documentation, recent events, or facts beyond your training data.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "query": PropertySchema(
                    type: "string",
                    description: "The search query"
                ),
                "max_results": PropertySchema(
                    type: "integer",
                    description: "Maximum results to return (default: 5, range: 1-10)"
                ),
            ],
            required: ["query"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let query = ToolArgExtractor.string(argsJSON, key: "query") else {
                return .error("Missing required argument: query")
            }

            let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                return .error("Search query cannot be empty")
            }

            let maxResults = min(max(
                ToolArgExtractor.int(argsJSON, key: "max_results") ?? 5, 1
            ), 10)

            do {
                let results = try await DuckDuckGoClient.search(
                    query: trimmed,
                    maxResults: maxResults
                )

                if results.isEmpty {
                    return .text("No results found for: \(trimmed)")
                }

                let formatted = results.enumerated().map { i, r in
                    "[\(i + 1)] \(r.title)\n    URL: \(r.url)\n    \(r.snippet)"
                }.joined(separator: "\n\n")

                return .text("Search results for: \(trimmed)\n\n\(formatted)")
            } catch {
                Log.agent.warning("[WebSearch] Search failed for '\(trimmed)': \(error)")
                return .error("Web search failed: \(error.localizedDescription)")
            }
        }
    )
}
