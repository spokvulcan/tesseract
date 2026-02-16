import Foundation
import MLXLMCommon
import Tokenizers

// MARK: - Protocol

protocol AgentTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parameters: [ToolParameter] { get }
    func execute(arguments: [String: JSONValue]) async throws -> String
}

extension AgentTool {
    var toolSpec: ToolSpec {
        var properties: [String: any Sendable] = [:]
        var required: [String] = []
        for param in parameters {
            properties[param.name] = param.schema
            if param.isRequired {
                required.append(param.name)
            }
        }
        return [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": [
                    "type": "object",
                    "properties": properties,
                    "required": required,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as ToolSpec
    }
}

// MARK: - Errors

enum ToolRegistryError: LocalizedError {
    case unknownTool(String)

    var errorDescription: String? {
        switch self {
        case .unknownTool(let name):
            "Unknown tool: \(name)"
        }
    }
}
