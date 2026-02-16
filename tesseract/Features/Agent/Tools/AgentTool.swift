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

// MARK: - JSONValue Extraction Helpers

extension [String: JSONValue] {
    func string(for key: String) -> String? {
        guard let value = self[key] else { return nil }
        switch value {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return String(b)
        case .null: return nil
        default: return nil
        }
    }

    func int(for key: String) -> Int? {
        guard let value = self[key] else { return nil }
        switch value {
        case .int(let i): return i
        case .double(let d): return Int(d)
        case .string(let s): return Int(s)
        case .null: return nil
        default: return nil
        }
    }

    func bool(for key: String) -> Bool? {
        guard let value = self[key] else { return nil }
        switch value {
        case .bool(let b): return b
        case .int(let i): return i != 0
        case .string(let s): return s == "true" || s == "1"
        case .null: return nil
        default: return nil
        }
    }
}

// MARK: - Tool Errors

enum AgentToolError: LocalizedError {
    case missingArgument(String)
    case invalidArgument(String, String)

    var errorDescription: String? {
        switch self {
        case .missingArgument(let name):
            "Missing required argument: \(name)"
        case .invalidArgument(let name, let reason):
            "Invalid argument '\(name)': \(reason)"
        }
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
