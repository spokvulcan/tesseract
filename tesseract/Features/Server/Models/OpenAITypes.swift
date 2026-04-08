//
//  OpenAITypes.swift
//  tesseract
//

import Foundation

/// Namespace for OpenAI-compatible API types.
/// Avoids name collisions with internal types (e.g. `ToolCall` from MLXLMCommon).
nonisolated enum OpenAI {

    // MARK: - Chat Completion Request

    struct ChatCompletionRequest: Codable, Sendable {
        var model: String?
        var messages: [ChatMessage]
        var tools: [ToolDefinition]?
        var stream: Bool?
        var max_tokens: Int?
        var max_completion_tokens: Int?
        var temperature: Double?
        var top_p: Double?
        var reasoning_effort: String?
        var stream_options: StreamOptions?
        var stop: StopSequence?

        nonisolated var effectiveMaxTokens: Int? {
            max_completion_tokens ?? max_tokens
        }
    }

    enum StopSequence: Codable, Sendable {
        case single(String)
        case multiple([String])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let string = try? container.decode(String.self) {
                self = .single(string)
            } else {
                self = .multiple(try container.decode([String].self))
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .single(let string):
                try container.encode(string)
            case .multiple(let array):
                try container.encode(array)
            }
        }

        var sequences: [String] {
            switch self {
            case .single(let s): [s]
            case .multiple(let a): a
            }
        }
    }

    struct StreamOptions: Codable, Sendable {
        var include_usage: Bool?
    }

    // MARK: - Chat Message

    struct ChatMessage: Codable, Sendable {
        var role: ChatRole
        var content: MessageContent?
        var tool_calls: [ToolCall]?
        var tool_call_id: String?
        var reasoning_content: String?
        var reasoning: String?

        init(
            role: ChatRole,
            content: MessageContent? = nil,
            tool_calls: [ToolCall]? = nil,
            tool_call_id: String? = nil,
            reasoning_content: String? = nil,
            reasoning: String? = nil
        ) {
            self.role = role
            self.content = content
            self.tool_calls = tool_calls
            self.tool_call_id = tool_call_id
            self.reasoning_content = reasoning_content
            self.reasoning = reasoning
        }

        nonisolated var resolvedReasoningContent: String? {
            let value = reasoning_content ?? reasoning
            let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
            return (trimmed?.isEmpty ?? true) ? nil : trimmed
        }
    }

    enum ChatRole: String, Codable, Sendable {
        case system
        case user
        case assistant
        case tool
    }

    enum MessageContent: Codable, Sendable {
        case text(String)
        case parts([ContentPart])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let string = try? container.decode(String.self) {
                self = .text(string)
            } else {
                self = .parts(try container.decode([ContentPart].self))
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .text(let string):
                try container.encode(string)
            case .parts(let parts):
                try container.encode(parts)
            }
        }

        nonisolated var textValue: String? {
            switch self {
            case .text(let s):
                return s
            case .parts(let parts):
                let texts = parts.compactMap(\.text)
                return texts.isEmpty ? nil : texts.joined(separator: "\n")
            }
        }
    }

    // MARK: - Content Parts

    struct ContentPart: Codable, Sendable {
        var type: ContentPartType
        var text: String?
        var image_url: ImageURL?
    }

    enum ContentPartType: String, Codable, Sendable {
        case text
        case image_url
    }

    struct ImageURL: Codable, Sendable {
        var url: String
    }

    // MARK: - Tool Definitions

    struct ToolDefinition: Codable, Sendable {
        var type: String
        var function: FunctionDefinition
    }

    struct FunctionDefinition: Codable, Sendable {
        var name: String
        var description: String?
        var parameters: AnyCodableValue?
    }

    // MARK: - Tool Calls

    struct ToolCall: Codable, Sendable {
        var id: String?
        var type: String?
        var function: FunctionCall?
        var index: Int?
    }

    struct FunctionCall: Codable, Sendable {
        var name: String?
        var arguments: String?
    }

    // MARK: - Chat Completion Response (non-streaming)

    struct ChatCompletionResponse: Codable, Sendable {
        var id: String
        var object: String = "chat.completion"
        var model: String
        var created: Int
        var system_fingerprint: String?
        var choices: [ChatCompletionChoice]
        var usage: Usage?
    }

    struct ChatCompletionChoice: Codable, Sendable {
        var index: Int
        var finish_reason: FinishReason?
        var message: ResponseMessage
    }

    struct ResponseMessage: Codable, Sendable {
        var role: ChatRole
        var content: String?
        var reasoning_content: String?
        var tool_calls: [ToolCall]?

        init(
            role: ChatRole,
            content: String? = nil,
            reasoning_content: String? = nil,
            tool_calls: [ToolCall]? = nil
        ) {
            self.role = role
            self.content = content
            self.reasoning_content = reasoning_content
            self.tool_calls = tool_calls
        }
    }

    enum FinishReason: String, Codable, Sendable {
        case stop
        case length
        case tool_calls
    }

    // MARK: - Chat Completion Chunk (streaming)

    struct ChatCompletionChunk: Codable, Sendable {
        var id: String
        var object: String = "chat.completion.chunk"
        var model: String
        var created: Int
        var system_fingerprint: String?
        var choices: [ChatCompletionChunkChoice]
        var usage: Usage?
    }

    struct ChatCompletionChunkChoice: Codable, Sendable {
        var index: Int
        var delta: ChunkDelta
        var finish_reason: FinishReason?
    }

    struct ChunkDelta: Codable, Sendable {
        var role: ChatRole?
        var content: String?
        var reasoning_content: String?
        var tool_calls: [ToolCall]?

        init(
            role: ChatRole? = nil,
            content: String? = nil,
            reasoning_content: String? = nil,
            tool_calls: [ToolCall]? = nil
        ) {
            self.role = role
            self.content = content
            self.reasoning_content = reasoning_content
            self.tool_calls = tool_calls
        }
    }

    // MARK: - Usage

    struct Usage: Codable, Sendable {
        var prompt_tokens: Int
        var completion_tokens: Int
        var total_tokens: Int
        var prompt_tokens_details: PromptTokensDetails?
    }

    struct PromptTokensDetails: Codable, Sendable {
        var cached_tokens: Int?
    }

    // MARK: - Models Endpoint

    struct ModelListResponse: Codable, Sendable {
        var object: String = "list"
        var data: [ModelObject]
    }

    struct ModelObject: Codable, Sendable {
        var id: String
        var object: String = "model"
        var type: String?
        var owned_by: String?
        var max_context_length: Int?
        var loaded_context_length: Int?
        var state: String?
    }
}
