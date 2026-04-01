import Foundation
import MLXLMCommon
import Tokenizers

/// Error thrown when a required special token is not found in the tokenizer vocabulary.
enum AgentTokenizerError: LocalizedError {
    case missingSpecialToken(String)

    var errorDescription: String? {
        switch self {
        case .missingSpecialToken(let token):
            "Required special token '\(token)' not found in tokenizer vocabulary"
        }
    }
}

/// Wraps the HuggingFace tokenizer, providing named access to ChatML special
/// tokens and encode/decode operations.
///
/// All special token IDs are resolved once at initialization from the tokenizer
/// vocabulary. Encode and decode delegate to the model container's actor-isolated tokenizer.
struct AgentTokenizer: Sendable {

    /// Resolved special token IDs for ChatML format.
    struct SpecialTokens: Sendable {
        /// `<|im_start|>` — ChatML message/role start delimiter (also BOS).
        let imStart: Int
        /// `<|im_end|>` — ChatML message end delimiter (also EOS).
        let imEnd: Int
        /// `<|endoftext|>` — end-of-text marker.
        let endOfText: Int
        /// `<think>` — chain-of-thought reasoning block start.
        let thinkStart: Int
        /// `</think>` — chain-of-thought reasoning block end.
        let thinkEnd: Int
        /// `<tool_call>` — tool invocation block start.
        let toolCallStart: Int
        /// `</tool_call>` — tool invocation block end.
        let toolCallEnd: Int
    }

    let specialTokens: SpecialTokens

    private let container: ModelContainer

    /// Resolves all special tokens from the model container's tokenizer.
    ///
    /// - Throws: ``AgentTokenizerError/missingSpecialToken(_:)`` if any required token
    ///   is absent from the vocabulary.
    init(container: ModelContainer) async throws {
        self.container = container
        self.specialTokens = try await Self.resolveSpecialTokens(from: container)
    }

    // MARK: - Encode

    /// Encodes text into token IDs (includes special tokens like BOS by default).
    func encode(_ text: String) async -> [Int] {
        await container.encode(text)
    }

    /// Encodes text into token IDs with explicit control over special token insertion.
    func encode(_ text: String, addSpecialTokens: Bool) async -> [Int] {
        await container.perform { context in
            context.tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        }
    }

    // MARK: - Decode

    /// Decodes token IDs back to text.
    func decode(_ tokens: [Int]) async -> String {
        await container.decode(tokens: tokens)
    }

    /// Decodes token IDs back to text, optionally stripping special tokens from the output.
    func decode(_ tokens: [Int], skipSpecialTokens: Bool) async -> String {
        await container.perform { context in
            context.tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
        }
    }

    // MARK: - Token Classification

    /// Returns `true` if the token signals end of generation (`<|im_end|>` or `<|endoftext|>`).
    func isEndOfGeneration(_ id: Int) -> Bool {
        id == specialTokens.imEnd || id == specialTokens.endOfText
    }

    // MARK: - Private

    /// Resolves all seven ChatML special tokens from the tokenizer vocabulary.
    /// Uses inline string literals to avoid actor-isolation issues with static properties
    /// inside `@Sendable` closures.
    private static func resolveSpecialTokens(from container: ModelContainer) async throws -> SpecialTokens {
        try await container.perform { context in
            let tokenizer = context.tokenizer

            guard let imStart = tokenizer.convertTokenToId("<|im_start|>") else {
                throw AgentTokenizerError.missingSpecialToken("<|im_start|>")
            }
            guard let imEnd = tokenizer.convertTokenToId("<|im_end|>") else {
                throw AgentTokenizerError.missingSpecialToken("<|im_end|>")
            }
            guard let endOfText = tokenizer.convertTokenToId("<|endoftext|>") else {
                throw AgentTokenizerError.missingSpecialToken("<|endoftext|>")
            }
            guard let thinkStart = tokenizer.convertTokenToId("<think>") else {
                throw AgentTokenizerError.missingSpecialToken("<think>")
            }
            guard let thinkEnd = tokenizer.convertTokenToId("</think>") else {
                throw AgentTokenizerError.missingSpecialToken("</think>")
            }
            guard let toolCallStart = tokenizer.convertTokenToId("<tool_call>") else {
                throw AgentTokenizerError.missingSpecialToken("<tool_call>")
            }
            guard let toolCallEnd = tokenizer.convertTokenToId("</tool_call>") else {
                throw AgentTokenizerError.missingSpecialToken("</tool_call>")
            }

            return SpecialTokens(
                imStart: imStart,
                imEnd: imEnd,
                endOfText: endOfText,
                thinkStart: thinkStart,
                thinkEnd: thinkEnd,
                toolCallStart: toolCallStart,
                toolCallEnd: toolCallEnd
            )
        }
    }
}
