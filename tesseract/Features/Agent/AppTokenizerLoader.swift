import Foundation
import Hub
import MLXLMCommon
import Tokenizers

/// Loads and classifies the same effective configuration, before adaptation
/// erases the decoder semantics. Both ordinary and PARO model loaders use
/// this boundary; unknown tokenizer implementations retain naive streaming.
nonisolated struct AppTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let configuration = LanguageModelConfigurationFromHub(modelFolder: directory)
        guard let config = try await configuration.tokenizerConfig else {
            throw Tokenizers.TokenizerError.missingConfig
        }
        let data = try await configuration.tokenizerData
        // Match AutoTokenizer.from(modelFolder:), including its choice of
        // concrete tokenizer rather than the remote tokenizer-class registry.
        return try LoadedTokenizer(config: config, data: data)
    }
}

/// A capability of the constructed tokenizer, not a guess from token samples.
nonisolated protocol ByteLevelTokenizing: MLXLMCommon.Tokenizer {
    var byteLevelDecoding: ByteLevelDecoding? { get }
}

nonisolated struct ByteLevelDecoding: Sendable {
    let addedTokens: Set<String>

    private init(addedTokens: Set<String>) {
        self.addedTokens = addedTokens
    }

    static func recognize(config: Config, data: Config) -> Self? {
        // Mirror PreTrainedTokenizer's effective added-token set: malformed
        // entries without an ID or content do not participate in decoding.
        let addedTokens = Set(
            data.addedTokens.array(or: []).compactMap { token -> String? in
                guard token.id.integer() != nil else { return nil }
                return token.content.string()
            })
        guard data.decoder.type.string() == "ByteLevel",
            !config.cleanUpTokenizationSpaces.boolean(or: true),
            let vocabulary = data.model.vocab.dictionary(),
            vocabulary.keys.allSatisfy({ spelling in
                addedTokens.contains(spelling.description)
                    || spelling.description.unicodeScalars.allSatisfy {
                        LinearStreamingDetokenizer.byteLevelAlphabet[$0] != nil
                    }
            })
        else { return nil }
        return Self(addedTokens: addedTokens)
    }
}

/// Keeps the decoder capability with the tokenizer it describes, and preserves
/// the chat-template rendering seam used by Emitted Path Resolve.
private nonisolated struct LoadedTokenizer: MLXLMCommon.ChatTemplateRendering, ByteLevelTokenizing {
    private let upstream: PreTrainedTokenizer
    let byteLevelDecoding: ByteLevelDecoding?

    init(config: Config, data: Config) throws {
        upstream = try PreTrainedTokenizer(tokenizerConfig: config, tokenizerData: data)
        byteLevelDecoding = ByteLevelDecoding.recognize(config: config, data: data)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }

    func renderChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> String {
        do {
            return try upstream.renderChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
