import Foundation
import MLXLMCommon

/// The GPU-free routing decisions for capturing a **leaf** Snapshot Admission:
/// which token path a future continuation can hydrate. Tokenizer-affine, no
/// live KV cache — the actor executes the Metal capture/`admit` from the
/// decision this produces.
///
/// Today this owns the **reusable-prefix probes** shared by the canonical-user
/// and tool-continuation leaf modes. Leaf-mode selection still lives on
/// `LLMActor.selectHTTPLeafStoreMode` (already a tested pure function), and the
/// Metal capture stays in the actor.
nonisolated enum LeafAdmissionBuilder {

    /// The synthetic continuation a reusable-prefix probe appends to discover
    /// the shared token path. A stop turn re-renders as a user continuation; a
    /// tool-call turn re-renders as a tool-result continuation.
    enum Continuation: Sendable {
        case userTurn
        case toolResult

        var probeMessage: [String: any Sendable] {
            switch self {
            case .userTurn:
                ["role": Chat.Message.Role.user.rawValue, "content": "Aqkz_strip_probe"]
            case .toolResult:
                ["role": Chat.Message.Role.tool.rawValue, "content": "Aqkz_tool_probe"]
            }
        }
    }

    /// The reusable token prefix for the stored assistant turn: the shared
    /// prefix between the turn's isolated render (no generation prompt) and the
    /// render that appends a single synthetic continuation. That shared prefix
    /// is the exact token path the immediate continuation can hydrate.
    ///
    /// Returns `nil` when the probe diverges — no common prefix, or the prefix
    /// is not strictly shorter than the continuation (the appended turn added
    /// no tokens, so there is no distinct boundary to align to).
    static func reusablePrefix(
        continuation: Continuation,
        storedConversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        tokenizer: any Tokenizer
    ) throws -> [Int]? {
        let baseMessages = storedConversation.promptMessages
        let storedTokens = try tokenizer.applyChatTemplate(
            messages: baseMessages,
            tools: toolSpecs,
            additionalContext: ["add_generation_prompt": false]
        )
        let continuationTokens = try tokenizer.applyChatTemplate(
            messages: baseMessages + [continuation.probeMessage],
            tools: toolSpecs,
            additionalContext: ["add_generation_prompt": false]
        )

        let common = zip(storedTokens, continuationTokens).prefix { $0 == $1 }.count
        guard common > 0, common <= storedTokens.count, common < continuationTokens.count else {
            return nil
        }
        return Array(continuationTokens[0..<common])
    }
}
