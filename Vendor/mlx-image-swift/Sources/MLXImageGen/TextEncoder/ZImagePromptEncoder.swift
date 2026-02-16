import Foundation
import MLX
import Tokenizers

/// Tokenize + encode prompt for Z-Image using Qwen3 text encoder.
/// Extracts valid (non-padding) tokens from the second-to-last hidden state.
enum ZImagePromptEncoder {
    private static let padTokenString = "<|endoftext|>"
    private static let fallbackPadTokenId: Int32 = 151643

    /// Encode a prompt into caption features for the Z-Image transformer.
    /// - Returns: [numValidTokens, 2560] caption features
    static func encodePrompt(
        prompt: String,
        tokenizer: Tokenizer,
        textEncoder: Qwen3TextEncoder,
        maxSequenceLength: Int = 512
    ) throws -> MLXArray {
        // Apply Qwen3 chat template (enable_thinking=true for Z-Image)
        let messages: [[String: any Sendable]] = [["role": "user", "content": prompt]]
        let rawIds = try tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: true,
            maxLength: maxSequenceLength,
            tools: nil,
            additionalContext: ["enable_thinking": true]
        )

        // Pad to maxSequenceLength with proper attention mask
        let padId = Int32(tokenizer.convertTokenToId(padTokenString) ?? Int(fallbackPadTokenId))
        let seqLen = min(rawIds.count, maxSequenceLength)

        var tokenIds = rawIds.prefix(seqLen).map { Int32($0) }
        var maskValues = [Int32](repeating: 1, count: seqLen)
        if seqLen < maxSequenceLength {
            tokenIds += [Int32](repeating: padId, count: maxSequenceLength - seqLen)
            maskValues += [Int32](repeating: 0, count: maxSequenceLength - seqLen)
        }

        let inputIds = MLXArray(tokenIds).expandedDimensions(axis: 0)
        let attentionMask = MLXArray(maskValues).expandedDimensions(axis: 0)

        // Get second-to-last hidden state
        let hiddenState = textEncoder.getSecondToLastHiddenState(
            inputIds: inputIds,
            attentionMask: attentionMask
        )

        // Extract valid (non-padding) tokens: [1, seqLen, 2560] → [numValid, 2560]
        let numValid = Int(MLX.sum(attentionMask).item(Int32.self))
        return hiddenState[0, ..<numValid, 0...]
    }
}
