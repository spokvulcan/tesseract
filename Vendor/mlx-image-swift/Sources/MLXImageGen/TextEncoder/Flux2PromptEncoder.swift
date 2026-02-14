import Foundation
import MLX
import Tokenizers

enum Flux2PromptEncoder {
    private static let padTokenString = "<|endoftext|>"
    private static let fallbackPadTokenId: Int32 = 151643

    static func encodePrompt(
        prompt: String,
        tokenizer: Tokenizer,
        textEncoder: Qwen3TextEncoder,
        maxSequenceLength: Int = 512,
        hiddenStateLayers: [Int] = [9, 18, 27]
    ) throws -> (promptEmbeds: MLXArray, textIds: MLXArray) {
        let promptEmbeds = try getQwen3PromptEmbeds(
            prompt: prompt,
            tokenizer: tokenizer,
            textEncoder: textEncoder,
            maxSequenceLength: maxSequenceLength,
            hiddenStateLayers: hiddenStateLayers
        )
        let textIds = prepareTextIds(promptEmbeds)
        return (promptEmbeds, textIds)
    }

    private static func getQwen3PromptEmbeds(
        prompt: String,
        tokenizer: Tokenizer,
        textEncoder: Qwen3TextEncoder,
        maxSequenceLength: Int,
        hiddenStateLayers: [Int]
    ) throws -> MLXArray {
        // Apply Qwen3 chat template (matches Python mflux: enable_thinking=False, add_generation_prompt=True)
        let messages: [[String: any Sendable]] = [["role": "user", "content": prompt]]
        let rawIds = try tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: true,
            maxLength: maxSequenceLength,
            tools: nil,
            additionalContext: ["enable_thinking": false]
        )
        NSLog("[MLXImageGen] Chat template produced %d tokens (max %d)", rawIds.count, maxSequenceLength)

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
        NSLog("[MLXImageGen] inputIds shape: %@, attentionMask shape: %@ (%d real, %d pad)",
              "\(inputIds.shape)", "\(attentionMask.shape)", seqLen, maxSequenceLength - seqLen)

        return textEncoder.getPromptEmbeds(
            inputIds: inputIds,
            attentionMask: attentionMask,
            hiddenStateLayers: hiddenStateLayers
        )
    }

    static func prepareTextIds(_ x: MLXArray) -> MLXArray {
        let (batchSize, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))
        var outIds = [MLXArray]()
        for _ in 0..<batchSize {
            let t = MLXArray.zeros([seqLen], type: Int32.self)
            let h = MLXArray.zeros([seqLen], type: Int32.self)
            let w = MLXArray.zeros([seqLen], type: Int32.self)
            let tokenIds = MLXArray(0..<Int32(seqLen))
            let coords = MLX.stacked([t, h, w, tokenIds], axis: 1)
            outIds.append(coords)
        }
        return MLX.stacked(outIds, axis: 0)
    }
}
