// TesseractSpeech — sentence segmentation, absorbed from the app's v1
// TextSegmenter (ADR-0038: segmentation moves behind the seam). Pure.

import Foundation
import NaturalLanguage

struct TextSegment: Sendable, Equatable {
    let index: Int
    let text: String
}

enum Segmenter {
    private enum Defaults {
        static let targetTokensPerSegment = 100
        static let tokensPerWordEstimate: Double = 1.3
        /// A lead segment grows past `leadTokens` until it holds this many,
        /// so a Reference Take is never a one-word title.
        static let minimumLeadTokens = 16
    }

    /// The lead segment's budget when it will become a Reference Take: its
    /// first sentence or two, about 40 words, so every later segment has
    /// little to re-read (ADR-0072).
    static let referenceLeadTokens = 52

    /// Sentences grouped into segments of about `targetTokens`. With
    /// `leadTokens`, the first segment closes at that smaller budget instead.
    static func segment(
        _ text: String,
        targetTokens: Int = Defaults.targetTokensPerSegment,
        leadTokens: Int? = nil
    ) -> [TextSegment] {
        let sentences = splitIntoSentences(text)
        guard sentences.count > 1 else {
            return [TextSegment(index: 0, text: text)]
        }

        var segments: [TextSegment] = []
        var currentSentences: [String] = []
        var currentTokenEstimate = 0

        for sentence in sentences {
            let sentenceTokens = estimateTokens(sentence)
            let lead = segments.isEmpty ? leadTokens : nil
            let limit = lead ?? targetTokens
            let canClose =
                lead == nil
                ? !currentSentences.isEmpty
                : currentTokenEstimate >= Defaults.minimumLeadTokens

            if currentTokenEstimate + sentenceTokens > limit && canClose {
                segments.append(TextSegment(index: segments.count, text: currentSentences.joined()))
                currentSentences = []
                currentTokenEstimate = 0
            }

            currentSentences.append(sentence)
            currentTokenEstimate += sentenceTokens
        }

        if !currentSentences.isEmpty {
            segments.append(TextSegment(index: segments.count, text: currentSentences.joined()))
        }

        return segments
    }

    private static func estimateTokens(_ text: String) -> Int {
        let words = text.split { $0.isWhitespace || $0.isNewline }.count
        return Int(Double(words) * Defaults.tokensPerWordEstimate)
    }

    private static func splitIntoSentences(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            sentences.append(String(text[range]))
            return true
        }

        return sentences.isEmpty ? [text] : sentences
    }
}
