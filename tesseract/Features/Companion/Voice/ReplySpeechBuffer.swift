//
//  ReplySpeechBuffer.swift
//  tesseract
//
//  The **Reply Speech Buffer** (ADR-0097): Jarvis's reply as the model writes
//  it, cut into what can be spoken — whole sentences, never one half written,
//  so the voice starts on the first sentence and never renders a sentence's
//  prosody before its end is known. Machine sub-state of the **Voice Session
//  Machine**, like the endpointer: pure, no clock, no I/O.
//

import Foundation

nonisolated struct ReplySpeechBuffer {

    /// Written but not yet handed to speech.
    private(set) var pending = ""

    mutating func append(_ delta: String) {
        pending += delta
    }

    /// Every whole sentence written so far, taken out as one piece — nil
    /// until one is complete. What follows the last sentence end stays.
    mutating func takeSentences() -> String? {
        guard let end = Self.lastSentenceEnd(in: pending) else { return nil }
        let piece = String(pending[..<end])
        pending = String(pending[end...])
        return Self.speakable(piece)
    }

    /// Everything held, complete as it stands: a text part closed (a tool
    /// call follows it), or the reply ended.
    mutating func takeAll() -> String? {
        let piece = pending
        pending = ""
        return Self.speakable(piece)
    }

    mutating func reset() {
        pending = ""
    }

    /// Just past the last sentence end in `text`: a line break, or a word
    /// that ends a sentence as a take's words do (`TakeToken.endsSentence`:
    /// never an abbreviation, an initialism or a pause) with the whitespace
    /// after it already written. A dot still waiting for its next character
    /// ends nothing yet.
    static func lastSentenceEnd(in text: String) -> String.Index? {
        let lineEnd = text.lastIndex(where: \.isNewline).map(text.index(after:))
        let sentenceEnd = TakeText.tokens(text).last {
            $0.endsSentence && $0.range.upperBound < text.endIndex
        }?.range.upperBound
        return [lineEnd, sentenceEnd].compactMap { $0 }.max()
    }

    private static func speakable(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
