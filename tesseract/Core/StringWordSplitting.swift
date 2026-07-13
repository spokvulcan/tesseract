//
//  StringWordSplitting.swift
//  tesseract
//
//  One definition of "a word" for the Speech feature: whitespace- and
//  newline-separated, empty runs omitted. Used by Word Timeline (the rendered word
//  model); the v2 engine's Segmenter (TesseractSpeech) applies the same
//  whitespace rule for its token-count estimation.
//

import Foundation

extension StringProtocol {
    /// Split into whitespace- and newline-separated words, omitting empty runs.
    /// `nonisolated` so the `nonisolated` `WordTimeline` value can call it under the
    /// project's default-MainActor isolation.
    nonisolated func splitIntoWords() -> [SubSequence] {
        split(omittingEmptySubsequences: true, whereSeparator: { $0.isWhitespace || $0.isNewline })
    }
}
