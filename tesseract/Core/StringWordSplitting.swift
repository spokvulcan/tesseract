//
//  StringWordSplitting.swift
//  tesseract
//
//  "A word" for the Speech feature: runs of Characters that
//  `Character.separatesWords` (TesseractSpeech) doesn't split, empty runs
//  omitted. The engine numbers the words it times by the same rule, and the
//  Word Timeline, the Read-Along and the Reader's text ranges all count by it.
//

import Foundation
import TesseractSpeech

extension StringProtocol {
    /// Split into words as the speech engine counts them, omitting empty runs.
    /// `nonisolated` so the `nonisolated` `WordTimeline` value can call it under the
    /// project's default-MainActor isolation.
    nonisolated func splitIntoWords() -> [SubSequence] {
        split(omittingEmptySubsequences: true, whereSeparator: \.separatesWords)
    }
}
