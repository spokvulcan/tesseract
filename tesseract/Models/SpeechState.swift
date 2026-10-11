//
//  SpeechState.swift
//  tesseract
//

enum SpeechState: Equatable, Sendable {
    case idle
    case capturingText
    case generating(progress: String)
    case streaming
    case streamingLongForm(segment: Int, of: Int)
    case paused(segment: Int, of: Int)
    case playing
    case error(String)

    /// The engine is doing (or holding) work — everything but a settled
    /// `idle`/`error`. Exhaustive on purpose: a new case must pick a side
    /// here, once, for every caller that gates on speech activity.
    var isActive: Bool {
        switch self {
        case .idle, .error:
            false
        case .capturingText, .generating, .streaming, .streamingLongForm, .paused, .playing:
            true
        }
    }

    /// The voice is speaking: generating, streaming or playing — not
    /// settled, paused or still reading the selection. The menu bar's glyph
    /// (whose resolver is nonisolated) and the Models section's Voice row
    /// both read it. Exhaustive for the same reason as `isActive`.
    nonisolated var isSpeaking: Bool {
        switch self {
        case .idle, .capturingText, .paused, .error:
            false
        case .generating, .streaming, .streamingLongForm, .playing:
            true
        }
    }
}
