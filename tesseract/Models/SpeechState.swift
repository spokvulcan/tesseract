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
}
