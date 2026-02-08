//
//  SpeechState.swift
//  tesseract
//

enum SpeechState: Equatable, Sendable {
    case idle
    case capturingText
    case loadingModel
    case generating(progress: String)
    case streaming
    case streamingLongForm(segment: Int, of: Int)
    case paused(segment: Int, of: Int)
    case playing
    case error(String)
}
