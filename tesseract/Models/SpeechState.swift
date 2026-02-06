//
//  SpeechState.swift
//  tesseract
//

enum SpeechState: Equatable, Sendable {
    case idle
    case capturingText
    case loadingModel
    case generating(progress: String)
    case playing
    case error(String)
}
