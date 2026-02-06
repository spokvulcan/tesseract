//
//  TTSParameters.swift
//  tesseract
//

struct TTSParameters: Codable, Sendable, Equatable {
    var temperature: Float = 0.6
    var topP: Float = 0.8
    var repetitionPenalty: Float = 1.3
    var repetitionContextSize: Int = 20
    var maxTokens: Int = 4096

    static let `default` = TTSParameters()
}
