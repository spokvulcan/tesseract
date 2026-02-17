//
//  AgentVoiceState.swift
//  tesseract
//

enum AgentVoiceState: Equatable, Sendable {
    case idle
    case recording
    case transcribing
    case error(String)
}
