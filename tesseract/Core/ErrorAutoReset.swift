//
//  ErrorAutoReset.swift
//  tesseract
//

/// How long a transient error stays on screen before its surface goes back
/// to idle. Dictation, Voice Input and speech each own their reset, because
/// each error lives on a different surface, but they all read this one
/// duration so they can't drift apart.
nonisolated enum ErrorAutoReset {
    static let delay: Duration = .seconds(3)
}
