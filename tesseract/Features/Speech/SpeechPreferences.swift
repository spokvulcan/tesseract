//
//  SpeechPreferences.swift
//  tesseract
//
//  The owner's choices for the Speech page's reading and its overlay, as the
//  typed values the settings store's raw strings decode into. Each has a
//  label for its picker; the raw values are the persisted keys' contents.
//

import SwiftUI

/// The Reader's typeface.
enum ReaderTypeface: String, CaseIterable, Identifiable, Sendable {
    case serif, sans

    var id: String { rawValue }

    var label: String {
        switch self {
        case .serif: "Serif"
        case .sans: "Sans"
        }
    }

    var design: Font.Design {
        switch self {
        case .serif: .serif
        case .sans: .default
        }
    }
}

/// What lights up as text is read: Apple's Read & Speak offers the same four.
enum ReadAlongHighlight: String, CaseIterable, Identifiable, Sendable {
    case words, sentences, both, none

    var id: String { rawValue }

    var label: String {
        switch self {
        case .words: "Words"
        case .sentences: "Sentences"
        case .both: "Words and Sentences"
        case .none: "None"
        }
    }

    var showsWord: Bool { self == .words || self == .both }
    var showsSentence: Bool { self == .sentences || self == .both }
}

/// How the **Speech Overlay** shows what is being read, outside the app.
enum SpeechOverlayStyle: String, CaseIterable, Identifiable, Sendable {
    /// Two lines hanging from the notch; pause and stop on hover.
    case island
    /// Large two-line captions at the bottom of the screen.
    case captions
    case off

    var id: String { rawValue }

    var label: String {
        switch self {
        case .island: "Island"
        case .captions: "Captions"
        case .off: "Off"
        }
    }
}

enum SpeechOverlaySize: String, CaseIterable, Identifiable, Sendable {
    case small, medium, large

    var id: String { rawValue }

    var label: String {
        switch self {
        case .small: "S"
        case .medium: "M"
        case .large: "L"
        }
    }

    /// The island's text size; captions read three points larger.
    var points: CGFloat {
        switch self {
        case .small: 15
        case .medium: 18
        case .large: 22
        }
    }
}

/// The colour of the word being read, in the overlay.
enum SpeechOverlayTint: String, CaseIterable, Identifiable, Sendable {
    case accent, yellow, white

    var id: String { rawValue }

    var label: String {
        switch self {
        case .accent: "Orange"
        case .yellow: "Yellow"
        case .white: "White"
        }
    }

    /// Fixed colours: the overlay is always dark, whatever the appearance.
    var color: Color {
        switch self {
        case .accent: Color(red: 0.96, green: 0.65, blue: 0.26)
        case .yellow: Color(red: 1.0, green: 0.86, blue: 0.2)
        case .white: .white
        }
    }
}

/// When the overlay appears.
enum SpeechOverlayScope: String, CaseIterable, Identifiable, Sendable {
    /// Except while the Speech page is in front: the page already shows
    /// what is being read.
    case automatic
    case always

    var id: String { rawValue }

    var label: String {
        switch self {
        case .automatic: "When the Speech page isn't in front"
        case .always: "Always"
        }
    }
}
