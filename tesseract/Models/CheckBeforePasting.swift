//
//  CheckBeforePasting.swift
//  tesseract
//

import Foundation

/// The **check-before-pasting** setting (PRD #612): whether a finished take
/// waits in the **Lens** for ↩ before it pastes, so a word can be fixed
/// before it lands. Stored raw (`SettingsManager.checkBeforePastingRaw`) so
/// an unrecognized persisted value degrades to the default.
nonisolated enum CheckBeforePasting: String, CaseIterable, Identifiable, Sendable {
    /// A take waits only when ⇧ was tapped while recording. The default.
    case whenShiftTapped
    /// Every take waits.
    case always
    /// No take waits: each one pastes on release, ⇧ or not.
    case never

    var id: String { rawValue }

    /// Picker label in the Dictation pane.
    var displayName: String {
        switch self {
        case .whenShiftTapped: "When I tap ⇧"
        case .always: "Always"
        case .never: "Never"
        }
    }

    /// Whether a finished take waits in the Lens instead of pasting.
    /// `shiftTapped` is whether ⇧ was tapped while the take was recorded.
    func takeWaits(shiftTapped: Bool) -> Bool {
        switch self {
        case .whenShiftTapped: shiftTapped
        case .always: true
        case .never: false
        }
    }
}
