//
//  RecordingHighlightSurface.swift
//  tesseractTests
//
//  A hermetic, in-memory `WordHighlightSurface` for tests — a *peer implementation*
//  of `TTSNotchPanelController`, not a mock (ADR-0003/0004). It records the call
//  sequence the coordinator drives so the segment-boundary switch — and its ordering
//  against the ADR-0003 virtual playback clock — is assertable for the first time
//  (every prior `SpeechCoordinator` test passed `notchOverlay: nil`). No `NSPanel`,
//  no `NSScreen`, no `Timer`.
//

import Foundation

@testable import Tesseract_Agent

@MainActor
final class RecordingHighlightSurface: WordHighlightSurface {

    /// One recorded call, capturing what tests assert on (text, forwarded token
    /// offsets, and the Segment Window).
    enum Call: Equatable {
        case show(text: String, tokenCharOffsets: [Int])
        case switchText(text: String, tokenCharOffsets: [Int], segmentBase: TimeInterval)
        case updateTotalDuration(TimeInterval)
        case markSegmentComplete
        case markGenerationComplete
        case dismiss
    }

    private(set) var calls: [Call] = []

    func show(text: String, tokenCharOffsets: [Int], playbackTimeProvider: @escaping () -> TimeInterval) {
        calls.append(.show(text: text, tokenCharOffsets: tokenCharOffsets))
    }

    func switchText(_ text: String, tokenCharOffsets: [Int], segmentBase: TimeInterval) {
        calls.append(.switchText(text: text, tokenCharOffsets: tokenCharOffsets, segmentBase: segmentBase))
    }

    func updateTotalDuration(_ duration: TimeInterval) {
        calls.append(.updateTotalDuration(duration))
    }

    func markSegmentComplete() { calls.append(.markSegmentComplete) }
    func markGenerationComplete() { calls.append(.markGenerationComplete) }
    func dismiss() { calls.append(.dismiss) }

    // MARK: - Query helpers for assertions

    /// Texts passed to `show` / `switchText`, in order — the segments this surface was
    /// asked to display.
    var displayedTexts: [String] {
        calls.compactMap {
            switch $0 {
            case .show(let text, _): return text
            case .switchText(let text, _, _): return text
            default: return nil
            }
        }
    }

    /// Index of the first `switchText`, or `nil` if no switch has happened yet.
    var firstSwitchIndex: Int? {
        calls.firstIndex {
            if case .switchText = $0 { return true }
            return false
        }
    }

    var didSwitch: Bool { firstSwitchIndex != nil }
}
