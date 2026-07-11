//
//  DictationPerf.swift
//  tesseract
//

import Foundation
import os

/// Dictation latency instrumentation — the v1 baseline the overlay rewrite
/// must hold (ticket #286, map #283). Measurement only, no behavior.
///
/// Two consumption paths:
/// - OSSignposter intervals ("PressToVisible", "ReleaseToResolved") and
///   "PanelResize" events for Instruments hitch-hunting.
/// - Measured-millisecond log lines, harvestable offline:
///   `log show --last 1h --predicate 'subsystem == "app.tesseract.agent"
///    AND category == "DictationPerf"'`
///
/// All marks live on the MainActor — the press/release path's own actor — so
/// marking never hops. Sub-spans (`record`) buffer until the terminal
/// `markResolved` flushes them into one line, keeping one dictation = one
/// summary line.
@MainActor
enum DictationPerf {
    static let log = OSLog(subsystem: "app.tesseract.agent", category: "DictationPerf")
    static let signposter = OSSignposter(logHandle: log)
    private static let logger = PublicLogger(
        subsystem: "app.tesseract.agent", category: "DictationPerf")

    private static var pressStart: DispatchTime?
    private static var pressInterval: OSSignpostIntervalState?
    private static var releaseStart: DispatchTime?
    private static var releaseInterval: OSSignpostIntervalState?
    private static var spans: [String] = []

    /// Hotkey-down on a path that will show the overlay — starts press→visible.
    static func markPress() {
        pressStart = .now()
        pressInterval = signposter.beginInterval("PressToVisible")
    }

    /// First panel show after a press — ends press→visible. Re-shows without a
    /// pending press (processing/error transitions, launch prewarm) no-op.
    static func markPanelShown() {
        guard let start = pressStart else { return }
        pressStart = nil
        if let interval = pressInterval {
            signposter.endInterval("PressToVisible", interval)
            pressInterval = nil
        }
        logger.info("press→visible \(msSince(start))ms")
    }

    /// Capture end (hotkey release or max-duration auto-stop) — starts
    /// release→resolved.
    static func markRelease() {
        releaseStart = .now()
        releaseInterval = signposter.beginInterval("ReleaseToResolved")
        spans = []
    }

    /// A measured sub-span of the release path (stop, session, inject); flushed
    /// into the terminal `markResolved` line.
    static func record(span name: String, ms: Int64) {
        spans.append("\(name)=\(ms)ms")
    }

    /// Terminal outcome of the release path — ends release→resolved and logs
    /// the one summary line with all recorded sub-spans.
    static func markResolved(_ outcome: String) {
        guard let start = releaseStart else { return }
        releaseStart = nil
        if let interval = releaseInterval {
            signposter.endInterval("ReleaseToResolved", interval)
            releaseInterval = nil
        }
        let detail = spans.isEmpty ? "" : " (\(spans.joined(separator: " ")))"
        spans = []
        logger.info("release→\(outcome) \(msSince(start))ms\(detail)")
    }

    /// Signpost-only event for stage-driven panel frame changes, so Instruments
    /// can correlate resize commits with animation hitches.
    static func panelResize(animated: Bool) {
        signposter.emitEvent("PanelResize", "animated=\(animated)")
    }

    nonisolated static func msSince(_ start: DispatchTime) -> Int64 {
        Int64((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
    }
}
