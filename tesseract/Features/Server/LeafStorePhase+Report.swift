//
//  LeafStorePhase+Report.swift
//  tesseract
//
//  The **Leaf Store** phase's per-request account: which path stored the
//  leaf and where the post-EOS time went. The drive prints it as the
//  notice-level `Leaf store —` line and folds its seconds into the trace
//  corpus.
//

import Foundation

nonisolated extension LeafStorePhase {
    /// The per-request account of the phase — the drive prints it as the
    /// notice-level `Leaf store —` line (persisted, so a post-EOS stall is
    /// attributable from `log show` alone) and folds its seconds into the
    /// trace corpus.
    struct Report: Sendable {
        enum Path: String, Sendable {
            /// **Live Leaf Capture**: the live final cache, no prefill.
            case live
            /// Boundary restore + canonical residual re-prefill.
            case boundary
            /// `directLeaf`: the live final cache under a non-thinking
            /// template (the pre-existing live path).
            case direct
            /// No leaf stored — `skipReason` says why.
            case skipped
        }

        var mode = "unkeyed"
        var path: Path = .skipped
        var skipReason: String?
        /// The `LiveLeafCapture.FallbackReason` wire token when a boundary
        /// mode ran the boundary executor because the live path was refused.
        var liveFallbackReason: String?
        var leafOffset: Int?
        /// Tokens prefilled on the GPU after generation ended — the
        /// boundary residual, `0` on the live and direct paths.
        var residualTokens = 0
        var timings = Timings()

        /// `key=value` fields for the drive's log line.
        var logFields: String {
            var fields = ["mode=\(mode)", "path=\(path.rawValue)"]
            if let skipReason { fields.append("skip=\(skipReason)") }
            if let liveFallbackReason { fields.append("liveFallback=\(liveFallbackReason)") }
            if let leafOffset { fields.append("leafOffset=\(leafOffset)") }
            fields.append("residualTokens=\(residualTokens)")
            fields.append(timings.logFields)
            return fields.joined(separator: " ")
        }
    }

    /// Wall milliseconds per stage of the tail, zero where a stage did not
    /// run on the chosen path.
    struct Timings: Sendable {
        /// Stored-conversation re-render + tokenize (CPU).
        var renderMs = 0.0
        /// Boundary snapshot restore (boundary path only).
        var restoreMs = 0.0
        /// Residual re-prefill (boundary path only).
        var prefillMs = 0.0
        /// Leaf snapshot deep copy.
        var captureMs = 0.0
        /// SSD payload extraction (full or extension suffix).
        var payloadMs = 0.0
        /// Radix-tree admission on the MainActor.
        var admitMs = 0.0

        var logFields: String {
            "renderMs=\(Self.format(renderMs)) restoreMs=\(Self.format(restoreMs)) "
                + "prefillMs=\(Self.format(prefillMs)) captureMs=\(Self.format(captureMs)) "
                + "payloadMs=\(Self.format(payloadMs)) admitMs=\(Self.format(admitMs))"
        }

        static func format(_ ms: Double) -> String {
            String(format: "%.1f", ms)
        }
    }

    /// Milliseconds elapsed since a `Date.timeIntervalSinceReferenceDate` mark.
    static func millisecondsSince(_ start: TimeInterval) -> Double {
        (Date.timeIntervalSinceReferenceDate - start) * 1000
    }
}
