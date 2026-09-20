//
//  LeafStorePhase+Report.swift
//  tesseract
//
//  The **Leaf Store** phase's per-request account: which path stored the
//  leaf, where the leaf came from, why a turn took the boundary path, and
//  where the post-EOS time went. A prefix-cache diagnostics payload
//  (`leafStore`), so it renders like every other cache event, reaches the
//  telemetry sinks and the events drawer, and is emitted at notice level by
//  the drive — persisted, so a post-EOS stall is attributable from
//  `log show` alone.
//

import Foundation

nonisolated extension LeafStorePhase {
    /// The per-request account of the phase — the drive logs it as the
    /// `leafStore` event and folds its seconds into the trace corpus.
    struct Report: PrefixCacheDiagnostics.Payload {
        enum Path: String, Sendable {
            /// The fast path: the live final cache under the fed path, no
            /// prefill (**Live Leaf Capture**).
            case live
            /// Boundary restore + canonical residual re-prefill.
            case boundary
            /// `directLeaf` on the boundary route: the live final cache
            /// under a non-thinking template's canonical stored path (the
            /// pre-existing render-trusting path).
            case direct
            /// An aborted request returned its unchanged original leaf.
            case rewind
            /// No leaf stored — `skipReason` says why.
            case skipped
        }

        /// Where the stored leaf's cache state came from (ADR-0063 decision
        /// 13; ADR-0064 adds `handoff`, `copy` and `rewind`).
        enum Source: String, Sendable {
            /// The live final cache, deep-copied at its own offset.
            case live
            /// The leaf took the cache objects themselves rather than a deep
            /// copy of them: the finished generation's, or — on the boundary
            /// path — the request-private cache its residual re-prefill ran
            /// on. `path` says which.
            case handoff
            case copy, rewind
            /// A restored boundary snapshot extended by the canonical
            /// residual re-prefill, deep-copied into the leaf.
            case boundary

            /// Whether the leaf was deep-copied out of a live cache
            /// (`HybridCacheSnapshot.capture`), so its KV was resident
            /// twice for a moment — what the **Active-Inference Reserve**
            /// doubles a lane for (#522). A moved leaf reports `handoff`,
            /// on the boundary path as on the live one; a `copy` is a
            /// restore by copy whose source body the tree already counts;
            /// a `rewind` copied nothing.
            var capturesByCopy: Bool {
                switch self {
                case .live, .boundary: true
                case .handoff, .copy, .rewind: false
                }
            }

            /// The source a stored leaf reports, from the path that stored
            /// it, the request's restore mode and whether the capture moved
            /// the generation's objects. Shared by the report and the leaf
            /// admission so the reserve observes exactly what the
            /// `leafStore` event says.
            static func stored(
                path: Path, restoreMode: String?, handedOff: Bool
            ) -> Source? {
                switch path {
                case .live, .direct:
                    restoreMode == "copy" || restoreMode == "failedCopy"
                        ? .copy : handedOff ? .handoff : .live
                case .boundary: handedOff ? .handoff : .boundary
                case .rewind: .rewind
                case .skipped: nil
                }
            }
        }

        enum CopyReason: String, Sendable {
            case quantized, imageKeySpace, checkpoint, immutableBody, pendingFullPayload,
                untrimmable, rotating, warmBody
        }

        var handedOff = false
        var restoreMode: String?
        var restoreCopyReason: CopyReason?
        /// Seconds the restore waited for a pending full payload before
        /// falling back to a copy, or before the writer released the body
        /// and the check-out succeeded (#523). `0` when nothing waited.
        var restoreCopyWaitSeconds: TimeInterval = 0
        var copyReason: CopyReason?
        var mode = "unkeyed"
        var path: Path = .skipped
        var skipReason: String?
        /// The leaf's source, once one was stored: the live final cache
        /// under the fast path and the direct route, a restored boundary
        /// snapshot under the boundary route.
        var source: Source? {
            Source.stored(path: path, restoreMode: restoreMode, handedOff: handedOff)
        }
        /// Why the turn took the boundary route (the `liveLeafCapture` skip
        /// token), whether or not a leaf was stored there.
        var boundaryReason: String?
        var leafOffset: Int?
        /// Tokens prefilled on the GPU after generation ended — the
        /// boundary residual, `0` on the live and direct paths.
        var residualTokens = 0
        /// The stored-conversation render (CPU): to bytes on the fast path,
        /// render + tokenize on the boundary path.
        var renderSeconds: TimeInterval = 0
        /// Routing on the boundary path only: the reusable-prefix probe and
        /// **Snapshot Resolution** of the restore boundary.
        var planSeconds: TimeInterval = 0
        /// The executor's stages.
        var timings = Timings()
        /// Set by the drive: the whole phase, and the span from generation
        /// end to the drive's finish (the client's wait for its terminal
        /// chunk).
        var leafStoreSeconds: TimeInterval?
        var tailSeconds: TimeInterval?
        /// Emitted Path registration (ADR-0063): what the index learned
        /// from this turn, or why it learned nothing.
        var emittedPathRegistered: EmittedPathRegistration.Registered?
        var emittedPathSkip: String?
        var emittedPathRegisterSeconds: TimeInterval = 0
        /// The request's resolves against the index, folded in by the drive.
        var emittedPathResolves: EmittedPathRequestTelemetry.Summary?

        let eventName = "leafStore"

        var fields: [(String, String)] {
            let ms = PrefixCacheDiagnostics.milliseconds
            var fields = [("mode", mode), ("path", path.rawValue)]
            if let source { fields.append(("source", source.rawValue)) }
            if let copyReason = restoreCopyReason ?? copyReason {
                fields.append(("copyReason", copyReason.rawValue))
            }
            if restoreCopyWaitSeconds > 0 {
                fields.append(("copyWaitMs", ms(restoreCopyWaitSeconds)))
            }
            if let restoreMode { fields.append(("restoreMode", restoreMode)) }
            if let skipReason { fields.append(("skip", skipReason)) }
            if let boundaryReason { fields.append(("boundary", boundaryReason)) }
            if let leafOffset { fields.append(("leafOffset", "\(leafOffset)")) }
            fields += [
                ("residualTokens", "\(residualTokens)"),
                ("renderMs", ms(renderSeconds)),
                ("planMs", ms(planSeconds)),
                ("restoreMs", ms(timings.restoreSeconds)),
                ("prefillMs", ms(timings.prefillSeconds)),
                ("captureMs", ms(timings.captureSeconds)),
                ("payloadMs", ms(timings.payloadSeconds)),
                ("admitMs", ms(timings.admitSeconds)),
            ]
            if let leafStoreSeconds { fields.append(("leafStoreMs", ms(leafStoreSeconds))) }
            if let tailSeconds { fields.append(("postGenerationMs", ms(tailSeconds))) }
            if let emittedPathRegistered {
                fields.append(("emittedPath", "registered"))
                fields.append(("emittedPathLength", "\(emittedPathRegistered.pathLength)"))
            } else if let emittedPathSkip {
                fields.append(("emittedPath", "skipped"))
                fields.append(("emittedPathSkip", emittedPathSkip))
            }
            fields.append(("emittedPathRegisterMs", ms(emittedPathRegisterSeconds)))
            if let resolves = emittedPathResolves {
                fields.append(("emittedPathResolves", "\(resolves.resolves)"))
                fields.append(("emittedPathHits", "\(resolves.hits)"))
                if let prefix = resolves.requestEdgeIndexedPrefix {
                    fields.append(("emittedPathRequestPrefix", "\(prefix)"))
                }
                if let suffix = resolves.requestEdgeSuffixTokens {
                    fields.append(("emittedPathRequestSuffix", "\(suffix)"))
                }
            }
            return fields
        }

        /// Record why the turn registered no Emitted Path — through the
        /// diagnostics net and into this account.
        mutating func recordEmittedPathSkip(
            _ reason: EmittedPathRegistration.SkipReason,
            fields: [(String, String)] = [],
            in diagnostics: PrefixCacheDiagnostics.Context
        ) {
            EmittedPathRegistration.emitSkip(
                EmittedPathRegistration.Skip(reason, fields: fields), in: diagnostics)
            emittedPathSkip = reason.rawValue
        }

        /// Fold a registration outcome (already emitted) into the account.
        mutating func absorbEmittedPath(
            _ outcome: EmittedPathRegistration.Outcome, registerSeconds: TimeInterval
        ) {
            emittedPathRegisterSeconds = registerSeconds
            switch outcome {
            case .registered(let registered):
                emittedPathRegistered = registered
                emittedPathSkip = nil
            case .skipped(let skip):
                emittedPathRegistered = nil
                emittedPathSkip = skip.reason.rawValue
            }
        }

        /// Emit a decidable skip and record its reason — the one way a skip
        /// reaches both the diagnostics net and this account.
        mutating func recordSkip(
            _ record: LeafSkipLog, in diagnostics: PrefixCacheDiagnostics.Context
        ) {
            record.emit(in: diagnostics)
            skipReason = record.reason
        }

        /// Fold what an executor produced into the account. `path` is the
        /// executor that ran; it reads `.skipped` when no leaf was captured.
        mutating func absorb(_ capture: LeafCapture, path: Path) {
            self.path = capture.leafOffset != nil ? path : .skipped
            skipReason = capture.skipReason
            leafOffset = capture.leafOffset
            residualTokens = capture.residualTokens
            handedOff = capture.handedOff
            copyReason = capture.copyReason
            timings = capture.timings
        }
    }

    /// Wall seconds per executor stage, zero where a stage did not run on
    /// the chosen path.
    struct Timings: Sendable {
        /// Boundary snapshot restore (boundary path only).
        var restoreSeconds: TimeInterval = 0
        /// Residual re-prefill (boundary path only).
        var prefillSeconds: TimeInterval = 0
        /// Leaf snapshot deep copy.
        var captureSeconds: TimeInterval = 0
        /// SSD payload preparation — an extension's detaching only (suffix
        /// slices plus the whole recurrent state); the host copy is deferred
        /// to the SSD writer (**Deferred Payload Extraction**).
        var payloadSeconds: TimeInterval = 0
        /// Radix-tree admission on the MainActor.
        var admitSeconds: TimeInterval = 0
    }

    /// Seconds elapsed since a `Date.timeIntervalSinceReferenceDate` mark.
    static func secondsSince(_ start: TimeInterval) -> TimeInterval {
        Date.timeIntervalSinceReferenceDate - start
    }
}
