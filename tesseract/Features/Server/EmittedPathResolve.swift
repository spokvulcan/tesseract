//
//  EmittedPathResolve.swift
//  tesseract
//
//  The **Emitted Path Resolve** (ADR-0063, decisions 4/5): the composition
//  the Conversation Render runs for every text-only render spelling —
//  render to bytes, look the history up in the Emitted Path Index from the
//  last end-of-turn marker backwards, and concatenate the deepest hit's path
//  with the canonical encode of the bytes after its marker.
//
//  Dark launch (this ticket): the composition is computed and shadow-checked
//  against the canonical full encode on every hit, but the render verbs
//  still return the canonical tokens. The shadow check counts and logs any
//  difference; the expected count is zero, and #476 removes the check when
//  the composition becomes the served tokens.
//
//  `EmittedPathRequestTelemetry` is the per-request account: a reference the
//  Conversation Render copies share, so the request-edge resolve, the
//  planner's last-user render and the Leaf Store's renders all land on one
//  record that the completion report and the trace corpus read at the end.
//

import Foundation
import MLXLMCommon

nonisolated enum EmittedPathResolve {

    enum Composition: Equatable, Sendable {
        /// `tokens` = the index path + the canonical encode of the suffix
        /// after the marker; `indexedPrefix` ids came from the index.
        case indexed(
            tokens: [Int], indexedPrefix: Int, suffixTokens: Int, markerDepth: Int,
            markerCount: Int)
        case miss(EmittedPathIndex.MissReason)
    }

    /// Resolves `renderedBytes` against `index` for `fingerprint` and, on a
    /// hit, encodes only the bytes after the hit's marker (a special token is
    /// a hard pretoken boundary, so the standalone encode equals the slice of
    /// the full encode).
    static func compose(
        index: EmittedPathIndex,
        fingerprint: String,
        marker: EndOfTurnMarker,
        renderedBytes: [UInt8],
        tokenizer: any Tokenizer
    ) -> Composition {
        switch index.resolve(
            fingerprint: fingerprint, renderedBytes: renderedBytes, marker: marker.bytes)
        {
        case .miss(let reason):
            return .miss(reason)
        case .hit(let resolution):
            // Render bytes are the template's own UTF-8: lossless by construction.
            // swiftlint:disable:next optional_data_string_conversion
            let suffix = String(decoding: renderedBytes[resolution.prefixEnd...], as: UTF8.self)
            let suffixTokens =
                suffix.isEmpty ? [] : tokenizer.encode(text: suffix, addSpecialTokens: false)
            return .indexed(
                tokens: resolution.path + suffixTokens,
                indexedPrefix: resolution.path.count,
                suffixTokens: suffixTokens.count,
                markerDepth: resolution.markerDepth,
                markerCount: resolution.markerCount
            )
        }
    }

    // MARK: - Shadow check

    struct ShadowDifference: Equatable, Sendable {
        /// First differing offset (or the shorter length when one is a
        /// prefix of the other).
        let offset: Int
        let composedCount: Int
        let canonicalCount: Int
        let composedContext: [Int]
        let canonicalContext: [Int]
    }

    static let shadowContextWidth = 4

    /// `nil` when the composition equals the canonical encode.
    static func shadow(composed: [Int], canonical: [Int]) -> ShadowDifference? {
        let shared = zip(composed, canonical).prefix { $0 == $1 }.count
        guard shared < composed.count || shared < canonical.count else { return nil }
        let width = shadowContextWidth
        let lower = max(0, shared - width)
        return ShadowDifference(
            offset: shared,
            composedCount: composed.count,
            canonicalCount: canonical.count,
            composedContext: Array(composed[lower..<min(composed.count, shared + width)]),
            canonicalContext: Array(canonical[lower..<min(canonical.count, shared + width)])
        )
    }

    // MARK: - Events

    /// One resolve attempt, hit or miss. `spelling` names the render verb.
    struct ResolveEvent: PrefixCacheDiagnostics.Payload {
        let spelling: String
        let composition: Composition
        let canonicalTokens: Int
        let resolveSeconds: Double

        let eventName = "emittedPathResolve"

        var fields: [(String, String)] {
            var fields: [(String, String)] = [("spelling", spelling)]
            switch composition {
            case .indexed(_, let indexedPrefix, let suffixTokens, let depth, let count):
                fields.append(("result", "hit"))
                fields.append(("indexedPrefix", "\(indexedPrefix)"))
                fields.append(("suffixTokens", "\(suffixTokens)"))
                fields.append(("markerDepth", "\(depth)"))
                fields.append(("markerCount", "\(count)"))
            case .miss(let reason):
                fields.append(("result", "miss"))
                fields.append(("reason", reason.rawValue))
            }
            fields.append(("canonicalTokens", "\(canonicalTokens)"))
            fields.append(("resolveMs", PrefixCacheDiagnostics.milliseconds(resolveSeconds)))
            return fields
        }
    }

    /// A composition that differs from the canonical encode — the claim the
    /// dark launch exists to test, failing.
    struct ShadowEvent: PrefixCacheDiagnostics.Payload {
        let spelling: String
        let difference: ShadowDifference

        let eventName = "emittedPathShadow"

        var fields: [(String, String)] {
            [
                ("spelling", spelling),
                ("offset", "\(difference.offset)"),
                ("composedTokens", "\(difference.composedCount)"),
                ("canonicalTokens", "\(difference.canonicalCount)"),
                ("composedContext", EmittedPathResolve.idList(difference.composedContext)),
                ("canonicalContext", EmittedPathResolve.idList(difference.canonicalContext)),
            ]
        }
    }

    /// Ids as `[a,b,c]` — the character class the diagnostics renderer
    /// leaves unescaped.
    static func idList(_ ids: [Int]) -> String {
        "[" + ids.map(String.init).joined(separator: ",") + "]"
    }

    static func line(_ payload: some PrefixCacheDiagnostics.Payload) -> String {
        payload.eventName + " "
            + payload.fields.map { "\($0.0)=\($0.1)" }.joined(separator: " ")
    }
}

// MARK: - Per-request telemetry

/// The request's Emitted Path account. `@unchecked Sendable`: the counters
/// are NSLock-guarded; the class exists so every `ConversationRender` copy a
/// request makes writes to the same record.
nonisolated final class EmittedPathRequestTelemetry: @unchecked Sendable {

    /// The render spellings, named as the events name them.
    enum Spelling: String, Sendable {
        case request
        case lastUserPrefix
        case continuation
        case base
        case admissionProbe
        case agentEdge
    }

    struct Summary: Equatable, Sendable {
        var resolves = 0
        var hits = 0
        /// Indexed prefix of the request-edge resolve, when it hit.
        var requestEdgeIndexedPrefix: Int?
        /// Suffix tokens the request-edge resolve encoded, when it hit.
        var requestEdgeSuffixTokens: Int?
        /// Miss reason of the request-edge resolve, when it missed.
        var requestEdgeMissReason: String?
        /// Why the request edge never consulted the index (an image-bearing
        /// or unknown-fingerprint render), when it did not.
        var requestEdgeSkipReason: String?
        /// The most recent resolve of any spelling: its indexed prefix on a
        /// hit, or its miss reason — what an offline walk reads back.
        var lastIndexedPrefix: Int?
        var lastMissReason: String?
        var shadowDifferences = 0
    }

    private let diagnostics: PrefixCacheDiagnostics.Context?
    private let lock = NSLock()
    private var summaryStorage = Summary()

    init(diagnostics: PrefixCacheDiagnostics.Context?) {
        self.diagnostics = diagnostics
    }

    var summary: Summary { lock.withLock { summaryStorage } }

    /// Records one resolve and its shadow check. The request-edge resolve
    /// persists in `log show` (`.notice`); the other spellings stay in the
    /// diagnostics net; a shadow difference is a warning wherever it lands.
    func record(
        spelling: Spelling,
        composition: EmittedPathResolve.Composition,
        canonical: [Int],
        resolveSeconds: Double,
        index: EmittedPathIndex
    ) {
        var difference: EmittedPathResolve.ShadowDifference?
        if case .indexed(let tokens, _, _, _, _) = composition {
            difference = EmittedPathResolve.shadow(composed: tokens, canonical: canonical)
        }
        lock.withLock {
            summaryStorage.resolves += 1
            switch composition {
            case .indexed(_, let indexedPrefix, let suffixTokens, _, _):
                summaryStorage.hits += 1
                summaryStorage.lastIndexedPrefix = indexedPrefix
                summaryStorage.lastMissReason = nil
                if spelling == .request {
                    summaryStorage.requestEdgeIndexedPrefix = indexedPrefix
                    summaryStorage.requestEdgeSuffixTokens = suffixTokens
                }
            case .miss(let reason):
                summaryStorage.lastIndexedPrefix = nil
                summaryStorage.lastMissReason = reason.rawValue
                if spelling == .request {
                    summaryStorage.requestEdgeMissReason = reason.rawValue
                }
            }
            if difference != nil { summaryStorage.shadowDifferences += 1 }
        }
        if difference != nil { index.noteShadowDifference() }

        let resolveEvent = EmittedPathResolve.ResolveEvent(
            spelling: spelling.rawValue, composition: composition,
            canonicalTokens: canonical.count, resolveSeconds: resolveSeconds)
        let resolveLevel: PrefixCacheDiagnostics.Level = spelling == .request ? .notice : .info
        emit(resolveEvent, level: resolveLevel)
        if let difference {
            emit(
                EmittedPathResolve.ShadowEvent(
                    spelling: spelling.rawValue, difference: difference), level: .warning)
        }
    }

    /// A render that never consulted the index — logged for the request
    /// edge only, once per request, so an image-bearing request's "resolves
    /// nothing" is visible with its reason.
    func recordSkip(spelling: Spelling, reason: String) {
        lock.withLock {
            if spelling == .request { summaryStorage.requestEdgeSkipReason = reason }
        }
        if let diagnostics {
            diagnostics.logSkip(
                stage: "emittedPathResolve", reason: reason,
                extraFields: [("spelling", spelling.rawValue)])
        } else {
            Log.server.info(
                "emitted-path skip stage=emittedPathResolve reason=\(reason) spelling=\(spelling.rawValue)"
            )
        }
    }

    private func emit(
        _ payload: some PrefixCacheDiagnostics.Payload, level: PrefixCacheDiagnostics.Level
    ) {
        if let diagnostics {
            diagnostics.log(payload, level: level)
            return
        }
        let line = "emitted-path " + EmittedPathResolve.line(payload)
        switch level {
        case .warning, .error: Log.server.warning(line)
        case .notice: Log.server.notice(line)
        case .info: Log.server.info(line)
        case .debug: Log.server.debug(line)
        }
    }
}
