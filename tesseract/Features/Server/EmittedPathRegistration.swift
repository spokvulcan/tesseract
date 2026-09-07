//
//  EmittedPathRegistration.swift
//  tesseract
//
//  Registration of one finished turn's **Emitted Path** (ADR-0063,
//  decisions 2/3/9): build the path from the fed prompt ids, the generated
//  ids and the stop id; gate it on the fidelity check; hash the stored
//  conversation's render through its last end-of-turn marker; register it.
//  The Leaf Store phase calls this first on its fast path: a turn the
//  **Live Leaf Capture** stores live under the fed ids registers those ids
//  under the stored render's key, and the next request resolves to them
//  (`EmittedPathResolve`). A turn on the boundary path registers nothing —
//  its leaf is keyed on the canonical re-render, not the fed path.
//
//  Skips are typed (`SkipReason`) and logged through the request's
//  diagnostics net at stage `emittedPathRegister`, so the corpus replay and
//  `log show` can account for every stored turn.
//

import Foundation
import MLXLMCommon

nonisolated enum EmittedPathRegistration {

    static let stage = "emittedPathRegister"

    /// Why a stored turn registered nothing: the fast path's guards,
    /// mirrored from `LiveLeafCapture.FallbackReason`, with the boundary
    /// path's render rule beside them; the fidelity gate; and the render
    /// and template preconditions.
    enum SkipReason: String, Sendable {
        case nonIdentityKeySpace
        case noGeneratedTokens
        case cacheOffsetOutsideLivePath
        case intervened
        case fidelityRejected
        /// The turn took the boundary path under a think-stripping template
        /// at a new-user-message boundary: the next request re-renders the
        /// turn, so its fed ids are not what any later render resolves to.
        case thinkStrippingUserBoundary
        /// No Emitted Path Index engaged for this render: an unkeyed
        /// completion, an unknown fingerprint, or an image-bearing render.
        case ineligibleRender
        /// The template has no single-token end-of-turn marker, or the
        /// stored render does not end on one.
        case noEndOfTurnMarker
        /// The template's end-of-turn marker is not a hard boundary for
        /// this tokenizer: encoding the bytes after it on their own differs
        /// from encoding them in context, so no composition can be served.
        case suffixEncodeUnstable
        /// The stored render produced no bytes (a non-rendering tokenizer).
        case renderUnavailable
        /// The path alone exceeds the index's byte budget.
        case pathTooLarge
    }

    /// The registration skip a boundary-path turn logs: the same reason the
    /// `leafStore` event's `boundary` field carries, as a camel-case wire
    /// name.
    static func skipReason(for fallback: LiveLeafCapture.FallbackReason) -> SkipReason {
        switch fallback {
        case .intervened: .intervened
        case .nonIdentityKeySpace: .nonIdentityKeySpace
        case .noGeneratedTokens: .noGeneratedTokens
        case .cacheOffsetOutsideLivePath: .cacheOffsetOutsideLivePath
        case .thinkStrippingUserBoundary: .thinkStrippingUserBoundary
        }
    }

    struct Inputs {
        let index: EmittedPathIndex
        let fingerprint: String
        let marker: EndOfTurnMarker
        let tokenizer: any Tokenizer
        /// The stored conversation's render, no generation prompt.
        let storedRenderBytes: [UInt8]
        /// The assistant message the Leaf Store appended.
        let storedMessage: HTTPPrefixCacheMessage
        let promptKeyPath: [Int]
        let generatedTokens: [Int]
        let stoppedOn: Int?
        let toolCallFormat: ToolCallFormat
        let tools: [ToolSpec]?
        let startsInsideThinkBlock: Bool
    }

    struct Registered: Equatable, Sendable {
        /// Bytes of the stored render the key covers (through the marker).
        let prefixBytes: Int
        let pathLength: Int
        let promptTokens: Int
        let generatedTokens: Int
        let appendedEndOfTurn: Bool
        /// Set when the key was already registered (last writer wins).
        let previousPathLength: Int?
        let evicted: Int
    }

    struct Skip: Sendable {
        let reason: SkipReason
        let fields: [(String, String)]
        let fidelity: EmittedPathFidelity.Mismatch?

        init(
            _ reason: SkipReason, fields: [(String, String)] = [],
            fidelity: EmittedPathFidelity.Mismatch? = nil
        ) {
            self.reason = reason
            self.fields = fields
            self.fidelity = fidelity
        }
    }

    enum Outcome: Sendable {
        case registered(Registered)
        case skipped(Skip)
    }

    static func register(_ inputs: Inputs) -> Outcome {
        let path = EmittedPath.make(
            promptKeyPath: inputs.promptKeyPath,
            generatedTokens: inputs.generatedTokens,
            stoppedOn: inputs.stoppedOn,
            endOfTurnID: inputs.marker.tokenID
        )
        if case .mismatch(let mismatch) = EmittedPathFidelity.check(
            contentIDs: path.contentIDs,
            tokenizer: inputs.tokenizer,
            toolCallFormat: inputs.toolCallFormat,
            tools: inputs.tools,
            startsInsideThinkBlock: inputs.startsInsideThinkBlock,
            stored: inputs.storedMessage
        ) {
            inputs.index.noteFidelityRejection()
            return .skipped(Skip(.fidelityRejected, fields: mismatch.fields, fidelity: mismatch))
        }
        guard
            let prefixEnd = EndOfTurnMarker.lastOccurrenceEnd(
                of: inputs.marker.bytes, in: inputs.storedRenderBytes)
        else {
            return .skipped(Skip(.noEndOfTurnMarker))
        }
        let hash = EmittedPathIndex.hash(of: inputs.storedRenderBytes[..<prefixEnd])
        switch inputs.index.register(fingerprint: inputs.fingerprint, hash: hash, ids: path.ids) {
        case .rejectedTooLarge:
            return .skipped(Skip(.pathTooLarge, fields: [("pathLength", "\(path.ids.count)")]))
        case .inserted(let evicted):
            return .registered(
                Registered(
                    prefixBytes: prefixEnd, pathLength: path.ids.count,
                    promptTokens: path.promptCount, generatedTokens: inputs.generatedTokens.count,
                    appendedEndOfTurn: path.appendedEndOfTurn, previousPathLength: nil,
                    evicted: evicted))
        case .replaced(let previousLength, let evicted):
            return .registered(
                Registered(
                    prefixBytes: prefixEnd, pathLength: path.ids.count,
                    promptTokens: path.promptCount, generatedTokens: inputs.generatedTokens.count,
                    appendedEndOfTurn: path.appendedEndOfTurn, previousPathLength: previousLength,
                    evicted: evicted))
        }
    }

    // MARK: - Events

    /// A registration: the persisted (`.notice`) account of what the index
    /// learned from this turn.
    struct RegisterEvent: PrefixCacheDiagnostics.Payload {
        let registered: Registered
        let registerSeconds: Double

        let eventName = "emittedPathRegister"

        var fields: [(String, String)] {
            var fields: [(String, String)] = [
                ("prefixBytes", "\(registered.prefixBytes)"),
                ("pathLength", "\(registered.pathLength)"),
                ("promptTokens", "\(registered.promptTokens)"),
                ("generatedTokens", "\(registered.generatedTokens)"),
                ("appendedEndOfTurn", "\(registered.appendedEndOfTurn)"),
                ("evicted", "\(registered.evicted)"),
                ("registerMs", PrefixCacheDiagnostics.milliseconds(registerSeconds)),
            ]
            if let previous = registered.previousPathLength {
                fields.append(("overwrote", "true"))
                fields.append(("previousPathLength", "\(previous)"))
            }
            return fields
        }
    }

    /// The same key registered again (decision 6): both lengths, so a
    /// branch that replaced a longer path is visible.
    struct OverwriteEvent: PrefixCacheDiagnostics.Payload {
        let pathLength: Int
        let previousPathLength: Int

        let eventName = "emittedPathOverwrite"

        var fields: [(String, String)] {
            [
                ("pathLength", "\(pathLength)"),
                ("previousPathLength", "\(previousPathLength)"),
            ]
        }
    }

    /// The fidelity gate refusing a registration — a warning: the emission
    /// and the template's rendering of it disagree.
    struct FidelityEvent: PrefixCacheDiagnostics.Payload {
        let mismatch: EmittedPathFidelity.Mismatch

        let eventName = "emittedPathFidelity"

        var fields: [(String, String)] {
            [("result", "mismatch")] + mismatch.fields
        }
    }

    /// Emit the outcome's events through the request's diagnostics net.
    static func emit(
        _ outcome: Outcome,
        registerSeconds: Double,
        in diagnostics: PrefixCacheDiagnostics.Context
    ) {
        switch outcome {
        case .registered(let registered):
            diagnostics.log(
                RegisterEvent(registered: registered, registerSeconds: registerSeconds),
                level: .notice)
            if let previous = registered.previousPathLength {
                diagnostics.log(
                    OverwriteEvent(pathLength: registered.pathLength, previousPathLength: previous),
                    level: .notice)
            }
        case .skipped(let skip):
            emitSkip(skip, in: diagnostics)
        }
    }

    static func emitSkip(_ skip: Skip, in diagnostics: PrefixCacheDiagnostics.Context) {
        if let mismatch = skip.fidelity {
            diagnostics.log(FidelityEvent(mismatch: mismatch), level: .warning)
        }
        diagnostics.logSkip(
            stage: stage, reason: skip.reason.rawValue,
            level: skip.fidelity == nil ? .info : .warning,
            extraFields: skip.fields)
    }
}
