//
//  EmittedPathRegistration.swift
//  tesseract
//
//  Registration of one finished turn's **Emitted Path** (ADR-0063,
//  decisions 2/3/9): build the path from the fed prompt ids, the generated
//  ids and the stop id; gate it on the fidelity check; hash the stored
//  conversation's render through its last end-of-turn marker; register it.
//  The Leaf Store phase calls this at the end of a stored turn. In this
//  ticket's dark launch it runs only after the **Live Leaf Capture** has
//  decided live — the fed path proved canonical — so the registered ids
//  equal the canonical encode and the resolve's shadow check is expected to
//  find no difference. #476 lifts that guard and serves the composition.
//
//  Skips are typed (`SkipReason`) and logged through the request's
//  diagnostics net at stage `emittedPathRegister`, so the corpus replay and
//  `log show` can account for every stored turn.
//

import Foundation
import MLXLMCommon

nonisolated enum EmittedPathRegistration {

    static let stage = "emittedPathRegister"

    /// Why a stored turn registered nothing. The first five are the ticket's
    /// guards, mirrored from `LiveLeafCapture.FallbackReason`; the rest are
    /// the dark launch's own and the render/template preconditions.
    enum SkipReason: String, Sendable {
        case nonIdentityKeySpace
        case noGeneratedTokens
        case cacheOffsetOutsideLivePath
        case intervened
        case fidelityRejected
        /// Dark launch: the Live Leaf Capture did not decide live (its
        /// comparison diverged, the render was shorter, the template's
        /// direct path skips the comparison, or the leaf store skipped
        /// before the decision).
        case notProvenLive
        /// No Emitted Path Index engaged for this render: an unkeyed
        /// completion, an unknown fingerprint, or an image-bearing render.
        case ineligibleRender
        /// The template has no single-token end-of-turn marker, or the
        /// stored render does not end on one.
        case noEndOfTurnMarker
        /// The stored render produced no bytes (a non-rendering tokenizer).
        case renderUnavailable
        /// The path alone exceeds the index's byte budget.
        case pathTooLarge
    }

    /// The registration guard a refused live capture maps to.
    static func skipReason(for fallback: LiveLeafCapture.FallbackReason) -> (
        reason: SkipReason, detail: String?
    ) {
        switch fallback {
        case .intervened: (.intervened, nil)
        case .nonIdentityKeySpace: (.nonIdentityKeySpace, nil)
        case .noGeneratedTokens: (.noGeneratedTokens, nil)
        case .cacheOffsetOutsideLivePath: (.cacheOffsetOutsideLivePath, nil)
        case .liveLongerThanStored: (.notProvenLive, "liveLongerThanStored")
        case .divergence: (.notProvenLive, "divergence")
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
