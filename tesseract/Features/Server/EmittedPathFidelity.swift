//
//  EmittedPathFidelity.swift
//  tesseract
//
//  The fidelity gate on Emitted Path registration (ADR-0063, decision 9):
//  the emitted ids, detokenized and run through the same stream pipeline
//  the server ran on them — the vendor `ToolCallProcessor`, the app's
//  `ToolCallParser` think-splitting, the `GenerationAccumulator` fold —
//  must reconstruct the assistant message the Leaf Store is about to store.
//  The comparison is the stored message's own normalization: content and
//  reasoning trimmed, tool-call arguments as canonical (sorted-key) JSON,
//  so JSON spacing and parameter order are equal as parsed values.
//
//  A mismatch means the model's emission and the template's rendering of
//  it disagree beyond that normalization — the index would serve a path
//  whose text is not the history the next request renders. Registration
//  is refused, the leaf still stores, and the next turn re-prefills from
//  the first differing token, exactly as today.
//

import Foundation
import MLXLMCommon

nonisolated enum EmittedPathFidelity {

    enum Field: String, Sendable {
        case thinking
        case content
        case toolCalls
    }

    struct Mismatch: Equatable, Sendable {
        let field: Field
        /// Characters (tool calls: count) on the emitted side.
        let emittedLength: Int
        /// Characters (tool calls: count) on the stored side.
        let storedLength: Int
        /// First differing character (tool calls: index), when both sides
        /// exist.
        let firstDifference: Int?

        var fields: [(String, String)] {
            [
                ("field", field.rawValue),
                ("emittedLength", "\(emittedLength)"),
                ("storedLength", "\(storedLength)"),
                ("firstDifference", firstDifference.map { "\($0)" } ?? "nil"),
            ]
        }
    }

    enum Verdict: Equatable, Sendable {
        case match
        case mismatch(Mismatch)
    }

    /// The assistant message the server would have stored for `contentIDs`
    /// — the generated ids without the stop id — replayed chunk by chunk
    /// through the stream pipeline: the streaming detokenizer's chunks
    /// (`LinearStreamingDetokenizer`, the live loop's naive chunks in linear
    /// time), the vendor processor (whose tag state machine handles one
    /// boundary per chunk, so a whole turn in one chunk would not parse as
    /// the stream did), the app parser, the accumulator.
    /// `startsInsideThinkBlock` is the request's (the generation prompt
    /// opened a `<think>` block the model closes).
    static func replay(
        contentIDs: [Int],
        tokenizer: any Tokenizer,
        toolCallFormat: ToolCallFormat,
        tools: [ToolSpec]?,
        startsInsideThinkBlock: Bool
    ) -> HTTPPrefixCacheMessage {
        var detokenizer = LinearStreamingDetokenizer(tokenizer: tokenizer)
        let processor = ToolCallProcessor(format: toolCallFormat, tools: tools)
        var deltaTracker = ToolCallDeltaTracker(format: toolCallFormat)
        let parser = ToolCallParser(startsInsideThinkBlock: startsInsideThinkBlock)
        var accumulator = GenerationAccumulator()
        var libraryParsedToolCalls = false

        // The stream loop's rule: once the library parsed a call, the app
        // parser's tool events (the leaked wrapper tags) are suppressed.
        func fold(_ event: ToolCallParser.Event) {
            if libraryParsedToolCalls {
                switch event {
                case .toolCall, .malformedToolCall, .toolCallDelta: return
                default: break
                }
            }
            accumulator.ingest(AgentGeneration(parserEvent: event))
        }
        func emitChunk(_ text: String) {
            for event in parser.processChunk(text) { fold(event) }
        }
        func drainLibraryCalls() {
            while !processor.toolCalls.isEmpty {
                let call = processor.toolCalls.removeFirst()
                libraryParsedToolCalls = true
                accumulator.ingest(.toolCall(call))
            }
        }

        func feed(_ raw: String) {
            if let text = processor.processChunk(raw) { emitChunk(text) }
            if deltaTracker.observe(raw) != nil { libraryParsedToolCalls = true }
            drainLibraryCalls()
        }
        for id in contentIDs {
            for raw in detokenizer.append(token: id) { feed(raw) }
        }
        for raw in detokenizer.finish() { feed(raw) }
        // End-of-stream recovery, as the loop does it: buffered content
        // parses as calls; the residual is text unless the delta stream
        // already carried exactly those bytes (an in-flight tagged block).
        let residual = processor.processEOS(returnBufferedText: true)
        drainLibraryCalls()
        if let residual, !deltaTracker.deltasCarriedBuffer {
            emitChunk(residual)
        }
        for event in parser.finalize() { fold(event) }

        return .assistant(
            content: accumulator.text,
            reasoning: accumulator.thinking ?? "",
            toolCalls: accumulator.toolCalls.map {
                HTTPPrefixCacheToolCall(name: $0.function.name, arguments: $0.function.arguments)
            }
        )
    }

    /// `replay` compared with `stored`, field by field, so a mismatch names
    /// what differed and by how much (never the text itself — the event is
    /// a log line).
    static func check(
        contentIDs: [Int],
        tokenizer: any Tokenizer,
        toolCallFormat: ToolCallFormat,
        tools: [ToolSpec]?,
        startsInsideThinkBlock: Bool,
        stored: HTTPPrefixCacheMessage
    ) -> Verdict {
        let emitted = replay(
            contentIDs: contentIDs, tokenizer: tokenizer, toolCallFormat: toolCallFormat,
            tools: tools, startsInsideThinkBlock: startsInsideThinkBlock)
        return compare(emitted: emitted, stored: stored)
    }

    static func compare(emitted: HTTPPrefixCacheMessage, stored: HTTPPrefixCacheMessage) -> Verdict
    {
        if emitted.reasoning != stored.reasoning {
            return .mismatch(
                textMismatch(
                    .thinking, emitted: emitted.reasoning ?? "", stored: stored.reasoning ?? ""))
        }
        if emitted.content != stored.content {
            return .mismatch(
                textMismatch(.content, emitted: emitted.content, stored: stored.content))
        }
        if emitted.toolCalls != stored.toolCalls {
            let first = Array(zip(emitted.toolCalls, stored.toolCalls)).firstIndex { $0 != $1 }
            return .mismatch(
                Mismatch(
                    field: .toolCalls,
                    emittedLength: emitted.toolCalls.count,
                    storedLength: stored.toolCalls.count,
                    firstDifference: first
                        ?? (emitted.toolCalls.count == stored.toolCalls.count
                            ? nil : min(emitted.toolCalls.count, stored.toolCalls.count))))
        }
        return .match
    }

    private static func textMismatch(_ field: Field, emitted: String, stored: String) -> Mismatch {
        // Scalars, not `Character`s: canonical-equivalence-blind, like the
        // byte comparisons elsewhere in the cache.
        let emittedScalars = Array(emitted.unicodeScalars)
        let storedScalars = Array(stored.unicodeScalars)
        let shared = zip(emittedScalars, storedScalars).prefix { $0 == $1 }.count
        return Mismatch(
            field: field,
            emittedLength: emittedScalars.count,
            storedLength: storedScalars.count,
            firstDifference: shared)
    }
}
