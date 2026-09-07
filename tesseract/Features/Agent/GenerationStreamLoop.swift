import Foundation
import MLXLMCommon
import os

/// Turns raw model output into agent events without rewriting reasoning.
/// Owns parser finalization, external cancellation, and completion metrics.
///
/// `nonisolated` so it can be driven from both an actor (`LLMActor`) and the
/// MainActor (`AgentEngine`) with no isolation hop, and so the sink can fold a
/// non-`Sendable` accumulator inline.
nonisolated struct GenerationStreamLoop {

    /// The normalized minimal handle the loop consumes. Callers adapt their richer
    /// handles down to this at the edge; the rich prefill metadata never crosses.
    nonisolated struct RawGenerationHandle: Sendable {
        let stream: AsyncStream<RawGeneration>
        let cancel: @Sendable () -> Void
        let waitForCompletion: @Sendable () async -> Void

        init(
            stream: AsyncStream<RawGeneration>,
            cancel: @escaping @Sendable () -> Void,
            waitForCompletion: @escaping @Sendable () async -> Void
        ) {
            self.stream = stream
            self.cancel = cancel
            self.waitForCompletion = waitForCompletion
        }
    }

    /// Per-event push, called inline on the driving task. Deliberately not
    /// `@Sendable` so a caller can fold non-`Sendable` state; never sees `.info`.
    typealias Sink = (AgentGeneration) -> Void

    struct Outcome: Sendable {
        /// Terminal `.info` captured from the stream, not pushed to the sink. Each
        /// caller re-yields it as the terminal stream event its downstream reads.
        let completionInfo: AgentGeneration.Info?
        let cancelled: Bool
        /// Loop-owned silent-close surface; the agent caller ignores it.
        let diagnostics: Diagnostics
    }

    struct Diagnostics: Sendable {
        /// Raw chunks kept unjoined. `rawChunksJoined` materializes the
        /// concatenation lazily so neither the cancel path (where no caller reads
        /// it) nor a library-parsed agent turn (where `hasUnparsedToolCallMarkers`
        /// short-circuits on `libraryParsedToolCalls` before any scan) pays for an
        /// O(total tokens) join it immediately discards.
        let rawChunkParts: [String]
        let finalizeState: ToolCallParser.FinalizeState
        let libraryParsedToolCalls: Bool

        var rawChunksJoined: String { rawChunkParts.joined() }

        /// Raw output carries `<tool_call>` / `<function` markers the vendor
        /// library never turned into a `.toolCall` event — the "library missed a
        /// tool call" signal both callers warn on. Joins only when the vendor
        /// emitted no tool call (otherwise the marker scan is moot).
        var hasUnparsedToolCallMarkers: Bool {
            guard !libraryParsedToolCalls else { return false }
            let joined = rawChunksJoined
            return joined.contains("tool_call") || joined.contains("<function")
        }
    }

    private struct HandleBox {
        let handle: RawGenerationHandle
        /// Deduplicates calls to the raw handle’s `cancel()`.
        var cancelIssued = false
        /// Sticky once `cancelCurrent` was called; signals the loop to stop and
        /// report `cancelled`.
        var externalCancel = false
    }

    private let box: OSAllocatedUnfairLock<HandleBox>
    private let startsInsideThinkBlock: Bool

    /// Pre-formatted `key=value` correlation token (e.g. `request_id=…` /
    /// `generation_id=…`) appended to the loop's own diagnostic warnings. The
    /// loop is request-agnostic, so the caller supplies the id; logging stays in
    /// one place instead of being re-duplicated into each caller.
    private let logContext: String

    /// `" \(logContext)"` when set, else empty — appended to warning lines.
    private var logSuffix: String { logContext.isEmpty ? "" : " " + logContext }

    init(
        initial: RawGenerationHandle,
        startsInsideThinkBlock: Bool,
        logContext: String = ""
    ) {
        self.box = OSAllocatedUnfairLock(initialState: HandleBox(handle: initial))
        self.startsInsideThinkBlock = startsInsideThinkBlock
        self.logContext = logContext
    }

    /// Cancels the raw handle; idempotent.
    /// Available before `run` so the caller can wire it into its own external
    /// cancel synchronously.
    var cancelCurrent: @Sendable () -> Void {
        let box = self.box
        return {
            let handle = box.withLock { state -> RawGenerationHandle? in
                state.externalCancel = true
                if state.cancelIssued { return nil }
                state.cancelIssued = true
                return state.handle
            }
            handle?.cancel()
        }
    }

    func run(sink: Sink) async throws -> Outcome {
        let parser = ToolCallParser(startsInsideThinkBlock: startsInsideThinkBlock)
        var rawChunkParts: [String] = []
        var libraryParsedToolCalls = false
        // The ToolCallProcessor drops its in-flight buffer at EOS if it can't
        // decode it (the producer suppresses the residual for tagged blocks).
        // Accumulate every `.toolCallBufferDelta` so we can surface the lost
        // content as `.malformedToolCall`; a successful `.toolCall` consumed
        // the buffer and resets it.
        var libraryToolCallBufferAccum = ""
        /// Name-lock for the vendor's in-flight buffer, mirroring
        /// `ToolCallParser.toolCallCurrentName`: locked once from the
        /// accumulated body, reset when a parsed `.toolCall` consumes it.
        var libraryToolCallName: String?
        var libraryToolCallEventCount = 0
        var completionInfo: AgentGeneration.Info?
        var cancelled = false

        // A stop is requested by either cooperative task cancellation or an
        // external `cancelCurrent()`.
        func stopRequested() -> Bool {
            Task.isCancelled || box.withLock { $0.externalCancel }
        }

        // Issue `cancel()` to whichever handle is currently live, at most once.
        func cancelLiveHandleOnce() {
            let toCancel = box.withLock { state -> RawGenerationHandle? in
                if state.cancelIssued { return nil }
                state.cancelIssued = true
                return state.handle
            }
            toCancel?.cancel()
        }

        // Snapshot the loop's silent-close surface. Call AFTER `finalize()` on the
        // natural path so `finalizeState` reflects the flushed parser; on the
        // cancel path finalize is skipped, so it
        // snapshots the un-finalized parser. The raw chunks are kept unjoined and
        // concatenated lazily by `Diagnostics.rawChunksJoined`.
        func makeDiagnostics() -> Diagnostics {
            Diagnostics(
                rawChunkParts: rawChunkParts,
                finalizeState: parser.snapshotFinalizeState(),
                libraryParsedToolCalls: libraryParsedToolCalls
            )
        }

        // Suppress app-parser tool events once the vendor owns tool parsing.
        func emitParserEvent(_ event: ToolCallParser.Event, allowToolEvents: Bool) {
            if !allowToolEvents {
                switch event {
                case .toolCall, .malformedToolCall, .toolCallDelta:
                    return
                default:
                    break
                }
            }
            sink(AgentGeneration(parserEvent: event))
        }

        let stream = box.withLock { $0.handle.stream }
        if stopRequested() {
            cancelled = true
        } else {
            for await item in stream {
                if stopRequested() {
                    cancelled = true
                    break
                }
                switch item {
                case .chunk(let text):
                    rawChunkParts.append(text)
                    for event in parser.processChunk(text) {
                        emitParserEvent(event, allowToolEvents: !libraryParsedToolCalls)
                    }
                case .toolCall(let call):
                    libraryParsedToolCalls = true
                    libraryToolCallEventCount += 1
                    libraryToolCallBufferAccum = ""
                    libraryToolCallName = nil
                    sink(.toolCall(call))
                case .toolCallBufferDelta(let delta):
                    libraryParsedToolCalls = true
                    libraryToolCallBufferAccum += delta
                    if libraryToolCallName == nil {
                        libraryToolCallName = ToolCallNameLock.extract(
                            from: libraryToolCallBufferAccum)
                    }
                    sink(.toolCallDelta(name: libraryToolCallName, argumentsDelta: delta))
                case .info(let vinfo):
                    completionInfo = AgentGeneration.Info(vinfo)
                }
            }
            // External cancellation can finish a stream without yielding an item.
            if stopRequested() { cancelled = true }
        }

        // On a stop, make sure the live handle is cancelled exactly once before we
        // wait on it — otherwise a cooperative `Task.isCancelled` cancel (no
        // `cancelCurrent`) would block on a still-running generation.
        if cancelled {
            cancelLiveHandleOnce()
        }

        // Wait for whichever handle is currently live to finish (exactly once).
        let waitForCompletion = box.withLock { $0.handle.waitForCompletion }
        await waitForCompletion()

        if cancelled {
            // On cancel we skip the finalize flush and malformed-EOS surfacing —
            // the caller discards partial output and clears its cache.
            return Outcome(
                completionInfo: completionInfo,
                cancelled: true,
                diagnostics: makeDiagnostics()
            )
        }

        for event in parser.finalize() {
            emitParserEvent(event, allowToolEvents: !libraryParsedToolCalls)
        }

        // Surface the processor's dropped in-flight buffer: when the model
        // emitted `<tool_call>…` then hit EOS before the close tag, no
        // `.toolCall` ever fired and the client would otherwise see
        // `finish_reason=stop` with no signal that a tool call was attempted.
        let droppedBuffer = libraryToolCallBufferAccum
        libraryToolCallBufferAccum = ""
        if libraryParsedToolCalls, libraryToolCallEventCount == 0, !droppedBuffer.isEmpty {
            let wrappedBuffer = Self.wrapMalformedToolCallBuffer(droppedBuffer)
            Log.agent.warning(
                "ToolCallProcessor dropped unparseable buffer at EOS — "
                    + "bufferLen=\(droppedBuffer.count) wrappedLen=\(wrappedBuffer.count) "
                    + "head=\(String(wrappedBuffer.prefix(120)).debugDescription) "
                    + "tail=\(String(wrappedBuffer.suffix(80)).debugDescription)"
                    + logSuffix
            )
            sink(.malformedToolCall(wrappedBuffer))
        }

        return Outcome(
            completionInfo: completionInfo,
            cancelled: false,
            diagnostics: makeDiagnostics()
        )
    }

    /// Wrap a vendor-dropped in-flight tool-call buffer with `<tool_call>` /
    /// `</tool_call>` tags so clients can always detect a tool-call attempt even
    /// when the model was interrupted before emitting the close tag. Idempotent.
    nonisolated static func wrapMalformedToolCallBuffer(_ buffer: String) -> String {
        var wrapped = buffer
        if !wrapped.hasPrefix("<tool_call>") {
            wrapped = "<tool_call>\n" + wrapped
        }
        if !wrapped.hasSuffix("</tool_call>") {
            if !wrapped.hasSuffix("\n") { wrapped.append("\n") }
            wrapped.append("</tool_call>")
        }
        return wrapped
    }
}

// MARK: - Handle normalization at the caller edge

extension GenerationStreamLoop.RawGenerationHandle {
    /// Collapse a `{ stream, completion }` pair from ``TokenGenerationLoop``:
    /// `cancel` and `waitForCompletion` drive the underlying generation `Task`.
    nonisolated init(stream: AsyncStream<RawGeneration>, completion: Task<Void, Never>) {
        self.init(
            stream: stream,
            cancel: { completion.cancel() },
            waitForCompletion: { await completion.value }
        )
    }

    /// The agent's already-minimal handle maps 1:1 (also the continuation return
    /// type for both callers).
    nonisolated init(_ start: HTTPServerRawGenerationStart) {
        self.init(
            stream: start.stream,
            cancel: start.cancel,
            waitForCompletion: start.waitForCompletion
        )
    }

    // The Server Completion module adds one more normalization in
    // `ServerCompletion.swift`: its private prefill bundle collapses to
    // `{ stream, cancel, wait }` there, so the bundle never crosses the seam.
}

// MARK: - Late-bound cancel bridge

/// A cancel hook handed to a caller *before* the `GenerationStreamLoop` that
/// backs it exists. Both generation callers must return their `start.cancel`
/// synchronously, yet the loop's `cancelCurrent` isn't available until the
/// driving `Task` has launched and built the loop. This box bridges that gap:
/// the task `fill`s it once the loop exists; every external cancel site reads
/// through it. Calling it before `fill` is a no-op (the loop hasn't begun
/// consuming, so there is nothing to cancel yet).
///
/// Extracted so the "deferred-cancel-before-the-loop" dance lives in one place
/// instead of being hand-copied as an `OSAllocatedUnfairLock<…?>` into each
/// caller.
nonisolated struct LateBoundCancel: Sendable {
    private let box = OSAllocatedUnfairLock<(@Sendable () -> Void)?>(initialState: nil)

    /// Install the real cancel once the loop exists. Called once, from the task.
    func fill(_ cancel: @escaping @Sendable () -> Void) {
        box.withLock { $0 = cancel }
    }

    /// Invoke the installed cancel if present; idempotent before `fill`.
    func callAsFunction() {
        (box.withLock { $0 })?()
    }
}
