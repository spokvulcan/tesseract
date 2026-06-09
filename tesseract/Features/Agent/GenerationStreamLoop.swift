import Foundation
import MLXLMCommon
import os

/// The streaming-generation spine: turns one raw model `AsyncStream<Generation>`
/// into the agent's `AgentGeneration` event stream under the thinking-loop
/// safeguard. Owns the loop, the four-case switch, the `ToolCallParser` lifecycle,
/// the `ThinkingSafeguardObserver` intervention triple, the continuation swap, and
/// the cross-swap external cancel. Each caller keeps its own per-event ``Sink`` and
/// its own post-loop projection of the returned ``Outcome``.
///
/// `nonisolated` so it can be driven from both an actor (`LLMActor`) and the
/// MainActor (`AgentEngine`) with no isolation hop, and so the sink can fold a
/// non-`Sendable` accumulator inline.
nonisolated struct GenerationStreamLoop {

    /// The normalized minimal handle the loop consumes. Callers adapt their richer
    /// handles down to this at the edge; the rich prefill metadata never crosses.
    nonisolated struct RawGenerationHandle: Sendable {
        let stream: AsyncStream<Generation>
        let cancel: @Sendable () -> Void
        let waitForCompletion: @Sendable () async -> Void

        init(
            stream: AsyncStream<Generation>,
            cancel: @escaping @Sendable () -> Void,
            waitForCompletion: @escaping @Sendable () async -> Void
        ) {
            self.stream = stream
            self.cancel = cancel
            self.waitForCompletion = waitForCompletion
        }
    }

    /// The one real port: re-prefills from the safe prefix and returns a fresh
    /// handle. A closure, not a protocol — it has a single requirement.
    typealias ContinuationStarter =
        @Sendable (_ safePrefix: String) async throws -> RawGenerationHandle

    /// Per-event push, called inline on the driving task. Deliberately not
    /// `@Sendable` so a caller can fold non-`Sendable` state; never sees `.info`.
    typealias Sink = (AgentGeneration) -> Void

    struct Outcome: Sendable {
        /// Terminal `.info` captured from the stream, not pushed to the sink. Each
        /// caller re-yields it as the terminal stream event its downstream reads.
        let completionInfo: AgentGeneration.Info?
        let intervened: Bool
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
        var handle: RawGenerationHandle
        /// Per-handle dedup of `cancel()`; reset on a continuation swap.
        var cancelIssued = false
        /// Sticky once `cancelCurrent` was called; signals the loop to stop and
        /// report `cancelled`.
        var externalCancel = false
    }

    private let box: OSAllocatedUnfairLock<HandleBox>
    private let startsInsideThinkBlock: Bool
    private let safeguardConfig: ThinkingRepetitionDetector.Config

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
        safeguard: ThinkingRepetitionDetector.Config,
        logContext: String = ""
    ) {
        self.box = OSAllocatedUnfairLock(initialState: HandleBox(handle: initial))
        self.startsInsideThinkBlock = startsInsideThinkBlock
        self.safeguardConfig = safeguard
        self.logContext = logContext
    }

    /// Cancels whichever raw handle is currently live (across swaps); idempotent.
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

    func run(continuation: ContinuationStarter?, sink: Sink) async throws -> Outcome {
        var parser = ToolCallParser(startsInsideThinkBlock: startsInsideThinkBlock)
        let safeguard = ThinkingSafeguardObserver(config: safeguardConfig)
        var rawChunkParts: [String] = []
        var libraryParsedToolCalls = false
        // Vendor's ToolCallProcessor silently drops its in-flight buffer at EOS if
        // it can't decode it. Accumulate every `.toolCallBufferDelta` so we can
        // surface the lost content as `.malformedToolCall`; a successful `.toolCall`
        // consumed the buffer and resets it.
        var libraryToolCallBufferAccum = ""
        var libraryToolCallEventCount = 0
        var completionInfo: AgentGeneration.Info?
        var cancelled = false
        // Set when an intervention closed the think block but no continuation
        // swapped in (no starter, or the starter threw). The truncation triple is
        // already emitted and the parser was NOT re-init, so it still holds the
        // degenerate post-trigger thinking — finalize must be skipped or it would
        // forward that text AFTER `.thinkEnd`, re-polluting the truncated reasoning.
        var interventionClosedWithoutSwap = false

        typealias Intervention = (safePrefix: String, reason: ThinkingRepetitionDetector.Reason)

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
        // cancel and intervention-without-swap paths finalize is skipped, so it
        // snapshots the un-finalized parser. The raw chunks are kept unjoined and
        // concatenated lazily by `Diagnostics.rawChunksJoined`.
        func makeDiagnostics() -> Diagnostics {
            Diagnostics(
                rawChunkParts: rawChunkParts,
                finalizeState: parser.snapshotFinalizeState(),
                libraryParsedToolCalls: libraryParsedToolCalls
            )
        }

        // Forward a parser event to the sink under the safeguard. When the vendor
        // library already parsed tool calls, its `<tool_call>` wrapper tags leak
        // back as chunks; suppress the app parser's tool events. Returns the
        // intervention payload when the safeguard fired.
        func emitParserEvent(
            _ event: ToolCallParser.Event,
            allowToolEvents: Bool
        ) -> Intervention? {
            if !allowToolEvents {
                switch event {
                case .toolCall, .malformedToolCall, .toolCallDelta:
                    return nil
                default:
                    break
                }
            }
            switch safeguard.observe(parserEvent: event) {
            case .forward:
                sink(AgentGeneration(parserEvent: event))
                return nil
            case .intervene(let safe, let reason):
                // Replace the degenerate thinking with the clean prefix, emit the
                // hand-off phrase, close `</think>`. Downstream consumers reset
                // their thinking accumulators on `.thinkTruncate`.
                sink(.thinkTruncate(safePrefix: safe))
                sink(.thinking(safeguardConfig.injectionMessage))
                sink(.thinkEnd)
                Log.agent.warning(
                    "Thinking-loop intervention — reason=\(reason.rawValue) "
                    + "safe_prefix_chars=\(safe.count)"
                    + logSuffix
                )
                return (safe, reason)
            }
        }

        var currentStream = box.withLock { $0.handle.stream }

        if stopRequested() {
            // Cancel arrived before `run` began consuming (the "available
            // pre-`run`" contract). Stop before touching the stream.
            cancelled = true
        } else {
            streamLoop: while true {
                var intervention: Intervention? = nil

                for await item in currentStream {
                    if stopRequested() {
                        cancelled = true
                        break
                    }

                    switch item {
                    case .chunk(let text):
                        rawChunkParts.append(text)
                        for event in parser.processChunk(text) {
                            if let fired = emitParserEvent(
                                event,
                                allowToolEvents: !libraryParsedToolCalls
                            ) {
                                intervention = fired
                                break
                            }
                        }

                    case .toolCall(let call):
                        libraryParsedToolCalls = true
                        libraryToolCallEventCount += 1
                        // The close-tag parse consumed whatever the deltas were
                        // building toward — reset so a later interrupted call's
                        // malformed surface doesn't include this parsed block.
                        libraryToolCallBufferAccum = ""
                        sink(.toolCall(call))

                    case .toolCallBufferDelta(let delta):
                        libraryParsedToolCalls = true
                        libraryToolCallBufferAccum += delta
                        sink(.toolCallDelta(name: nil, argumentsDelta: delta))

                    case .info(let vinfo):
                        // Captured into the Outcome, never pushed to the sink.
                        completionInfo = AgentGeneration.Info(
                            promptTokenCount: vinfo.promptTokenCount,
                            generationTokenCount: vinfo.generationTokenCount,
                            promptTime: vinfo.promptTime,
                            generateTime: vinfo.generateTime,
                            stopReason: vinfo.stopReason
                        )
                    }

                    if intervention != nil { break }
                }

                // An external cancel may have finished the stream with no item
                // delivered to the body above — re-check once the iterator ends.
                if stopRequested() { cancelled = true }
                if cancelled { break streamLoop }

                guard let fired = intervention else {
                    break streamLoop  // natural end of the current stream
                }

                // Intervention fired; the truncation triple is already emitted.
                guard let continuation else {
                    // No starter ⇒ emit the truncation and stop. Don't flush the
                    // parser afterward (it still holds the truncated thinking).
                    interventionClosedWithoutSwap = true
                    break streamLoop
                }

                // Cancel + drain the current (old) handle before swapping.
                let old = box.withLock { state -> RawGenerationHandle in
                    state.cancelIssued = true
                    return state.handle
                }
                old.cancel()
                await old.waitForCompletion()

                do {
                    let newHandle = try await continuation(fired.safePrefix)
                    // Install the continuation handle. If an external cancel landed
                    // during the swap, honor it on the freshly-installed handle —
                    // the cross-swap invariant: cancel whichever handle is live.
                    let externalDuringSwap = box.withLock { state -> Bool in
                        state.handle = newHandle
                        state.cancelIssued = false
                        return state.externalCancel
                    }
                    if externalDuringSwap {
                        cancelLiveHandleOnce()
                        cancelled = true
                        break streamLoop
                    }
                    // Continuation picks up AFTER `</think>` — re-init the parser in
                    // out-of-think mode so its output is classified as text. Do NOT
                    // reset the safeguard: `hasIntervened` must survive to the
                    // result and the intervention limit must keep blocking a re-fire.
                    parser = ToolCallParser(startsInsideThinkBlock: false)
                    currentStream = newHandle.stream
                    continue streamLoop
                } catch {
                    Log.agent.error(
                        "Thinking-safeguard continuation failed: "
                        + "\(error.localizedDescription) — finishing with truncated response"
                    )
                    // Same as the no-starter case: the think block is closed and the
                    // parser was never re-init, so skip the post-loop finalize flush.
                    interventionClosedWithoutSwap = true
                    break streamLoop
                }
            }
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
                intervened: safeguard.hasIntervened,
                cancelled: true,
                diagnostics: makeDiagnostics()
            )
        }

        // Flush any remaining buffered text. A finalize-triggered safeguard event
        // cannot swap (the loop is over), matching the server path's discipline.
        // Skipped when an intervention closed the think block without a swap: the
        // un-reinit parser still holds the degenerate post-trigger thinking, and
        // flushing it would forward text AFTER the `.thinkEnd` already emitted —
        // re-polluting the reasoning the safeguard just truncated.
        if !interventionClosedWithoutSwap {
            for event in parser.finalize() {
                _ = emitParserEvent(event, allowToolEvents: !libraryParsedToolCalls)
            }
        }

        // Surface the vendor's dropped in-flight buffer: when the model emitted
        // `<tool_call>…` then hit EOS before the close tag, no `.toolCall` ever
        // fired and the client would otherwise see `finish_reason=stop` with no
        // signal that a tool call was attempted.
        let droppedBuffer = libraryToolCallBufferAccum
        libraryToolCallBufferAccum = ""
        if libraryParsedToolCalls, libraryToolCallEventCount == 0, !droppedBuffer.isEmpty {
            let wrappedBuffer = Self.wrapMalformedToolCallBuffer(droppedBuffer)
            Log.agent.warning(
                "Vendor ToolCallProcessor dropped unparseable buffer at EOS — "
                + "bufferLen=\(droppedBuffer.count) wrappedLen=\(wrappedBuffer.count) "
                + "head=\(String(wrappedBuffer.prefix(120)).debugDescription) "
                + "tail=\(String(wrappedBuffer.suffix(80)).debugDescription)"
                + logSuffix
            )
            sink(.malformedToolCall(wrappedBuffer))
        }

        return Outcome(
            completionInfo: completionInfo,
            intervened: safeguard.hasIntervened,
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
    /// Collapse a vendor-style `{ stream, completion }` pair: `cancel` and
    /// `waitForCompletion` drive the underlying generation `Task`.
    nonisolated init(stream: AsyncStream<Generation>, completion: Task<Void, Never>) {
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
