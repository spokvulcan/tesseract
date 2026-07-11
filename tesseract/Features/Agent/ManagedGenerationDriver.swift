//
//  ManagedGenerationDriver.swift
//  tesseract
//
//  The **Managed Generation Driver**: the envelope every managed generation
//  runs inside — the agent engine's wrap and the server's cache-aware
//  completion drive, which previously hand-copied it around
//  `GenerationStreamLoop`. It owns the safeguard configuration (and its
//  elevated-risk warning), the cross-swap cancel bridge fill, the loop run,
//  the shared outcome tail (terminal `.info` re-yield through the sink, the
//  completion log, the unparsed-tool-call warning), and the start-handle
//  contract both callers expose. Callers keep their own task/isolation
//  structure, per-event sink side effects, and post-outcome projections
//  (cache admissions, cache diagnostics, engine registries).
//

import Foundation

nonisolated struct ManagedGenerationDriver: Sendable {

    /// The loop's safeguard configuration — exposed because both callers
    /// also feed its `continuationHandOff` into their continuation starters.
    let safeguard: ThinkingRepetitionDetector.Config

    /// Whether the rendered prompt ends inside an open `<think>` block —
    /// exposed because the server's leaf-store mode selection keys on it.
    let startsInsideThinkBlock: Bool

    private let logContext: String

    /// Derives the safeguard from `parameters` and emits the elevated-risk
    /// warning — previously duplicated at both call sites.
    init(
        parameters: AgentGenerateParameters,
        startsInsideThinkBlock: Bool,
        logContext: String
    ) {
        self.safeguard = parameters.thinkingSafeguard
        self.startsInsideThinkBlock = startsInsideThinkBlock
        self.logContext = logContext
        parameters.warnIfThinkingLoopRiskElevated(startsThinking: startsInsideThinkBlock)
    }

    /// Run the spine: build the loop, fill `cancelBridge` (which the caller
    /// wired into its start handle synchronously at assembly), drive with the
    /// caller's sink, and apply the shared outcome tail. On a natural finish
    /// the terminal `.info` is re-yielded *through the sink* — downstream
    /// consumers read completion metrics from the stream, not a return
    /// value — and the shared completion log and unparsed-tool-call warning
    /// are emitted. A cancelled outcome gets no tail. Returns the outcome
    /// for caller-specific projections.
    func run(
        initial: GenerationStreamLoop.RawGenerationHandle,
        cancelBridge: LateBoundCancel,
        continuationStarter: GenerationStreamLoop.ContinuationStarter?,
        sink: GenerationStreamLoop.Sink
    ) async throws -> GenerationStreamLoop.Outcome {
        let loop = GenerationStreamLoop(
            initial: initial,
            startsInsideThinkBlock: startsInsideThinkBlock,
            safeguard: safeguard,
            logContext: logContext
        )
        cancelBridge.fill(loop.cancelCurrent)

        let outcome = try await loop.run(continuation: continuationStarter, sink: sink)
        guard !outcome.cancelled else { return outcome }

        if let info = outcome.completionInfo {
            sink(.info(info))
            Log.agent.info(
                "Generation complete — \(info.generationTokenCount) tokens, "
                    + "\(String(format: "%.1f", info.tokensPerSecond)) tok/s, "
                    + "stopReason=\(describeStopReason(info.stopReason))"
            )
        }
        if outcome.diagnostics.hasUnparsedToolCallMarkers {
            Log.agent.warning(
                "Raw output contains tool call markers but no .toolCall events were emitted by library"
            )
        }
        return outcome
    }

    /// The start-handle contract every managed generation exposes: `cancel`
    /// fires the live-handle bridge (whichever raw handle is current after an
    /// intervention swap) *and* the driving task; `waitForCompletion` awaits
    /// the task (the loop awaits the live handle internally before
    /// returning); and stream termination — consumer gone or natural finish —
    /// triggers the same cancel, idempotent by the bridge's per-handle dedup.
    static func makeStart(
        stream: AsyncThrowingStream<AgentGeneration, Error>,
        continuation: AsyncThrowingStream<AgentGeneration, Error>.Continuation,
        cachedTokenCount: Int,
        diagnostics: HTTPServerGenerationStart.Diagnostics = .unavailable,
        cancelBridge: LateBoundCancel,
        task: Task<Void, Never>
    ) -> HTTPServerGenerationStart {
        let start = HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: cachedTokenCount,
            cancel: {
                cancelBridge()
                task.cancel()
            },
            waitForCompletion: {
                _ = await task.result
            },
            diagnostics: diagnostics
        )
        continuation.onTermination = { _ in start.cancel() }
        return start
    }
}
