import Foundation

/// Wraps ``ThinkingRepetitionDetector`` with one-intervention-per-request
/// semantics. Feed each ``ToolCallParser/Event`` through ``observe(parserEvent:)``
/// and act on the returned outcome:
/// - `.forward` → yield the event downstream as usual.
/// - `.intervene(safePrefix:reason:)` → kick off the path's intervention flow
///   (emit `.thinkTruncate` + `.thinking(injection)` + `.thinkEnd`, cancel
///   upstream, optionally start a continuation).
///
/// After an intervention is returned, subsequent calls always return `.forward`
/// — we never trigger twice in the same request. Callers that want a fresh
/// state (e.g. in tests) should construct a new instance or call ``reset()``.
nonisolated final class ThinkingSafeguardObserver {

    enum Outcome: Sendable {
        case forward
        case intervene(
            safePrefix: String,
            reason: ThinkingRepetitionDetector.Reason
        )
    }

    private let config: ThinkingRepetitionDetector.Config
    private let limit: Int
    private let detector: ThinkingRepetitionDetector
    private var interventionsIssued: Int = 0

    init(
        config: ThinkingRepetitionDetector.Config,
        limit: Int = 1
    ) {
        self.config = config
        self.limit = limit
        self.detector = ThinkingRepetitionDetector(config: config)
    }

    /// Observe a parser event. Only `.thinking` events are fed to the detector;
    /// everything else (text, tool calls, think-start/end, reclassify) passes
    /// through unchanged.
    func observe(parserEvent: ToolCallParser.Event) -> Outcome {
        guard config.enabled, interventionsIssued < limit else { return .forward }
        guard case .thinking(let chunk) = parserEvent else { return .forward }

        switch detector.ingest(chunk: chunk) {
        case .continue:
            return .forward
        case .intervene(let reason, let safePrefix):
            interventionsIssued += 1
            return .intervene(safePrefix: safePrefix, reason: reason)
        }
    }

    /// Variant for code paths that have `AgentGeneration` events instead of raw
    /// parser events — same semantics, but observes `.thinking(String)` on the
    /// higher-level enum.
    func observe(agentEvent: AgentGeneration) -> Outcome {
        guard config.enabled, interventionsIssued < limit else { return .forward }
        guard case .thinking(let chunk) = agentEvent else { return .forward }

        switch detector.ingest(chunk: chunk) {
        case .continue:
            return .forward
        case .intervene(let reason, let safePrefix):
            interventionsIssued += 1
            return .intervene(safePrefix: safePrefix, reason: reason)
        }
    }

    func reset() {
        detector.reset()
        interventionsIssued = 0
    }

    var hasIntervened: Bool { interventionsIssued > 0 }
}
