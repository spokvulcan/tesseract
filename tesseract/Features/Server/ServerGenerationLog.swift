import Foundation
import MLXLMCommon
import Observation

/// Live, in-app observability log of generations flowing through the local
/// inference engine. Surfaces per-request phase, metrics, and a token-by-token
/// stream into SwiftUI via `@Observable`.
@MainActor
@Observable
final class ServerGenerationLog {

    // MARK: - Config

    static let maxTraces: Int = 20
    /// Byte budgets for the head and tail of any single accumulated text
    /// span. Using UTF-8 byte budgets keeps the `cappedAppend` fast-path
    /// check and the slicing consistent.
    static let textHeadBytes: Int = 8 * 1024
    static let textTailBytes: Int = 24 * 1024
    /// Minimum interval between streaming scroll bumps during decode. Rate-
    /// limits the scroll driver without affecting the append path — spans
    /// update on every flushed batch.
    static let scrollThrottleMs: Double = 100

    /// Interval at which buffered text/thinking chunks are drained into the
    /// observed `trace.spans`. Caps SwiftUI's observation rate to ≤30 Hz
    /// during streaming, well under the LazyVStack layout budget. Without
    /// this coalescing, raw 80+ tok/s chunk arrival triggered a re-entrant
    /// `LazyLayoutViewCache.invalidateSize` → `_postWindowNeedsUpdate-
    /// Constraints` AppKit trap under bench load.
    static let coalesceIntervalMs: Double = 33

    // MARK: - Public state

    private(set) var traces: [RequestTrace] = []
    private(set) var streamingVersion: Int = 0
    var selectedTraceID: UUID? = nil

    // MARK: - Private

    private var lastScrollBumpAt: Date?
    private var sequence: Int = 0

    /// Per-trace coalescing buffer for `.text` / `.thinking` chunks. Kept
    /// off `@Observable` storage so a chunk append doesn't notify SwiftUI —
    /// the scheduled flush batches them into a single observed mutation.
    @ObservationIgnored
    private var pendingAppends: [UUID: PendingAppend] = [:]

    private struct PendingAppend {
        var pendingText: String = ""
        var pendingThinking: String = ""
        var flushScheduled: Bool = false
    }

    // MARK: - Ingress

    @discardableResult
    func startRequest(
        completionID: String,
        model: String,
        stream: Bool,
        sessionAffinity: String?,
        startedAt: Date = Date()
    ) -> TraceHandle {
        sequence += 1
        let trace = RequestTrace(
            id: UUID(),
            sequence: sequence,
            completionID: completionID,
            model: model,
            stream: stream,
            sessionAffinity: sessionAffinity,
            startedAt: startedAt
        )
        traces.append(trace)
        if traces.count > Self.maxTraces {
            let overflow = traces.count - Self.maxTraces
            traces.removeFirst(overflow)
        }
        selectedTraceID = trace.id
        bumpScroll()
        return TraceHandle(id: trace.id)
    }

    func markLeaseAcquired(handle: TraceHandle, at: Date = Date()) {
        update(handle) { trace in
            trace.leaseAcquiredAt = at
            if trace.phase == .queued {
                trace.phase = .lookingUp
            }
        }
    }

    func markCacheLookup(
        handle: TraceHandle,
        reason: String,
        cachedTokens: Int,
        sharedPrefixLength: Int,
        promptTokens: Int,
        lookupMs: Double,
        restoreMs: Double,
        prefillMs: Double
    ) {
        update(handle) { trace in
            trace.cacheReason = reason
            trace.cachedTokens = cachedTokens
            trace.sharedPrefixLength = sharedPrefixLength
            trace.promptTokens = promptTokens
            trace.lookupMs = lookupMs
            trace.restoreMs = restoreMs
            trace.prefillMs = prefillMs
            if trace.phase == .queued || trace.phase == .lookingUp {
                trace.phase = .prefilling
            }
        }
    }

    func ingest(handle: TraceHandle, event: AgentGeneration) {
        switch event {
        case .text(let chunk):
            enqueueAppend(handle: handle, chunk: chunk, isThinking: false)
        case .thinking(let chunk):
            enqueueAppend(handle: handle, chunk: chunk, isThinking: true)
        case .thinkStart, .thinkEnd, .thinkReclassify:
            return  // Not a span mutation — skip flush AND scroll bump.
        case .toolCall(let call):
            // Phase-transition / metadata events: drain any buffered chunks
            // first so they appear BEFORE the transition in the log.
            flushPending(handle: handle)
            update(handle) { trace in
                trace.markFirstTokenIfNeeded()
                trace.appendToolCall(
                    name: call.function.name,
                    arguments: call.function.arguments
                )
            }
            throttledScrollBump()
        case .malformedToolCall(let raw):
            flushPending(handle: handle)
            update(handle) { $0.appendMalformedToolCall(raw) }
            throttledScrollBump()
        case .info(let info):
            flushPending(handle: handle)
            update(handle) { trace in
                trace.promptTokens = info.promptTokenCount
                trace.generationTokens = info.generationTokenCount
                trace.tokensPerSecond = info.tokensPerSecond
            }
            throttledScrollBump()
        case .thinkTruncate(let safePrefix):
            flushPending(handle: handle)
            update(handle) { $0.replaceThinkingWithSafePrefix(safePrefix) }
            // Suppress the scroll bump — the span just shrank from ~16K chars
            // to safePrefix-size in a single tick, so an auto-scroll-to-bottom
            // is meaningless and risks racing SwiftUI's LazyVStack prefetch
            // pass against stale ScrollView geometry.
        }
    }

    func complete(handle: TraceHandle, finishReason: String, at: Date = Date()) {
        flushPending(handle: handle)
        update(handle) { trace in
            trace.phase = .completed
            trace.finishReason = finishReason
            trace.completedAt = at
        }
        bumpScroll()
    }

    func fail(handle: TraceHandle, error: String, at: Date = Date()) {
        flushPending(handle: handle)
        update(handle) { trace in
            trace.phase = .failed
            trace.errorMessage = error
            trace.completedAt = at
        }
        bumpScroll()
    }

    func cancel(handle: TraceHandle, at: Date = Date()) {
        flushPending(handle: handle)
        update(handle) { trace in
            trace.phase = .cancelled
            trace.completedAt = at
        }
        bumpScroll()
    }

    func clear() {
        pendingAppends.removeAll()
        traces.removeAll()
        selectedTraceID = nil
        bumpScroll()
    }

    // MARK: - Helpers

    private func update(_ handle: TraceHandle, _ mutate: (inout RequestTrace) -> Void) {
        guard let idx = traces.firstIndex(where: { $0.id == handle.id }) else { return }
        mutate(&traces[idx])
    }

    /// Accumulate a text/thinking chunk off the observed path. On the first
    /// chunk of a fresh coalesce window, schedule a flush `coalesceIntervalMs`
    /// in the future; subsequent chunks within the window piggy-back on the
    /// same scheduled flush. If a chunk of the OPPOSITE kind arrives while a
    /// same-kind buffer is live, the existing buffer is drained synchronously
    /// first so on-wire order (thinking vs. text) is preserved even when a
    /// type-switch lands in the middle of a window.
    private func enqueueAppend(handle: TraceHandle, chunk: String, isThinking: Bool) {
        guard !chunk.isEmpty else { return }

        if let pending = pendingAppends[handle.id] {
            let otherBuffered = isThinking
                ? !pending.pendingText.isEmpty
                : !pending.pendingThinking.isEmpty
            if otherBuffered {
                flushPending(handle: handle)
            }
        }

        var pending = pendingAppends[handle.id] ?? PendingAppend()
        if isThinking {
            pending.pendingThinking += chunk
        } else {
            pending.pendingText += chunk
        }
        let needsSchedule = !pending.flushScheduled
        pending.flushScheduled = true
        pendingAppends[handle.id] = pending

        if needsSchedule {
            let delay = UInt64(Self.coalesceIntervalMs * 1_000_000)
            Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: delay)
                self?.flushPending(handle: handle)
            }
        }
    }

    /// Drain any buffered text/thinking into a single observed mutation on
    /// the trace. Safe to call multiple times — no-op when nothing is
    /// pending. Called on phase transitions, terminal events, and from the
    /// scheduled flush task. Visible to tests so they can observe the
    /// buffered state synchronously.
    func flushPending(handle: TraceHandle) {
        guard let pending = pendingAppends.removeValue(forKey: handle.id) else {
            return
        }
        let thinking = pending.pendingThinking
        let text = pending.pendingText
        guard !thinking.isEmpty || !text.isEmpty else { return }

        update(handle) { trace in
            if !thinking.isEmpty {
                trace.markFirstTokenIfNeeded()
                trace.appendThinking(thinking)
            }
            if !text.isEmpty {
                trace.markFirstTokenIfNeeded()
                trace.appendText(text)
            }
        }
        throttledScrollBump()
    }

    private func throttledScrollBump() {
        let now = Date()
        if let last = lastScrollBumpAt,
           now.timeIntervalSince(last) * 1000 < Self.scrollThrottleMs {
            return
        }
        lastScrollBumpAt = now
        streamingVersion &+= 1
    }

    private func bumpScroll() {
        lastScrollBumpAt = Date()
        streamingVersion &+= 1
    }
}

/// Opaque cross-isolation reference to a specific request's trace.
nonisolated struct TraceHandle: Sendable, Hashable {
    let id: UUID
}

struct RequestTrace: Identifiable, Equatable {

    enum Phase: Equatable, Sendable {
        case queued
        case lookingUp
        case prefilling
        case decoding
        case completed
        case failed
        case cancelled

        var isTerminal: Bool {
            switch self {
            case .completed, .failed, .cancelled: return true
            default: return false
            }
        }
    }

    enum Span: Equatable, Hashable, Sendable, Identifiable {
        case text(id: UUID, content: String)
        case thinking(id: UUID, content: String)
        case toolCall(id: UUID, name: String, argumentsJSON: String)
        case malformedToolCall(id: UUID, raw: String)

        var id: UUID {
            switch self {
            case .text(let id, _),
                 .thinking(let id, _),
                 .toolCall(let id, _, _),
                 .malformedToolCall(let id, _):
                return id
            }
        }
    }

    let id: UUID
    let sequence: Int
    let completionID: String
    let model: String
    let stream: Bool
    let sessionAffinity: String?
    let startedAt: Date

    var phase: Phase = .queued
    var spans: [Span] = []

    var leaseAcquiredAt: Date?
    var firstTokenAt: Date?
    var completedAt: Date?

    var cacheReason: String?
    var cachedTokens: Int = 0
    var sharedPrefixLength: Int = 0
    var lookupMs: Double?
    var restoreMs: Double?
    var prefillMs: Double?
    var promptTokens: Int?
    var generationTokens: Int = 0
    var tokensPerSecond: Double = 0

    var finishReason: String?
    var errorMessage: String?

    /// Time-to-first-token, measured from lease acquisition (which is
    /// the point we hold GPU rights). Nil until the first token arrives.
    var ttftMs: Double? {
        guard let firstTokenAt, let leaseAcquiredAt else { return nil }
        return firstTokenAt.timeIntervalSince(leaseAcquiredAt) * 1000
    }

    var isActive: Bool { !phase.isTerminal }

    var elapsedFromStart: TimeInterval {
        (completedAt ?? Date()).timeIntervalSince(startedAt)
    }

    mutating func markFirstTokenIfNeeded() {
        if firstTokenAt == nil {
            firstTokenAt = Date()
            phase = .decoding
        }
    }

    mutating func appendText(_ chunk: String) {
        if case .text(let id, let existing) = spans.last {
            spans[spans.count - 1] = .text(
                id: id,
                content: Self.cappedAppend(existing, chunk)
            )
        } else {
            spans.append(.text(id: UUID(), content: Self.cappedAppend("", chunk)))
        }
    }

    mutating func appendThinking(_ chunk: String) {
        if case .thinking(let id, let existing) = spans.last {
            spans[spans.count - 1] = .thinking(
                id: id,
                content: Self.cappedAppend(existing, chunk)
            )
        } else {
            spans.append(.thinking(id: UUID(), content: Self.cappedAppend("", chunk)))
        }
    }

    /// Replace the most recent thinking span's content with `safePrefix`.
    /// Called when the thinking-loop safeguard fires so the activity log
    /// surface reflects the cleaned-up reasoning rather than the degen garbage.
    mutating func replaceThinkingWithSafePrefix(_ safePrefix: String) {
        if case .thinking(let id, _) = spans.last {
            spans[spans.count - 1] = .thinking(
                id: id,
                content: Self.cappedAppend("", safePrefix)
            )
        } else {
            spans.append(.thinking(id: UUID(), content: Self.cappedAppend("", safePrefix)))
        }
    }

    mutating func appendToolCall(name: String, arguments: [String: JSONValue]) {
        spans.append(.toolCall(
            id: UUID(),
            name: name,
            argumentsJSON: ToolArgumentNormalizer.encode(arguments)
        ))
    }

    mutating func appendMalformedToolCall(_ raw: String) {
        spans.append(.malformedToolCall(id: UUID(), raw: raw))
    }

    /// Concatenate all `.text` spans into a single string for "Copy full output".
    var concatenatedText: String {
        spans.compactMap { span -> String? in
            if case .text(_, let content) = span { return content }
            return nil
        }.joined()
    }

    static func cappedAppend(_ current: String, _ chunk: String) -> String {
        let headBytes = ServerGenerationLog.textHeadBytes
        let tailBytes = ServerGenerationLog.textTailBytes
        let budget = headBytes + tailBytes

        // Fast path: under budget. Covers the common case on every token.
        if current.utf8.count + chunk.utf8.count <= budget {
            return current + chunk
        }

        let merged = current + chunk
        // Slice on grapheme-aligned prefix/suffix — `String.prefix(_:)` /
        // `suffix(_:)` work on characters, which never splits a codepoint.
        // We overshoot the character count from the byte budget since typical
        // generation is ASCII-heavy (≈1 byte per char).
        let head = merged.prefix(headBytes)
        let tail = merged.suffix(tailBytes)
        let elidedBytes = merged.utf8.count - head.utf8.count - tail.utf8.count
        return String(head)
            + "\n--- elided \(elidedBytes) bytes ---\n"
            + String(tail)
    }
}
