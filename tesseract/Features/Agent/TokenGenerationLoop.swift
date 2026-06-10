import Foundation
import MLX
import MLXLMCommon

/// The raw generation event stream produced by the app's own token loop —
/// the app-level replacement for the vendor fork's `Generation` enum
/// (ADR-0006). Upstream's `Generation` lacks `.toolCallBufferDelta`, and the
/// agent's live tool-call argument streaming plus malformed-tool-call
/// recovery both depend on it, so the app owns the event type and the loop
/// that produces it.
nonisolated enum RawGeneration: Sendable {
    /// A generated text chunk (already filtered through the tool-call
    /// processor — never contains in-flight tool-call bytes).
    case chunk(String)

    /// A tool call parsed from the generated output.
    case toolCall(ToolCall)

    /// An append-only delta of text buffered inside an in-flight
    /// `<tool_call>…</tool_call>` block (before the close tag is emitted).
    /// The authoritative `.toolCall` event still fires once at close with the
    /// parsed payload — consumers that only care about the final parse can
    /// keep ignoring this case. Consumers that want to render arguments live
    /// (e.g. a UI activity log) can concatenate these deltas to show the
    /// tool-call body filling in character-by-character.
    case toolCallBufferDelta(String)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)
}

/// The app-owned token→events generation loop: iterates a `TokenIterator`
/// (or any `TokenIteratorProtocol`), detokenizes, routes text through the
/// upstream `ToolCallProcessor`, and emits ``RawGeneration`` events.
///
/// Port of the fork's `generateLoopTask` + `TextToolTokenLoopHandler`
/// (ADR-0006): the package's generation event stream is no longer used, so
/// future stream features land here instead of in a vendored patch.
nonisolated enum TokenGenerationLoop {

    /// Start the loop on a detached task and return the event stream plus the
    /// task. Cancelling the task (or the stream) stops generation; await the
    /// task to know when the model and cache are no longer being touched.
    ///
    /// Generic over `TokenIteratorProtocol` so unit tests can drive the loop
    /// with a scripted token source; production passes a `TokenIterator`.
    static func start<TokenSource: TokenIteratorProtocol>(
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: any Tokenizer,
        iterator: consuming TokenSource,
        tools: [ToolSpec]? = nil
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        let (stream, continuation) = AsyncStream<RawGeneration>.makeStream()

        // The loop runs off-actor (plain detached-context Task), exactly like
        // the vendor loop did; every value it touches crosses once via these
        // boxes. The iterator owns live KV caches and the tokenizer is not
        // Sendable — both are used only inside the task.
        let iteratorBox = UnsafeSendableBox(iterator)
        let tokenizerBox = UnsafeSendableBox(tokenizer)
        let toolsBox = UnsafeSendableBox(tools)
        let configurationBox = UnsafeSendableBox(modelConfiguration)

        let task = Task {
            var it = iteratorBox.value
            let tokenizer = tokenizerBox.value
            let configuration = configurationBox.value

            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            let format = configuration.toolCallFormat ?? .json
            let processor = ToolCallProcessor(format: format, tools: toolsBox.value)
            var deltaTracker = ToolCallDeltaTracker(format: format)

            // Complete EOS token set from all sources (mirrors the vendor
            // loop's buildStopTokenIds).
            var stopTokenIds = configuration.eosTokenIds
            if let tokenizerEOS = tokenizer.eosTokenId {
                stopTokenIds.insert(tokenizerEOS)
            }
            for token in configuration.extraEOSTokens {
                if let id = tokenizer.convertTokenToId(token) {
                    stopTokenIds.insert(id)
                }
            }

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            // Forward one detokenized chunk through the processor and emit
            // the resulting events. Returns false when the stream consumer
            // terminated.
            func emitChunkEvents(_ rawChunk: String) -> Bool {
                if let text = processor.processChunk(rawChunk) {
                    if case .terminated = continuation.yield(.chunk(text)) {
                        return false
                    }
                }
                if let delta = deltaTracker.observe(rawChunk) {
                    if case .terminated = continuation.yield(.toolCallBufferDelta(delta)) {
                        return false
                    }
                }
                while !processor.toolCalls.isEmpty {
                    let call = processor.toolCalls.removeFirst()
                    deltaTracker.didEmitToolCall()
                    if case .terminated = continuation.yield(.toolCall(call)) {
                        return false
                    }
                }
                return true
            }

            while let token = it.next() {
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }

                if promptTime == 0 {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - start
                    start = now
                }

                if token == tokenizer.unknownTokenId || stopTokenIds.contains(token) {
                    stopReason = .stop
                    break
                }

                tokenCount += 1
                detokenizer.append(token: token)
                if let chunk = detokenizer.next() {
                    if !emitChunkEvents(chunk) {
                        stopReason = .cancelled
                        break
                    }
                }
            }

            if stopReason == nil {
                if Task.isCancelled {
                    stopReason = .cancelled
                } else if let maxTokens = it.maxTokens, it.tokenCount >= maxTokens {
                    stopReason = .length
                } else {
                    stopReason = .cancelled
                }
            }

            // End-of-stream recovery: parse any buffered content as tool
            // call(s). A residual from an unclosed *tagged* block is
            // deliberately not re-emitted as text — the delta stream above
            // already carried it, and the consumer owns malformed-tool-call
            // surfacing. A residual from upstream's bare-JSON fallback is
            // ordinary trailing text the tracker never saw, so it is
            // preserved as a regular chunk.
            let residual = processor.processEOS(returnBufferedText: true)
            for call in processor.toolCalls {
                if case .terminated = continuation.yield(.toolCall(call)) {
                    break
                }
            }
            if let residual, !deltaTracker.isMidToolCall {
                _ = continuation.yield(.chunk(residual))
            }

            let now = Date.timeIntervalSinceReferenceDate
            let generateTime = now - start

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: promptTime + it.promptPrefillTime,
                generationTime: generateTime,
                stopReason: stopReason ?? .cancelled
            )
            _ = continuation.yield(.info(info))

            // Synchronize with the MLX stream to ensure pending GPU work is
            // complete before the task reports finished — callers rely on
            // "task done ⇒ model/cache untouched from here on".
            MLX.Stream().synchronize()

            continuation.finish()
        }

        // When the consumer cancels the stream, cancel the underlying task.
        continuation.onTermination = { termination in
            if case .cancelled = termination {
                task.cancel()
            }
        }

        return (stream, task)
    }
}

/// Re-derives the fork's `toolCallBufferDelta` stream from the raw chunk
/// sequence: a small mirror of the upstream `ToolCallProcessor`'s *tagged*
/// collection states, fed the same chunks. UI-only — the processor stays
/// authoritative for parses; this only decides which already-seen bytes to
/// surface live. Bare-JSON fallback collection intentionally produces no
/// deltas (the fork had no such state either).
///
/// Delta timing mirrors the fork: nothing is emitted while a start tag is
/// only partially matched; once the full start tag is confirmed, the first
/// delta carries the whole buffer (open tag included); the chunk that
/// completes the close tag emits no delta for its remaining bytes (the
/// authoritative `.toolCall` covers it).
nonisolated struct ToolCallDeltaTracker {
    private enum State {
        case normal
        case potential
        case collecting
    }

    private let startTag: String?
    private let endTag: String?
    private var state: State = .normal
    private var buffer = ""
    private var emittedChars = 0

    /// True when end-of-stream would land inside an unclosed (or only
    /// partially opened) tool-call block — the case where the EOS residual
    /// belongs to the delta stream rather than to regular text.
    var isMidToolCall: Bool { state != .normal }

    init(format: ToolCallFormat) {
        let parser = format.createParser()
        self.startTag = parser.startTag
        self.endTag = parser.endTag
    }

    /// The processor parsed a complete tool call (e.g. via `processEOS`
    /// recovery) — make sure the tracker doesn't keep treating the stream
    /// position as mid-call.
    mutating func didEmitToolCall() {
        if state == .normal {
            buffer = ""
            emittedChars = 0
        }
    }

    /// Feed the same raw chunk that was fed to the `ToolCallProcessor`.
    /// Returns newly-buffered in-flight tool-call text to surface as a
    /// `.toolCallBufferDelta`, or `nil` when the buffer did not grow.
    mutating func observe(_ chunk: String) -> String? {
        guard let startTag, let endTag, let startChar = startTag.first else {
            return nil
        }

        var delta = ""
        var input = Substring(chunk)

        while true {
            switch state {
            case .normal:
                guard let idx = input.firstIndex(of: startChar) else {
                    input = Substring("")
                    break
                }
                buffer = String(input[idx...])
                emittedChars = 0
                input = Substring("")
                state = .potential
                continue

            case .potential:
                buffer += input
                input = Substring("")
                if buffer.hasPrefix(startTag) {
                    state = .collecting
                    continue
                }
                if startTag.hasPrefix(buffer) {
                    // Still ambiguous — wait for more text.
                    break
                }
                // False positive: the processor flushes the buffer as plain
                // text without re-scanning it; mirror that.
                buffer = ""
                emittedChars = 0
                state = .normal

            case .collecting:
                buffer += input
                input = Substring("")
                if let closeRange = buffer.range(of: endTag) {
                    // Close tag arrived: the authoritative `.toolCall` covers
                    // the block; no delta for the bytes since the last one.
                    // Re-scan any trailing text for another tool call.
                    let trailing = Substring(buffer[closeRange.upperBound...])
                    buffer = ""
                    emittedChars = 0
                    state = .normal
                    if !trailing.isEmpty {
                        input = trailing
                        continue
                    }
                } else if buffer.count > emittedChars {
                    delta += String(buffer.dropFirst(emittedChars))
                    emittedChars = buffer.count
                }
            }
            break
        }

        return delta.isEmpty ? nil : delta
    }
}
