import Foundation
import MLX
import MLXLMCommon

/// The raw generation event stream produced by the app's token-event mapping —
/// the app-level replacement for the vendor fork's `Generation` enum
/// (ADR-0006). Upstream's `Generation` lacks `.toolCallBufferDelta`, and the
/// agent's live tool-call argument streaming plus malformed-tool-call
/// recovery both depend on it, so the app owns the event type and the
/// mapping that produces it.
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

/// The app's generation entry point: drives upstream's public raw-token loop
/// (`generateTokenTask`) and maps its `TokenGeneration` stream into
/// ``RawGeneration`` events — detokenize, route text through the upstream
/// `ToolCallProcessor`, and surface in-flight tool-call bytes as
/// `.toolCallBufferDelta`s.
///
/// The upstream task owns the iterator: stop-token handling, max-tokens,
/// prompt/generation timing, cancellation, and the final MLX stream
/// synchronize all live there (ADR-0006 follow-up — the app no longer
/// carries a copy of that loop). Only the event mapping is app-owned.
///
/// Exception (PRD #72): the cache-aware Server Completion path decodes
/// through the app-owned ``StateThreadedTokenIterator``, and upstream's raw
/// entry point only accepts its concrete `TokenIterator` — so that one
/// overload of `start` drives an app-side replica of the upstream raw loop
/// (same stop-token set, timing split, and stream synchronize).
nonisolated enum TokenGenerationLoop {

    /// Start generation and return the event stream plus the mapping task.
    /// Cancelling the task (or the stream) stops generation; await the task
    /// to know when the model and cache are no longer being touched.
    static func start(
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: any Tokenizer,
        iterator: consuming TokenIterator,
        tools: [ToolSpec]? = nil
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        let (tokens, generationTask) = generateTokenTask(
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfiguration,
            tokenizer: tokenizer,
            iterator: iterator
        )
        return events(
            from: tokens,
            generationTask: generationTask,
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfiguration,
            tokenizer: tokenizer,
            tools: tools
        )
    }

    /// Cache-aware overload: same event mapping, driven by the state-threaded
    /// decode loop instead of upstream's concrete-`TokenIterator` task.
    static func start(
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: any Tokenizer,
        iterator: consuming StateThreadedTokenIterator,
        tools: [ToolSpec]? = nil
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        let (tokens, generationTask) = rawTokenTask(
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfiguration,
            tokenizer: tokenizer,
            iterator: iterator
        )
        return events(
            from: tokens,
            generationTask: generationTask,
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfiguration,
            tokenizer: tokenizer,
            tools: tools
        )
    }

    /// Batch Engine pool-lane overload (PRD #173, ADR-0022): identical
    /// event mapping and raw-loop semantics, but every `iterator.next()`
    /// runs inside a granted decode step — the engine interleaves lanes at
    /// token granularity and all Metal work stays grant-serialized.
    static func start(
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: any Tokenizer,
        iterator: consuming StateThreadedTokenIterator,
        engine: BatchEngine,
        laneID: UUID,
        tools: [ToolSpec]? = nil
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        let box = UnsafeMutableSendableBox(iterator)
        let (tokens, generationTask) = steppedRawTokenTask(
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfiguration,
            tokenizer: tokenizer,
            engine: engine,
            laneID: laneID,
            nextToken: { box.value.next() },
            iteratorStats: {
                (box.value.tokenCount, box.value.maxTokens, box.value.promptPrefillTime)
            }
        )
        return events(
            from: tokens,
            generationTask: generationTask,
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfiguration,
            tokenizer: tokenizer,
            tools: tools
        )
    }

    /// The stepped raw loop's core, over closures so the loop semantics are
    /// unit-testable without a model (`TokenGenerationLoopSteppedTests`).
    /// Mirrors `rawTokenTask` exactly — stop-token set, timing split,
    /// authoritative `.info` — plus a final grant-wrapped synchronize so the
    /// "task done ⇒ model/cache untouched" contract holds while sibling
    /// lanes may still be decoding.
    static func steppedRawTokenTask(
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: any Tokenizer,
        engine: BatchEngine,
        laneID: UUID,
        nextToken: @escaping @Sendable () -> Int?,
        iteratorStats:
            @escaping @Sendable () -> (
                tokenCount: Int, maxTokens: Int?, promptPrefillTime: TimeInterval
            )
    ) -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

        var stopTokenIds = modelConfiguration.eosTokenIds
        if let tokenizerEOS = tokenizer.eosTokenId {
            stopTokenIds.insert(tokenizerEOS)
        }
        for token in modelConfiguration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                stopTokenIds.insert(id)
            }
        }
        let unknownTokenId = tokenizer.unknownTokenId
        let resolvedStopTokenIds = stopTokenIds

        let task = Task {
            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            decode: while true {
                let token: Int?
                do {
                    token = try await engine.step(lane: laneID, kind: .decode) { _ in
                        nextToken()
                    }
                } catch {
                    // Lane torn down or the step cancelled while pending.
                    stopReason = .cancelled
                    break decode
                }
                guard let token else { break }
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }
                if promptTime == 0 {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - start
                    start = now
                }
                if token == unknownTokenId || resolvedStopTokenIds.contains(token) {
                    stopReason = .stop
                    break
                }
                tokenCount += 1
                if case .terminated = continuation.yield(.token(token)) {
                    stopReason = .cancelled
                    break
                }
            }

            let stats = iteratorStats()
            let resolvedStopReason: GenerateStopReason =
                stopReason
                ?? {
                    if Task.isCancelled { return .cancelled }
                    if let maxTokens = stats.maxTokens, stats.tokenCount >= maxTokens {
                        return .length
                    }
                    return .cancelled
                }()

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: promptTime + stats.promptPrefillTime,
                generationTime: Date.timeIntervalSinceReferenceDate - start,
                stopReason: resolvedStopReason
            )
            _ = continuation.yield(.info(info))

            // Match upstream's end-of-loop synchronize, inside a grant when
            // the lane is still live — a sibling may be mid-decode. When the
            // lane is already gone (cancel path) fall back to the direct
            // wait-only call, exactly today's behavior.
            do {
                _ = try await engine.step(lane: laneID, kind: .decode) { _ in
                    Stream().synchronize()
                }
            } catch {
                Stream().synchronize()
            }
            continuation.finish()
        }

        continuation.onTermination = { termination in
            if case .cancelled = termination {
                task.cancel()
            }
        }

        return (stream, task)
    }

    /// App-side replica of upstream's raw token loop (`generateLoopTask`'s
    /// `TokenGeneration` shape): per-token cancellation checks, the combined
    /// stop-token set (`eosTokenIds` + tokenizer EOS + `extraEOSTokens`), the
    /// prompt/generation timing split, and the final MLX stream synchronize.
    /// Exists because upstream's public raw entry point is hardcoded to its
    /// concrete `TokenIterator`.
    private static func rawTokenTask(
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: any Tokenizer,
        iterator: consuming StateThreadedTokenIterator
    ) -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()

        var stopTokenIds = modelConfiguration.eosTokenIds
        if let tokenizerEOS = tokenizer.eosTokenId {
            stopTokenIds.insert(tokenizerEOS)
        }
        for token in modelConfiguration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                stopTokenIds.insert(id)
            }
        }
        let unknownTokenId = tokenizer.unknownTokenId

        let iteratorBox = UnsafeSendableBox(iterator)
        let task = Task {
            var iterator = iteratorBox.value

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            while let token = iterator.next() {
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }
                if promptTime == 0 {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - start
                    start = now
                }
                if token == unknownTokenId || stopTokenIds.contains(token) {
                    stopReason = .stop
                    break
                }
                tokenCount += 1
                if case .terminated = continuation.yield(.token(token)) {
                    stopReason = .cancelled
                    break
                }
            }

            let resolvedStopReason: GenerateStopReason =
                stopReason
                ?? {
                    if Task.isCancelled { return .cancelled }
                    if let maxTokens = iterator.maxTokens, iterator.tokenCount >= maxTokens {
                        return .length
                    }
                    return .cancelled
                }()

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: promptTime + iterator.promptPrefillTime,
                generationTime: Date.timeIntervalSinceReferenceDate - start,
                stopReason: resolvedStopReason
            )
            _ = continuation.yield(.info(info))

            // Match upstream: ensure pending MLX work is complete before the
            // caller's "task done ⇒ model/cache untouched" contract kicks in.
            Stream().synchronize()
            continuation.finish()
        }

        continuation.onTermination = { termination in
            if case .cancelled = termination {
                task.cancel()
            }
        }

        return (stream, task)
    }

    /// Map an upstream raw-token stream into ``RawGeneration`` events.
    /// Internal seam: unit tests drive this with a scripted
    /// `AsyncStream<TokenGeneration>` instead of a live model.
    ///
    /// - Parameters:
    ///   - generationTask: the upstream task producing `tokens`; cancelled
    ///     when the consumer stops early, awaited before the returned task
    ///     finishes (it synchronizes the MLX stream, preserving the caller
    ///     contract "task done ⇒ model/cache untouched from here on").
    ///   - promptTokenCount: only used for the synthesized `.info` when
    ///     iteration is cancelled before upstream's authoritative `.info`
    ///     arrives.
    static func events(
        from tokens: AsyncStream<TokenGeneration>,
        generationTask: Task<Void, Never>?,
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: any Tokenizer,
        tools: [ToolSpec]? = nil
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        let (stream, continuation) = AsyncStream<RawGeneration>.makeStream()

        let task = Task {
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            let format = modelConfiguration.toolCallFormat ?? .json
            let processor = ToolCallProcessor(format: format, tools: tools)
            var deltaTracker = ToolCallDeltaTracker(format: format)

            var info: GenerateCompletionInfo?
            var generatedTokens = 0
            var consumerTerminated = false

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
                    if case .terminated = continuation.yield(.toolCall(call)) {
                        return false
                    }
                }
                return true
            }

            tokenLoop: for await event in tokens {
                switch event {
                case .token(let token):
                    generatedTokens += 1
                    detokenizer.append(token: token)
                    if let chunk = detokenizer.next() {
                        if !emitChunkEvents(chunk) {
                            consumerTerminated = true
                            break tokenLoop
                        }
                    }
                case .info(let completion):
                    info = completion
                }
            }

            // The consumer abandoned the stream, or this task was cancelled
            // mid-iteration — stop the upstream generation promptly.
            if consumerTerminated || Task.isCancelled {
                generationTask?.cancel()
            }

            // End-of-stream recovery: parse any buffered content as tool
            // call(s). The residual is re-emitted as a regular chunk unless
            // the delta stream above already carried exactly those bytes (an
            // in-flight *tagged* block): partial start tags and bare-JSON
            // buffers produce no deltas, so suppressing their residual would
            // silently drop trailing text (#67 review).
            let residual = processor.processEOS(returnBufferedText: true)
            for call in processor.toolCalls {
                if case .terminated = continuation.yield(.toolCall(call)) {
                    break
                }
            }
            if let residual, !deltaTracker.deltasCarriedBuffer {
                _ = continuation.yield(.chunk(residual))
            }

            // Upstream's `.info` is authoritative (token counts, prompt
            // prefill time, stop reason). It is only missing when iteration
            // ended before the upstream stream finished — a cancellation.
            _ = continuation.yield(
                .info(
                    info
                        ?? GenerateCompletionInfo(
                            promptTokenCount: promptTokenCount,
                            generationTokenCount: generatedTokens,
                            promptTime: 0,
                            generationTime: 0,
                            stopReason: .cancelled
                        )))

            // The upstream loop synchronizes the MLX stream before finishing.
            await generationTask?.value
            continuation.finish()
        }

        // When the consumer cancels the stream, cancel the mapping task (which
        // cancels the upstream generation task in turn).
        continuation.onTermination = { termination in
            if case .cancelled = termination {
                task.cancel()
            }
        }

        return (stream, task)
    }
}

/// Re-derives the fork's `toolCallBufferDelta` stream from the raw chunk
/// sequence: a mirror of the upstream `ToolCallProcessor`'s collection
/// states, fed the same chunks. UI-only — the processor stays authoritative
/// for parses; this only decides which already-seen bytes to surface live.
///
/// Both collection modes of the `.json`-format processor are mirrored:
///
/// - **Tagged** `<tool_call>…</tool_call>` collection produces deltas.
///   Nothing is emitted while a start tag is only partially matched; once
///   the full start tag is confirmed, the first delta carries the whole
///   pending block (open tag included); the chunk containing the close tag
///   still deltas its body bytes up to the tag (the Argument Transcoder
///   needs the complete body on this channel), while the close tag itself
///   is never a delta (the authoritative `.toolCall` covers it).
/// - **Bare-JSON fallback** (a `{` that looks like a JSON object, preferred
///   by the processor when it precedes any `<`) is tracked for state parity
///   but intentionally produces no deltas (the fork had no such state
///   either). Without this mirror, a `<` inside buffered JSON would be
///   misread as a tag start and processor-buffered bytes would be re-emitted
///   as spurious deltas (#67 review).
///
/// Known divergence, shared with the fork: when a *closed* tagged block
/// fails to parse, the processor flushes it back as plain text after the
/// deltas already carried it — those bytes appear on both channels.
nonisolated struct ToolCallDeltaTracker {
    private enum State {
        case normal
        case potential
        case collecting
        case collectingJSON
    }

    private let startTag: String?
    private let endTag: String?
    private let supportsBareJSONFallback: Bool
    /// Mirror of the processor's safety valve for pathological unmatched
    /// JSON-like buffers (`ToolCallProcessor.maxJSONFallbackBufferLength`).
    private let maxJSONFallbackBufferLength = 32_768

    private var state: State = .normal
    /// `.potential` only: the partially-matched start-tag bytes (bounded by
    /// the tag length while at rest between chunks).
    private var buffer = ""
    /// `.collecting` only: trailing already-emitted characters kept so a
    /// close tag spanning a chunk boundary is still recognized (bounded by
    /// the close-tag length).
    private var tailWindow = ""

    // `.collectingJSON` only: incremental mirror of the processor's
    // `JSONLeadingObjectScanner` — O(1) state instead of a re-scanned buffer.
    private var jsonResolvedValid = false
    /// Bytes buffered until the object-prefix validity resolves (the
    /// processor flushes-and-rescans them when the prefix turns out not to
    /// be a JSON object). Tiny in practice: `{` plus any whitespace run.
    private var jsonPrefix = ""
    private var jsonDepth = 0
    private var jsonInString = false
    private var jsonEscaped = false
    private var jsonLength = 0

    /// True when the bytes the processor is buffering at end-of-stream were
    /// already surfaced through the delta stream — the only case where the
    /// EOS residual must not be re-emitted as a regular chunk. Partial start
    /// tags (`.potential`) and bare-JSON buffers produce no deltas.
    var deltasCarriedBuffer: Bool { state == .collecting }

    /// True when end-of-stream would land inside an unclosed (or only
    /// partially opened, or bare-JSON) tool-call block.
    var isMidToolCall: Bool { state != .normal }

    init(format: ToolCallFormat) {
        let parser = format.createParser()
        self.startTag = parser.startTag
        self.endTag = parser.endTag
        self.supportsBareJSONFallback = format == .json
    }

    // Feed the same raw chunk that was fed to the `ToolCallProcessor`.
    // Returns newly-buffered in-flight tool-call text to surface as a
    // `.toolCallBufferDelta`, or `nil` when nothing new should surface.
    // O(chunk) — no state re-scans the accumulated block.
    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable:next cyclomatic_complexity function_body_length
    mutating func observe(_ chunk: String) -> String? {
        guard let startTag, let endTag, let startChar = startTag.first else {
            return nil
        }

        var delta = ""
        var input = Substring(chunk)

        processing: while true {
            switch state {
            case .normal:
                guard !input.isEmpty else { break processing }
                let tagIndex = input.firstIndex(of: startChar)
                let jsonIndex = supportsBareJSONFallback ? input.firstIndex(of: "{") : nil
                switch (tagIndex, jsonIndex) {
                case (nil, nil):
                    break processing

                case (.some(let tag), nil):
                    buffer = ""
                    state = .potential
                    input = input[tag...]
                    continue processing

                case (nil, .some(let json)):
                    enterJSON()
                    input = input[input.index(after: json)...]
                    continue processing

                case (.some(let tag), .some(let json)):
                    // Mirror `taggedStartMode`: an earlier `{` wins only when
                    // it can begin a JSON object.
                    if json >= tag || Self.jsonPrefixLooksInvalid(input[json...]) {
                        buffer = ""
                        state = .potential
                        input = input[tag...]
                    } else {
                        enterJSON()
                        input = input[input.index(after: json)...]
                    }
                    continue processing
                }

            case .potential:
                buffer += input
                input = Substring("")
                if buffer.hasPrefix(startTag) {
                    // Start tag confirmed: the whole pending block (open tag
                    // included) moves to collecting, which decides whether it
                    // is delta'd or already closed within this chunk.
                    let pending = buffer
                    buffer = ""
                    tailWindow = ""
                    state = .collecting
                    input = Substring(pending)
                    continue processing
                }
                if startTag.hasPrefix(buffer) {
                    // Still ambiguous — wait for more text.
                    break processing
                }
                // False positive: the processor flushes the buffer as plain
                // text without re-scanning it; mirror that.
                buffer = ""
                state = .normal
                break processing

            case .collecting:
                guard !input.isEmpty else { break processing }
                var pending = tailWindow
                pending += input
                if let closeRange = pending.range(of: endTag) {
                    // Close tag arrived: forward the not-yet-emitted bytes
                    // that precede it (the leading `tailWindow` characters
                    // went out with an earlier chunk) so the delta stream
                    // carries the complete block body — the server-side
                    // Argument Transcoder streams arguments from these
                    // deltas, so a body tail withheld here would be lost
                    // from the wire. The close tag itself is never a delta
                    // (the authoritative `.toolCall` covers the block).
                    // Re-scan any trailing text for another tool call.
                    let preClose = pending[..<closeRange.lowerBound]
                    if preClose.count > tailWindow.count {
                        delta += preClose.dropFirst(tailWindow.count)
                    }
                    let trailing = pending[closeRange.upperBound...]
                    tailWindow = ""
                    state = .normal
                    input = trailing
                    continue processing
                }
                delta += input
                tailWindow = String(pending.suffix(endTag.count - 1))
                break processing

            case .collectingJSON:
                guard !input.isEmpty else { break processing }
                // Mirror the processor's safety valve: a pathological
                // unmatched JSON-like buffer is flushed back as plain text.
                if jsonLength + input.count > maxJSONFallbackBufferLength {
                    resetJSON()
                    state = .normal
                    break processing
                }
                jsonLength += input.count

                var index = input.startIndex
                while index < input.endIndex {
                    let character = input[index]

                    if !jsonResolvedValid {
                        if character.isWhitespace {
                            jsonPrefix.append(character)
                            index = input.index(after: index)
                            continue
                        }
                        guard character == "\"" || character == "}" else {
                            // Not a JSON object after all: the processor
                            // flushes the whole buffer as text, re-parsing it
                            // only when it contains a full start tag.
                            let flushed = jsonPrefix + String(input[index...])
                            resetJSON()
                            state = .normal
                            if flushed.contains(startTag) {
                                input = Substring(flushed)
                                continue processing
                            }
                            break processing
                        }
                        jsonResolvedValid = true
                        jsonPrefix = ""
                        // The resolving character is structural — fall
                        // through to the depth tracking below.
                    }

                    if jsonInString {
                        if jsonEscaped {
                            jsonEscaped = false
                        } else if character == "\\" {
                            jsonEscaped = true
                        } else if character == "\"" {
                            jsonInString = false
                        }
                    } else {
                        switch character {
                        case "\"":
                            jsonInString = true
                        case "{":
                            jsonDepth += 1
                        case "}":
                            jsonDepth -= 1
                            if jsonDepth == 0 {
                                // Complete object: whether the processor
                                // parses it as a tool call or flushes it as
                                // text, its collection state resets and the
                                // trailing text is re-scanned.
                                let trailing = input[input.index(after: index)...]
                                resetJSON()
                                state = .normal
                                input = trailing
                                continue processing
                            }
                        default:
                            break
                        }
                    }

                    index = input.index(after: index)
                }
                break processing
            }
        }

        return delta.isEmpty ? nil : delta
    }

    // MARK: - Bare-JSON mirror helpers

    private mutating func enterJSON() {
        state = .collectingJSON
        jsonResolvedValid = false
        jsonPrefix = "{"
        jsonDepth = 1
        jsonInString = false
        jsonEscaped = false
        jsonLength = 1
    }

    private mutating func resetJSON() {
        jsonResolvedValid = false
        jsonPrefix = ""
        jsonDepth = 0
        jsonInString = false
        jsonEscaped = false
        jsonLength = 0
    }

    /// Mirror of `JSONLeadingObjectScanner.evaluatePrefix` on a chunk slice
    /// starting at a candidate `{`: invalid when the first non-whitespace
    /// character after the `{` can't begin an object body. "Needs more" (the
    /// chunk ends first) is not invalid.
    private static func jsonPrefixLooksInvalid(_ text: Substring) -> Bool {
        var index = text.index(after: text.startIndex)
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
        guard index < text.endIndex else { return false }
        let first = text[index]
        return first != "\"" && first != "}"
    }
}
