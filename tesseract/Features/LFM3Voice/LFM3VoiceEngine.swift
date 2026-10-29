//
//  LFM3VoiceEngine.swift
//  tesseract
//
//  **LFM3 Voice**'s model: LFM3-Audio hears the owner, writes the answer and
//  speaks it in its own voice, text and speech interleaved as the model card's
//  spoken chat with tools writes them. It is its own load beside the agent's,
//  so a voice turn never waits on the agent's container, and it holds the
//  codec that turns the speech frames into sound.
//

import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// What an answer sends out while it is written.
nonisolated enum LFM3VoiceOutput: Sendable {
    /// Visible text: tool calls and the speech markers are left out.
    case text(String)
    /// Mono samples of the answer's voice, at ``LFM3VoiceEngine/sampleRate``.
    case speech([Float])
}

/// How a written answer ended: the tools it calls, and whether the model
/// spoke it.
nonisolated struct LFM3VoiceAnswer: Sendable {
    let toolCalls: [ToolCall]
    let spoke: Bool
}

/// A prompt rendered on the main actor (the chat's messages are there) and
/// read once, inside the container.
nonisolated struct LFM3VoicePrompt: @unchecked Sendable {
    let input: UserInput
}

nonisolated enum LFM3VoiceError: LocalizedError {
    case notLoaded
    case notLFM3Audio

    var errorDescription: String? {
        switch self {
        case .notLoaded: "LFM3-Audio isn't loaded."
        case .notLFM3Audio: "The selected model isn't an LFM3-Audio checkpoint."
        }
    }
}

/// An answer that may call a tool opens with `<tool_call>` or
/// `<|mixed_start|>` and nothing else, the call weighed down by
/// ``LFM3VoiceEngine/toolCallBias``.
nonisolated struct LFM3VoiceOpening: LogitProcessor {
    let toolCall: Int
    let mixedStart: Int
    private var opened = false

    init(toolCall: Int, mixedStart: Int) {
        self.toolCall = toolCall
        self.mixedStart = mixedStart
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard !opened else { return logits }
        var offsets = [Float](repeating: -.infinity, count: logits.dim(-1))
        guard offsets.indices.contains(toolCall), offsets.indices.contains(mixedStart) else {
            return logits
        }
        offsets[toolCall] = LFM3VoiceEngine.toolCallBias
        offsets[mixedStart] = 0
        return logits + MLXArray(offsets).asType(logits.dtype)
    }

    mutating func didSample(token: MLXArray) {
        opened = true
    }
}

actor LFM3VoiceEngine {

    /// The codec's output rate.
    nonisolated static let sampleRate = 24_000

    /// The card's prompt for a spoken chat with tools is "Respond with
    /// function calls, and interleaved text and audio." Under it, and under
    /// every rewording tried, `<tool_call>` is the first token's top choice
    /// even for "hi" (2026-10-28, 40 requests with the app's agenda, profile
    /// and web tools). This wording spreads the two kinds furthest apart, and
    /// ``toolCallBias`` splits them.
    nonisolated static let systemPrompt = """
        Respond with interleaved text and audio. Call a function only when the request \
        needs one of these tools; otherwise answer directly.

        List of tools:
        """

    /// Added to `<tool_call>`'s logit at an answer's first token, where it
    /// competes with `<|mixed_start|>` alone. Under ``systemPrompt`` the gap
    /// was −1.0 to +9.0 for 18 of 20 requests that need no tool and +9.6 to
    /// +16.0 for all 20 that need one; "I'm feeling a bit tired today" (+12.9)
    /// and "Who wrote Romeo and Juliet?" (+15.2) still call one.
    nonisolated static let toolCallBias: Float = -9.3

    /// The prompt that reads text aloud, for an answer written without its
    /// speech. The card's voices are "female" and "male".
    nonisolated static let readPrompt = "Perform TTS in American English using a female voice."

    /// The most text tokens and speech frames in one answer: about 80 s of
    /// speech.
    nonisolated static let maxSlots = 1_200

    /// The model and its codec. The codec is an MLX module; only the
    /// container's closures use it, one at a time.
    private nonisolated struct Loaded: @unchecked Sendable {
        let container: ModelContainer
        let codec: LFM3AudioCodec
    }

    private var loading: (directory: URL, task: Task<Loaded, Error>)?

    // MARK: - Load

    /// Loads the checkpoint in `directory` and warms it, unless it is loaded
    /// or loading already. Another checkpoint replaces the current one.
    func load(from directory: URL) async throws {
        if loading?.directory != directory {
            unload()
            let task = Task { () throws -> Loaded in
                let start = Date()
                let loaded = Loaded(
                    container: try await VLMModelFactory.shared.loadContainer(
                        from: directory, using: AppTokenizerLoader()),
                    codec: try LFM3AudioCodec.load(modelDirectory: directory))
                try await Self.warmUp(loaded)
                Log.agent.notice(
                    "LFM3 Voice loaded \(directory.lastPathComponent) in \(String(format: "%.1f", Date().timeIntervalSince(start)))s"
                )
                return loaded
            }
            loading = (directory, task)
        }
        _ = try await current()
    }

    /// Drops the model and the codec.
    func unload() {
        guard let loading else { return }
        loading.task.cancel()
        self.loading = nil
        Memory.clearCache()
        Log.agent.info("LFM3 Voice unloaded \(loading.directory.lastPathComponent)")
    }

    private func current() async throws -> Loaded {
        guard let loading else { throw LFM3VoiceError.notLoaded }
        do {
            return try await loading.task.value
        } catch {
            if self.loading?.directory == loading.directory { self.loading = nil }
            throw error
        }
    }

    /// `body` on the model, inside its container; MLX errors throw rather
    /// than end the process.
    private nonisolated static func run<R: Sendable>(
        _ loaded: Loaded, _ body: @Sendable (ModelContext, LFM3Audio) async throws -> R
    ) async throws -> R {
        try await withError {
            try await loaded.container.perform { context in
                guard let model = context.model as? LFM3Audio else {
                    throw LFM3VoiceError.notLFM3Audio
                }
                return try await body(context, model)
            }
        }
    }

    /// A second of silence through the ear and two frames through the voice
    /// and the codec, so the first turn runs on compiled kernels.
    private nonisolated static func warmUp(_ loaded: Loaded) async throws {
        try await run(loaded) { context, model in
            let silence = try await context.processor.prepare(
                input: UserInput(chat: [
                    .system("Perform ASR in English."),
                    .user("", audios: [.array(MLXArray.zeros([16_000]))]),
                ]))
            var hearing = try LFM3AudioGenerator(
                model: model, input: silence, output: .text, maxSlots: 2)
            while hearing.next() != nil {}
            let line = try await context.processor.prepare(
                input: UserInput(chat: [.system(readPrompt), .user("Hi.")]))
            var speaking = try LFM3AudioGenerator(
                model: model, input: line, output: .speech, maxSlots: 2)
            var voice = Voice(codec: loaded.codec, outputs: nil)
            try voice.speak(&speaking) { _ in }
        }
    }

    // MARK: - Hear

    /// The words in a take of mono samples at `sampleRate`, heard in
    /// `language` (its English name), or in whatever language it is in when
    /// `nil`.
    func transcribe(_ samples: [Float], sampleRate: Double, language: String?) async throws
        -> String
    {
        let loaded = try await current()
        let clip = AudioConverter.resample(samples, from: sampleRate, to: 16_000)
        let task = language.map { "Perform ASR in \($0)." } ?? "Perform ASR."
        return try await Self.run(loaded) { context, model in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [
                    .system(task), .user("", audios: [.array(MLXArray(clip))]),
                ]))
            // Speech is a few tokens a second. On noise the model can loop
            // to any cap (2026-10-28 measurement), so the take sets it.
            let cap = 32 + 8 * clip.count / 16_000
            var generator = try LFM3AudioGenerator(
                model: model, input: input, output: .text, maxSlots: cap)
            var tokens = [Int]()
            var voice = Voice(codec: loaded.codec, outputs: nil)
            try voice.speak(&generator) { tokens.append($0) }
            return context.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    // MARK: - Answer

    /// Writes one answer to `prompt`: a tool call, or text spoken as it is
    /// written. Without `mayCallTools` it must answer, spoken. Text and speech
    /// go to `outputs` as they come, which finishes when the answer ends.
    func answer(
        _ prompt: LFM3VoicePrompt, mayCallTools: Bool,
        outputs: AsyncStream<LFM3VoiceOutput>.Continuation
    ) async throws -> LFM3VoiceAnswer {
        defer { outputs.finish() }
        let loaded = try await current()
        return try await Self.run(loaded) { context, model in
            let config = model.config
            let endOfTurn = context.tokenizer.convertTokenToId("<|im_end|>") ?? 124_900
            let input = try await context.processor.prepare(input: prompt.input)
            var components = GenerationComponents()
            if mayCallTools, let toolCall = context.tokenizer.convertTokenToId("<tool_call>") {
                components = components.appendingLogitProcessor {
                    LFM3VoiceOpening(toolCall: toolCall, mixedStart: config.mixedStartTokenId)
                }
            }
            var generator = try LFM3AudioGenerator(
                model: model, input: input, output: mayCallTools ? .automatic : .interleaved,
                maxSlots: Self.maxSlots, endOfTurnTokenIds: [endOfTurn], components: components)

            let markers: Set<Int> = [
                config.mixedStartTokenId, config.textEndTokenId, config.audioStartTokenId,
                config.audioOutTokenId, endOfTurn,
            ]
            var detokenizer = LinearStreamingDetokenizer(
                tokenizer: context.tokenizer, delivery: .live)
            let calls = ToolCallProcessor(
                format: context.configuration.toolCallFormat ?? .xmlFunction,
                tools: prompt.input.tools)
            func show(_ chunks: [String]) {
                for chunk in chunks {
                    if let shown = calls.processChunk(chunk), !shown.isEmpty {
                        outputs.yield(.text(shown))
                    }
                }
            }
            var voice = Voice(codec: loaded.codec, outputs: outputs)
            try voice.speak(&generator) { token in
                if !markers.contains(token) { show(detokenizer.append(token: token)) }
            }
            show(detokenizer.finish())
            if let rest = calls.processEOS(returnBufferedText: true), !rest.isEmpty {
                outputs.yield(.text(rest))
            }
            return LFM3VoiceAnswer(toolCalls: calls.toolCalls, spoke: voice.spoke)
        }
    }

    // MARK: - Read aloud

    /// Speaks `text` in the voice an answer speaks in, for an answer the model
    /// wrote without speaking it.
    func read(_ text: String, outputs: AsyncStream<LFM3VoiceOutput>.Continuation) async throws {
        defer { outputs.finish() }
        let loaded = try await current()
        try await Self.run(loaded) { context, model in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.system(Self.readPrompt), .user(text)]))
            // About 5 frames a spoken word; the cap stops speech that never ends.
            let words = text.split(whereSeparator: \.isWhitespace).count
            var generator = try LFM3AudioGenerator(
                model: model, input: input, output: .speech,
                maxSlots: min(Self.maxSlots, 40 + 10 * words))
            var voice = Voice(codec: loaded.codec, outputs: outputs)
            try voice.speak(&generator) { _ in }
        }
    }

    /// A generator's slots to their end: text tokens to the caller, speech
    /// frames into sound. The first frame decodes alone for a quick first
    /// sound, then five at a time (0.4 s), each piece final once it is out.
    private nonisolated struct Voice {
        let codec: LFM3AudioCodec
        /// Where the sound goes; `nil` decodes it and drops it (the warm-up).
        let outputs: AsyncStream<LFM3VoiceOutput>.Continuation?
        private var stream: LFM3AudioCodec.Stream?
        private var pending = [[Int32]]()
        private(set) var spoke = false

        init(codec: LFM3AudioCodec, outputs: AsyncStream<LFM3VoiceOutput>.Continuation?) {
            self.codec = codec
            self.outputs = outputs
        }

        mutating func speak(_ generator: inout LFM3AudioGenerator, text: (Int) -> Void) throws {
            while let slot = generator.next() {
                try Task.checkCancellation()
                switch slot {
                case .text(let token): text(token)
                case .speech(let codes): try add(codes)
                case .endOfSpeech: try end()
                }
            }
            try end()
        }

        private mutating func add(_ codes: [Int32]) throws {
            let fresh = stream == nil
            if fresh { stream = try codec.makeStream() }
            spoke = true
            pending.append(codes)
            if fresh || pending.count >= 5 { try flush(isFinal: false) }
        }

        /// The end of the speech so far: what is held back comes out.
        private mutating func end() throws {
            guard stream != nil else { return }
            try flush(isFinal: true)
            stream = nil
        }

        private mutating func flush(isFinal: Bool) throws {
            guard let stream else { return }
            let samples = try stream.decode(LFM3AudioCodec.codes(frames: pending), isFinal: isFinal)
                .asArray(Float.self)
            pending.removeAll()
            if !samples.isEmpty { outputs?.yield(.speech(samples)) }
        }
    }
}

extension LFM3VoiceEngine {
    /// `text` read aloud, its speech handed to `onSpeech` as it comes.
    @MainActor
    func read(_ text: String, onSpeech: ([Float]) -> Void) async throws {
        let (outputs, continuation) = AsyncStream.makeStream(of: LFM3VoiceOutput.self)
        async let reading: Void = read(text, outputs: continuation)
        for await case .speech(let samples) in outputs {
            onSpeech(samples)
        }
        try await reading
    }
}
