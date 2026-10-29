//
//  LFM3VoiceE2ERunner.swift
//  tesseract
//
//  `--lfm3-voice-e2e [--bench-model-id ID]`: **LFM3 Voice**'s turns on the
//  real LFM3-Audio, headless. Each request is spoken by the model's own voice,
//  heard back through the voice mode's ear, and answered by its turns, with
//  the real agenda and memory tools over in-memory stores and canned web
//  tools. The report lands in `lfm3-voice-e2e/latest.log` under the bench
//  output: what was heard, every call and result, the answer, and when its
//  first sound came.
//

import Foundation
import MLXLMCommon

@MainActor
struct LFM3VoiceE2ERunner {

    private struct Scenario {
        let request: String
        /// The tool the request needs, or nil for one it answers directly.
        let tool: String?
    }

    private static let scenarios = [
        Scenario(request: "Hi there! How's it going?", tool: nil),
        Scenario(request: "What's on my calendar today?", tool: "agenda"),
        Scenario(
            request: "Remind me to call my sister at six thirty tonight.", tool: "add_reminder"),
        Scenario(request: "Tell me a joke.", tool: nil),
        Scenario(
            request: "Can you look up the weather in Paris for tomorrow?", tool: "browser.search"),
        Scenario(request: "Remember that my favorite coffee is a flat white.", tool: "remember"),
    ]

    let runner: BenchmarkRunner

    func run() async throws {
        let config = runner.activeConfig
        let reportDir = config.outputDir.appendingPathComponent("lfm3-voice-e2e")
        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        var report = [String]()
        func say(_ line: String) {
            print(line)
            report.append(line)
        }

        // The voice mode's checkpoint unless the flags name another.
        let directory =
            config.modelID == nil && config.modelDir == nil
            ? ModelDefinition.all.first { $0.id == "lfm3-audio-2b-8bit" }?.directory(
                in: ModelDownloadManager.modelStorageURL)
            : try runner.resolveModelDirectory()
        guard let directory else { throw Failure.noCheckpoint }

        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(
            "lfm3-voice-e2e-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        let engine = LFM3VoiceEngine()
        let ear = LFM3VoiceTranscriber(engine: engine)
        let player = CountingPlayback()
        let playback = LFM3VoicePlayback(player: player)
        let agendaStore = InMemoryAgendaStore()
        let tools =
            createAgendaTools(agenda: Agenda(store: agendaStore))
            + createProfileTools(
                profile: ProfileStore(url: nil),
                index: RecallIndex(
                    conversationsDirectory: scratch,
                    databaseURL: scratch.appendingPathComponent("recall.sqlite")))
            + Self.webTools
        var chat = [any AgentMessageProtocol & Sendable]()
        let turns = LFM3VoiceTurns(
            engine: engine, playback: playback,
            tools: { tools.filter { LFM3VoiceTurns.toolNames.contains($0.name) } },
            chatMessages: { chat },
            commit: { messages in
                chat += messages
                return true
            },
            onError: { say("error: \($0)") })

        let loadStart = Date()
        try await engine.load(from: directory)
        say(
            String(
                format: "loaded %@ in %.1fs", directory.lastPathComponent,
                Date().timeIntervalSince(loadStart)))

        var right = 0
        for scenario in Self.scenarios {
            say("\n» \(scenario.request)")
            var spoken = [Float]()
            try await engine.read(scenario.request) { spoken += $0 }
            let rate = Double(LFM3VoiceEngine.sampleRate)
            let hearStart = Date()
            let heard = try await ear.transcribe(
                AudioData(samples: spoken, sampleRate: rate, duration: Double(spoken.count) / rate),
                language: "en"
            ).text
            say(String(format: "  heard (%.2fs): %@", Date().timeIntervalSince(hearStart), heard))

            let before = chat.count
            player.reset()
            let turnStart = Date()
            let answer: String = await withCheckedContinuation { continuation in
                turns.replyCompleted = { continuation.resume(returning: $0) }
                turns.send(heard)
            }
            let wall = Date().timeIntervalSince(turnStart)
            var called = [String]()
            for message in chat[before...] {
                if let assistant = message.asAssistant {
                    for call in assistant.toolCalls {
                        called.append(call.name)
                        say("  call \(call.name) \(call.argumentsJSON)")
                    }
                } else if let result = message.asToolResult {
                    say(
                        "  result \(result.isError ? "(error) " : "")\(result.content.textContent.prefix(160))"
                    )
                }
            }
            say("  answer: \(answer)")
            say(
                String(
                    format: "  spoken %.1fs of audio, first sound %@, turn %.2fs",
                    player.seconds,
                    player.firstChunkAt.map {
                        String(format: "%.2fs", $0.timeIntervalSince(turnStart))
                    } ?? "never", wall))
            let toolRight = scenario.tool.map { called.contains($0) } ?? called.isEmpty
            let ok = toolRight && player.seconds > 0.5 && !answer.isEmpty
            if ok { right += 1 }
            say(
                "  \(ok ? "PASS" : "FAIL")\(toolRight ? "" : " (expected \(scenario.tool ?? "no tool"), called \(called))")"
            )
        }
        say("\nreminders in the store: \(agendaStore.reminders.values.map(\.title))")
        say("\(right)/\(Self.scenarios.count) scenarios passed")
        try (report.joined(separator: "\n") + "\n").write(
            to: reportDir.appendingPathComponent("latest.log"), atomically: true, encoding: .utf8)
        await engine.unload()
        if right < Self.scenarios.count {
            throw Failure.scenariosFailed(passed: right, of: Self.scenarios.count)
        }
    }

    /// The web tools, answering from a script: the harness stays offline.
    private static var webTools: [AgentToolDefinition] {
        func canned(_ name: String, _ description: String, _ parameter: String, _ result: String)
            -> AgentToolDefinition
        {
            AgentToolDefinition(
                name: name, label: name, description: description,
                parameterSchema: JSONSchema(
                    type: "object",
                    orderedProperties: [
                        (
                            name: parameter,
                            schema: PropertySchema(type: "string", description: parameter)
                        )
                    ],
                    required: [parameter]),
                execute: { _, _, _, _ in .text(result) })
        }
        return [
            canned(
                "browser.search",
                "Search the web anonymously and return titles, URLs, and snippets.", "query",
                "1. Paris weather tomorrow: sunny, 14°C, light wind (weather.com)."),
            canned(
                "browser.fetch", "Fetch a URL anonymously and return it as distilled Markdown.",
                "url", "Paris, Thursday: sunny, high 14°C, low 6°C."),
        ]
    }

    enum Failure: Error, CustomStringConvertible {
        case noCheckpoint
        case scenariosFailed(passed: Int, of: Int)

        var description: String {
            switch self {
            case .noCheckpoint: "no LFM3-Audio checkpoint in the catalog"
            case .scenariosFailed(let passed, let total): "\(passed)/\(total) scenarios passed"
            }
        }
    }
}

/// Counts what would play, and plays out at once.
@MainActor
private final class CountingPlayback: AudioPlayback {
    var onPlaybackFinished: (@MainActor @Sendable () -> Void)?
    private(set) var samples = 0
    private(set) var firstChunkAt: Date?
    private var rate = LFM3VoiceEngine.sampleRate

    var seconds: Double { Double(samples) / Double(rate) }
    var totalScheduledDuration: TimeInterval { seconds }

    func reset() {
        samples = 0
        firstChunkAt = nil
    }

    func play(samples: [Float], sampleRate: Int) {}
    func startStreaming(sampleRate: Int) { rate = sampleRate }
    func appendChunk(samples chunk: [Float]) {
        if firstChunkAt == nil { firstChunkAt = Date() }
        samples += chunk.count
    }
    func finishStreaming() { onPlaybackFinished?() }
    func pause() {}
    func resume() {}
    func currentPlaybackTime() -> TimeInterval { seconds }
    func stop() {}
    func setPlaybackRate(_ rate: Float) {}
}
