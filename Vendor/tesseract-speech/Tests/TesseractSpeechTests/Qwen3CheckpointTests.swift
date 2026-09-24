// The disk rule for a loadable Qwen3-TTS checkpoint, and the production
// adapter's refusal to fetch one. Both run on temp directories; the adapter
// tests fail before any weights load, so no MLX and no GPU.

import Foundation
import Testing
@testable import TesseractSpeech

/// A temp directory laid out like a model store folder.
private struct CheckpointFixture {
    let directory: URL

    init() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-checkpoint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    func remove() { try? FileManager.default.removeItem(at: directory) }

    func place(_ path: String, bytes: Int = 16) throws {
        try place(path, data: Data(count: bytes))
    }

    func place(_ path: String, json: Any) throws {
        try place(path, data: JSONSerialization.data(withJSONObject: json))
    }

    func place(_ path: String, data: Data) throws {
        let file = directory.appendingPathComponent(path)
        try FileManager.default.createDirectory(
            at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: file)
    }

    /// The single-shard layout of the VoiceDesign 6-bit/8-bit repos.
    func placeComplete() throws {
        try place("config.json", json: ["model_type": "qwen3_tts"])
        try place("model.safetensors")
        try place(
            "model.safetensors.index.json",
            json: ["weight_map": ["talker.a": "model.safetensors"]])
        try place("tokenizer_config.json")
        try place("vocab.json")
        try place("merges.txt")
        try place("speech_tokenizer/config.json")
        try place("speech_tokenizer/model.safetensors")
    }

    var missing: [String] { Qwen3Checkpoint.missingFiles(in: directory) }
}

@Suite struct Qwen3CheckpointTests {

    @Test func completeCheckpointMissesNothing() throws {
        let fixture = try CheckpointFixture()
        defer { fixture.remove() }
        try fixture.placeComplete()
        #expect(fixture.missing.isEmpty)
    }

    @Test func emptyDirectoryListsEverything() throws {
        let fixture = try CheckpointFixture()
        defer { fixture.remove() }
        #expect(
            fixture.missing == [
                "config.json", "model.safetensors", "tokenizer_config.json", "vocab.json",
                "merges.txt", "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors",
            ])
    }

    /// A nested speech-tokenizer file is not the model. This is the shape of
    /// an interrupted sharded download: every small file plus the speech
    /// tokenizer, none of the talker shards.
    @Test func speechTokenizerWithoutTalkerShardsIsIncomplete() throws {
        let fixture = try CheckpointFixture()
        defer { fixture.remove() }
        try fixture.placeComplete()
        try FileManager.default.removeItem(
            at: fixture.directory.appendingPathComponent("model.safetensors"))
        try fixture.place(
            "model.safetensors.index.json",
            json: [
                "weight_map": [
                    "talker.a": "model-00001-of-00002.safetensors",
                    "talker.b": "model-00002-of-00002.safetensors",
                ]
            ])

        #expect(
            fixture.missing == [
                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            ])

        try fixture.place("model-00001-of-00002.safetensors")
        #expect(fixture.missing == ["model-00002-of-00002.safetensors"])
    }

    @Test func emptyOrUnreadableFilesDoNotCount() throws {
        let fixture = try CheckpointFixture()
        defer { fixture.remove() }
        try fixture.placeComplete()
        try fixture.place("speech_tokenizer/model.safetensors", bytes: 0)
        try fixture.place("config.json", data: Data("{\"model_type\":".utf8))

        #expect(fixture.missing == ["config.json", "speech_tokenizer/model.safetensors"])
    }

    /// The vendor writes tokenizer.json from vocab.json + merges.txt on
    /// first load; either form is enough.
    @Test func tokenizerJSONStandsInForVocabAndMerges() throws {
        let fixture = try CheckpointFixture()
        defer { fixture.remove() }
        try fixture.placeComplete()
        for name in ["vocab.json", "merges.txt"] {
            try FileManager.default.removeItem(at: fixture.directory.appendingPathComponent(name))
        }
        #expect(fixture.missing == ["vocab.json", "merges.txt"])

        try fixture.place("tokenizer.json")
        #expect(fixture.missing.isEmpty)
    }

    @Test func speechTokenizerMustBeADirectory() throws {
        let fixture = try CheckpointFixture()
        defer { fixture.remove() }
        try fixture.placeComplete()
        try FileManager.default.removeItem(
            at: fixture.directory.appendingPathComponent("speech_tokenizer"))
        try fixture.place("speech_tokenizer")

        #expect(
            fixture.missing == ["speech_tokenizer/config.json", "speech_tokenizer/model.safetensors"])
    }
}

/// The production adapter loads only from the directory it's given. Before,
/// a missing or partial checkpoint sent it to the hub: it deleted the folder
/// and downloaded a snapshot, inside the engine's GPU lease.
@Suite struct Qwen3SynthesizerOfflineTests {

    private let spec = TTSModelSpec.voiceDesign17B(.q6)

    private func expectModelUnavailable(
        _ body: () async throws -> Void, sourceLocation: SourceLocation = #_sourceLocation
    ) async {
        let error = await #expect(throws: SpeechEngineError.self, sourceLocation: sourceLocation) {
            try await body()
        }
        guard let error else { return }  // #expect recorded the miss
        guard case .modelUnavailable(let detail) = error else {
            Issue.record("expected modelUnavailable, got \(String(describing: error))",
                sourceLocation: sourceLocation)
            return
        }
        #expect(detail.contains(spec.repo), sourceLocation: sourceLocation)
    }

    @Test func absentCheckpointIsUnavailableAndNothingIsCreated() async throws {
        let store = try CheckpointFixture()
        defer { store.remove() }
        let directory = store.directory.appendingPathComponent("never-downloaded")
        let synthesizer = Qwen3Synthesizer(checkpointDirectory: { _ in directory })

        await expectModelUnavailable { try await synthesizer.checkAvailable(spec) }
        await expectModelUnavailable { try await synthesizer.load(spec, onPhase: nil) }

        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(await synthesizer.audioFormat() == nil, "nothing loaded")
    }

    @Test func partialCheckpointIsLeftInPlace() async throws {
        let fixture = try CheckpointFixture()
        defer { fixture.remove() }
        try fixture.place("config.json", json: ["model_type": "qwen3_tts"])
        try fixture.place("speech_tokenizer/model.safetensors")
        let directory = fixture.directory
        let synthesizer = Qwen3Synthesizer(checkpointDirectory: { _ in directory })

        await expectModelUnavailable { try await synthesizer.load(spec, onPhase: nil) }

        // The download manager owns this folder: the adapter neither deletes
        // the partial download nor adds to it.
        let files = FileManager.default.subpaths(atPath: fixture.directory.path) ?? []
        #expect(
            Set(files) == ["config.json", "speech_tokenizer", "speech_tokenizer/model.safetensors"])
    }
}
