import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

/// The Emitted Path Index replay gate (ADR-0063, ticket #475): the recorded
/// sessions walked through the **Canonical-Echo Fidelity** harness with a
/// private index learning every echoed turn — the Leaf Store's registration
/// simulated on the canonical encode — and every next request resolving at
/// its edge. The dark launch's claims, on real recordings with the real
/// tokenizer and template:
///
/// - every boundary a live-stored turn would have produced registers (the
///   only tolerated non-registration is a prompt that is not a token prefix
///   of the stored render — the junction merge the Live Leaf Capture would
///   have refused, so no live turn exists to register);
/// - every registered boundary's next request resolves with a non-zero
///   indexed prefix;
/// - the shadow check finds zero differences between the composition and
///   the canonical encode (special token = hard pretoken boundary).
///
/// Opt-in like the fidelity corpus gate — same variables, see
/// `docs/testing.md`:
///
///     TEST_RUNNER_TESSERACT_FIDELITY_CORPUS=~/projects/tesseract-traces/2026-09-06-emitted-path \
///     TEST_RUNNER_TESSERACT_FIDELITY_MODEL="~/Library/…/models/mlx-community_Qwen3.8-27B-4bit" \
///     xcodebuild test … -only-testing:tesseractTests/EmittedPathReplayCorpusTests
@MainActor
struct EmittedPathReplayCorpusTests {

    private nonisolated static var corpusRoot: String? {
        ProcessInfo.processInfo.environment["TESSERACT_FIDELITY_CORPUS"]
    }
    private nonisolated static var modelRoot: String? {
        ProcessInfo.processInfo.environment["TESSERACT_FIDELITY_MODEL"]
    }

    @Test(.enabled(if: corpusRoot != nil && modelRoot != nil))
    func everyStoredTurnRegistersAndEveryNextRequestResolves() async throws {
        let corpus = URL(
            fileURLWithPath: NSString(string: Self.corpusRoot!).expandingTildeInPath)
        let modelDirectory = URL(
            fileURLWithPath: NSString(string: Self.modelRoot!).expandingTildeInPath)

        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: modelDirectory)
        let identity = ModelIdentity(directory: modelDirectory)
        let toolCallFormat = Self.declaredToolCallFormat(modelDirectory: modelDirectory)

        let recordingsDirectory = corpus.appendingPathComponent("http-completions")
        let recordingFiles = try FileManager.default
            .contentsOfDirectory(at: recordingsDirectory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasSuffix("-request.json") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        try #require(!recordingFiles.isEmpty, "no recordings under \(recordingsDirectory.path)")

        var sessions: [String: [(model: String, request: CanonicalEchoFidelity.RecordedRequest)]] =
            [:]
        var sessionOrder: [String] = []
        var undecodable: [String] = []
        for file in recordingFiles {
            guard let recording = Self.decodeRecording(at: file) else {
                undecodable.append(file.lastPathComponent)
                continue
            }
            let key = recording.session ?? "unaffiliated"
            if sessions[key] == nil { sessionOrder.append(key) }
            sessions[key, default: []].append(
                (
                    recording.model,
                    CanonicalEchoFidelity.RecordedRequest(
                        messages: recording.body.messages,
                        tools: recording.body.tools
                    )
                ))
        }

        // One index across the walk, as one loaded model has one.
        let learning = CanonicalEchoFidelity.EmittedPathLearning(
            fingerprint: "replay:\(modelDirectory.lastPathComponent)",
            toolCallFormat: toolCallFormat,
            promptStartsThinking: identity.promptStartsThinking)

        var total = CanonicalEchoFidelity.EmittedPathSessionSummary()
        for key in sessionOrder {
            let entries = sessions[key] ?? []
            let report = await CanonicalEchoFidelity.walkSession(
                requests: entries.map(\.request),
                sessionAffinity: key,
                modelID: entries.first?.model ?? "unknown",
                tokenizer: tokenizer,
                learning: learning
            )
            print(CanonicalEchoFidelity.renderText(report))
            guard let summary = report.emittedPathSummary else { continue }
            total.boundaries += summary.boundaries
            total.registered += summary.registered
            total.registrationSkips.merge(summary.registrationSkips, uniquingKeysWith: +)
            total.nextResolved += summary.nextResolved
            total.nextMisses.merge(summary.nextMisses, uniquingKeysWith: +)
            total.shadowDifferences += summary.shadowDifferences
        }
        let stats = learning.index.statsSnapshot()
        print(
            "emitted-path corpus total: sessions=\(sessionOrder.count) "
                + "boundaries=\(total.boundaries) registered=\(total.registered) "
                + "skips=\(total.registrationSkips) nextResolved=\(total.nextResolved) "
                + "nextMisses=\(total.nextMisses) shadowDifferences=\(total.shadowDifferences) "
                + "undecodable=\(undecodable.count) index=[\(EmittedPathIndex.summary(of: stats))]")

        #expect(total.boundaries > 0, "corpus produced no checkable boundaries")
        // Every simulated live-stored turn registers: the only tolerated
        // skip is a prompt that is not a token prefix of the stored render.
        let unexplained = total.registrationSkips.filter { $0.key != "promptNotTokenPrefix" }
        #expect(unexplained.isEmpty, "registration skips: \(unexplained)")
        #expect(total.registered > 0, "no boundary registered")
        #expect(
            total.nextResolved == total.registered,
            "next requests that missed the index: \(total.nextMisses)")
        #expect(total.shadowDifferences == 0, "shadow check found differences")
        #expect(stats.shadowDifferences == 0)
        #expect(stats.fidelityRejections == 0)
    }

    /// The tool-call format the loaded model declares (the vendor's
    /// `ChatConventionsProviding` on the model class, which the server reads
    /// from the session configuration): the Qwen3.5 family — Qwen3.8
    /// included — declares `.qwen35`; anything else here falls back to the
    /// server's own default.
    private static func declaredToolCallFormat(modelDirectory: URL) -> ToolCallFormat {
        guard
            let data = try? Data(contentsOf: modelDirectory.appendingPathComponent("config.json")),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modelType = json["model_type"] as? String
        else { return .json }
        return modelType.hasPrefix("qwen3_5") ? .qwen35 : .json
    }

    /// `HTTPRequestLogger` recordings are the raw request body prefixed with
    /// a `// session=…` comment line.
    private static func decodeRecording(
        at url: URL
    ) -> (session: String?, model: String, body: OpenAI.ChatCompletionRequest)? {
        guard var raw = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var session: String?
        if raw.hasPrefix("//"), let newline = raw.firstIndex(of: "\n") {
            let header = String(raw[..<newline])
            if let range = header.range(of: "session=") {
                session = header[range.upperBound...]
                    .split(separator: " ").first.map(String.init)
            }
            raw = String(raw[raw.index(after: newline)...])
        }
        guard let data = raw.data(using: .utf8),
            let body = try? JSONDecoder().decode(OpenAI.ChatCompletionRequest.self, from: data)
        else { return nil }
        return (session, body.model ?? "unknown", body)
    }
}
