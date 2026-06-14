import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

/// The corpus-mode **Canonical-Echo Fidelity** gate (PRD #94): replays a
/// recorded session corpus through the real normalization + probe machinery
/// with the real model tokenizer and template, and fails on any boundary
/// whose derived leaf/speculation path is not a token-identical prefix of the
/// next request's render.
///
/// Opt-in (the corpus contains user project content and lives outside the
/// repo). Invocation — see `docs/testing.md`:
///
///     TESSERACT_FIDELITY_CORPUS=~/projects/tesseract-traces/2026-06-12-interrupt-rewind \
///     TESSERACT_FIDELITY_MODEL="~/Library/…/models/z-lab_Qwen3.5-4B-PARO" \
///     scripts/dev.sh test --filter CanonicalEchoFidelityCorpusTests
@MainActor
struct CanonicalEchoFidelityCorpusTests {

    private nonisolated static var corpusRoot: String? {
        ProcessInfo.processInfo.environment["TESSERACT_FIDELITY_CORPUS"]
    }
    private nonisolated static var modelRoot: String? {
        ProcessInfo.processInfo.environment["TESSERACT_FIDELITY_MODEL"]
    }

    @Test(.enabled(if: corpusRoot != nil && modelRoot != nil))
    func corpusHasZeroMismatchedBoundaries() async throws {
        let corpus = URL(
            fileURLWithPath: NSString(
                string: Self.corpusRoot!
            ).expandingTildeInPath)
        let modelDirectory = URL(
            fileURLWithPath: NSString(
                string: Self.modelRoot!
            ).expandingTildeInPath)

        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: modelDirectory)

        let recordingsDirectory = corpus.appendingPathComponent("http-completions")
        let recordingFiles = try FileManager.default
            .contentsOfDirectory(at: recordingsDirectory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasSuffix("-request.json") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        try #require(!recordingFiles.isEmpty, "no recordings under \(recordingsDirectory.path)")

        // Group recordings by session affinity, preserving file order.
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

        var totalBoundaries = 0
        var totalMismatches = 0
        for key in sessionOrder {
            let entries = sessions[key] ?? []
            let report = await CanonicalEchoFidelity.walkSession(
                requests: entries.map(\.request),
                sessionAffinity: key,
                modelID: entries.first?.model ?? "unknown",
                tokenizer: tokenizer
            )
            totalBoundaries += report.boundaries.count
            totalMismatches += report.mismatchCount
            print(CanonicalEchoFidelity.renderText(report))
        }
        print(
            "corpus total: sessions=\(sessionOrder.count) boundaries=\(totalBoundaries) "
                + "mismatches=\(totalMismatches) undecodable=\(undecodable.count)")

        #expect(totalBoundaries > 0, "corpus produced no checkable boundaries")
        #expect(totalMismatches == 0, "canonical-echo fidelity gate failed")
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
