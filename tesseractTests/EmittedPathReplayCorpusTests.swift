import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

/// The Emitted Path Index replay gate (ADR-0063, tickets #475/#476/#477):
/// the recorded sessions walked through the **Canonical-Echo Fidelity**
/// harness with a private index learning every echoed turn — the Leaf
/// Store's registration simulated on the canonical encode, the leaf source
/// decided exactly as the live fast path decides it — and every next
/// request resolving at its edge. The claims, on real recordings with the
/// real tokenizer and template, judged per turn by `EmittedPathReplayGate`:
///
/// - in a tool stretch the leaf source is `live`; a stop turn is `live` or
///   the explained think-stripping boundary;
/// - every live turn registers (the only tolerated skip is a prompt that is
///   not a token prefix of the stored render — a turn the harness cannot
///   simulate fed ids for);
/// - the fidelity gate rejected nothing and no key was registered twice
///   (asserted on the index's own counters, not only logged);
/// - every next request resolves the whole registered path and prefills
///   its new messages plus at most the glue allowance;
/// - the simulated post-EOS CPU tail stays under budget below 20k tokens.
///
/// Every recorded request renders under the context the server resolved
/// for it (the request's `reasoning_effort` against the template's
/// defaults, the app's preserve-thinking default), so the walk feeds the
/// bytes the build fed.
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
    func everyTurnPassesTheReplayGate() async throws {
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
                        tools: recording.body.tools,
                        renderContext: Self.resolveRenderContext(recording.body, identity: identity)
                    )
                ))
        }

        // One index across the walk, as one loaded model has one.
        let learning = CanonicalEchoFidelity.EmittedPathLearning(
            fingerprint: "replay:\(modelDirectory.lastPathComponent)",
            toolCallFormat: toolCallFormat,
            promptStartsThinking: identity.promptStartsThinking)

        var total = CanonicalEchoFidelity.EmittedPathSessionSummary()
        var failures: [EmittedPathReplayGate.Failure] = []
        var turns = 0
        var exempt = 0
        var slowestTail = 0.0
        var longestPath = 0
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
            let sessionFailures = EmittedPathReplayGate.check(report)
            for failure in sessionFailures { print("GATE \(failure)") }
            failures += sessionFailures
            for boundary in report.boundaries {
                guard let turn = EmittedPathReplayGate.TurnAccount(boundary) else { continue }
                turns += 1
                // A turn the harness cannot simulate is exempt from the
                // prefill, glue and tail rules; the total names how many.
                if turn.verdict.registration == EmittedPathReplayGate.promptNotTokenPrefix {
                    exempt += 1
                }
                slowestTail = max(slowestTail, turn.verdict.tailSeconds ?? 0)
                longestPath = max(longestPath, turn.verdict.pathLength ?? 0)
                print(
                    "TURN request#\(turn.requestIndex) " + EmittedPathReplayGate.account(of: turn))
            }
            guard let summary = report.emittedPathSummary else { continue }
            total.boundaries += summary.boundaries
            total.registered += summary.registered
            total.registrationSkips.merge(summary.registrationSkips, uniquingKeysWith: +)
            total.nextResolved += summary.nextResolved
            total.nextMisses.merge(summary.nextMisses, uniquingKeysWith: +)
            total.nextSuffixTokens += summary.nextSuffixTokens
            total.sources.merge(summary.sources, uniquingKeysWith: +)
            total.overwrites += summary.overwrites
            total.fidelityRejections += summary.fidelityRejections
        }
        let stats = learning.index.statsSnapshot()
        print(
            "emitted-path corpus total: recordings=\(recordingFiles.count) "
                + "sessions=\(sessionOrder.count) boundaries=\(total.boundaries) "
                + "registered=\(total.registered) exempt=\(exempt) "
                + "skips=\(total.registrationSkips) "
                + "sources=\(total.sources) nextResolved=\(total.nextResolved) "
                + "nextMisses=\(total.nextMisses) nextSuffixTokens=\(total.nextSuffixTokens) "
                + "slowestTailMs=\(String(format: "%.1f", slowestTail * 1000)) "
                + "longestPath=\(longestPath) undecodable=\(undecodable.count) "
                + "index=[\(EmittedPathIndex.summary(of: stats))]")

        #expect(total.boundaries > 0, "corpus produced no checkable boundaries")
        #expect(turns == total.boundaries)
        #expect(total.registered > 0, "no boundary registered")
        // The per-turn gate: every failure names its turn and account.
        #expect(
            failures.isEmpty,
            Comment(
                rawValue: "\(failures.count) gate failures:\n"
                    + failures.map(\.description).joined(separator: "\n")))
        // The index's own counters, not only the walk's log: nothing was
        // rejected by the fidelity gate, no key was registered twice.
        #expect(stats.fidelityRejections == 0, "fidelity rejections: \(stats.fidelityRejections)")
        #expect(stats.overwrites == 0, "same-key overwrites: \(stats.overwrites)")
        #expect(total.fidelityRejections == 0)
        #expect(total.overwrites == 0)
        #expect(
            total.nextResolved == total.registered,
            "next requests that missed the index: \(total.nextMisses)")

        Self.checkLiveTails(in: corpus)
    }

    /// When the corpus directory carries the build's own completion traces
    /// (`trace-*.jsonl`, the durable per-completion record), the live
    /// post-EOS tail every registered turn paid is gated too: the offline
    /// walk can only simulate the CPU half.
    private static func checkLiveTails(in corpus: URL) {
        let traceFiles = CompletionTraceLog.traceFiles(in: corpus)
        guard !traceFiles.isEmpty else {
            print(
                "emitted-path corpus: no trace-*.jsonl beside the recordings; live tail unchecked")
            return
        }
        let records = CompletionTraceLog.readRecords(at: traceFiles)
        var over: [String] = []
        var checked = 0
        for record in records {
            guard let emitted = record.emittedPath, emitted.registered,
                let pathLength = emitted.pathLength,
                pathLength < EmittedPathReplayGate.tailBudgetPathTokens,
                let tail = record.tailSeconds
            else { continue }
            checked += 1
            if tail >= EmittedPathReplayGate.tailBudgetSeconds {
                over.append(
                    "request=\(record.requestID) pathLength=\(pathLength) "
                        + "tailMs=\(String(format: "%.1f", tail * 1000))")
            }
        }
        print("emitted-path corpus: live tails checked=\(checked) over=\(over.count)")
        #expect(
            over.isEmpty,
            Comment(rawValue: "live tails over budget:\n" + over.joined(separator: "\n")))
    }

    /// The render context the server resolved for the recorded request —
    /// `CompletionHandler`'s own resolution, with the preserve-thinking
    /// render on as the recorded sessions ran it.
    private nonisolated static func resolveRenderContext(
        _ body: OpenAI.ChatCompletionRequest, identity: ModelIdentity
    ) -> TemplateRenderContext {
        CompletionHandler.resolveRenderContext(
            for: body, preserveThinking: true, template: identity)
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
