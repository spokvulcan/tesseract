import Foundation
import Testing

@testable import Tesseract_Agent

/// Source-shape pin for the **Conversation Render** module (ticket #473 of
/// issue #471): one place in the server applies the chat template, so the
/// **Emitted Path Resolve** (ADR-0063) can be added in one spot and cover
/// every spelling. The scan reads the app sources off disk — the same
/// `#filePath` locator `ServerCompletionExtractSnapshotPayloadsTests` uses —
/// and fails on any non-comment call to `applyChatTemplate(` or
/// `renderChatTemplate(` outside the module.
///
/// The module is two files: the verbs (`ConversationRender.swift`) and the
/// **Render+Token Cache** they resolve through (`RenderTokenCache.swift`,
/// "the implementation below it" per the glossary), whose miss path renders
/// to bytes and encodes.
///
/// Excluded by name — the ticket's "benchmarks and non-server processors":
/// - `Features/Agent/Benchmark/` — standalone benchmarks that time the
///   fused call against the split on purpose.
/// - `ParoQuantLoader.swift` — the in-tree PARO `UserInputProcessor`, whose
///   `prepare` is the processor fallback a bypassing request keeps (vendor
///   processors do the same from `Vendor/`, outside the scan); ADR-0063
///   decision 5 narrows that path in #475.
/// - `ProofreadModel.swift` — the dictation proofread processor.
/// - `LLMActor.swift` — the agent's raw-prompt inspector (`formatRawPrompt`),
///   a display of the rendered string, never a prompt the server feeds.
///
/// Doc comments legitimately name the calls, so comment lines are dropped
/// before matching; the pattern requires the open paren of a call.
@Suite struct ConversationRenderSourceShapeTests {

    /// The module's verbs: every server template application goes through them.
    private static let moduleFile = "ConversationRender.swift"

    /// The module: the verbs plus the resolve arm below them.
    private static let moduleFiles: Set<String> = [moduleFile, "RenderTokenCache.swift"]

    /// Non-server processors and inspectors allowed their own call, by name.
    private static let excludedByName: Set<String> = [
        "ParoQuantLoader.swift",
        "ProofreadModel.swift",
        "LLMActor.swift",
    ]

    /// Benchmarks, excluded by directory name.
    private static let excludedDirectories: Set<String> = ["Benchmark"]

    /// A call to either template entry — the name followed by its open paren.
    /// Not `\b`: Swift's default word boundaries follow UAX #29, under which
    /// `tokenizer.applyChatTemplate` is one word, so a member call would never
    /// match; an explicit non-identifier character (or line start) is the
    /// boundary instead.
    private static var templateCall: Regex<(Substring, Substring)> {
        /(?:^|[^A-Za-z0-9_])(applyChatTemplate|renderChatTemplate)\s*\(/
    }

    private static var appSourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // tesseractTests
            .deletingLastPathComponent()  // project root
            .appendingPathComponent("tesseract")
    }

    private static func appSources() throws -> [URL] {
        let enumerator = try #require(
            FileManager.default.enumerator(
                at: appSourceRoot,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            ))
        var files: [URL] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            files.append(url)
        }
        return files.sorted { $0.path < $1.path }
    }

    /// Non-comment lines that call the template, numbered.
    private static func templateCallLines(in source: String) -> [Int] {
        var hits: [Int] = []
        for (index, line) in source.components(separatedBy: "\n").enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("//") { continue }
            if line.contains(templateCall) {
                hits.append(index + 1)
            }
        }
        return hits
    }

    private static func isAllowed(_ url: URL) -> Bool {
        moduleFiles.contains(url.lastPathComponent)
            || excludedByName.contains(url.lastPathComponent)
            || !excludedDirectories.isDisjoint(with: url.pathComponents)
    }

    @Test func onlyTheConversationRenderModuleAppliesTheChatTemplate() throws {
        let sources = try Self.appSources()
        // Guard the locator: a wrong root would pass vacuously.
        #expect(
            sources.count > 100,
            Comment(rawValue: "expected the whole app source tree, got \(sources.count)"))

        var offenders: [String] = []
        for url in sources where !Self.isAllowed(url) {
            let source = try String(contentsOf: url, encoding: .utf8)
            let lines = Self.templateCallLines(in: source)
            if !lines.isEmpty {
                let relative = url.path.replacingOccurrences(
                    of: Self.appSourceRoot.deletingLastPathComponent().path + "/", with: "")
                offenders.append("\(relative):\(lines.map(String.init).joined(separator: ","))")
            }
        }
        #expect(
            offenders.isEmpty,
            Comment(
                rawValue: "chat template applied outside the Conversation Render module: "
                    + offenders.joined(separator: "; ")))
    }

    /// The scan is not vacuous: the module itself applies the template, and
    /// the matcher sees it.
    @Test func theModuleItselfAppliesTheTemplate() throws {
        let module = try #require(
            try Self.appSources().first { $0.lastPathComponent == Self.moduleFile })
        let source = try String(contentsOf: module, encoding: .utf8)
        #expect(!Self.templateCallLines(in: source).isEmpty)
    }

    /// Comment lines that mention the call by name do not count.
    @Test func commentMentionsAreNotCalls() {
        let source = """
            /// Falls back to `applyChatTemplate(messages:tools:additionalContext:)`.
            // renderChatTemplate( is what the miss path runs
            let tokens = tokenizer.encode(text: "applyChatTemplate", addSpecialTokens: false)
            """
        #expect(Self.templateCallLines(in: source).isEmpty)
        #expect(
            Self.templateCallLines(in: "let x = try tokenizer.applyChatTemplate(messages: m)")
                == [1])
    }
}
