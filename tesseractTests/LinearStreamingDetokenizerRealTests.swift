import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

/// `LinearStreamingDetokenizer` against the real Qwen3.8 tokenizer — the
/// byte-level BPE the live loop streams through — on the shapes a turn
/// takes: prose, a tagged tool call, CJK and emoji, code, and a long run
/// without a newline. The chunk sequence must equal the naive one, the byte
/// path must be the one that produced it (this is where the alphabet meets
/// a real vocabulary), and neither guard may fire. Skipped unless the model
/// directory is on disk.
struct LinearStreamingDetokenizerRealTests {

    private nonisolated static var modelDirectory: URL {
        let path =
            ProcessInfo.processInfo.environment["TESSERACT_TOKENIZE_CACHE_MODEL"]
            ?? "~/Library/Application Support/models/mlx-community_Qwen3.8-27B-4bit"
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    }

    private nonisolated static var modelAvailable: Bool {
        FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent("tokenizer_config.json").path)
    }

    private static func loadTokenizer() async throws -> any MLXLMCommon.Tokenizer {
        try await #huggingFaceTokenizerLoader().load(from: modelDirectory)
    }

    private static let samples: [String] = [
        "The quick brown fox jumps over the lazy dog. Then it rests.\n\nA second paragraph.",
        "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"/tmp/a b.txt\", "
            + "\"limit\": 10, \"filters\": {\"kind\": [\"x\", \"y\"]}}}\n</tool_call>",
        "日本語のテキストと絵文字 😀🏳️‍🌈🇺🇸、そして café — naïve façade. Ünïcödé.",
        "func f(x: Int) -> Int {\n\tlet y = x * 2\n\treturn y\n}\n// done\r\nend",
        "Let me think about this.\n</think>\n\nHere is the answer: 42.",
    ]

    @Test(.enabled(if: modelAvailable))
    func theChunksEqualTheNaiveChunksOnEveryShape() async throws {
        let tokenizer = try await Self.loadTokenizer()
        for sample in Self.samples {
            let tokens = tokenizer.encode(text: sample, addSpecialTokens: false)
            var naive = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            var naiveChunks: [String] = []
            for token in tokens {
                naive.append(token: token)
                if let chunk = naive.next() { naiveChunks.append(chunk) }
            }
            var linear = LinearStreamingDetokenizer(tokenizer: tokenizer)
            var linearChunks: [String] = []
            for token in tokens { linearChunks += linear.append(token: token) }
            linearChunks += linear.finish()
            #expect(linearChunks == naiveChunks, Comment(rawValue: sample))
            #expect(linear.decodesFromBytes, Comment(rawValue: sample))
            #expect(linear.resyncs == 0 && linear.fallbacks == 0, Comment(rawValue: sample))
        }
    }

    @Test(.enabled(if: modelAvailable))
    func aLongRunWithoutNewlinesStaysUnderTheTailBudget() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let text = String(
            repeating: "{\"path\": \"/tmp/file.txt\", \"limit\": 10, \"text\": \"value\"}, ",
            count: 400)
        let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
        #expect(tokens.count > 6_000, "\(tokens.count) tokens")
        let start = DispatchTime.now().uptimeNanoseconds
        var linear = LinearStreamingDetokenizer(tokenizer: tokenizer)
        var chunks: [String] = []
        for token in tokens { chunks += linear.append(token: token) }
        chunks += linear.finish()
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        print("linear detokenizer: \(tokens.count) tokens in \(seconds) s")
        #expect(chunks.joined() == text)
        #expect(linear.decodesFromBytes)
        #expect(linear.resyncs == 0 && linear.fallbacks == 0)
        #expect(seconds < EmittedPathReplayGate.tailBudgetSeconds, "\(seconds) s")
    }
}
