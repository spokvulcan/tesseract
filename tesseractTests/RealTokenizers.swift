import Foundation
import MLXHuggingFace
import MLXLMCommon
import Synchronization
import Tokenizers

@testable import Tesseract_Agent

// MARK: - RealTokenizers

/// The real tokenizers the `*RealTests` suites read, each loaded once per test
/// process and shared by every test that asks for the same loader and
/// directory.
///
/// A load parses a 12–19 MB `tokenizer.json` and builds the BPE tables, about
/// 2 s in a Debug build, and every test case used to pay it: over 40 loads a
/// run, most of the real-tokenizer suites' time. A loaded tokenizer is
/// immutable and built to be shared, as the app shares one across its callers:
/// `BPETokenizer` builds its tables eagerly for concurrent `encode` callers,
/// and `PreTrainedTokenizer` locks its compiled-template cache.
nonisolated enum RealTokenizers {
    /// Loaded through the vendor bridge, `#huggingFaceTokenizerLoader()`.
    static func huggingFace(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        try await shared("huggingFace", directory) {
            try await #huggingFaceTokenizerLoader().load(from: directory)
        }
    }

    /// Loaded through the production loader, ``AppTokenizerLoader``.
    static func app(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        try await shared("app", directory) {
            try await AppTokenizerLoader().load(from: directory)
        }
    }

    private static let loads = Mutex<[String: Task<any MLXLMCommon.Tokenizer, any Error>]>([:])

    private static func shared(
        _ loader: String, _ directory: URL,
        load: @escaping @Sendable () async throws -> any MLXLMCommon.Tokenizer
    ) async throws -> any MLXLMCommon.Tokenizer {
        let key = "\(loader):\(directory.standardizedFileURL.path)"
        let task = loads.withLock { loads in
            if let task = loads[key] { return task }
            // Detached, so the parse runs on the global executor even when a
            // main-actor suite asks first.
            let task = Task.detached { try await load() }
            loads[key] = task
            return task
        }
        return try await task.value
    }
}
