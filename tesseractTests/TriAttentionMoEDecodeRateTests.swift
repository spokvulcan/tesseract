import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Decode-rate regression gate for Qwen3.5-family MoE under TriAttention.
///
/// The HTTP-path sparse-KV runtime regressed by 15–25× on Qwen3.6-35B-A3B
/// past the retention budget (~14K tokens) after commit `30104e5b` broadened
/// TriAttention eligibility to MoE. This suite pins a 50 tok/s floor across
/// the context regime that exhibits the cliff, so any future regression of
/// the same shape fails loudly instead of silently degrading user-visible
/// decode throughput.
///
/// Opt-in: tests skip (and log) when the MoE model is not on disk. Run
/// manually with the MoE artifact downloaded via the Tesseract model
/// manager.
@MainActor
@Suite(.serialized)
struct TriAttentionMoEDecodeRateTests {

    /// Directory name inside the app's Application Support model store.
    /// The absolute path resolves through `NSHomeDirectory()` so the test
    /// works both when the xctest binary runs inside the Tesseract Agent
    /// sandbox (home is the container) and outside it.
    private static let moeModelDirectoryName = "unsloth_Qwen3.6-35B-A3B-UD-MLX-4bit"

    private static let moeModelID = "qwen3.6-35b-a3b-ud"

    /// Five context points spanning the regime where the regression appears.
    /// 8K is still under the 12K budget; 14K/16K/20K/28K cross it. Pre-fix
    /// decode on Qwen3.6-35B-A3B with TriAttention enabled drops from ~65
    /// tok/s at 8K to 4 tok/s at 14K and 2 tok/s past 27K.
    private static let promptTokenTargets = [8_192, 14_336, 16_384, 20_480, 28_672]

    /// Conservative floor. Dense Qwen3.5-27B PARO + TriAttention decodes at
    /// 35–45 tok/s; MoE pre-regression was 63–72 tok/s. 50 tok/s sits safely
    /// above the dense floor and far above the 4 tok/s regression signature,
    /// so the test catches the cliff with margin to spare.
    private static let decodeRateFloor: Double = 50.0

    private static let maxOutputTokens = 256

    // MARK: - Seed paragraphs for deterministic long-context prompt construction.
    //
    // Same shape as PrefillStepBenchmarkSupport — repeat-and-truncate by
    // token count against the loaded tokenizer, so every run at a given
    // target yields an identical byte sequence.

    nonisolated static let systemSeedParagraph = """
        You are running a deterministic TriAttention decode-rate regression test on a \
        Qwen3.5-family MoE checkpoint. The prompt block intentionally contains a very \
        long, repeated system instruction so the context crosses the sparse-KV retention \
        budget. Prefer concise answers, keep style factual, and never call tools. This \
        paragraph is repeated to build a stable long prefix for the decode-rate sweep, \
        not to change the task.
        """

    nonisolated static let userSeedParagraph = """
        Please continue the passage above with a short, deterministic summary. The user \
        message is padded to push total prompt tokens past the TriAttention retention \
        budget so the sparse-KV scoring path is exercised during decode. Keep the \
        response brief and factual; do not invoke any tools.
        """

    // MARK: - Tests

    /// Primary regression gate: TriAttention enabled on MoE, five context points.
    @Test
    func decodeRateOnMoE_TriAttentionEnabled_meetsFloorAcrossContextRegime() async throws {
        guard let modelDir = Self.resolveMoEModelOrSkip() else { return }

        let engine = AgentEngine()
        try await engine.loadModel(
            from: modelDir,
            visionMode: false,
            triAttention: TriAttentionConfiguration(enabled: true)
        )

        var results: [(promptTokens: Int, genTokens: Int, tokPerSec: Double)] = []

        for target in Self.promptTokenTargets {
            let (systemPrompt, userMessage) = try await Self.buildPromptPair(
                engine: engine,
                targetSystemTokens: target
            )

            let params = Self.makeGenerateParameters(triAttentionEnabled: true)
            let start = try await engine.generateServerTextCompletion(
                modelID: Self.moeModelID,
                systemPrompt: systemPrompt,
                messages: [.user(content: userMessage, images: [])],
                toolSpecs: nil,
                prefixCacheConversation: nil,
                parameters: params
            )

            var info: AgentGeneration.Info?
            for try await event in start.stream {
                if case .info(let i) = event { info = i }
            }

            guard let info else {
                Issue.record("No .info event emitted at target=\(target)")
                continue
            }
            results.append((info.promptTokenCount, info.generationTokenCount, info.tokensPerSecond))
            print(
                "[decode-rate] target=\(target) promptTokens=\(info.promptTokenCount) "
                    + "genTokens=\(info.generationTokenCount) "
                    + "tokPerSec=\(String(format: "%.2f", info.tokensPerSecond))"
            )
        }

        for result in results {
            #expect(
                result.tokPerSec >= Self.decodeRateFloor,
                "decode rate \(result.tokPerSec) tok/s below floor \(Self.decodeRateFloor) at promptTokens=\(result.promptTokens)"
            )
        }
    }

    // NOTE: A TriAttention-disabled baseline at p≈14K will be reintroduced
    // as a separate test once the primary regression is fixed. Running both
    // tests in the same xcodebuild invocation requires ~2× 35B model loads
    // which exceeds 48GB unified memory; scheduling them as separate runs
    // is the right shape. Kept as a TODO until the decode-rate floor holds.

    // MARK: - Helpers

    /// Returns the MoE model directory if present; otherwise logs a skip
    /// message and returns nil. Callers early-return on nil — Swift Testing
    /// doesn't have a native skip primitive on this toolchain, so a no-op
    /// return is the clean way to mark the test as not applicable.
    ///
    /// The resolved path is `~/Library/Application Support/models/<name>`
    /// through `NSHomeDirectory()`. When run inside the Tesseract Agent
    /// xctest bundle, `NSHomeDirectory()` is the sandbox container, so this
    /// resolves to
    /// `~/Library/Containers/app.tesseract.agent/Data/Library/Application Support/models/<name>` —
    /// the same directory the Tesseract model manager writes into.
    private static func resolveMoEModelOrSkip() -> URL? {
        let modelDir = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Application Support/models", isDirectory: true)
            .appendingPathComponent(moeModelDirectoryName, isDirectory: true)
        let configPath = modelDir.appendingPathComponent("config.json", isDirectory: false).path
        guard FileManager.default.fileExists(atPath: configPath) else {
            print(
                "[decode-rate] SKIP — MoE model not present at \(modelDir.path). "
                    + "Download via the Tesseract model manager to enable this test."
            )
            return nil
        }
        return modelDir
    }

    /// Produce `(systemPrompt, userMessage)` such that `systemPrompt` tokenizes
    /// to approximately `targetSystemTokens` tokens and `userMessage` is a
    /// short fixed trigger. We grow the system prompt (not the user message)
    /// because the retention-budget cliff depends on total context length,
    /// and the system prompt carries it with the least noise from chat-
    /// template framing.
    private static func buildPromptPair(
        engine: AgentEngine,
        targetSystemTokens: Int
    ) async throws -> (systemPrompt: String, userMessage: String) {
        let systemPrompt = try await engine.withModelContainer { container in
            await container.perform { context in
                PrefillStepBenchmarkSupport.buildText(
                    targetTokens: targetSystemTokens,
                    measureTextTokens: { text in
                        context.tokenizer.encode(text: text, addSpecialTokens: false).count
                    },
                    seedParagraph: systemSeedParagraph
                )
            }
        }
        // Short, fixed user trigger — same across every context point so
        // the only variable is the system-prompt token count.
        return (systemPrompt, userSeedParagraph)
    }

    /// Sampling params mirroring `qwen36Thinking` — the documented preset for
    /// this model family — but capped at `maxOutputTokens` so the test
    /// finishes in bounded wall time. Greedy decoding is avoided because
    /// Qwen3.6 thinking-mode loops under low temperature.
    private static func makeGenerateParameters(
        triAttentionEnabled: Bool
    ) -> AgentGenerateParameters {
        var params = AgentGenerateParameters.qwen36Thinking
        params.maxTokens = maxOutputTokens
        params.triAttention =
            triAttentionEnabled
            ? TriAttentionConfiguration(enabled: true)
            : .v1Disabled
        return params
    }
}
