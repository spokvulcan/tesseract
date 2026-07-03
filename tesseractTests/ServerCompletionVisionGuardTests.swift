import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXLMCommon
import Testing
import UniformTypeIdentifiers

@testable import Tesseract_Agent

/// Counts toy-model forwards across isolations — the "allocate" side of the
/// ADR-0014 ordering assertion: a rejected request must never have run one.
private nonisolated final class ForwardCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var _count = 0

    var hasForwarded: Bool { lock.withLock { _count > 0 } }

    func onForward(_ offset: Int) {
        lock.withLock { _count += 1 }
    }
}

/// Vision-guard ordering at both generation arms (PRD #137, user story 12;
/// ADR-0014): an image whose priced patch count would put the global ViT's
/// `[heads, ΣP, ΣP]` attention matrix past the Metal buffer limit must be
/// rejected as a typed error *before* the tower allocates — pinned here as
/// "before any model forward at all", over the 2D-token toy variant.
@Suite struct ServerCompletionVisionGuardTests {

    /// A grid whose patch count prices the ViT attention matrix in the
    /// hundreds of petabytes — over any real device's buffer limit, so the
    /// suite needs no Metal-limit override to trip the guard.
    private static let oversizedFrame = THW(1, 8192, 8192)

    /// A real 1×1 PNG: the keyed path proves attachments `CIImage`-decodable
    /// before prepare, so the conversation must carry genuine image bytes.
    private static func tinyPNG() throws -> Data {
        let context = try #require(
            CGContext(
                data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ))
        let image = try #require(context.makeImage())
        let data = NSMutableData()
        let destination = try #require(
            CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
        return data as Data
    }

    @MainActor
    private static func parameters() -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        parameters.kvBits = nil
        return parameters
    }

    /// The keyed arm, end to end through the module's public entry: a
    /// recognized vision family (identity interpreted from a `qwen3_5`
    /// config), a keyed image request, and a grid the guard must reject.
    /// The typed error surfaces from `start`, and the session shows cache
    /// creation but no prefill verb — with zero forwards on the model.
    @Test func keyedArmRejectsOversizedVisionTowerBeforeAnyForward() async throws {
        let tokenizer = ToySequencingTokenizer()
        let forwards = ForwardCounter()
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0], onForward: forwards.onForward),
            tokenizer: tokenizer,
            vision: ToyUserInputProcessor.VisionStub(
                padTokenId: 400, padRunLength: 4, frame: Self.oversizedFrame
            )
        )
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5",
                "image_token_id": 400,
                "vision_config": ["num_heads": 16, "spatial_merge_size": 2],
            ],
            chatTemplate: nil
        )
        let fixture = ServerCompletionFixture(provider: provider, identity: identity)
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                HTTPPrefixCacheMessage(
                    role: .user, content: "look",
                    images: [HTTPPrefixCacheImage(data: try Self.tinyPNG())]
                )
            ]
        )

        do {
            _ = try await fixture.start(conversation: conversation, parameters: Self.parameters())
            Issue.record("oversized vision request must throw, not start generating")
        } catch AgentEngineError.generationFailed {
            // The ADR-0014 typed rejection.
        }

        // Reject before allocate: the guard fired after cache creation but
        // before the vision continuation was even queried — and the model
        // never ran a forward.
        #expect(provider.recorder.verbs == [.prepare, .newCache])
        #expect(forwards.hasForwarded == false)
        await fixture.drain()
    }

    /// The unkeyed arm through its internal entry (the shape the PR A image
    /// test drives), with a rejecting profile: same typed error, same
    /// no-forward ordering.
    @Test func unkeyedArmRejectsOversizedVisionTowerBeforeAnyForward() async throws {
        let tokenizer = FakeChatMLTokenizer()
        let forwards = ForwardCounter()
        let prompt = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "Hi"]], tools: nil, additionalContext: nil
        )
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: prompt, onForward: forwards.onForward),
            tokenizer: tokenizer
        )

        await #expect(throws: AgentEngineError.self) {
            try await provider.withSession { session in
                let prepared = try await session.prepare(
                    UserInput(messages: [["role": "user", "content": "Hi"]])
                )
                let fullInput = LMInput(
                    text: prepared.text,
                    image: LMInput.ProcessedImage(
                        pixels: MLXArray.zeros([4, 3]),
                        frames: [Self.oversizedFrame]
                    )
                )
                _ = try await ServerCompletion.makeUnkeyedGeneration(
                    session: session,
                    fullInput: fullInput,
                    fullTokens: LLMActor.extractTokenSequence(fullInput.text.tokens),
                    reason: .placeholderRunCountMismatch,
                    parameters: GenerateParameters(temperature: 0),
                    toolSpecs: nil,
                    partitionKey: CachePartitionKey(
                        modelID: "toy/model", kvBits: nil, kvGroupSize: 64
                    ),
                    fullAttentionScratchProfile: nil,
                    visionAttentionScratchProfile: ModelIdentity.FullAttentionScratchProfile(
                        attentionHeads: 16, bytesPerElement: 2
                    ),
                    ssdEnabled: false,
                    diagnosticsContext: PrefixCacheDiagnostics.Context(
                        requestID: UUID(), modelID: "toy/model", kvBits: nil, kvGroupSize: 64
                    ),
                    progressHandler: nil
                )
            }
        }

        #expect(provider.recorder.verbs == [.prepare, .newCache])
        #expect(forwards.hasForwarded == false)
    }
}
