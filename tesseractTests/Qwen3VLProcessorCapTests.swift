import CoreImage
import MLXLMCommon
import MLXVLM
import Testing

/// The vendored Qwen3VL processor's per-image **Vision Token Budget** cap
/// (ADR-0014). A PARO-style config ships a huge `longest_edge` (≈16 MP), so
/// without the default clamp a full-resolution screenshot feeds tens of
/// thousands of patches into the global O(patches²) ViT. These assert observable
/// output — the produced grid's patch/token count — never the clamp arithmetic,
/// so they survive a refactor of *how* the budget is computed.
///
/// Scope: these exercise the **image** `preprocess` entry only. ADR-0014 applies
/// the identical `min(config.maxPixels, 2560·factor²)` clamp to the per-frame
/// **video** path in `prepare`, but video input is product-unreachable
/// (CONTEXT.md → "no audio/video input path") and driving it needs a real
/// `AVAsset`, so the video limb is covered by structural parity with the
/// image clamp asserted here, not by a separate executing assertion.
struct Qwen3VLProcessorCapTests {

    /// Budget in patches: 2,560 vision tokens × mergeSize²(4) = 10,240 patches.
    private static let budgetPatches = 10_240

    /// PARO-shaped processor config: factor = patch_size(16) × merge_size(2) =
    /// 32, and a `longest_edge` far above the 2,560-token budget so the default
    /// ceiling is what bites. `preprocess` needs no weights — only the config —
    /// so a stub tokenizer satisfies the initializer.
    private static func paroProcessor() throws -> Qwen3VLProcessor {
        let json = #"""
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "merge_size": 2,
              "patch_size": 16,
              "temporal_patch_size": 2,
              "image_processor_type": "Qwen2VLImageProcessor",
              "size": { "longest_edge": 16777216, "shortest_edge": 3136 }
            }
            """#
        let config = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self,
            from: Data(json.utf8)
        )
        return Qwen3VLProcessor(config, tokenizer: FakeChatMLTokenizer())
    }

    /// A solid CIImage of the given pixel size — no decode, no disk.
    private static func solidImage(width: Int, height: Int) -> CIImage {
        CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5))
            .cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    @Test func capsAFullResolutionScreenshotToTheTokenBudget() throws {
        // A 7.74 MP "Retina screenshot" (3456 × 2234) is 30,240 patches (7,560
        // tokens) uncapped; the default budget clamps it to ≤ 2,560 tokens.
        let processor = try Self.paroProcessor()
        let (_, grid) = try processor.preprocess(
            images: [Self.solidImage(width: 3456, height: 2234)],
            processing: nil
        )

        #expect(grid.product <= Self.budgetPatches)
        #expect(grid.product / 4 <= 2_560)  // vision tokens = patches / mergeSize²
    }

    @Test func explicitProcessingMaxPixelsRaisesTheCeiling() throws {
        // ADR-0008: a client may ask for a higher budget. The same screenshot
        // with a larger explicit `maxPixels` resolves to strictly more patches
        // than the default ceiling allows — proving the default is a default,
        // not a hard wall.
        let processor = try Self.paroProcessor()
        let image = Self.solidImage(width: 3456, height: 2234)

        let (_, defaultGrid) = try processor.preprocess(images: [image], processing: nil)
        let (_, raisedGrid) = try processor.preprocess(
            images: [image],
            processing: UserInput.Processing(maxPixels: 6_000_000)
        )

        #expect(defaultGrid.product <= Self.budgetPatches)
        #expect(raisedGrid.product > defaultGrid.product)
    }
}
