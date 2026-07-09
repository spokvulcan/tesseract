import AppKit
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM
import Testing

/// The vendored Qwen3VL image path must hand the model **gamma-encoded sRGB**
/// values — the byte values the HF reference preprocesses — not CoreImage's
/// linear-light working-space values. The 2026-07-09 incident: the port
/// lacked the sRGB tone-curve step, so near-black content (dark-theme text at
/// luminance 23/255 on a 6/255 background) reached the ViT with ~12× less
/// contrast than the reference and the model could not read it, while the
/// same image through Qwen's reference deployment read fine.
struct Qwen3VLProcessorToneCurveTests {

    /// Same PARO-shaped config as `Qwen3VLProcessorCapTests`.
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

    /// A 64×64 PNG-decoded image: background gray level 6 with a centered
    /// 32×32 block at gray level 23 — the exact luminances of the incident
    /// screenshot's background and faint headline. Encoded to PNG and decoded
    /// via `CIImage(data:)` so it takes the same color-management path as a
    /// real attachment.
    private static func faintBlockImage() throws -> CIImage {
        let side = 64, block = 32
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: side, pixelsHigh: side,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        for y in 0..<side {
            for x in 0..<side {
                let inBlock = (16..<16 + block).contains(x) && (16..<16 + block).contains(y)
                let level = CGFloat(inBlock ? 23 : 6) / 255
                rep.setColor(
                    NSColor(deviceRed: level, green: level, blue: level, alpha: 1),
                    atX: x, y: y)
            }
        }
        let png = rep.representation(using: .png, properties: [:])!
        return try #require(CIImage(data: png))
    }

    @Test func darkContentKeepsGammaEncodedContrast() throws {
        let processor = try Self.paroProcessor()
        let (pixels, _) = try processor.preprocess(
            images: [Self.faintBlockImage()], processing: nil)

        // Patchify rearranges spatial layout but preserves values, so global
        // extrema are layout-independent assertions. Reference (HF byte-value
        // preprocessing, mean/std 0.5): block = (23/255 − 0.5)/0.5 ≈ −0.820,
        // background = (6/255 − 0.5)/0.5 ≈ −0.953. The linear-space failure
        // mode produced ≈ −0.984 / −0.996 — far outside these tolerances.
        let maxValue = pixels.max().item(Float.self)
        let minValue = pixels.min().item(Float.self)
        #expect(abs(maxValue - (-0.820)) < 0.05, "faint block reached the model at \(maxValue)")
        #expect(abs(minValue - (-0.953)) < 0.05, "background reached the model at \(minValue)")
    }
}
