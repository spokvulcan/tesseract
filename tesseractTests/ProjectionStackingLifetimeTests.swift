import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

/// Run alone: peak counters are process-global. This is a bounded traversal
/// investigation, not a production loading optimization or a 27B measurement.
@Suite(.serialized)
struct ProjectionStackingLifetimeTests {
    private struct Measurement: Codable {
        let traversal: String
        let beforeBytes: Int
        let afterBytes: Int
        let peakIncreaseBytes: Int
        let foldedBytes: Int
        let bitwiseEqual: Bool
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TESSERACT_PROJECTION_LIFETIME_EVIDENCE"] == "1"
        ))
    func incrementalTraversalAvoidsRetainingEveryReplacedProjection() throws {
        let flat = measure(incremental: false)
        let incremental = measure(incremental: true)
        let report = try JSONEncoder().encode([flat, incremental])
        print("PROJECTION_LIFETIME=" + (try #require(String(bytes: report, encoding: .utf8))))
        #expect(flat.bitwiseEqual && incremental.bitwiseEqual)
        #expect(flat.foldedBytes == incremental.foldedBytes)
        #expect(flat.peakIncreaseBytes > incremental.peakIncreaseBytes + flat.foldedBytes / 2)
    }

    private func measure(incremental: Bool) -> Measurement {
        MLXRandom.seed(506)
        let blocks = (0..<8).map { _ in Qwen3NextMLP(dimensions: 512, hiddenDimensions: 1024) }
        let model = Sequential(layers: blocks)
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
        let foldedBytes = blocks.reduce(0) { count, block in
            count
                + [block.gateProj, block.upProj].reduce(0) {
                    $0 + $1.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
                }
        }
        let input = MLXRandom.normal([1, 4, 512])
        let expected = model(input).asData(access: .copy).data
        Memory.clearCache()
        let before = Memory.activeMemory
        Memory.peakMemory = 0
        let stacked: Int
        if incremental {
            // Prototype only: visit replaces each block before enumerating its
            // children, so no flat array holds all of the original projections.
            var count = 0
            model.visit { _, module in
                if let folding = module as? SameInputProjectionStacking,
                    folding.stackSameInputProjections()
                {
                    count += 1
                }
            }
            if count > 0 { model.invalidateCompiledTraces() }
            stacked = count
        } else {
            stacked = stackSameInputProjections(in: model)
        }
        #expect(stacked == blocks.count)
        let peak = Memory.peakMemory
        let after = Memory.activeMemory
        let actual = model(input).asData(access: .copy).data
        return Measurement(
            traversal: incremental ? "incremental" : "flat", beforeBytes: before, afterBytes: after,
            peakIncreaseBytes: peak - before, foldedBytes: foldedBytes,
            bitwiseEqual: actual == expected)
    }
}
