import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import Tesseract_Agent

/// The toy model as a DFlash2 target (ADR-0079): the vendor
/// `DFlash2SpeculativeTokenIterator` runs over it unchanged, so the DFlash2
/// arm is testable on both speculative arms without a downloaded model.
///
/// A prefill is the toy's own forward. A verify pass moves the attention
/// caches to the block's position and runs the same forward, so its rows
/// land where the iterator commits them and its logits are the toy's
/// scripted predictions at those positions; the rows past the accepted
/// prefix are scratch the next pass overwrites, as the real target's
/// `writeRows` leaves them. Hidden states are zeros: the scripted drafter
/// reads positions, not features. Recurrent layers are not supported, so a
/// toy with `recurrentElements > 0` refuses the cache.
nonisolated extension ToyLanguageModel: DFlash2TargetModel {
    var dflash2LayerCount: Int { kvHeads.count }
    var dflash2Embedding: Embedding { Embedding(embeddingCount: vocabSize, dimensions: headDim) }
    var dflash2Head: Linear? { nil }

    func dflash2SupportsCache(_ cache: [any KVCache]) -> Bool {
        cache.allSatisfy { $0 is KVCacheSimple }
    }

    func dflash2Prefill(
        _ tokens: MLXArray, cache: [any KVCache], captureLayers: [Int]
    ) -> (logits: MLXArray, hidden: [MLXArray]) {
        let logits = self(tokens, cache: cache)
        return (logits, captureLayers.map { _ in MLXArray.zeros([1, tokens.dim(-1), 1]) })
    }

    func dflash2Verify(_ request: DFlash2VerifyRequest, cache: [any KVCache]) -> DFlash2VerifyResult
    {
        // The position may be lazy (the pipelined round builds this pass
        // before the previous accept is known); the toy syncs on it. The
        // iterator commits offsets a round behind the passes it schedules,
        // so the offset may sit below the position (the rows between were
        // written by the previous pass and are accepted) or above it (the
        // previous pass's rejected rows): either way the pass writes at the
        // position.
        let position = request.position.item(Int.self)
        for case let layer as KVCacheSimple in cache {
            layer.offset = position
        }
        let logits = self(request.tokens, cache: cache)
        return DFlash2VerifyResult(
            logits: logits,
            hidden: request.captureLayers.map { _ in
                MLXArray.zeros([1, request.tokens.dim(-1), 1])
            },
            recurrentCaptures: [])
    }
}

/// A DFlash2 drafter that proposes the toy's own scripted continuation, so
/// greedy rounds accept every draft, unless `missEvery` makes every n-th
/// drafted token wrong, which exercises partial acceptance and the rewind of
/// undrained drafts. It keeps no context cache and ignores hidden states:
/// the anchor's position is all it needs.
nonisolated final class ScriptedDFlash2Drafter: Module, DFlash2DrafterModel {
    let script: [Int]
    let eosTokenId: Int
    let vocabSize: Int
    let missEvery: Int
    let targetLayerCount: Int
    let contextWindow: Int
    private let lock = NSLock()
    private var drafted = 0
    private var proposals = 0

    init(for model: ToyLanguageModel, missEvery: Int = 0, contextWindow: Int = 16) {
        self.script = model.script
        self.eosTokenId = model.eosTokenId
        self.vocabSize = model.vocabSize
        self.missEvery = missEvery
        self.targetLayerCount = model.kvHeads.count
        self.contextWindow = contextWindow
        super.init()
    }

    var blockSize: Int { DFlash2Support.blockSize }
    var maskTokenId: Int { vocabSize - 1 }
    var targetLayerIds: [Int] { [0] }

    /// Proposals made so far, across every iterator this drafter served.
    var proposalCount: Int { lock.withLock { proposals } }

    func makeState() -> DFlash2DrafterState {
        DFlash2DrafterState(contextCaches: [])
    }

    func propose(
        block: MLXArray,
        targetHidden: MLXArray,
        contextPosition: Int,
        validRows: MLXArray,
        temperature: Float,
        target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2Proposal {
        // The anchor sits right after the context's committed rows.
        let anchorPosition = contextPosition + validRows.item(Int.self)
        let width = block.dim(1) - 1
        let tokens: [Int32] = (1...width).map { offset in
            let position = anchorPosition + offset
            let truth = position < script.count ? script[position] : eosTokenId
            let miss = lock.withLock {
                drafted += 1
                return missEvery > 0 && drafted % missEvery == 0
            }
            return Int32(miss ? (truth + 1) % (vocabSize - 1) : truth)
        }
        lock.withLock { proposals += 1 }
        let drafts = MLXArray(tokens).expandedDimensions(axis: 0)
        return DFlash2Proposal(
            tokens: drafts,
            candidates: drafts.expandedDimensions(axis: -1),
            probabilities: MLXArray.ones([1, width, 1]))
    }
}

extension Speculation {
    /// The DFlash2 draft resident, scripted over `model`.
    nonisolated static func scriptedDFlash2(
        over model: ToyLanguageModel, missEvery: Int = 0
    ) -> Speculation {
        Speculation(dflash2Drafter: ScriptedDFlash2Drafter(for: model, missEvery: missEvery))
    }
}
