import Foundation
@preconcurrency import MLX
import MLXNN

public func mossApplyRepetitionPenalty(
    logits: MLXArray,
    previousTokenIDs: MLXArray?,
    penalty: Float = 1.0
) -> MLXArray {
    guard penalty != 1.0, let previousTokenIDs, previousTokenIDs.size > 0 else {
        return logits
    }

    let vocabSize = logits.dim(logits.ndim - 1)
    let uniqueTokenIDs = Array(Set(previousTokenIDs.asArray(Int32.self).map(Int.init)))
        .filter { $0 >= 0 && $0 < vocabSize }
    guard !uniqueTokenIDs.isEmpty else { return logits }

    let tokenIDs = MLXArray(uniqueTokenIDs.map { Int32($0) }).reshaped([1, -1])
    let selected = takeAlong(logits, tokenIDs, axis: -1)
    let penalized = MLX.where(
        selected .< 0,
        selected * MLXArray(penalty),
        selected / MLXArray(penalty)
    )
    return putAlong(logits, tokenIDs, values: penalized, axis: -1)
}

public func mossApplyTopK(_ logits: MLXArray, topK: Int?) -> MLXArray {
    guard let topK, topK > 0 else { return logits }
    let vocabSize = logits.dim(logits.ndim - 1)
    let k = min(topK, vocabSize)
    guard k < vocabSize else { return logits }

    let kth = min(k - 1, max(vocabSize - 1, 0))
    let maskIndices = argPartition(-logits, kth: kth, axis: -1)[0..., k...]
    let negInf = MLXArray.full(maskIndices.shape, values: MLXArray(-Float.infinity), dtype: logits.dtype)
    return putAlong(logits, maskIndices, values: negInf, axis: -1)
}

public func mossApplyTopP(_ logits: MLXArray, topP: Float?) -> MLXArray {
    guard let topP, topP > 0, topP < 1 else { return logits }
    let vocabSize = logits.dim(logits.ndim - 1)
    guard vocabSize > 1 else { return logits }

    let logProbs = logSoftmax(logits, axis: -1)
    let sortedIndices = argSort(logProbs, axis: -1)
    let sortedProbs = exp(takeAlong(logProbs, sortedIndices, axis: -1))
    let cumulativeProbs = MLX.cumsum(sortedProbs, axis: -1)

    let arangeIndices = MLXArray(0 ..< vocabSize).reshaped([1, -1]).asType(.int32)
    let inverseIndices = putAlong(
        MLXArray.zeros(sortedIndices.shape, type: Int32.self),
        sortedIndices.asType(.int32),
        values: arangeIndices,
        axis: -1
    )
    let cumulativeOriginalOrder = takeAlong(cumulativeProbs, inverseIndices, axis: -1)
    return MLX.where(
        cumulativeOriginalOrder .> MLXArray(1 - topP),
        logits,
        MLXArray(-Float.infinity).asType(logits.dtype)
    )
}

public func mossSampleNextToken(
    logits: MLXArray,
    doSample: Bool,
    temperature: Float = 1.0,
    topK: Int? = nil,
    topP: Float? = nil,
    previousTokenIDs: MLXArray? = nil,
    repetitionPenalty: Float = 1.0
) throws -> MLXArray {
    var scores = mossApplyRepetitionPenalty(
        logits: logits,
        previousTokenIDs: previousTokenIDs,
        penalty: repetitionPenalty
    )
    guard doSample else {
        return argMax(scores, axis: -1).asType(.int32)
    }
    guard temperature > 0 else {
        throw MossTTSNanoError.invalidInput("temperature must be positive when doSample is true")
    }

    if temperature != 1.0 {
        scores = scores / MLXArray(temperature)
    }
    scores = mossApplyTopK(scores, topK: topK)
    scores = mossApplyTopP(scores, topP: topP)
    return MLXRandom.categorical(scores).asType(.int32)
}

public func mossSampleAssistantTextToken(
    textLogits: MLXArray,
    audioAssistantSlotTokenID: Int,
    audioEndTokenID: Int,
    doSample: Bool,
    temperature: Float,
    topK: Int,
    topP: Float
) throws -> MLXArray {
    let candidateIDs = MLXArray([
        Int32(audioAssistantSlotTokenID),
        Int32(audioEndTokenID),
    ]).reshaped([1, 2])
    let candidateScores = takeAlong(textLogits, candidateIDs, axis: -1)
    let sampledCandidate = try mossSampleNextToken(
        logits: candidateScores,
        doSample: doSample,
        temperature: temperature,
        topK: min(topK, 2),
        topP: topP
    ).reshaped([1, 1])
    return takeAlong(candidateIDs, sampledCandidate, axis: -1).squeezed(axis: -1)
}
