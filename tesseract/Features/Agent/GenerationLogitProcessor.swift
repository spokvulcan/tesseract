//
//  GenerationLogitProcessor.swift
//  tesseract
//
//  The one seam every raw-generation path routes through to decide which
//  logit processor it attaches (issue #405, enforcing ADR-0053's "every
//  generation path" clause).
//

import Foundation
import MLXLMCommon

/// The **Generation Logit Processor** seam: the single place the app decides
/// which ``LogitProcessor`` a generation path attaches, so no call site
/// hand-picks one.
///
/// ADR-0053 shipped an output-only presence penalty and asserted "every
/// generation path routes through ``AgentLogitProcessors/processor(for:)``".
/// That property was previously re-implemented at four wiring sites
/// (single-shot prefill, chunked-prefill decode, state-threaded decode ×2).
/// A fifth path that forgot the call would regress it with a green suite.
/// Collapsing the decision here makes that decision made — and tested —
/// exactly once.
///
/// **Policy, not performer** (house style, ADR-0049/0050/0051): a pure
/// `nonisolated` decider returning the processor value; the paths own the
/// effect of building their iterator around it.
///
/// Two shapes, by who converts the KV cache (the **Decode Start** decides
/// that, ADR-0093). An iterator that holds the processor itself — the app's
/// state-threaded iterators, and the vendor's explicit-processor init over a
/// cache its path already converted — takes ``processor(for:)``. An iterator
/// built from parameters keeps the vendor's in-iterator KV plan and takes
/// the processor as a component (``components(for:)``).
nonisolated enum GenerationLogitProcessor {

    /// The processor a generation path must attach, or `nil` when no
    /// penalties are configured.
    static func processor(for parameters: GenerateParameters) -> (any LogitProcessor)? {
        AgentLogitProcessors.processor(for: parameters)
    }

    /// For an iterator built from parameters, which keeps the vendor's
    /// in-iterator KV plan: the parameters without the penalties, and the
    /// app processor that applies them as a generation component (the
    /// vendor's parameter-built penalties would seed from the prompt and
    /// double the app's).
    static func components(
        for parameters: GenerateParameters
    ) -> (parameters: GenerateParameters, components: GenerationComponents) {
        var stripped = parameters
        stripped.repetitionPenalty = nil
        stripped.presencePenalty = nil
        stripped.frequencyPenalty = nil
        guard let processor = AgentLogitProcessors.processor(for: parameters) else {
            return (stripped, GenerationComponents())
        }
        // One iterator = one generation, so a single resolved instance keeps
        // the fresh-state contract; the box routes the non-Sendable processor
        // into the @Sendable factory.
        let box = UnsafeSendableBox(processor)
        return (stripped, GenerationComponents().appendingLogitProcessor { box.value })
    }
}
