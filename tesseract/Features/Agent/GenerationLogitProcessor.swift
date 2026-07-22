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
/// (single-shot prefill, chunked-prefill decode, state-threaded decode ×2),
/// one of them carrying the `kvBits` carve-out by hand — the exact wiring
/// shape whose earlier lapse *was* the bug ADR-0053 fixed. A fifth path that
/// forgot the call would regress it with a green suite. Collapsing the
/// decision here makes that decision made — and tested — exactly once.
///
/// **Policy, not performer** (house style, ADR-0049/0050/0051): a pure
/// `nonisolated` decider returning the processor value; the paths own the
/// effect of building their iterator around it.
///
/// The `kvBits` carve-out. The explicit-processor `TokenIterator` init does no
/// in-iterator KV-cache quantization, so a path may only hand it an app
/// processor when it has already quantized the cache up front (or when
/// `kvBits` is nil, so there is nothing to quantize). The single-shot arm has
/// no up-front quantization point that reproduces the vendor's
/// quantize-after-prefill order, so it passes `pathQuantizesKVUpFront: false`
/// and reverts to the vendor's parameter-driven init (prompt-seeded presence
/// window) when `kvBits` is set — theoretical, since `kvBits` is nil on every
/// agent preset (#252). The chunked-prefill and state-threaded paths quantize
/// before their iterator, so they pass `true` and always get the app
/// processor.
nonisolated enum GenerationLogitProcessor {

    /// The processor a generation path must attach, or `nil` to attach none.
    ///
    /// `nil` means one of: no penalties are configured, or (with
    /// `pathQuantizesKVUpFront: false`) `kvBits` is set and the path must defer
    /// to the vendor init. Both resolve the same way at every call site — no
    /// explicit app processor — so callers need not tell them apart.
    ///
    /// - Parameters:
    ///   - parameters: the generation parameters.
    ///   - pathQuantizesKVUpFront: whether this path quantizes the KV cache
    ///     before building its iterator (so it can safely attach an app
    ///     processor even when `kvBits` is set). `false` only for the
    ///     single-shot arm.
    static func resolve(
        for parameters: GenerateParameters,
        pathQuantizesKVUpFront: Bool
    ) -> (any LogitProcessor)? {
        if !pathQuantizesKVUpFront, parameters.kvBits != nil {
            return nil
        }
        return AgentLogitProcessors.processor(for: parameters)
    }
}
