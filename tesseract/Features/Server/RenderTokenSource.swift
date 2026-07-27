//
//  RenderTokenSource.swift
//  tesseract
//
//  The single home for "may this render tokenize through the
//  **Render+Token Cache**, and has its token list already been computed this
//  request?" — one value replacing the five different spellings of that
//  predicate the C25–C31 seams grew (`imageKeying == nil` +
//  `?? "unfingerprinted"` on the request path, `keySpace.isIdentity, let
//  modelFingerprint` on the planner/leaf paths, a ternary in the admission
//  builder). Two of those five disagreed on the unknown-fingerprint case;
//  this type makes the answer one decision made in one place.
//
//  A `nil` `cacheFingerprint` means "always render+encode in full" — the
//  pre-C25 behavior. The cache itself never sees a synthetic key: an unknown
//  model fingerprint bypasses rather than sharing a bucket, because the
//  repeat path is the one resolve whose exactness rests on the key alone (a
//  byte-identical render under the same fingerprint returns the cached tokens
//  outright, with no empirical arbiter behind it).
//

import Foundation

nonisolated struct RenderTokenSource: Sendable {

    /// The fingerprint to resolve under, or `nil` to bypass the cache.
    let cacheFingerprint: String?

    /// The BASE conversation's render-space token list, when an earlier phase
    /// of this same request already computed the identical render (C31). Only
    /// ever the base render — a continuation render is a different
    /// conversation and is always computed.
    let baseRenderTokens: [Int]?

    /// Always render+encode in full: no cache, no plumbed render.
    static let uncached = RenderTokenSource(cacheFingerprint: nil)

    init(cacheFingerprint: String?, baseRenderTokens: [Int]? = nil) {
        self.cacheFingerprint = cacheFingerprint
        self.baseRenderTokens = baseRenderTokens
    }

    /// The planner / leaf-store / leaf-admission seams: engage the cache only
    /// on an identity (text-only) **Cache Key Space** under a known model
    /// fingerprint. Image-bearing key spaces need the real token list for
    /// their placeholder runs, so they always render.
    static func forIdentityKeySpace(
        _ keySpace: CacheKeySpace,
        modelFingerprint: String?,
        baseRenderTokens: [Int]? = nil
    ) -> RenderTokenSource {
        RenderTokenSource(
            cacheFingerprint: keySpace.isIdentity ? modelFingerprint : nil,
            baseRenderTokens: baseRenderTokens
        )
    }

    /// The request path (**Request Keying**, and the agent's
    /// `startRawGeneration`): engage the cache only for a media-free request
    /// on a model whose processor emits a flat 1-D token list.
    ///
    /// `producesFlatTextTokens` is the DIRECT property the cache path needs —
    /// `LMInput(tokens:)` here must reproduce what the processor would build,
    /// and a vision container's text-only `prepare` emits 2D `[batch, seq]`.
    /// It replaces the old `imageKeying == nil` proxy, which asked whether the
    /// app RECOGNIZES a vision container (`model_type` prefix `qwen3_5` plus a
    /// `vision_config`) — true of today's only VLM family by coincidence, and
    /// silently wrong for any VLM family added without an image-keying rule.
    static func forTextOnlyRequest(
        hasMedia: Bool,
        producesFlatTextTokens: Bool,
        modelFingerprint: String?
    ) -> RenderTokenSource {
        guard !hasMedia, producesFlatTextTokens else { return .uncached }
        return RenderTokenSource(cacheFingerprint: modelFingerprint)
    }

    /// Same eligibility, carrying a base render computed elsewhere this
    /// request.
    func withBaseRenderTokens(_ tokens: [Int]?) -> RenderTokenSource {
        RenderTokenSource(cacheFingerprint: cacheFingerprint, baseRenderTokens: tokens)
    }
}
