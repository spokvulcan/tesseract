//
//  ArbiterGPULease.swift
//  tesseract
//
//  The production adapter for the v2 speech engine's GPU-leasing port
//  (ADR-0038 §dependency strategy): every per-segment generation burst and
//  load/warmup acquires the same FIFO `GPULeaseQueue` the LLM stack uses, so
//  TTS and LLM work stay mutually exclusive on the GPU. Unlike the v1
//  `withExclusiveGPU(.tts)` path, no slot loading happens here — the engine
//  loads its own model under the lease (ADR-0039 lazy load).
//

import TesseractSpeech

nonisolated struct ArbiterGPULease: GPULeasing {
    let arbiter: InferenceArbiter

    // `@concurrent` matches the package requirement exactly: this module
    // builds with NonisolatedNonsendingByDefault, the package does not, and
    // the two spell a plain `async` closure differently.
    func withLease<T: Sendable>(
        _ body: @concurrent @Sendable () async throws -> T
    ) async throws -> T {
        try await arbiter.withSpeechGPULease(body)
    }
}
