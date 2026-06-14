//
//  LLMVisionRequirementTests.swift
//  tesseractTests
//
//  Pins the vision-load policy (ADR-0008, ADR-0013): the pure decision a lease
//  makes about whether to load the VLM container. Takes plain booleans so the
//  full matrix is testable without an arbiter, a model load, or a capability
//  probe. Companion to `ImageInputAvailabilityTests` — the composer affordance
//  and the load policy must agree (show affordances iff chat will load vision).
//

import Testing

@testable import Tesseract_Agent

struct LLMVisionRequirementTests {

    /// `.fromSettings` (chat UI, background agents): load vision only when the
    /// global opt-out is on *and* the model is capable. A migrated user who opts
    /// out gets text-only — no stale legacy flag can override the choice.
    @Test func fromSettingsHonorsOptOutAndCapability() {
        #expect(
            LLMVisionRequirement.fromSettings.wantsVision(
                useVisionWhenAvailable: true, isVisionCapable: true) == true)
        #expect(
            LLMVisionRequirement.fromSettings.wantsVision(
                useVisionWhenAvailable: false, isVisionCapable: true) == false)  // opt-out wins
        #expect(
            LLMVisionRequirement.fromSettings.wantsVision(
                useVisionWhenAvailable: true, isVisionCapable: false) == false)
        #expect(
            LLMVisionRequirement.fromSettings.wantsVision(
                useVisionWhenAvailable: false, isVisionCapable: false) == false)
    }

    /// `.visionIfCapable` (HTTP server, ADR-0008): capability alone — the global
    /// opt-out cannot silently break a configured client.
    @Test func visionIfCapableIgnoresOptOut() {
        #expect(
            LLMVisionRequirement.visionIfCapable.wantsVision(
                useVisionWhenAvailable: false, isVisionCapable: true) == true)
        #expect(
            LLMVisionRequirement.visionIfCapable.wantsVision(
                useVisionWhenAvailable: true, isVisionCapable: false) == false)
    }
}
