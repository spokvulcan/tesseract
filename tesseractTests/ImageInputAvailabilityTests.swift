//
//  ImageInputAvailabilityTests.swift
//  tesseractTests
//
//  Pins the image-input-availability projection (ADR-0013, PRD #112): the pure
//  decision the composer uses to show or hide image affordances. No model
//  loading, no capability probe — the projection takes plain booleans so it is
//  testable over the full (vision-capable, setting-on) matrix without a
//  capability probe or model fixture.
//

import Testing

@testable import Tesseract_Agent

struct ImageInputAvailabilityTests {

    // MARK: - The full matrix

    @Test
    func affordanceShownOnlyWhenCapableAndSettingOn() {
        #expect(
            ImageInputAvailability.showImageAffordance(
                isVisionCapable: true, useVisionWhenAvailable: true) == true)
    }

    @Test
    func capableButOptedOutHidesAffordance() {
        // The user turned the global opt-out off → chat runs the text-only
        // container, so images would silently drop. Hide the affordance.
        #expect(
            ImageInputAvailability.showImageAffordance(
                isVisionCapable: true, useVisionWhenAvailable: false) == false)
    }

    @Test
    func textOnlyModelHidesAffordanceRegardlessOfSetting() {
        #expect(
            ImageInputAvailability.showImageAffordance(
                isVisionCapable: false, useVisionWhenAvailable: true) == false)
        #expect(
            ImageInputAvailability.showImageAffordance(
                isVisionCapable: false, useVisionWhenAvailable: false) == false)
    }
}
