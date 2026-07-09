//
//  OnboardingModelPickTests.swift
//  tesseractTests
//
//  The hardware-aware zero-UI model pick of the Onboarding Tour (PRD #171,
//  ADR-0021): physical RAM in, recommended agent-tier model id out. Pure, in
//  the ModelCatalog style — prior art: `ModelCatalogTests`. Expected tiers come
//  from the catalogue's own stated requirements: the 48 GB+ tier is the
//  35B-A3B MoE PARO (#228), the 9B tier "~10 GB with voice", the 4B tier is the
//  floor default.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct OnboardingModelPickTests {

    private static let gib: UInt64 = 1 << 30

    @Test(arguments: [8, 16, 18] as [UInt64])
    func smallMachinesGetTheCompactTier(gigabytes: UInt64) {
        #expect(
            OnboardingModelPick.recommendedAgentModelID(
                physicalMemoryBytes: gigabytes * Self.gib)
                == "qwen3.5-4b-paro")
    }

    @Test(arguments: [24, 32, 36] as [UInt64])
    func midMachinesGetTheNineBTier(gigabytes: UInt64) {
        #expect(
            OnboardingModelPick.recommendedAgentModelID(
                physicalMemoryBytes: gigabytes * Self.gib)
                == "qwen3.5-9b-paro")
    }

    @Test(arguments: [48, 64, 128, 512] as [UInt64])
    func bigMachinesGetTheMoETier(gigabytes: UInt64) {
        #expect(
            OnboardingModelPick.recommendedAgentModelID(
                physicalMemoryBytes: gigabytes * Self.gib)
                == "qwen3.6-35b-a3b-paro")
    }

    @Test func boundariesBelongToTheHigherTier() {
        #expect(
            OnboardingModelPick.recommendedAgentModelID(
                physicalMemoryBytes: 24 * Self.gib - 1)
                == "qwen3.5-4b-paro")
        #expect(
            OnboardingModelPick.recommendedAgentModelID(
                physicalMemoryBytes: 48 * Self.gib - 1)
                == "qwen3.5-9b-paro")
    }

    /// Every recommendable id must be a real agent-category catalogue entry —
    /// the pick can never point the download queue at nothing.
    @Test func everyRecommendationExistsInTheAgentCatalogue() {
        for gigabytes in [4, 8, 16, 24, 32, 48, 96, 192] as [UInt64] {
            let id = OnboardingModelPick.recommendedAgentModelID(
                physicalMemoryBytes: gigabytes * Self.gib)
            let definition = ModelDefinition.withID(id)
            #expect(definition != nil, "\(gigabytes) GiB → \(id) not in catalogue")
            #expect(definition?.category == .agent)
        }
    }
}
