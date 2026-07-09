//
//  OnboardingModelPick.swift
//  tesseract
//

import Foundation

/// The Onboarding Tour's hardware-aware zero-UI model pick (ADR-0021): physical
/// RAM in, recommended agent-tier model id out. Pure, in the `ModelCatalog`
/// style — data in, answers out; the caller supplies the machine's RAM.
///
/// Tier boundaries derive from the catalogue's own stated requirements: the
/// 48 GB+ tier is the Qwen3.6-35B-A3B MoE PARO (#228 — in-app it prefills
/// faster than the unsloth build, decodes equal, runs 1.8 GB lighter, and is
/// vision-capable; it replaced the dense 27B PARO the tour launched with once
/// #217 removed the expected decode trade-off); the 9B tier "~10 GB with
/// voice", comfortable from 24 GB up; below that, the compact 4B default.
nonisolated enum OnboardingModelPick {

    private static let gib: UInt64 = 1 << 30

    static func recommendedAgentModelID(physicalMemoryBytes: UInt64) -> String {
        if physicalMemoryBytes >= 48 * gib {
            return "qwen3.6-35b-a3b-paro"
        }
        if physicalMemoryBytes >= 24 * gib {
            return "qwen3.5-9b-paro"
        }
        // ModelDefinition.defaultAgentModelID, spelled out because that static
        // is MainActor-isolated; the catalogue-existence test pins the literal.
        return "qwen3.5-4b-paro"
    }
}
