import Foundation

/// Which speculative-decoding drafters a model load may attach — the
/// "Speculative Decoding" picker in agent settings. A load-time policy, not a
/// per-request one: each case gates the corresponding drafter loader in
/// `LLMActor.loadModel`, and the generation-time preference between loaded
/// drafters (DFlash2 over MTP) stays in the engagement policies.
nonisolated enum SpeculationMode: String, CaseIterable, Sendable {
    /// Load every drafter the checkpoint supports. DFlash2 wins at
    /// generation time when both attach: deeper blocks (8 vs 2 effective)
    /// and it speculates under sampling presets too.
    case automatic
    /// Only the MTP head (greedy-only, models that ship `mtp.*` weights).
    case mtp
    /// Only the DFlash2 draft (Qwen3.8-27B with its draft downloaded).
    case dflash2
    /// No drafters — plain autoregressive decoding.
    case off

    var allowsMTP: Bool { self == .automatic || self == .mtp }
    var allowsDFlash2: Bool { self == .automatic || self == .dflash2 }

    /// Picker label in agent settings.
    var displayName: String {
        switch self {
        case .automatic: "Automatic"
        case .mtp: "MTP Only"
        case .dflash2: "DFlash2 Only"
        case .off: "Off"
        }
    }
}

/// The speculative algorithm that actually decoded a request — resolved per
/// request at iterator construction, unlike ``SpeculationMode`` which is the
/// load-time policy. Absence (a `nil` field) means plain autoregressive
/// decoding: drafters may be loaded and still not engage (warm cache hits,
/// image prompts, non-identity key spaces).
nonisolated enum SpeculativeArm: String, Sendable, Equatable {
    case mtp
    case dflash2

    /// Short badge text for the activity surfaces.
    var displayName: String {
        switch self {
        case .mtp: "MTP"
        case .dflash2: "DFlash2"
        }
    }
}
