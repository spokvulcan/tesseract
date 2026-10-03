//
//  VoiceLibrary.swift
//  tesseract
//
//  The voices the Speech page offers: built-in designs, the ones the owner
//  saved under a name, and designed voices already pinned on disk that were
//  never named. Choosing one sets the settings voice; the coordinator opens
//  it pinned to its Reference Take when there is one (ADR-0072).
//

import Foundation
import Observation

/// A designed voice the owner named and saved.
nonisolated struct SavedVoice: Codable, Sendable, Hashable, Identifiable {
    let id: UUID
    var name: String
    var description: String

    init(id: UUID = UUID(), name: String, description: String) {
        self.id = id
        self.name = name
        self.description = description
    }
}

/// One voice as the page shows it.
nonisolated struct VoiceOption: Identifiable, Hashable, Sendable {
    enum Kind: Hashable, Sendable {
        case builtIn
        case saved
        /// Designed and pinned, never named.
        case unnamed
        /// A CustomVoice checkpoint's own speaker.
        case preset
    }

    var id: String { description }
    let name: String
    /// What the engine is given: the design, or a preset's speaker name.
    let description: String
    /// A line under the name: what it sounds like.
    let detail: String
    let kind: Kind

    var isEditable: Bool { kind == .saved || kind == .unnamed }
}

@Observable @MainActor
final class VoiceLibrary {
    private let settings: any SpeechSettings
    private let pinnedVoices: PinnedVoiceStore
    let source: TTSVoiceSource

    /// Pinned voices without a name, most recent first; refreshed on demand.
    private(set) var unnamed: [VoiceOption] = []

    init(
        settings: any SpeechSettings, pinnedVoices: PinnedVoiceStore,
        source: TTSVoiceSource = ModelDefinition.textToSpeechVoiceSource
    ) {
        self.settings = settings
        self.pinnedVoices = pinnedVoices
        self.source = source
        refresh()
    }

    // MARK: - Listing

    /// The owner's voices: saved ones first, then designed-but-unnamed.
    var yourVoices: [VoiceOption] {
        guard source.supportsVoiceDesign else { return [] }
        let saved = settings.savedVoices.map {
            VoiceOption(
                name: $0.name, description: $0.description,
                detail: Self.detail(for: $0.description), kind: .saved)
        }
        let named = Set(saved.map(\.description))
        return saved + unnamed.filter { !named.contains($0.description) }
    }

    /// The voices that come with the app, or the checkpoint's speakers.
    var builtIn: [VoiceOption] {
        switch source {
        case .designed: Self.designedBuiltIns
        case .presets(let speakers):
            speakers.map {
                VoiceOption(name: $0.capitalized, description: $0, detail: "", kind: .preset)
            }
        }
    }

    var currentDescription: String { settings.ttsVoiceDescription }

    var currentName: String {
        name(for: currentDescription)
    }

    func name(for description: String) -> String {
        if description.isEmpty { return "Default voice" }
        if let option = (yourVoices + builtIn).first(where: { $0.description == description }) {
            return option.name
        }
        return VoiceDesign.shortName(for: description)
    }

    /// Re-reads the pinned voices (after a new take is kept).
    func refresh() {
        guard source.supportsVoiceDesign else {
            unnamed = []
            return
        }
        let builtInDescriptions = Set(Self.designedBuiltIns.map(\.description))
        var seen = Set<String>()
        unnamed = pinnedVoices.designedVoices(model: ModelDefinition.textToSpeechModelSpec)
            .compactMap { entry in
                guard !builtInDescriptions.contains(entry.description),
                    seen.insert(entry.description).inserted
                else { return nil }
                return VoiceOption(
                    name: VoiceDesign.shortName(for: entry.description),
                    description: entry.description, detail: Self.detail(for: entry.description),
                    kind: .unnamed)
            }
    }

    /// The owner's voices: what their description names, so near-identical
    /// designs read apart, or the description itself.
    private static func detail(for description: String) -> String {
        let summary = VoiceDesign.summary(of: description)
        return summary.isEmpty ? description : summary
    }

    // MARK: - Changes

    func select(_ voice: VoiceOption) {
        settings.ttsVoiceDescription = voice.description
    }

    /// Saves `description` under `name`, replacing an earlier save of it.
    func save(name: String, description: String) {
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        guard source.supportsVoiceDesign, !trimmed.isEmpty else { return }
        let display = name.trimmingCharacters(in: .whitespacesAndNewlines)
        var saved = settings.savedVoices.filter { $0.description != trimmed }
        saved.insert(
            SavedVoice(
                name: display.isEmpty ? VoiceDesign.shortName(for: trimmed) : display,
                description: trimmed),
            at: 0)
        settings.savedVoices = saved
    }

    func rename(_ voice: VoiceOption, to name: String) {
        let display = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !display.isEmpty else { return }
        if let index = settings.savedVoices.firstIndex(where: {
            $0.description == voice.description
        }) {
            settings.savedVoices[index].name = display
        } else {
            save(name: display, description: voice.description)
        }
    }

    /// Forgets the voice and its takes. If it was in use, the page falls
    /// back to the first built-in voice.
    func delete(_ voice: VoiceOption) {
        guard voice.isEditable else { return }
        settings.savedVoices.removeAll { $0.description == voice.description }
        pinnedVoices.remove(
            description: voice.description, model: ModelDefinition.textToSpeechModelSpec)
        if settings.ttsVoiceDescription == voice.description {
            settings.ttsVoiceDescription = Self.designedBuiltIns.first?.description ?? ""
        }
        refresh()
    }

    // MARK: - Built-in designs

    /// The first four keep the descriptions earlier versions shipped, so
    /// voices the owner already pinned stay the same person. The rest follow
    /// `VoiceDesign`'s shape and spread across who is speaking and timbre,
    /// which is what makes voices distinct.
    static let designedBuiltIns: [VoiceOption] = [
        VoiceOption(
            name: "Natural",
            description:
                "A natural, clear voice with a moderate pace and neutral tone, suitable for everyday conversations.",
            detail: "Clear and neutral, everyday pace", kind: .builtIn),
        VoiceOption(
            name: "Warm",
            description:
                "A warm, friendly female voice with a gentle tone and smooth cadence, comforting and approachable.",
            detail: "Friendly, gentle, smooth", kind: .builtIn),
        VoiceOption(
            name: "Deep",
            description:
                "A deep, resonant male narrator voice with a measured pace and authoritative presence.",
            detail: "Resonant narrator, measured", kind: .builtIn),
        VoiceOption(
            name: "Calm",
            description:
                "A calm, soothing voice with a slow, deliberate pace, perfect for reading and relaxation.",
            detail: "Slow and soothing, for long reads", kind: .builtIn),
        VoiceOption(
            name: "Bright",
            description:
                "Female, young adult, high pitch. A bright, clear voice, cheerful without laughing, speaking fluently at a brisk pace with clear articulation.",
            detail: "Young, quick, upbeat", kind: .builtIn),
        VoiceOption(
            name: "Anchor",
            description:
                "Male, middle-aged, medium pitch. A crisp, resonant voice, serious and composed, speaking fluently at a moderate, even pace with precise articulation.",
            detail: "Crisp diction, steady and formal", kind: .builtIn),
        VoiceOption(
            name: "Storyteller",
            description:
                "Male, senior, low pitch. A warm, slightly husky voice, gentle and engaged, speaking fluently at a slow, unhurried pace with clear articulation.",
            detail: "Older, husky, unhurried", kind: .builtIn),
        VoiceOption(
            name: "Velvet",
            description:
                "Female, middle-aged, low pitch. A deep, resonant voice, calm and steady, speaking fluently at a moderate, even pace with clear articulation.",
            detail: "Low and resonant, composed", kind: .builtIn),
        VoiceOption(
            name: "Mellow",
            description:
                "Male, young adult, medium pitch. A warm, mellow voice, calm and relaxed, speaking fluently at a moderate, even pace with clear articulation.",
            detail: "Young, relaxed, easy", kind: .builtIn),
        VoiceOption(
            name: "Close",
            description:
                "Female, young adult, medium pitch. A soft, breathy voice, gentle and quiet, speaking fluently at a slow pace with clear articulation.",
            detail: "Soft and close, for quiet reading", kind: .builtIn),
    ]
}
