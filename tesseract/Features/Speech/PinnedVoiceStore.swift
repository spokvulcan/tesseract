//
//  PinnedVoiceStore.swift
//  tesseract
//

import Foundation
import TesseractSpeech

/// The **Pinned Voice** of each designed voice, on disk, so a voice is the same
/// person after a relaunch (ADR-0072). Keyed by checkpoint, language and
/// description: a Reference Take only conditions the checkpoint and precision
/// that rendered it, and changing the description designs a new voice.
@MainActor
final class PinnedVoiceStore {
    private struct Entry: Codable {
        let key: String
        let voice: String
    }

    private let storageURL: URL
    private let maxVoices: Int
    /// Least recently saved first.
    private var entries: [Entry]

    /// - Parameters:
    ///   - directory: storage directory; defaults to the app-support home the
    ///     other small stores use. Injectable for tests.
    ///   - maxVoices: the store bound. A voice is a few KB.
    init(directory: URL? = nil, maxVoices: Int = 32) {
        let base =
            directory
            ?? FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first?.appendingPathComponent("Tesseract Agent", isDirectory: true)
            ?? FileManager.default.temporaryDirectory
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.storageURL = base.appendingPathComponent("pinned_voices.json")
        self.maxVoices = maxVoices
        self.entries =
            (try? JSONDecoder().decode([Entry].self, from: Data(contentsOf: storageURL))) ?? []
    }

    func voice(description: String?, language: String, model: TTSModelSpec) -> PinnedVoice? {
        let key = Self.key(description: description, language: language, model: model)
        guard let entry = entries.last(where: { $0.key == key }) else { return nil }
        // A voice from an older schema can't condition this engine; it reads
        // as absent and the next take replaces it.
        return try? PinnedVoice(validating: Data(entry.voice.utf8))
    }

    func save(_ voice: PinnedVoice, description: String?, language: String, model: TTSModelSpec) {
        guard let data = try? voice.serialized(), let json = String(data: data, encoding: .utf8)
        else { return }
        let key = Self.key(description: description, language: language, model: model)
        entries.removeAll { $0.key == key }
        entries.append(Entry(key: key, voice: json))
        if entries.count > maxVoices {
            entries.removeFirst(entries.count - maxVoices)
        }
        guard let encoded = try? JSONEncoder().encode(entries) else { return }
        try? encoded.write(to: storageURL, options: .atomic)
    }

    private static func key(description: String?, language: String, model: TTSModelSpec) -> String {
        "\(model.fingerprint)|\(language)|\(description ?? "")"
    }
}
