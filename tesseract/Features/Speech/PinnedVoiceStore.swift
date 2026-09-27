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

    /// The store bound. A voice is a few KB.
    private static let maxVoices = 32

    private let storageURL: URL
    /// Least recently saved first.
    private var entries: [Entry]

    /// - Parameter directory: storage directory; defaults to the app-support
    ///   home the other small stores use. Injectable for tests.
    init(directory: URL? = nil) {
        let base =
            directory
            ?? StorageEnvironment.applicationSupport.appendingPathComponent(
                "Tesseract Agent", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.storageURL = base.appendingPathComponent("pinned_voices.json")
        self.entries =
            (try? JSONDecoder().decode([Entry].self, from: Data(contentsOf: storageURL))) ?? []
    }

    func voice(description: String?, language: String, model: TTSModelSpec) -> PinnedVoice? {
        let key = Self.key(
            fingerprint: model.fingerprint, language: language, description: description)
        guard let entry = entries.last(where: { $0.key == key }) else { return nil }
        // A voice from an older schema can't condition this engine; it reads
        // as absent and the next take replaces it.
        return try? PinnedVoice(validating: Data(entry.voice.utf8))
    }

    /// Stores `voice` under its own checkpoint, language and description,
    /// replacing the take kept there. A take that is already stored is not
    /// written again.
    func save(_ voice: PinnedVoice) {
        guard let data = try? voice.serialized(), let json = String(data: data, encoding: .utf8)
        else { return }
        let key = Self.key(
            fingerprint: voice.modelFingerprint, language: voice.language ?? "",
            description: voice.voiceDescription)
        guard entries.last(where: { $0.key == key })?.voice != json else { return }
        entries.removeAll { $0.key == key }
        entries.append(Entry(key: key, voice: json))
        if entries.count > Self.maxVoices {
            entries.removeFirst(entries.count - Self.maxVoices)
        }
        guard let encoded = try? JSONEncoder().encode(entries) else { return }
        try? encoded.write(to: storageURL, options: .atomic)
    }

    private static func key(fingerprint: String, language: String, description: String?) -> String {
        "\(fingerprint)|\(language)|\(description ?? "")"
    }
}
