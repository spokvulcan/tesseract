import Foundation
import MLXLMCommon

/// Spools a request's audio clips to temp files for the duration of one
/// `prepare` call. The vendor's `UserInput.Audio.url` decode is the single
/// tested container-parsing + 16 kHz-mono-resample path (AVFoundation), and
/// it reads from a URL — so each clip's exact received bytes land in a
/// per-request temp directory, named by digest with the client-declared
/// container extension (sanitized at the wire boundary by
/// `MessageConverter.convertAudioContent`).
///
/// Lifetime is the keying phase: create, hand `userInputAudios` to
/// `prepare`, `cleanUp()` in a defer. Files are per-request (no shared
/// cache) — clip payloads are small next to the tensors prepare builds from
/// them, and a private directory per request means no cross-request
/// lifetime reasoning.
nonisolated struct AudioClipSpool {
    let userInputAudios: [UserInput.Audio]
    private let directory: URL?

    init(clips: [HTTPPrefixCacheAudio]) throws {
        guard !clips.isEmpty else {
            self.userInputAudios = []
            self.directory = nil
            return
        }
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("audio-spool-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var audios: [UserInput.Audio] = []
        audios.reserveCapacity(clips.count)
        for clip in clips {
            let file = directory.appendingPathComponent(
                "\(clip.digest.hexString).\(clip.format)")
            // Two clips with equal digests are byte-equal — the first write
            // already holds the right content.
            if !FileManager.default.fileExists(atPath: file.path) {
                try clip.data.write(to: file)
            }
            audios.append(.url(file))
        }
        self.userInputAudios = audios
        self.directory = directory
    }

    func cleanUp() {
        guard let directory else { return }
        try? FileManager.default.removeItem(at: directory)
    }
}
