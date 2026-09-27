//
//  VoiceTakePlayer.swift
//  tesseract
//
//  Replays a take from the voice designer's strip, so takes can be compared
//  before one is kept. Takes are a sentence or two, held in memory.
//

import AVFoundation
import Observation

@Observable @MainActor
final class VoiceTakePlayer: NSObject, AVAudioPlayerDelegate {
    private(set) var playingID: UUID?
    @ObservationIgnored private var player: AVAudioPlayer?

    func toggle(_ take: VoiceTake) {
        if playingID == take.id {
            stop()
            return
        }
        stop()
        guard
            let player = try? AVAudioPlayer(
                data: Self.wav(take.samples, sampleRate: take.sampleRate))
        else { return }
        player.delegate = self
        player.play()
        self.player = player
        playingID = take.id
    }

    func stop() {
        player?.stop()
        player = nil
        playingID = nil
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self] in
            self?.player = nil
            self?.playingID = nil
        }
    }

    /// 16-bit PCM mono WAV in memory.
    private static func wav(_ samples: [Float], sampleRate: Int) -> Data {
        let byteCount = samples.count * 2
        var data = Data(capacity: 44 + byteCount)
        func put<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: Array("RIFF".utf8))
        put(UInt32(36 + byteCount))
        data.append(contentsOf: Array("WAVEfmt ".utf8))
        put(UInt32(16))
        put(UInt16(1))
        put(UInt16(1))
        put(UInt32(sampleRate))
        put(UInt32(sampleRate * 2))
        put(UInt16(2))
        put(UInt16(16))
        data.append(contentsOf: Array("data".utf8))
        put(UInt32(byteCount))
        for sample in samples {
            put(Int16(max(-1, min(1, sample)) * 32_767))
        }
        return data
    }
}
