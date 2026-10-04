//
//  PhoneAudioSession.swift
//  tesseract-ios
//
//  The app's audio session: spoken audio for playback, active only while
//  something reads, so music another app was playing comes back when
//  reading stops.
//

import AVFoundation
import UIKit

@MainActor
final class PhoneAudioSession {
    private var isActive = false

    func activate() {
        guard !isActive else { return }
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .spokenAudio)
            try session.setActive(true)
            isActive = true
        } catch {
            Log.speech.error("[Phone] audio session didn't activate: \(error)")
        }
    }

    func deactivate() {
        guard isActive else { return }
        isActive = false
        do {
            try AVAudioSession.sharedInstance().setActive(
                false, options: .notifyOthersOnDeactivation)
        } catch {
            Log.speech.error("[Phone] audio session didn't deactivate: \(error)")
        }
    }
}

/// The Mac's playback adapter with the session around it: the session turns
/// on as playback starts and off when it stops or drains.
@MainActor
final class SessionPlayback: AudioPlayback {
    private let player = AudioPlaybackManager()
    private let session: PhoneAudioSession

    var onPlaybackFinished: (@MainActor @Sendable () -> Void)?

    init(session: PhoneAudioSession) {
        self.session = session
        player.onPlaybackFinished = { [weak self] in
            guard let self else { return }
            self.session.deactivate()
            self.onPlaybackFinished?()
        }
    }

    var totalScheduledDuration: TimeInterval { player.totalScheduledDuration }

    func play(samples: [Float], sampleRate: Int) {
        session.activate()
        player.play(samples: samples, sampleRate: sampleRate)
    }

    func startStreaming(sampleRate: Int) {
        session.activate()
        player.startStreaming(sampleRate: sampleRate)
    }

    func appendChunk(samples: [Float]) { player.appendChunk(samples: samples) }
    func finishStreaming() { player.finishStreaming() }
    func pause() { player.pause() }

    func resume() {
        session.activate()
        player.resume()
    }

    func currentPlaybackTime() -> TimeInterval { player.currentPlaybackTime() }
    func heardPlaybackTime() -> TimeInterval { player.heardPlaybackTime() }

    func stop() {
        player.stop()
        session.deactivate()
    }

    func setPlaybackRate(_ rate: Float) { player.setPlaybackRate(rate) }
}

/// The speech hotkey's text source, which the phone has no hotkey for: what
/// the pasteboard holds.
@MainActor
final class PasteboardTextExtractor: TextExtracting {
    func extractSelectedText() async throws -> String {
        UIPasteboard.general.string ?? ""
    }
}
