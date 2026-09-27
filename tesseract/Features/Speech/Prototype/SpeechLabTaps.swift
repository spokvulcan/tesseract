//
//  SpeechLabTaps.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  Two decorators the composition root wraps around the coordinator's own
//  collaborators, so the prototype sees every utterance without touching
//  `SpeechCoordinator`:
//
//  - `SpeechLabRecordingPlayback` forwards to the real playback manager and
//    copies each utterance's samples into a take (history, replay, export).
//  - `SpeechLabHighlightFanOut` forwards to the chosen overlay (the new
//    island/captions, today's notch as "Classic", or nothing) and feeds the
//    read-along clock the page highlights from.
//

import Foundation

@MainActor
final class SpeechLabRecordingPlayback: AudioPlayback {
    private let inner: AudioPlaybackManager
    private let lab: SpeechLab
    private var coordinatorFinished: (@MainActor @Sendable () -> Void)?

    init(wrapping inner: AudioPlaybackManager, lab: SpeechLab) {
        self.inner = inner
        self.lab = lab
        inner.onPlaybackFinished = { [weak self] in self?.playbackFinished() }
        lab.liveRateSink = { [weak inner] rate in inner?.setPlaybackRate(rate) }
    }

    var onPlaybackFinished: (@MainActor @Sendable () -> Void)? {
        get { coordinatorFinished }
        set { coordinatorFinished = newValue }
    }

    private func playbackFinished() {
        lab.tapPlaybackFinished()
        coordinatorFinished?()
    }

    var totalScheduledDuration: TimeInterval { inner.totalScheduledDuration }

    func play(samples: [Float], sampleRate: Int) {
        inner.play(samples: samples, sampleRate: sampleRate)
    }

    func startStreaming(sampleRate: Int) {
        lab.tapStartStreaming(sampleRate: sampleRate)
        inner.startStreaming(sampleRate: sampleRate)
        inner.setPlaybackRate(lab.playbackRate)
    }

    func appendChunk(samples: [Float]) {
        lab.tapAppend(samples: samples)
        inner.appendChunk(samples: samples)
    }

    func finishStreaming() {
        lab.tapFinishStreaming()
        inner.finishStreaming()
    }

    func pause() { inner.pause() }
    func resume() { inner.resume() }
    func currentPlaybackTime() -> TimeInterval { inner.currentPlaybackTime() }

    func stop() {
        lab.tapStop()
        inner.stop()
    }

    func playbackLevel() -> Float { inner.playbackLevel() }
    func setVolume(_ volume: Float) { inner.setVolume(volume) }
    var volume: Float { inner.volume }
}

@MainActor
final class SpeechLabHighlightFanOut: WordHighlightSurface {
    private let classic: any WordHighlightSurface
    private let lab: SpeechLab
    private var routesToClassic = false

    init(classic: any WordHighlightSurface, lab: SpeechLab) {
        self.classic = classic
        self.lab = lab
    }

    func show(
        text: String, tokenCharOffsets: [Int], playbackTimeProvider: @escaping () -> TimeInterval
    ) {
        lab.tapSegment(text: text, base: 0)
        lab.readAlong.startLiveClock(playbackTimeProvider)
        let style = lab.overlayStyleForLiveUtterance()
        routesToClassic = style == .classic
        switch style {
        case .classic:
            classic.show(
                text: text, tokenCharOffsets: tokenCharOffsets,
                playbackTimeProvider: playbackTimeProvider)
        case .island, .captions:
            lab.overlayController.utteranceBegan(style: style)
        case .off:
            break
        }
    }

    func switchText(_ text: String, tokenCharOffsets: [Int], segmentBase: TimeInterval) {
        lab.tapSegment(text: text, base: segmentBase)
        if routesToClassic {
            classic.switchText(text, tokenCharOffsets: tokenCharOffsets, segmentBase: segmentBase)
        }
    }

    func updateTotalDuration(_ duration: TimeInterval) {
        lab.tapScheduledDuration(duration)
        if routesToClassic { classic.updateTotalDuration(duration) }
    }

    func markSegmentComplete() {
        if routesToClassic { classic.markSegmentComplete() }
    }

    func markGenerationComplete() {
        lab.tapGenerationComplete()
        if routesToClassic { classic.markGenerationComplete() }
    }

    func dismiss() {
        lab.tapDismiss()
        classic.dismiss()
        routesToClassic = false
    }
}

extension SpeechLab {
    /// The overlay this utterance gets: the chosen style, unless the scope
    /// keeps the overlay for text from other apps and this came from here.
    func overlayStyleForLiveUtterance() -> OverlayPrefs.Style {
        if overlay.scope == .otherApps, liveSource != .selection { return .off }
        return overlay.style
    }
}
