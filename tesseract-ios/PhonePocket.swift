//
//  PhonePocket.swift
//  tesseract-ios
//
//  The system's side of reading in the pocket (#515): the audio session's
//  interruptions and route changes, the audio engine's configuration
//  changes, the app leaving the screen, the phone's temperature, the lock
//  screen's commands, and what Now Playing shows. Each becomes a
//  `PocketEvent` for the shared `PocketControls`, which decides.
//

import AVFoundation
import MediaPlayer
import Observation
import UIKit

@MainActor
final class PhonePocket {
    private let controls: PocketControls
    private let reading: PhoneReading
    private let library: ReaderLibrary
    private let settings: PhoneSettings
    private var observers: [NSObjectProtocol] = []
    private var outputChange: Task<Void, Never>?

    init(
        controls: PocketControls, reading: PhoneReading, library: ReaderLibrary,
        settings: PhoneSettings
    ) {
        self.controls = controls
        self.reading = reading
        self.library = library
        self.settings = settings
    }

    func start() {
        guard observers.isEmpty else { return }
        observe(AVAudioSession.interruptionNotification) { [weak self] info in
            guard let raw = info[AVAudioSessionInterruptionTypeKey],
                let type = AVAudioSession.InterruptionType(rawValue: raw)
            else { return }
            switch type {
            case .began:
                self?.controls.handle(.interruptionBegan)
            case .ended:
                let options = AVAudioSession.InterruptionOptions(
                    rawValue: info[AVAudioSessionInterruptionOptionKey] ?? 0)
                self?.controls.handle(
                    .interruptionEnded(shouldResume: options.contains(.shouldResume)))
            @unknown default:
                break
            }
        }
        observe(AVAudioSession.routeChangeNotification) { [weak self] info in
            guard let raw = info[AVAudioSessionRouteChangeReasonKey],
                AVAudioSession.RouteChangeReason(rawValue: raw) == .oldDeviceUnavailable
            else { return }
            self?.controls.handle(.outputLost)
        }
        observe(.AVAudioEngineConfigurationChange) { [weak self] _ in
            // A lost route also changes the engine; give its pause a moment
            // to land first, so unplugged headphones never restart on the
            // speaker.
            self?.outputChange?.cancel()
            self?.outputChange = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .milliseconds(300))
                guard !Task.isCancelled else { return }
                self?.controls.handle(.outputChanged)
            }
        }
        observe(UIApplication.didEnterBackgroundNotification) { [weak self] _ in
            self?.controls.handle(.movedToBackground)
        }
        observe(ProcessInfo.thermalStateDidChangeNotification) { [weak self] _ in
            self?.thermalStateChanged()
        }
        setUpRemoteCommands()
        followTheReading()
    }

    /// Runs `handle` on the main actor for each `name` notification, with
    /// the notification's number values (all these notifications carry).
    private func observe(
        _ name: Notification.Name, _ handle: @escaping @MainActor ([String: UInt]) -> Void
    ) {
        observers.append(
            NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) {
                note in
                var info: [String: UInt] = [:]
                for (key, value) in note.userInfo ?? [:] {
                    if let key = key as? String, let value = value as? UInt { info[key] = value }
                }
                MainActor.assumeIsolated { handle(info) }
            })
    }

    // MARK: - Heat

    /// Critical stops the reading and says why. Serious hands the reading to
    /// the System Voice, which is the only voice on the phone until the
    /// neural one arrives (#515, slice 4).
    private func thermalStateChanged() {
        let decision = ThermalPolicy.decision(for: ProcessInfo.processInfo.thermalState)
        reading.notice = ThermalPolicy.notice(for: decision)
        if decision == .pause, let reader = reading.current?.reader, reader.isReading {
            reader.stop()
        }
    }

    // MARK: - Remote commands

    private func setUpRemoteCommands() {
        let center = MPRemoteCommandCenter.shared()
        let commands: [(MPRemoteCommand, RemoteCommand)] = [
            (center.playCommand, .play), (center.pauseCommand, .pause),
            (center.togglePlayPauseCommand, .togglePlayPause),
            (center.previousTrackCommand, .previousTrack), (center.nextTrackCommand, .nextTrack),
        ]
        for (command, remote) in commands {
            command.isEnabled = true
            command.addTarget { [weak self] _ in
                MainActor.assumeIsolated { self?.controls.handle(.remote(remote)) }
                return .success
            }
        }
        center.changePlaybackPositionCommand.isEnabled = true
        center.changePlaybackPositionCommand.addTarget { [weak self] event in
            guard let event = event as? MPChangePlaybackPositionCommandEvent else {
                return .commandFailed
            }
            return MainActor.assumeIsolated {
                guard let self, let duration = self.lastDuration, duration > 0 else {
                    return .commandFailed
                }
                self.controls.handle(.remote(.seek(event.positionTime / duration)))
                return .success
            }
        }
        // Previous and next mean a sentence, so the interval skips stay off.
        center.skipForwardCommand.isEnabled = false
        center.skipBackwardCommand.isEnabled = false
    }

    // MARK: - Now Playing

    /// The whole reading's length at 1×, as Now Playing last showed it.
    private var lastDuration: TimeInterval?

    /// Updates Now Playing whenever the text, its sentence or its state
    /// changes: a sentence at a time, not a word.
    private func followTheReading() {
        withObservationTracking {
            if let current = reading.current {
                _ = current.reader.isReading
                _ = current.reader.isPaused
                _ = current.reader.bookmark
            }
            _ = settings.ttsPlaybackRate
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in
                self?.updateNowPlaying()
                self?.followTheReading()
            }
        }
        updateNowPlaying()
    }

    private func updateNowPlaying() {
        guard let (id, reader) = reading.current else { return }
        let rate = max(settings.ttsPlaybackRate, 0.5)
        // Time left at 1×, and the whole text's length from how far it is.
        let left = reader.timeLeft * rate
        let duration = left / max(1 - reader.progress, 0.001)
        lastDuration = duration
        let isPlaying = reader.isReading && !reader.isPaused
        MPNowPlayingInfoCenter.default().nowPlayingInfo = [
            MPMediaItemPropertyTitle: library.entry(id)?.title ?? "Tesseract",
            MPMediaItemPropertyArtist: "Tesseract",
            MPMediaItemPropertyPlaybackDuration: duration,
            MPNowPlayingInfoPropertyElapsedPlaybackTime: max(duration - left, 0),
            MPNowPlayingInfoPropertyPlaybackRate: isPlaying ? rate : 0,
            MPNowPlayingInfoPropertyDefaultPlaybackRate: 1.0,
            MPNowPlayingInfoPropertyMediaType: MPNowPlayingInfoMediaType.audio.rawValue,
        ]
    }
}
