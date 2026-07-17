//
//  CaptureRig.swift — a VPIO capture engine wired with the voice hold's
//  exact stopped-engine discipline (ADR-0041):
//
//    1. `setVoiceProcessingEnabled(true)` on the fresh, stopped engine;
//    2. formats queried only AFTER enabling (VP changes them);
//    3. the tap installed once, stopped;
//    4. the render side (mainMixer → output) connected with the IO format
//       pin verified by read-back, stopped;
//    5. `start()` last; while running only gate flips, player scheduling,
//       and node volume — never tap or format work.
//
//  The rig is the lab's stand-in for `AudioCaptureEngine` under a hold; the
//  discipline (not the code) is what it shares with the app.
//

import AVFoundation

final class CaptureRig {

    let engine = AVAudioEngine()
    let trace: LevelTrace
    let player = AVAudioPlayerNode()

    /// The Capture Gate: flipped while running, read by the tap block.
    final class Gate: @unchecked Sendable { var isOpen = true }
    let gate = Gate()

    private(set) var voiceProcessing = false
    private(set) var renderWired = false
    private(set) var configChangeCount = 0
    private var observer: NSObjectProtocol?

    let inputFormat: AVAudioFormat
    let playbackFormat: AVAudioFormat

    /// Wires the full hold graph, stopped. `hostPlayback` attaches the
    /// persistent player (Apple-sample-shaped: before start).
    init(hostPlayback: Bool, playbackSampleRate: Double = 24_000) throws {
        do {
            try engine.inputNode.setVoiceProcessingEnabled(true)
            voiceProcessing = true
        } catch {
            print("  [rig] VP refused (\(error.localizedDescription)) — raw capture")
        }

        // Formats only after the VP arm — enabling changes them.
        inputFormat = engine.inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0 else {
            throw LabError("input node reports no format (wedged device?)")
        }
        guard
            let tapFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: inputFormat.sampleRate,
                channels: 1, interleaved: false),
            let playFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: playbackSampleRate,
                channels: 1, interleaved: false)
        else { throw LabError("could not build mono Float32 formats") }
        playbackFormat = playFormat

        trace = LevelTrace(sampleRate: inputFormat.sampleRate)
        let trace = trace
        let gate = gate
        engine.inputNode.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) {
            buffer, _ in
            if gate.isOpen { trace.ingest(buffer) }
        }

        // Render side with the VP pin verified by read-back
        // (AVAudioIONode.h: input-node output format == output-node input
        // format, stopped-only).
        let ioFormat = engine.inputNode.outputFormat(forBus: 0)
        engine.connect(engine.mainMixerNode, to: engine.outputNode, format: ioFormat)
        let readBack = engine.outputNode.inputFormat(forBus: 0)
        renderWired =
            readBack.sampleRate == ioFormat.sampleRate
            && readBack.channelCount == ioFormat.channelCount
        if !renderWired {
            print(
                """
                  [rig] render pin mismatch \
                  (\(readBack.sampleRate)/\(readBack.channelCount) vs \
                  \(ioFormat.sampleRate)/\(ioFormat.channelCount)) — input-only
                """)
            engine.disconnectNodeOutput(engine.mainMixerNode)
        }

        if hostPlayback, renderWired {
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: playFormat)
        }

        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
        ) { [weak self] _ in
            self?.configChangeCount += 1
        }

        engine.prepare()
        try engine.start()
    }

    /// Schedules mono samples on the hosted player and plays.
    func play(_ samples: [Float], completion: (@Sendable () -> Void)? = nil) {
        guard let buffer = Self.buffer(samples, format: playbackFormat) else { return }
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
            completion?()
        }
        player.play()
    }

    func stop() {
        if let observer { NotificationCenter.default.removeObserver(observer) }
        observer = nil
        player.stop()
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
    }

    static func buffer(_ samples: [Float], format: AVAudioFormat) -> AVAudioPCMBuffer? {
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return nil }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { pointer in
            buffer.floatChannelData![0].update(from: pointer.baseAddress!, count: samples.count)
        }
        return buffer
    }
}

/// A plain second engine — the app's dedicated `AudioPlaybackManager` path.
final class DedicatedPlayer {
    let engine = AVAudioEngine()
    let player = AVAudioPlayerNode()
    let format: AVAudioFormat

    init(sampleRate: Double = 24_000) throws {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                channels: 1, interleaved: false)
        else { throw LabError("could not build playback format") }
        self.format = format
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        engine.prepare()
        try engine.start()
    }

    func play(_ samples: [Float], completion: (@Sendable () -> Void)? = nil) {
        guard let buffer = CaptureRig.buffer(samples, format: format) else { return }
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
            completion?()
        }
        player.play()
    }

    func stop() {
        player.stop()
        engine.stop()
    }
}

struct LabError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
