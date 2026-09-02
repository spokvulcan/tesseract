//
//  WhipAudio.swift
//  tesseract
//
//  The whip's voice, synthesised rather than sampled.
//
//  A whoosh has to track tip speed continuously — it rises in pitch and volume
//  as the thong accelerates and falls away as it slows — and no fixed sample can
//  follow that. So the sound is generated from the same numbers that drive the
//  drawing: filtered noise whose centre frequency and gain are the tip's speed,
//  plus a short burst when the physics reports a crack. Sound and picture read
//  one simulation, so they cannot drift apart.
//
//  Nothing is shipped as an asset; there are no audio files in this repo and
//  this adds none.
//

import AVFoundation
import Foundation
import Synchronization

/// Live parameters handed from the main actor to the audio render thread.
/// Floats travel as bit patterns because `Atomic` needs an integer
/// representation; relaxed ordering is right here — every value is a smoothly
/// varying control signal where reading last tick's number is inaudible.
private nonisolated final class WhipAudioParameters: Sendable {
    let tipSpeed = Atomic<UInt32>(0)
    let pendingCrack = Atomic<UInt32>(0)
    let outputGain = Atomic<UInt32>(Float(0).bitPattern)

    func setTipSpeed(_ value: Float) {
        tipSpeed.store(value.bitPattern, ordering: .relaxed)
    }

    func setOutputGain(_ value: Float) {
        outputGain.store(value.bitPattern, ordering: .relaxed)
    }

    /// Queues a crack. Written from the main actor, consumed exactly once by the
    /// render thread, which zeroes it as it picks it up.
    func queueCrack(intensity: Float) {
        pendingCrack.store(intensity.bitPattern, ordering: .relaxed)
    }

    func takeCrack() -> Float {
        let raw = pendingCrack.exchange(0, ordering: .relaxed)
        return Float(bitPattern: raw)
    }

    var currentTipSpeed: Float { Float(bitPattern: tipSpeed.load(ordering: .relaxed)) }
    var currentOutputGain: Float { Float(bitPattern: outputGain.load(ordering: .relaxed)) }
}

/// DSP state owned exclusively by the render thread. Unchecked `Sendable`
/// because the audio unit guarantees single-threaded access to the render
/// block, and a lock on the audio thread is exactly what must not happen.
private nonisolated final class WhipAudioScratch: @unchecked Sendable {
    /// Chamberlin state-variable filter state for the whoosh.
    var low: Float = 0
    var band: Float = 0
    /// Separate filter for the crack, so a crack during a swing is not smeared
    /// by the whoosh filter's state.
    var crackLow: Float = 0
    var crackBand: Float = 0

    /// One-pole smoothed control signals, to keep parameter changes from
    /// stepping audibly between render buffers.
    var smoothedSpeed: Float = 0
    var smoothedGain: Float = 0

    /// Crack envelope: amplitude and its per-sample decay.
    var crackEnvelope: Float = 0
    var crackTailEnvelope: Float = 0

    var randomState: UInt32 = 0x9E37_79B9
}

/// Owns the engine and the synthesis. Starts on the whip's first motion and
/// stops when it falls asleep, so an idle whip holds no audio device awake.
@MainActor
final class WhipAudio {

    private let parameters = WhipAudioParameters()
    private var engine: AVAudioEngine?
    private var sourceNode: AVAudioSourceNode?

    /// Peak tip speed the loudness curve is normalised against, points/s.
    /// `nonisolated` because the render block on the audio thread reads it.
    private nonisolated static let referenceSpeed: Float = 3200

    /// Whether the synthesiser should make any sound at all — the feature's own
    /// setting AND the app-wide Play Sounds preference.
    var isEnabled = true {
        didSet {
            guard isEnabled != oldValue else { return }
            if !isEnabled { stop() }
        }
    }

    // MARK: - Lifecycle

    /// Boots the engine if it is not already running. Cheap to call every tick.
    func start() {
        guard isEnabled, engine == nil else { return }

        let engine = AVAudioEngine()
        // Mono at the hardware rate: the whip is a point source and this avoids
        // a sample-rate conversion on the render path.
        let outputFormat = engine.outputNode.inputFormat(forBus: 0)
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: outputFormat.sampleRate,
                channels: 1,
                interleaved: false)
        else {
            Log.whip.error("whip audio: could not build a render format")
            return
        }

        let node = Self.makeSourceNode(
            format: format,
            scratch: WhipAudioScratch(),
            parameters: parameters)

        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)

        do {
            try engine.start()
            self.engine = engine
            self.sourceNode = node
        } catch {
            Log.whip.error("whip audio: engine failed to start — \(error.localizedDescription)")
            engine.detach(node)
        }
    }

    /// Tears the engine down so the audio device can idle. The whip sleeps far
    /// more than it swings, and a permanently running engine was the one part of
    /// this feature with a real standing power cost.
    func stop() {
        guard let engine else { return }
        parameters.setTipSpeed(0)
        parameters.setOutputGain(0)
        engine.stop()
        if let sourceNode {
            engine.detach(sourceNode)
        }
        self.sourceNode = nil
        self.engine = nil
    }

    // MARK: - Per-tick updates

    /// Feeds one physics tick to the synthesiser.
    func update(tipSpeed: CGFloat, cracked: Bool, crackIntensity: CGFloat) {
        guard isEnabled else { return }
        parameters.setTipSpeed(Float(tipSpeed))
        parameters.setOutputGain(1)
        if cracked {
            parameters.queueCrack(intensity: Float(crackIntensity))
        }
    }

    /// Fades the voice out without tearing the engine down — used while the whip
    /// is settling, so the whoosh dies away rather than cutting off.
    func silence() {
        parameters.setTipSpeed(0)
    }

    // MARK: - Render

    /// Builds the source node from a `nonisolated` context, and that is the
    /// whole point of this function existing.
    ///
    /// This project compiles with `SWIFT_DEFAULT_ACTOR_ISOLATION=MainActor`, so
    /// a closure written inline inside `start()` inherits MainActor isolation.
    /// CoreAudio then calls it on the render thread, Swift's
    /// `swift_task_checkIsolated` asserts it is not on the main queue, and the
    /// process takes `EXC_BREAKPOINT` within a second of the first sound. The
    /// audio render block must be genuinely isolation-free.
    private nonisolated static func makeSourceNode(
        format: AVAudioFormat,
        scratch: WhipAudioScratch,
        parameters: WhipAudioParameters
    ) -> AVAudioSourceNode {
        let sampleRate = Float(format.sampleRate)
        return AVAudioSourceNode(format: format) { _, _, frameCount, audioBufferList in
            render(
                frameCount: frameCount,
                audioBufferList: audioBufferList,
                scratch: scratch,
                parameters: parameters,
                sampleRate: sampleRate)
        }
    }

    /// The audio thread. No allocation, no locks, no Swift runtime calls that
    /// could take one — everything it touches is preallocated scratch.
    private nonisolated static func render(
        frameCount: AVAudioFrameCount,
        audioBufferList: UnsafeMutablePointer<AudioBufferList>,
        scratch: WhipAudioScratch,
        parameters: WhipAudioParameters,
        sampleRate: Float
    ) -> OSStatus {
        let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
        let targetSpeed = parameters.currentTipSpeed
        let outputGain = parameters.currentOutputGain

        let newCrack = parameters.takeCrack()
        if newCrack > 0 {
            // Retrigger rather than accumulate: two cracks in a row should be
            // two distinct snaps, not one that grows.
            scratch.crackEnvelope = min(1, newCrack)
            scratch.crackTailEnvelope = min(1, newCrack) * 0.42
        }

        // One-pole smoothing coefficients. The speed follows quickly (the whoosh
        // must track the swing) while the gain lags slightly so the onset is not
        // a click.
        let speedCoefficient: Float = 1 - exp(-1 / (0.010 * sampleRate))
        let gainCoefficient: Float = 1 - exp(-1 / (0.030 * sampleRate))

        // Snap decay is fast — this is the crack itself. The tail is the room
        // it happened in, and is what stops it sounding like a click.
        let snapDecay: Float = exp(-1 / (0.011 * sampleRate))
        let tailDecay: Float = exp(-1 / (0.085 * sampleRate))

        for frame in 0..<Int(frameCount) {
            scratch.smoothedSpeed += (targetSpeed - scratch.smoothedSpeed) * speedCoefficient
            scratch.smoothedGain += (outputGain - scratch.smoothedGain) * gainCoefficient

            let normalised = min(scratch.smoothedSpeed / referenceSpeed, 1.6)

            // White noise, cheap and deterministic.
            scratch.randomState = scratch.randomState &* 1_664_525 &+ 1_013_904_223
            let noise = Float(Int32(bitPattern: scratch.randomState)) / Float(Int32.max)

            // Whoosh: band-passed noise. Centre frequency and loudness both ride
            // the tip's speed, which is the whole point — this is air moving past
            // a thin object, and it has no sound of its own when still.
            let centre = 240 + 2500 * normalised
            let f = 2 * sin(Float.pi * min(centre, sampleRate * 0.24) / sampleRate)
            let q: Float = 0.62

            scratch.low += f * scratch.band
            let high = noise - scratch.low - q * scratch.band
            scratch.band += f * high

            // Squared so quiet swings stay genuinely quiet and only a real
            // effort gets loud.
            let whooshGain = normalised * normalised * 0.30 * scratch.smoothedGain
            var sample = scratch.band * whooshGain

            // Crack: a brighter, much shorter burst on its own filter.
            if scratch.crackEnvelope > 0.0001 || scratch.crackTailEnvelope > 0.0001 {
                let crackCentre: Float = 2600
                let cf = 2 * sin(Float.pi * min(crackCentre, sampleRate * 0.24) / sampleRate)
                scratch.crackLow += cf * scratch.crackBand
                let crackHigh = noise - scratch.crackLow - 0.9 * scratch.crackBand
                scratch.crackBand += cf * crackHigh

                sample += crackHigh * scratch.crackEnvelope * 0.55 * scratch.smoothedGain
                sample +=
                    scratch.crackBand * scratch.crackTailEnvelope * 0.30 * scratch.smoothedGain

                scratch.crackEnvelope *= snapDecay
                scratch.crackTailEnvelope *= tailDecay
            }

            // Soft clip. A crack at full intensity can exceed unity, and hard
            // clipping a noise burst sounds like a fault rather than a whip.
            let clipped = tanh(sample * 1.2)

            for buffer in buffers {
                guard let data = buffer.mData else { continue }
                data.assumingMemoryBound(to: Float.self)[frame] = clipped
            }
        }

        return noErr
    }
}
