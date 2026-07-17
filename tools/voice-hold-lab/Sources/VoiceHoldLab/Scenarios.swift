//
//  Scenarios.swift — the lab's E-scenarios (RUNBOOK.md documents procedure
//  and records results).
//
//  E1  hold wire/start/gate cycles + dynamic-attach config-change sub-check
//  E2  echo residual: hosted vs dedicated vs dedicated+duck (+ noise floor)
//  E3  running-engine discipline (safe ops never throw; crash class opt-in)
//  E4  device change under hold (interactive)
//  E5  pause/resume + duck-ramp transient profile
//  E6  wiring latency distribution
//  E7  begin/end churn
//

import AVFoundation
import Foundation

enum Scenarios {

    // MARK: E1

    static func e1() throws {
        print("E1 — hold wiring + capture gate + dynamic attach")
        let rig = try CaptureRig(hostPlayback: true)
        defer { rig.stop() }
        print("  wired: vp=\(rig.voiceProcessing) render=\(rig.renderWired)")

        for _ in 0..<20 {
            rig.gate.isOpen = false
            Thread.sleep(forTimeInterval: 0.02)
            rig.gate.isOpen = true
            Thread.sleep(forTimeInterval: 0.02)
        }
        print("  gate: 20 open/close cycles on the running engine — ok")

        // D3 sub-check: does player attach/connect on the RUNNING engine
        // echo a configuration-change notification? (If yes, per-utterance
        // node churn would trigger hold rebuilds in the app.)
        let before = rig.configChangeCount
        let extra = AVAudioPlayerNode()
        rig.engine.attach(extra)
        rig.engine.connect(extra, to: rig.engine.mainMixerNode, format: rig.playbackFormat)
        Thread.sleep(forTimeInterval: 0.3)
        extra.stop()
        rig.engine.detach(extra)
        Thread.sleep(forTimeInterval: 0.3)
        let fired = rig.configChangeCount - before
        print("  dynamic attach/detach fired \(fired) config-change notifications")
        print(
            fired == 0
                ? "  PASS: no config-change echo from runtime node churn"
                : "  WARN: runtime node churn echoes config changes — persistent node is mandatory"
        )
        print("  engine running: \(rig.engine.isRunning)")
    }

    // MARK: E2

    /// ONE rig for every mode, segments sliced from a single continuous
    /// trace. Back-to-back VP engine create/destroy cycles wedge CoreAudio
    /// input (the same pattern the app's kept-engine design exists to
    /// avoid) — and the single held engine is also exactly how the app
    /// runs: AEC converges once and stays converged across segments.
    static func e2() throws -> [String: [Float]] {
        print("E2 — echo residual at the VP mic, by playback path (one held rig)")
        print("  (quiet room + normal listening volume — the traces become fixtures)")

        let signal =
            Signals.chirp(seconds: 2, sampleRate: 24_000)
            + Signals.speechShaped(seconds: 6, sampleRate: 24_000)
        let signalBins = 8 * 20

        let rig = try CaptureRig(hostPlayback: true)
        let dedicated = try DedicatedPlayer()
        var marks: [String: Int] = [:]
        var cursor = 0

        func segment(_ name: String, seconds: Double, run: () -> Void) {
            marks[name] = cursor
            run()
            Thread.sleep(forTimeInterval: seconds)
            cursor += Int(seconds * 20)
        }

        segment("noiseFloor", seconds: 3) {}
        segment("hostedReply", seconds: 9) { rig.play(signal) }
        segment("gap1", seconds: 1) {}
        segment("dedicatedMin", seconds: 9) {
            rig.engine.inputNode.voiceProcessingOtherAudioDuckingConfiguration =
                AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                    enableAdvancedDucking: false, duckingLevel: .min)
            dedicated.play(signal)
        }
        segment("gap2", seconds: 1) {}
        segment("dedicatedDefault", seconds: 9) {
            rig.engine.inputNode.voiceProcessingOtherAudioDuckingConfiguration =
                AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                    enableAdvancedDucking: false, duckingLevel: .default)
            dedicated.play(signal)
        }
        dedicated.stop()
        rig.stop()

        let full = rig.trace.levels()
        guard full.count >= cursor - 20 else {
            throw LabError("trace too short (\(full.count) bins) — wedged input, re-run")
        }

        var traces: [String: [Float]] = [:]
        for (name, seconds) in [
            ("noiseFloor", 60), ("hostedReply", signalBins),
            ("dedicatedMin", signalBins), ("dedicatedDefault", signalBins),
        ] {
            let start = marks[name]!
            let end = min(start + seconds, full.count)
            let slice = start < end ? Array(full[start..<end]) : []
            traces[name] = slice
            print(
                "  " + slice.stats(label: name.padding(toLength: 16, withPad: " ", startingAt: 0)))
        }

        // First-second exposure (D6): the from-cold slice of the hosted
        // segment — AEC's first look at far-end audio in this process.
        let hosted = traces["hostedReply"] ?? []
        traces["hostedFirstSecond"] = Array(hosted.prefix(20))
        print("  " + (traces["hostedFirstSecond"] ?? []).stats(label: "hosted, 1st sec "))
        return traces
    }

    // MARK: E3

    static func e3(runCrashClass: Bool) throws {
        print("E3 — running-engine discipline (the 2026-07-17 crash class)")
        let rig = try CaptureRig(hostPlayback: true)
        defer { rig.stop() }

        // Everything the hold does while running — all documented-safe.
        rig.gate.isOpen = false
        rig.gate.isOpen = true
        rig.play(Signals.speechShaped(seconds: 0.5, sampleRate: 24_000))
        rig.player.volume = 0.25
        rig.player.pause()
        rig.player.play()
        rig.player.volume = 1.0
        Thread.sleep(forTimeInterval: 0.8)
        print("  PASS: gate flips, scheduling, pause/play, volume — no throw")

        guard runCrashClass else {
            print("  (crash-class repro is opt-in: --unsafe re-taps with a new")
            print("   format on the running VP engine — expect SIGABRT.)")
            return
        }
        print("  --unsafe: re-installing the tap with a different format on the")
        print("  RUNNING VP engine. This SHOULD crash (uncatchable NSException):")
        rig.engine.inputNode.removeTap(onBus: 0)
        let strange = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
        rig.engine.inputNode.installTap(onBus: 0, bufferSize: 1024, format: strange) { _, _ in }
        print("  UNEXPECTED: survived the crash-class operation")
    }

    // MARK: E4

    static func e4() throws {
        print("E4 — device change under hold (interactive)")
        print("  Switch the output/input device now (AirPods, display speakers…).")
        print("  Watching config-change notifications for 30 s…")
        let rig = try CaptureRig(hostPlayback: true)
        defer { rig.stop() }
        var last = 0
        for second in 1...30 {
            Thread.sleep(forTimeInterval: 1.0)
            if rig.configChangeCount != last {
                last = rig.configChangeCount
                print(
                    "  t=\(second)s config-change #\(last) — engine running: \(rig.engine.isRunning)"
                )
            }
        }
        print("  total: \(last) notifications (app: rebuild-under-hold path re-wires)")
    }

    // MARK: E5

    static func e5() throws -> [Float] {
        print("E5 — pause/resume + duck-ramp transient profile (hosted)")
        let rig = try CaptureRig(hostPlayback: true)
        let signal = Signals.speechShaped(seconds: 10, sampleRate: 24_000)
        rig.play(signal)

        Thread.sleep(forTimeInterval: 2.0)
        print("  t=2s duck to 0.25 over 100 ms")
        ramp(rig.player, to: 0.25, over: 0.1)
        Thread.sleep(forTimeInterval: 1.0)
        print("  t=3s fade back to 1.0 over 200 ms")
        ramp(rig.player, to: 1.0, over: 0.2)
        Thread.sleep(forTimeInterval: 1.0)
        print("  t=4s hard pause")
        rig.player.pause()
        Thread.sleep(forTimeInterval: 1.0)
        print("  t=5s resume")
        rig.player.play()
        Thread.sleep(forTimeInterval: 2.0)
        rig.stop()

        let levels = rig.trace.levels()
        print("  " + levels.stats(label: "transient trace"))
        // The interesting windows, in 50 ms bins from t=0.
        for (name, range) in [
            ("post-duck   (2.0–2.5s)", 40..<50),
            ("post-fadeup (3.0–3.5s)", 60..<70),
            ("post-resume (5.0–5.5s)", 100..<110),
        ] where levels.count >= range.upperBound {
            let slice = Array(levels[range])
            print("    " + slice.stats(label: name))
        }
        return levels
    }

    private static func ramp(
        _ player: AVAudioPlayerNode, to target: Float, over duration: TimeInterval
    ) {
        let start = player.volume
        let steps = max(1, Int(duration / 0.016))
        for step in 1...steps {
            Thread.sleep(forTimeInterval: 0.016)
            player.volume = start + (target - start) * Float(step) / Float(steps)
        }
    }

    // MARK: E6

    static func e6() throws {
        print("E6 — full hold wiring latency (build+VP+tap+render+start)")
        var timings: [Double] = []
        for i in 1...5 {
            let start = Date()
            let rig = try CaptureRig(hostPlayback: true)
            let ms = Date().timeIntervalSince(start) * 1000
            timings.append(ms)
            print(String(format: "  run %d: %.0f ms (vp=%@)", i, ms, "\(rig.voiceProcessing)"))
            rig.stop()
            Thread.sleep(forTimeInterval: 0.3)
        }
        print(String(format: "  median ≈ %.0f ms", timings.sorted()[timings.count / 2]))
    }

    // MARK: E7

    static func e7() throws {
        print("E7 — begin/end churn (rapid wire → play → discard cycles)")
        for i in 1...10 {
            let rig = try CaptureRig(hostPlayback: true)
            rig.play(Signals.speechShaped(seconds: 0.3, sampleRate: 24_000))
            Thread.sleep(forTimeInterval: Double(i % 3) * 0.05)
            rig.stop()
        }
        print("  PASS: 10 cycles, no throw, clean teardown each time")
    }

    // MARK: Owner-barge recording (fixture input)

    static func recordBarge() throws -> [Float] {
        print("record-barge — SPEAK OVER the playback, starting after ~3 s,")
        print("a normal interruption (\"hang on, one question…\"), ~3 s long.")
        let rig = try CaptureRig(hostPlayback: true)
        rig.play(Signals.speechShaped(seconds: 10, sampleRate: 24_000))
        for second in 1...10 {
            Thread.sleep(forTimeInterval: 1.0)
            print("  t=\(second)s\(second == 3 ? "  ← speak now" : "")")
        }
        rig.stop()
        let levels = rig.trace.levels()
        print("  " + levels.stats(label: "owner-barge trace"))
        return levels
    }
}
