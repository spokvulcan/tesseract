//
//  voice-hold-lab — the ADR-0041 runtime harness (see RUNBOOK.md).
//
//  Usage:
//    swift run --package-path tools/voice-hold-lab voice-hold-lab <command>
//
//  Commands:
//    e1 … e7        one scenario (e3 takes --unsafe; e4 is interactive)
//    all            e1 e2 e3 e5 e6 e7 (the non-interactive set)
//    emit-fixture   e2 + e5 [+ record-barge with --with-owner-barge], then
//                   write tesseractTests/VoiceLabFixtures.swift
//                   (--out <path> overrides)
//    record-barge   interactive owner-speaking-over-reply trace only
//

import AVFoundation
import Foundation

func requireMicrophone() {
    let status = AVCaptureDevice.authorizationStatus(for: .audio)
    if status == .authorized { return }
    print("requesting microphone access for this terminal…")
    let semaphore = DispatchSemaphore(value: 0)
    var granted = false
    AVCaptureDevice.requestAccess(for: .audio) { ok in
        granted = ok
        semaphore.signal()
    }
    semaphore.wait()
    guard granted else {
        print("microphone denied — grant the terminal access in System Settings → Privacy")
        exit(1)
    }
}

let arguments = CommandLine.arguments.dropFirst()
let command = arguments.first ?? "all"
let unsafeRepro = arguments.contains("--unsafe")
let withOwnerBarge = arguments.contains("--with-owner-barge")
let outPath =
    arguments.firstIndex(of: "--out").flatMap { index -> String? in
        let next = arguments.index(after: index)
        return next < arguments.endIndex ? arguments[next] : nil
    } ?? "tesseractTests/VoiceLabFixtures.swift"

requireMicrophone()
print("voice-hold-lab — ADR-0041 runtime harness\n")

do {
    switch command {
    case "e1": try Scenarios.e1()
    case "e2": _ = try Scenarios.e2()
    case "e3": try Scenarios.e3(runCrashClass: unsafeRepro)
    case "e4": try Scenarios.e4()
    case "e5": _ = try Scenarios.e5()
    case "e6": try Scenarios.e6()
    case "e7": try Scenarios.e7()
    case "record-barge": _ = try Scenarios.recordBarge()
    case "all":
        try Scenarios.e1()
        print("")
        _ = try Scenarios.e2()
        print("")
        try Scenarios.e3(runCrashClass: false)
        print("")
        _ = try Scenarios.e5()
        print("")
        try Scenarios.e6()
        print("")
        try Scenarios.e7()
    case "emit-fixture":
        var traces = try Scenarios.e2()
        print("")
        traces["resumeTransient"] = try Scenarios.e5()
        // The far-end envelopes of the deterministic signals, computed pure —
        // the replay tests' playbackLevel input, bin-aligned with the traces.
        let e2Signal =
            Signals.chirp(seconds: 2, sampleRate: 24_000)
            + Signals.speechShaped(seconds: 6, sampleRate: 24_000)
        traces["e2SignalEnvelope"] = Signals.envelope(e2Signal, sampleRate: 24_000)
        traces["e5SignalEnvelope"] = Signals.envelope(
            Signals.speechShaped(seconds: 10, sampleRate: 24_000), sampleRate: 24_000)
        if withOwnerBarge {
            print("")
            traces["ownerBargeOverReply"] = try Scenarios.recordBarge()
        }
        try FixtureWriter.write(traces: traces, to: outPath)
    default:
        print("unknown command: \(command)")
        exit(2)
    }
} catch {
    print("FAILED: \(error)")
    exit(1)
}
