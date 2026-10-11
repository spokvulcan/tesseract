//
//  MenuModelsSectionTests.swift
//  tesseractTests
//
//  The menu's Models section reads at a glance: what each model is doing,
//  its size once loaded, and the app's memory with the system's swap. The
//  Dictation row hears every push-to-talk the app makes.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct MenuModelsSectionTests {

    private let gigabytes: Int64 = 15_132_802_752

    @Test func eachRowSaysWhatTheModelIsDoing() {
        let size = ByteCountFormatter.string(fromByteCount: gigabytes, countStyle: .memory)
        let generating = ModelActivityRow(
            id: "llm", role: "Language model", name: "Qwen3.8 27B",
            state: .working("Jarvis · Triage"), bytes: gigabytes)
        #expect(generating.detail == "Jarvis · Triage · \(size)")
        let idle = ModelActivityRow(
            id: "llm", role: "Language model", name: "Qwen3.8 27B", state: .loaded, bytes: gigabytes
        )
        #expect(idle.detail == "Idle · \(size)")
        #expect(
            ModelActivityRow(id: "stt", role: "Dictation", name: "Whisper Turbo", state: .loaded)
                .detail == "Idle")
        #expect(
            ModelActivityRow(id: "tts", role: "Voice", name: "Voice Engine", state: .notLoaded)
                .detail == "Not loaded")
        #expect(
            ModelActivityRow(id: "tts", role: "Voice", name: "Voice Engine", state: .loading)
                .detail == "Loading…")
    }

    @Test func theMemoryLineShowsTheAppAndTheSwap() {
        var activity = ModelActivity.empty
        #expect(activity.memoryLine.isEmpty)
        activity.footprintBytes = 21 * 1_073_741_824
        activity.swapBytes = 8 * 1_073_741_824
        let app = ByteCountFormatter.string(
            fromByteCount: Int64(21 * 1_073_741_824), countStyle: .memory)
        let swap = ByteCountFormatter.string(
            fromByteCount: Int64(8 * 1_073_741_824), countStyle: .memory)
        #expect(activity.memoryLine == "Tesseract uses \(app) · swap \(swap)")
        activity.swapBytes = 0
        #expect(activity.memoryLine == "Tesseract uses \(app)")
    }

    /// The Dictation row reads every push-to-talk the container made, Today's
    /// composer's included, so a take on any surface reads as Listening.
    @MainActor @Test func theDictationRowHearsEveryPushToTalk() {
        let container = DependencyContainer()
        let surfaces = [
            container.agentVoiceInput, container.captureVoiceInput,
            container.panelVoiceInput, container.todayVoiceInput,
        ]
        for surface in surfaces {
            #expect(container.voiceInputs.contains { $0 === surface })
        }
    }

    /// The container builds a push-to-talk in one place, which registers it
    /// for the Dictation row, so a new surface can't be left out of it.
    @Test func theContainerBuildsEveryPushToTalkInOnePlace() throws {
        let app = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // tesseractTests
            .deletingLastPathComponent()  // project root
            .appendingPathComponent("tesseract/App")
        let container = try FileManager.default
            .contentsOfDirectory(at: app, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("DependencyContainer") }
            .map { try String(contentsOf: $0, encoding: .utf8) }
            .joined(separator: "\n")
        #expect(
            container.components(separatedBy: "AgentVoiceInputController(").count - 1 == 1,
            "Build a surface's push-to-talk with makeVoiceInput(), which registers it")
    }
}
