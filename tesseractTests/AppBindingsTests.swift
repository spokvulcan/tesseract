//
//  AppBindingsTests.swift
//  tesseractTests
//
//  Drives App Bindings purely through its interface — settings writes through
//  the facade (built over the in-memory Settings Store Adapter), input closures,
//  and the download-status publisher — and asserts only on the recorded effect
//  closures and their order. No real panels, server, models, or downloads.
//

import AppKit
import Combine
import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AppBindingsTests {

    @Test
    func startSeedsBorderGlowThemeBeforeSettingUpOverlayPanels() {
        let h = makeHarness { $0.glowTheme = .matrix }
        defer { h.bindings.stop() }

        h.bindings.start()

        #expect(
            Array(h.recorder.events.prefix(2)) == [
                "setBorderGlowTheme(matrix)",
                "setUpOverlayPanels",
            ])
    }

    @Test
    func oneDictationStateEmissionReachesPillBorderAndMenuBarExactlyOnce() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()

        // The initial emission pushes the current state once to each surface.
        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "pushDictationState") == [
                    "pushDictationStateToPill(idle)",
                    "pushDictationStateToBorder(idle)",
                    "pushDictationStateToMenuBar(idle)",
                ]
            })

        h.driver.dictationState = .recording

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "pushDictationState") == [
                    "pushDictationStateToPill(idle)",
                    "pushDictationStateToBorder(idle)",
                    "pushDictationStateToMenuBar(idle)",
                    "pushDictationStateToPill(recording)",
                    "pushDictationStateToBorder(recording)",
                    "pushDictationStateToMenuBar(recording)",
                ]
            })
    }

    @Test
    func audioLevelFansOutToBothOverlays() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "pushAudioLevel") == [
                    "pushAudioLevelToPill(0.0)",
                    "pushAudioLevelToBorder(0.0)",
                ]
            })

        h.driver.audioLevel = 0.5

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "pushAudioLevel") == [
                    "pushAudioLevelToPill(0.0)",
                    "pushAudioLevelToBorder(0.0)",
                    "pushAudioLevelToPill(0.5)",
                    "pushAudioLevelToBorder(0.5)",
                ]
            })
    }

    @Test
    func glowThemeChangeUpdatesTheBorderOverlayLive() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()
        h.settings.glowTheme = .ocean

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "setBorderGlowTheme").last
                    == "setBorderGlowTheme(ocean)"
            })
    }

    @Test
    func dictationHotkeyReBindsOnlyWhenTheComboActuallyChanges() async {
        let h = makeHarness()
        defer { h.bindings.stop() }
        // The wiring seeds the hotkey manager with the persisted combo before
        // start(), so the initial emission matches the current binding.
        h.driver.currentDictationHotkey = h.settings.hotkey

        h.bindings.start()

        let newCombo = KeyCombo(keyCode: 36, modifiers: [.command])
        h.settings.hotkey = newCombo

        // Exactly one re-bind: the changed combo. The initial (unchanged)
        // emission must not have produced one.
        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "updateDictationHotkey")
                    == ["updateDictationHotkey(\(newCombo.displayString))"]
            })
    }

    @Test
    func ttsAndAgentHotkeyChangesReBindTheirRegistrations() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()

        let ttsCombo = KeyCombo(keyCode: 17, modifiers: [.command, .shift])
        let agentCombo = KeyCombo(keyCode: 0, modifiers: [.option])
        h.settings.ttsHotkey = ttsCombo
        h.settings.agentHotkey = agentCombo

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "updateTTSHotkey").last
                    == "updateTTSHotkey(\(ttsCombo.displayString))"
                    && h.recorder.events(withPrefix: "updateAgentHotkey").last
                        == "updateAgentHotkey(\(agentCombo.displayString))"
            })
    }

    @Test
    func serverEnabledAtLaunchStartsItExactlyOnceAndTogglingOffStopsIt() async {
        let h = makeHarness { $0.isServerEnabled = true }
        defer { h.bindings.stop() }

        h.bindings.start()

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "startHTTPServer") == ["startHTTPServer"]
            })
        #expect(h.recorder.events(withPrefix: "stopHTTPServer").isEmpty)

        h.settings.isServerEnabled = false

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "stopHTTPServer") == ["stopHTTPServer"]
            })
        #expect(h.recorder.events(withPrefix: "startHTTPServer") == ["startHTTPServer"])
    }

    @Test
    func serverDisabledAtLaunchNeverStartsIt() async {
        let h = makeHarness { $0.isServerEnabled = false }
        defer { h.bindings.stop() }

        h.bindings.start()

        // Fence on another rule's initial emission so the server subscription
        // has demonstrably run before we assert it produced no start.
        #expect(
            await waitUntil {
                !h.recorder.events(withPrefix: "updateHTTPServerPort").isEmpty
            })
        #expect(h.recorder.events(withPrefix: "startHTTPServer").isEmpty)
    }

    @Test
    func serverPortChangeProducesAClampedUpdate() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()
        h.settings.serverPort = 0

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "updateHTTPServerPort").last
                    == "updateHTTPServerPort(1)"
            })
    }

    @Test
    func initialAgentModelEmissionNeverForcesAModelLoad() async {
        let h = makeHarness()
        defer { h.bindings.stop() }
        h.driver.isLLMSlotLoaded = false

        h.bindings.start()

        // Fence on a later-installed rule's initial emission, so the reload
        // guard has demonstrably seen — and dropped — its own.
        #expect(
            await waitUntil {
                !h.recorder.events(withPrefix: "updateHTTPServerPort").isEmpty
            })
        #expect(h.recorder.events(withPrefix: "reloadLLM").isEmpty)
    }

    @Test
    func agentModelChangeWhileAnLLMSlotIsLoadedReloadsTheModel() async {
        let h = makeHarness()
        defer { h.bindings.stop() }
        h.driver.isLLMSlotLoaded = false

        h.bindings.start()

        // Let the guard drop the initial emission while nothing is loaded.
        #expect(
            await waitUntil {
                !h.recorder.events(withPrefix: "updateHTTPServerPort").isEmpty
            })

        h.driver.isLLMSlotLoaded = true
        h.settings.selectedAgentModelID = "another-agent-model"

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "reloadLLM") == ["reloadLLMIfNeeded"]
            })
    }

    @Test
    func whisperModelAvailableAtLaunchIsLoaded() async {
        let h = makeHarness()
        defer { h.bindings.stop() }
        h.driver.whisperModelPath = URL(fileURLWithPath: "/models/whisper")

        h.bindings.start()

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "loadWhisperModel")
                    == ["loadWhisperModel(/models/whisper)"]
            })
    }

    @Test
    func whisperModelUnavailableAtLaunchIsNotLoaded() async {
        let h = makeHarness()
        defer { h.bindings.stop() }
        h.driver.whisperModelPath = nil

        h.bindings.start()

        for _ in 0..<100 { await Task.yield() }
        #expect(h.recorder.events(withPrefix: "loadWhisperModel").isEmpty)
    }

    @Test
    func whisperDownloadCompletionLoadsTheModelOnlyWhileEngineIsUnloaded() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()

        for _ in 0..<100 { await Task.yield() }
        #expect(h.recorder.events(withPrefix: "loadWhisperModel").isEmpty)

        // The download completes while the engine is unloaded → load.
        h.driver.whisperModelPath = URL(fileURLWithPath: "/models/whisper")
        h.statuses.send([ModelDefinition.defaultSpeechToTextModelID: .downloaded(sizeOnDisk: 1)])

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "loadWhisperModel")
                    == ["loadWhisperModel(/models/whisper)"]
            })

        // A re-download completing while the engine is already serving → skip.
        h.driver.isTranscriptionModelLoaded = true
        h.statuses.send([ModelDefinition.defaultSpeechToTextModelID: .notDownloaded])
        h.statuses.send([ModelDefinition.defaultSpeechToTextModelID: .downloaded(sizeOnDisk: 2)])

        for _ in 0..<100 { await Task.yield() }
        #expect(
            h.recorder.events(withPrefix: "loadWhisperModel")
                == ["loadWhisperModel(/models/whisper)"])
    }

    @Test
    func speechModelSelectionChangeHotSwapsTheLoadedModel() async {
        let h = makeHarness()
        defer { h.bindings.stop() }
        h.driver.whisperModelPath = URL(fileURLWithPath: "/models/whisper")
        h.driver.isTranscriptionModelLoaded = true

        h.bindings.start()

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "loadWhisperModel")
                    == ["loadWhisperModel(/models/whisper)"]
            })

        // Switching the selection loads the newly selected model even though
        // the engine is already serving — that is the hot-swap.
        h.driver.whisperModelPath = URL(fileURLWithPath: "/models/whisper-compact")
        h.settings.selectedSpeechToTextModelID = "whisper-large-v3-turbo-compact"

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "loadWhisperModel") == [
                    "loadWhisperModel(/models/whisper)",
                    "loadWhisperModel(/models/whisper-compact)",
                ]
            })
    }

    @Test
    func missingSelectedSpeechModelHealsSelectionToTheDownloadedVariantAndLoadsIt() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()

        // Only the non-selected compact variant exists on disk — e.g. a fresh
        // low-memory install that skipped the default, or the selected variant
        // was deleted.
        h.driver.whisperModelPath = URL(fileURLWithPath: "/models/whisper-compact")
        h.statuses.send(["whisper-large-v3-turbo-compact": .downloaded(sizeOnDisk: 1)])

        #expect(
            await waitUntil {
                h.settings.selectedSpeechToTextModelID == "whisper-large-v3-turbo-compact"
            })
        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "loadWhisperModel")
                    == ["loadWhisperModel(/models/whisper-compact)"]
            })
    }

    @Test
    func downloadingASecondVariantNeverOverridesAnAvailableSelection() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()

        let selectedID = ModelDefinition.defaultSpeechToTextModelID
        h.statuses.send([
            selectedID: .downloaded(sizeOnDisk: 1),
            "whisper-large-v3-turbo-compact": .downloaded(sizeOnDisk: 1),
        ])

        for _ in 0..<100 { await Task.yield() }
        #expect(h.settings.selectedSpeechToTextModelID == selectedID)
    }

    @Test
    func subscriptionsAreLiveWhileTheInitialWhisperLoadIsStillRunning() async {
        let gate = AsyncGate()
        let h = makeHarness(
            configureSettings: { $0.isServerEnabled = true },
            loadWhisperModel: { _ in await gate.wait() }
        )
        defer { h.bindings.stop() }
        h.driver.whisperModelPath = URL(fileURLWithPath: "/models/whisper")

        h.bindings.start()

        // The Whisper load is suspended at the gate — yet the server-enable
        // rule has already started the server. Launch is not gated on the load.
        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "startHTTPServer") == ["startHTTPServer"]
            })
        #expect(h.recorder.events(withPrefix: "loadWhisperModel").isEmpty)

        gate.open()

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "loadWhisperModel")
                    == ["loadWhisperModel(/models/whisper)"]
            })
    }

    @Test
    func stopCancelsEverySubscriptionAndTheWhisperLoad() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()
        #expect(
            await waitUntil {
                !h.recorder.events(withPrefix: "setPillOverlayEnabled").isEmpty
            })

        h.bindings.stop()
        for _ in 0..<100 { await Task.yield() }
        let countAfterStop = h.recorder.events.count

        h.settings.isServerEnabled = true
        h.settings.overlayStyle = .fullScreenBorder
        h.driver.dictationState = .recording
        h.statuses.send([ModelDefinition.defaultSpeechToTextModelID: .downloaded(sizeOnDisk: 1)])

        for _ in 0..<200 { await Task.yield() }
        #expect(h.recorder.events.count == countAfterStop)
    }

    @Test
    func overlayStyleRuleEnablesExactlyTheSelectedOverlay() async {
        let h = makeHarness()
        defer { h.bindings.stop() }

        h.bindings.start()

        // Initial emission applies the persisted style (pill by default).
        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "setPillOverlayEnabled") == [
                    "setPillOverlayEnabled(true)"
                ]
                    && h.recorder.events(withPrefix: "setBorderOverlayEnabled") == [
                        "setBorderOverlayEnabled(false)"
                    ]
            })

        h.settings.overlayStyle = .fullScreenBorder

        #expect(
            await waitUntil {
                h.recorder.events(withPrefix: "setPillOverlayEnabled")
                    == ["setPillOverlayEnabled(true)", "setPillOverlayEnabled(false)"]
                    && h.recorder.events(withPrefix: "setBorderOverlayEnabled")
                        == ["setBorderOverlayEnabled(false)", "setBorderOverlayEnabled(true)"]
            })
    }
}

// MARK: - Harness

/// Records every effect invocation as a formatted event string, in order.
@MainActor
private final class EffectRecorder {
    private(set) var events: [String] = []

    func callAsFunction(_ event: String) { events.append(event) }

    func events(withPrefix prefix: String) -> [String] {
        events.filter { $0.hasPrefix(prefix) }
    }
}

/// Observable backing for the module's tracked-read input closures — the test
/// peer of the dictation coordinator and audio capture engine.
@Observable @MainActor
private final class InputDriver {
    var dictationState: DictationState = .idle
    var audioLevel: Float = 0
    var currentDictationHotkey: KeyCombo = .optionSpace
    var isLLMSlotLoaded = false
    var whisperModelPath: URL?
    var isTranscriptionModelLoaded = false
}

@MainActor
private struct Harness {
    let bindings: AppBindings
    let settings: SettingsManager
    let driver: InputDriver
    let statuses: CurrentValueSubject<[String: ModelStatus], Never>
    let recorder: EffectRecorder
}

/// A manually opened gate for suspending an effect closure mid-flight.
@MainActor
private final class AsyncGate {
    private var isOpen = false
    private var continuations: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { continuations.append($0) }
    }

    func open() {
        isOpen = true
        for continuation in continuations { continuation.resume() }
        continuations = []
    }
}

@MainActor
private func makeHarness(
    configureSettings: (SettingsManager) -> Void = { _ in },
    loadWhisperModel: (@MainActor (URL) async -> Void)? = nil
) -> Harness {
    let settings = SettingsManager(store: InMemorySettingsStore())
    configureSettings(settings)
    let driver = InputDriver()
    let statuses = CurrentValueSubject<[String: ModelStatus], Never>([:])
    let recorder = EffectRecorder()
    let bindings = AppBindings(
        settings: settings,
        inputs: .init(
            dictationState: { driver.dictationState },
            audioLevel: { driver.audioLevel },
            currentDictationHotkey: { driver.currentDictationHotkey },
            isLLMSlotLoaded: { driver.isLLMSlotLoaded },
            whisperModelPath: { driver.whisperModelPath },
            isTranscriptionModelLoaded: { driver.isTranscriptionModelLoaded },
            modelDownloadStatuses: statuses.eraseToAnyPublisher()
        ),
        effects: .init(
            setBorderGlowTheme: { recorder("setBorderGlowTheme(\($0.rawValue))") },
            setUpOverlayPanels: { recorder("setUpOverlayPanels") },
            setPillOverlayEnabled: { recorder("setPillOverlayEnabled(\($0))") },
            setBorderOverlayEnabled: { recorder("setBorderOverlayEnabled(\($0))") },
            pushDictationStateToPill: { recorder("pushDictationStateToPill(\($0))") },
            pushDictationStateToBorder: { recorder("pushDictationStateToBorder(\($0))") },
            pushDictationStateToMenuBar: { recorder("pushDictationStateToMenuBar(\($0))") },
            pushAudioLevelToPill: { recorder("pushAudioLevelToPill(\($0))") },
            pushAudioLevelToBorder: { recorder("pushAudioLevelToBorder(\($0))") },
            updateDictationHotkey: { recorder("updateDictationHotkey(\($0.displayString))") },
            updateTTSHotkey: { recorder("updateTTSHotkey(\($0.displayString))") },
            updateAgentHotkey: { recorder("updateAgentHotkey(\($0.displayString))") },
            startHTTPServer: { recorder("startHTTPServer") },
            stopHTTPServer: { recorder("stopHTTPServer") },
            updateHTTPServerPort: { recorder("updateHTTPServerPort(\($0))") },
            reloadLLMIfNeeded: { recorder("reloadLLMIfNeeded") },
            loadWhisperModel: { url in
                if let loadWhisperModel { await loadWhisperModel(url) }
                recorder("loadWhisperModel(\(url.path))")
            }
        )
    )
    return Harness(
        bindings: bindings,
        settings: settings,
        driver: driver,
        statuses: statuses,
        recorder: recorder
    )
}

/// Pumps the main actor until `condition` holds, giving the module's
/// observation tasks a chance to run. Returns the final condition value.
@MainActor
private func waitUntil(_ condition: @MainActor () -> Bool) async -> Bool {
    for _ in 0..<2000 {
        if condition() { return true }
        await Task.yield()
    }
    return condition()
}
