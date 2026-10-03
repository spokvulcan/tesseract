//
//  PhoneContainer.swift
//  tesseract-ios
//
//  The iPhone app's composition root: pure wiring, like the Mac's
//  `DependencyContainer`. The speech code is the Mac's (the coordinator, the
//  Read-Along, the Reader); what differs is the voice behind the engine and
//  the adapters around it. Until the neural voice comes to the phone (#515,
//  slice 4) the system voice reads.
//

import Foundation
import TesseractSpeech

@MainActor
final class PhoneContainer {
    let settings = PhoneSettings()
    let library = ReaderLibrary()
    /// The **Read-Along** (ADR-0076): the one clock of which word is heard.
    let readAlong = SpeechReadAlong()
    let audioSession = PhoneAudioSession()

    lazy var engine = SpeechEnginePresenter(
        engine: SpeechEngine(model: .customVoice06B, synthesizer: SystemVoiceSynthesizer()))

    lazy var coordinator = SpeechCoordinator(
        textExtractor: PasteboardTextExtractor(),
        engine: engine,
        // The system voice is always there, so speech never waits on a download.
        voiceEngineStatus: { .downloaded(sizeOnDisk: 0) },
        playback: SessionPlayback(session: audioSession),
        settings: settings,
        notchOverlay: readAlong)

    lazy var reading = PhoneReading(
        library: library, coordinator: coordinator, readAlong: readAlong, settings: settings)

    lazy var intake = PhoneIntake(library: library)

    lazy var pocket = PhonePocket(
        controls: PocketControls { [unowned self] in self.reading.current?.reader },
        reading: reading, library: library, settings: settings)

    init() {
        if library.isNew {
            library.add(PhoneWelcome.text, title: PhoneWelcome.title)
        }
        pocket.start()
    }
}
