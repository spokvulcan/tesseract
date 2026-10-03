//
//  PhoneReading.swift
//  tesseract-ios
//
//  The Library's open texts: a Reader for each, kept while it reads, so
//  leaving a text's page doesn't stop it and the Library can show what is
//  being read. Opening another text drops the Readers that aren't reading.
//

import Foundation
import Observation

@Observable @MainActor
final class PhoneReading {
    private var readers: [UUID: SpeechReader] = [:]

    @ObservationIgnored private let library: ReaderLibrary
    @ObservationIgnored private let coordinator: SpeechCoordinator
    @ObservationIgnored private let readAlong: SpeechReadAlong
    @ObservationIgnored private let settings: PhoneSettings

    init(
        library: ReaderLibrary, coordinator: SpeechCoordinator, readAlong: SpeechReadAlong,
        settings: PhoneSettings
    ) {
        self.library = library
        self.coordinator = coordinator
        self.readAlong = readAlong
        self.settings = settings
    }

    /// The text being read, if one is.
    var nowReading: (id: UUID, reader: SpeechReader)? {
        readers.first { $0.value.isReading }.map { ($0.key, $0.value) }
    }

    /// The Reader of text `id`: the one already open, else a new one. Nil
    /// when the Library no longer has the text.
    func open(_ id: UUID) -> SpeechReader? {
        if let reader = readers[id] { return reader }
        guard let entry = library.entry(id) else { return nil }
        readers = readers.filter { $0.value.isReading }
        let reader = SpeechReader(
            coordinator: coordinator, readAlong: readAlong, settings: settings,
            store: library.store(for: id))
        reader.language = entry.language
        reader.start()
        readers[id] = reader
        return reader
    }

    /// Text `id` is leaving the Library: stop it if it reads.
    func close(_ id: UUID) {
        readers[id]?.stop()
        readers[id] = nil
    }
}
