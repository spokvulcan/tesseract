//
//  SpeechLab.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  The in-memory model the five Speech page variants share, so flipping
//  between them keeps your text, your takes and your settings:
//
//  - the draft text;
//  - every take the engine renders, recorded off the playback path by
//    `SpeechLabRecordingPlayback` (so the hotkey and the assistant land here
//    too), for replay, export and drag-out;
//  - the read-along clock (`SpeechReadAlong`) fed by `SpeechLabHighlightFanOut`;
//  - the overlay preferences and the overlay itself;
//  - the voice list (built-in presets plus the designed voices already
//    pinned on disk).
//
//  Nothing here persists: a relaunch starts with no takes. That is the
//  point — persistence is one of the questions the prototype asks.
//

import AppKit
import Foundation
import Observation
import SwiftUI
import TesseractSpeech
import os

// MARK: - Take

nonisolated struct SpeechTake: Identifiable, Equatable, Sendable {
    nonisolated enum Source: String, Sendable {
        /// Spoken from the Speech page.
        case page
        /// Selected text in another app, via the global hotkey.
        case selection
        /// Something the app itself said (the assistant, auto-speak).
        case app
        /// "Try another take": a new rendering of the voice.
        case voiceTake
        /// Rendered to audio without playing it.
        case render

        var label: String {
            switch self {
            case .page: "Speech page"
            case .selection: "Selected text"
            case .app: "Assistant"
            case .voiceTake: "Voice take"
            case .render: "Rendered"
            }
        }

        var symbol: String {
            switch self {
            case .page: "text.cursor"
            case .selection: "selection.pin.in.out"
            case .app: "sparkles"
            case .voiceTake: "person.wave.2"
            case .render: "waveform.badge.checkmark"
            }
        }
    }

    let id: UUID
    let createdAt: Date
    let text: String
    let source: Source
    let blockID: UUID?
    let voiceDescription: String
    let language: String
    let parameters: TTSParameters
    let seed: Int
    let audio: TakeAudio
    let segments: [ReadAlongMap.Segment]
    /// Document words before the spoken text (a read that started mid-text).
    let wordOffset: Int
    /// False when speech was stopped before the whole text rendered.
    let isComplete: Bool
    /// The voice's Reference Take this audio continued (or became): keeping
    /// it again makes the voice this person from now on.
    var pinnedVoice: PinnedVoice?

    var duration: TimeInterval { audio.duration }

    var voiceName: String { SpeechLab.voiceName(for: voiceDescription) }

    /// A short title from the text's opening words.
    var title: String {
        let words = text.split(whereSeparator: \.isWhitespace).prefix(7).joined(separator: " ")
        return words.isEmpty ? "Untitled take" : words
    }

    static func == (lhs: SpeechTake, rhs: SpeechTake) -> Bool {
        lhs.id == rhs.id && lhs.audio === rhs.audio && lhs.isComplete == rhs.isComplete
    }
}

/// The take being recorded right now — coarse, observable facts only; the
/// samples themselves stay out of observation.
struct LiveTake: Equatable {
    let id: UUID
    let source: SpeechTake.Source
    let blockID: UUID?
    let wordOffset: Int
    let requestText: String?
    let startedAt: Date
    var duration: TimeInterval = 0
    var peaks: [Float] = []
    var isGenerationFinished = false
}

// MARK: - Voices

nonisolated struct LabVoice: Identifiable, Hashable, Sendable {
    nonisolated enum Kind: Hashable, Sendable {
        case preset
        case yours
    }

    var id: String { description }
    let name: String
    let description: String
    let kind: Kind
    /// A short line for cards and menus.
    let blurb: String
}

// MARK: - Overlay preferences

struct OverlayPrefs: Equatable {
    enum Style: String, CaseIterable, Identifiable {
        case island, captions, classic, off
        var id: String { rawValue }
        var label: String {
            switch self {
            case .island: "Island"
            case .captions: "Captions"
            case .classic: "Classic"
            case .off: "Off"
            }
        }
        var detail: String {
            switch self {
            case .island: "Two lines under the notch, controls on hover."
            case .captions: "Large captions at the bottom of the screen."
            case .classic: "Today's teleprompter panel, unchanged."
            case .off: "Speak without anything on screen."
            }
        }
        var symbol: String {
            switch self {
            case .island: "capsule.tophalf.filled"
            case .captions: "captions.bubble"
            case .classic: "text.alignleft"
            case .off: "eye.slash"
            }
        }
    }

    enum Scope: String, CaseIterable, Identifiable {
        case everything, otherApps
        var id: String { rawValue }
        var label: String {
            switch self {
            case .everything: "All speech"
            case .otherApps: "Only text from other apps"
            }
        }
    }

    enum Size: String, CaseIterable, Identifiable {
        case small, medium, large
        var id: String { rawValue }
        var label: String {
            switch self {
            case .small: "S"
            case .medium: "M"
            case .large: "L"
            }
        }
        var points: CGFloat {
            switch self {
            case .small: 15
            case .medium: 18
            case .large: 23
            }
        }
    }

    enum Tint: String, CaseIterable, Identifiable {
        case accent, yellow, white
        var id: String { rawValue }
        var label: String {
            switch self {
            case .accent: "Orange"
            case .yellow: "Yellow"
            case .white: "White"
            }
        }
        var color: Color {
            switch self {
            case .accent: Color(red: 0.96, green: 0.65, blue: 0.26)
            case .yellow: Color(red: 1.0, green: 0.86, blue: 0.2)
            case .white: .white
            }
        }
    }

    var style: Style = .island
    var scope: Scope = .everything
    var size: Size = .medium
    var tint: Tint = .accent
    var showsControls = true
}

// MARK: - Status

enum SpeechLabStatus: Equatable {
    case idle
    case loadingModel(String)
    case capturing
    case preparing
    case speaking(segment: Int, of: Int)
    case paused(segment: Int, of: Int)
    case rendering(Double)
    case error(String)

    var isActive: Bool {
        switch self {
        case .idle, .error: false
        default: true
        }
    }

    var label: String {
        switch self {
        case .idle: "Ready"
        case .loadingModel(let text): text.isEmpty ? "Loading the voice model…" : text
        case .capturing: "Reading the selection…"
        case .preparing: "Preparing…"
        case .speaking(let segment, let total):
            total > 1 ? "Speaking · part \(segment) of \(total)" : "Speaking"
        case .paused: "Paused"
        case .rendering(let progress): "Rendering · \(Int(progress * 100))%"
        case .error(let message): message
        }
    }
}

// MARK: - The lab

@Observable @MainActor
final class SpeechLab {
    static let shared = SpeechLab()

    // Wiring, set once by the composition root.
    @ObservationIgnored private(set) weak var settings: SettingsManager?
    @ObservationIgnored private(set) weak var coordinator: SpeechCoordinator?
    @ObservationIgnored private(set) weak var enginePresenter: SpeechEnginePresenter?

    /// The text on the page. Shared, so switching variants keeps it.
    /// Only the editors read it; everything else reads `draftIsEmpty`, so a
    /// keystroke redraws the editor and its word count, nothing more.
    var draft: String = SpeechLab.sampleText {
        didSet {
            let empty = !draft.contains { !$0.isWhitespace }
            if empty != draftIsEmpty { draftIsEmpty = empty }
        }
    }

    private(set) var draftIsEmpty = false

    /// Newest first.
    private(set) var takes: [SpeechTake] = []
    private(set) var liveTake: LiveTake?
    private(set) var render: RenderState?

    let readAlong = SpeechReadAlong()
    let player = SpeechTakePlayer()
    var overlay = OverlayPrefs()

    /// Live and replay speed. Live speech time-stretches without changing pitch.
    var playbackRate: Float = 1.0 {
        didSet {
            player.rate = playbackRate
            liveRateSink?(playbackRate)
        }
    }

    private(set) var yourVoices: [LabVoice] = []

    @ObservationIgnored var liveRateSink: ((Float) -> Void)?
    @ObservationIgnored private var pendingTag: Tag?
    @ObservationIgnored private var recordingSamples: [Float] = []
    @ObservationIgnored private var recordingSegments: [ReadAlongMap.Segment] = []
    @ObservationIgnored private var recordingSampleRate = 24_000
    @ObservationIgnored private var recordingSnapshot: VoiceSnapshot?
    @ObservationIgnored private var peakTail: [Float] = []
    @ObservationIgnored lazy var overlayController = SpeechLabOverlayController(lab: self)
    @ObservationIgnored private var renderSession: (key: String, session: SpeechSession)?

    private struct Tag {
        let text: String
        let source: SpeechTake.Source
        let wordOffset: Int
        let blockID: UUID?
        let date: Date
    }

    private struct VoiceSnapshot {
        let description: String
        let language: String
        let parameters: TTSParameters
        let seed: Int
    }

    struct RenderState: Equatable {
        let id: UUID
        let blockID: UUID?
        var progress: Double
    }

    /// Keep memory bounded: 24 kHz float mono is ~5.8 MB a minute.
    private static let maxTakes = 60

    private init() {
        player.onWillPlay = { [weak self] in
            guard let self, let coordinator = self.coordinator, coordinator.state.isActive else {
                return
            }
            coordinator.stop()
        }
        player.onClockChange = { [weak self] take, clock in
            guard let self else { return }
            if let take, let clock {
                self.readAlong.beginReplay(take: take, clock: clock)
            } else {
                self.readAlong.endReplay()
            }
        }
        loadYourVoices()
        // Launch flag for checking speed without opening a menu.
        if UserDefaults.standard.object(forKey: "speechPrototype.rate") != nil {
            playbackRate = Float(UserDefaults.standard.double(forKey: "speechPrototype.rate"))
        }
    }

    func attach(
        settings: SettingsManager, coordinator: SpeechCoordinator, engine: SpeechEnginePresenter
    ) {
        self.settings = settings
        self.coordinator = coordinator
        self.enginePresenter = engine
    }

    // MARK: - Speaking

    /// Speak `text` from the page. `wordOffset`: document words before it,
    /// so the read-along can highlight the right place in a longer text.
    func speak(
        _ text: String, source: SpeechTake.Source = .page, wordOffset: Int = 0, blockID: UUID? = nil
    ) {
        guard let coordinator, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        player.stop()
        pendingTag = Tag(
            text: text, source: source, wordOffset: wordOffset, blockID: blockID, date: .now)
        coordinator.speakText(text, userInitiated: true)
    }

    /// "Try another take": re-render the current voice from the opening of
    /// `sample` and keep the new take from now on (ADR-0072).
    func tryAnotherTake(sample: String) {
        guard let coordinator else { return }
        player.stop()
        let trimmed = sample.trimmingCharacters(in: .whitespacesAndNewlines)
        pendingTag = Tag(
            text: trimmed, source: .voiceTake, wordOffset: 0, blockID: nil, date: .now)
        coordinator.tryAnotherTake(sampleFrom: trimmed)
    }

    func stop() {
        coordinator?.stop()
        player.stop()
    }

    func pause() { coordinator?.pause() }
    func resume() { coordinator?.resume() }

    func togglePause() {
        guard let coordinator else { return }
        if case .paused = coordinator.state { coordinator.resume() } else { coordinator.pause() }
    }

    var status: SpeechLabStatus {
        if let render { return .rendering(render.progress) }
        if let enginePresenter, enginePresenter.isLoading {
            return .loadingModel(enginePresenter.loadingStatus)
        }
        guard let coordinator else { return .idle }
        switch coordinator.state {
        case .idle: return .idle
        case .capturingText: return .capturing
        case .generating: return .preparing
        case .streaming, .playing: return .speaking(segment: 1, of: 1)
        case .streamingLongForm(let segment, let total):
            return .speaking(segment: segment, of: total)
        case .paused(let segment, let total): return .paused(segment: segment, of: total)
        case .error(let message): return .error(message)
        }
    }

    var isPaused: Bool {
        if case .paused = coordinator?.state { return true }
        return false
    }

    /// Something is audible or about to be: live speech or a replay.
    var isSounding: Bool { status.isActive || player.isPlaying }

    // MARK: - Draft facts (cheap: one UTF-8 pass, no allocation)

    var draftWordCount: Int { Self.wordCount(draft) }

    /// A rough spoken length: ~15 characters a second at the default pace.
    var draftEstimatedSeconds: Int { Self.estimatedSeconds(draft) }

    nonisolated static func wordCount(_ text: String) -> Int {
        var count = 0
        var inWord = false
        for byte in text.utf8 {
            let isSpace = byte == 0x20 || byte == 0x0A || byte == 0x09 || byte == 0x0D
            if isSpace {
                inWord = false
            } else if !inWord {
                inWord = true
                count += 1
            }
        }
        return count
    }

    nonisolated static func estimatedSeconds(_ text: String) -> Int {
        Int((Double(text.utf8.count) / 15).rounded())
    }

    // MARK: - Voices

    nonisolated static let presets: [LabVoice] = [
        LabVoice(
            name: "Natural",
            description:
                "A natural, clear voice with a moderate pace and neutral tone, suitable for everyday conversations.",
            kind: .preset, blurb: "Clear and neutral, everyday pace"),
        LabVoice(
            name: "Warm",
            description:
                "A warm, friendly female voice with a gentle tone and smooth cadence, comforting and approachable.",
            kind: .preset, blurb: "Friendly, gentle, smooth"),
        LabVoice(
            name: "Deep",
            description:
                "A deep, resonant male narrator voice with a measured pace and authoritative presence.",
            kind: .preset, blurb: "Resonant narrator, measured"),
        LabVoice(
            name: "Calm",
            description:
                "A calm, soothing voice with a slow, deliberate pace, perfect for reading and relaxation.",
            kind: .preset, blurb: "Slow and soothing, for long reads"),
        LabVoice(
            name: "Bright",
            description:
                "A bright, energetic young female voice with a quick pace and a smile in every sentence.",
            kind: .preset, blurb: "Young, quick, upbeat"),
        LabVoice(
            name: "Anchor",
            description:
                "A crisp, confident male news anchor voice with precise diction and a steady, professional pace.",
            kind: .preset, blurb: "Crisp diction, steady and formal"),
        LabVoice(
            name: "Storyteller",
            description:
                "An older male storyteller with a warm, slightly raspy voice, unhurried and full of expression.",
            kind: .preset, blurb: "Older, raspy, expressive"),
        LabVoice(
            name: "Close",
            description:
                "A soft, breathy female voice speaking close to the microphone, quiet and intimate.",
            kind: .preset, blurb: "Soft and close, almost a whisper"),
    ]

    var allVoices: [LabVoice] { Self.presets + yourVoices }

    var currentVoiceDescription: String { settings?.ttsVoiceDescription ?? "" }

    var currentVoiceName: String {
        yourVoices.first { $0.description == currentVoiceDescription }?.name
            ?? Self.voiceName(for: currentVoiceDescription)
    }

    var currentLanguage: String { settings?.ttsLanguage ?? TTSLanguage.english.rawValue }

    func select(_ voice: LabVoice) {
        settings?.ttsVoiceDescription = voice.description
    }

    func setVoiceDescription(_ description: String) {
        settings?.ttsVoiceDescription = description
        rememberVoice(description)
    }

    nonisolated static func voiceName(for description: String) -> String {
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "Default voice" }
        if let preset = presets.first(where: { $0.description == trimmed }) { return preset.name }
        return shortName(trimmed)
    }

    /// "A warm, slightly raspy older man…" → "Warm, slightly raspy".
    nonisolated static func shortName(_ description: String) -> String {
        var words = description.split(whereSeparator: \.isWhitespace).map(String.init)
        if let first = words.first, ["a", "an", "the"].contains(first.lowercased()) {
            words.removeFirst()
        }
        let head = words.prefix(3).joined(separator: " ")
            .trimmingCharacters(in: CharacterSet(charactersIn: ",.;:"))
        guard let first = head.first else { return "Custom voice" }
        return first.uppercased() + head.dropFirst()
    }

    /// Adds a designed description to "your voices" if it is new.
    func rememberVoice(_ description: String) {
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
            !Self.presets.contains(where: { $0.description == trimmed }),
            !yourVoices.contains(where: { $0.description == trimmed })
        else { return }
        yourVoices.insert(
            LabVoice(
                name: Self.shortName(trimmed), description: trimmed, kind: .yours,
                blurb: "Designed by you"),
            at: 0)
    }

    /// Save a designed voice under a name of your choosing.
    func nameVoice(_ description: String, name: String) {
        let trimmed = description.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !Self.presets.contains(where: { $0.description == trimmed }) else {
            return
        }
        let display = name.trimmingCharacters(in: .whitespaces)
        yourVoices.removeAll { $0.description == trimmed }
        yourVoices.insert(
            LabVoice(
                name: display.isEmpty ? Self.shortName(trimmed) : display, description: trimmed,
                kind: .yours, blurb: "Designed by you"),
            at: 0)
    }

    /// The designed voices already pinned on disk (ADR-0072): each has a
    /// Reference Take, so it is the same person every time.
    private func loadYourVoices() {
        struct Entry: Decodable { let key: String }
        let url = StorageEnvironment.applicationSupport
            .appendingPathComponent("Tesseract Agent", isDirectory: true)
            .appendingPathComponent("pinned_voices.json")
        guard let data = try? Data(contentsOf: url),
            let entries = try? JSONDecoder().decode([Entry].self, from: data)
        else { return }
        var seen = Set(Self.presets.map(\.description))
        var voices: [LabVoice] = []
        for entry in entries.reversed() {
            let parts = entry.key.split(
                separator: "|", maxSplits: 2, omittingEmptySubsequences: false)
            guard parts.count == 3 else { continue }
            let description = String(parts[2])
            guard !description.isEmpty, !seen.contains(description) else { continue }
            seen.insert(description)
            voices.append(
                LabVoice(
                    name: Self.shortName(description), description: description, kind: .yours,
                    blurb: "Pinned · \(parts[1])"))
        }
        yourVoices = voices
    }

    // MARK: - Voice takes

    /// "Try another take" renders of `description`, newest first — the
    /// audition strip. Each carries the Reference Take it made.
    func voiceTakes(for description: String) -> [SpeechTake] {
        takes.filter {
            $0.voiceDescription == description && $0.source == .voiceTake && $0.pinnedVoice != nil
        }
    }

    /// The strip entry whose Reference Take the voice uses now.
    func keptTakeID(for description: String) -> UUID? {
        let key = "\(description)#\(keptVersion)#\(takes.count)"
        if let cached = keptCache, cached.key == key { return cached.id }
        let current = PinnedVoiceStore().voice(
            description: description.isEmpty ? nil : description, language: currentLanguage,
            model: ModelDefinition.textToSpeechModelSpec)
        let id = voiceTakes(for: description).first {
            $0.pinnedVoice?.codeFrames == current?.codeFrames
                && $0.pinnedVoice?.referenceText == current?.referenceText
        }?.id
        keptCache = (key, id)
        return id
    }

    @ObservationIgnored private var keptCache: (key: String, id: UUID?)?

    /// Make `take`'s Reference Take the voice from now on ("Use take").
    func keepVoice(of take: SpeechTake) {
        guard let pinned = take.pinnedVoice, let coordinator else { return }
        player.stop()
        keptVersion += 1
        Task {
            await coordinator.keep(pinned)
            keptVersion += 1
        }
    }

    /// Bumped when a kept take changes, so strips re-read which is kept.
    private(set) var keptVersion = 0

    // MARK: - Takes

    func takes(forBlock id: UUID) -> [SpeechTake] {
        takes.filter { $0.blockID == id }
    }

    func delete(_ take: SpeechTake) {
        if player.takeID == take.id { player.stop() }
        takes.removeAll { $0.id == take.id }
    }

    func clearTakes() {
        player.stop()
        takes.removeAll()
    }

    /// Insert an externally-built take (a render, a merged export).
    func add(_ take: SpeechTake) {
        takes.insert(take, at: 0)
        if takes.count > Self.maxTakes { takes.removeLast(takes.count - Self.maxTakes) }
    }

    // MARK: - Recording taps (called by the decorators)

    func tapStartStreaming(sampleRate: Int) {
        finalizeRecording(complete: false)
        player.stop()
        let tag = consumeTag()
        let fallbackText = coordinator?.currentText ?? ""
        let source: SpeechTake.Source = tag?.source ?? (fallbackText.isEmpty ? .app : .selection)
        recordingSnapshot = snapshot()
        recordingSamples = []
        recordingSamples.reserveCapacity(sampleRate * 30)
        recordingSegments = []
        recordingSampleRate = sampleRate
        peakTail = []
        let requestText = tag?.text ?? (fallbackText.isEmpty ? nil : fallbackText)
        liveTake = LiveTake(
            id: UUID(), source: source, blockID: tag?.blockID, wordOffset: tag?.wordOffset ?? 0,
            requestText: requestText, startedAt: .now)
        readAlong.prepareLive(
            takeID: liveTake?.id, wordOffset: tag?.wordOffset ?? 0, requestText: requestText,
            source: source)
    }

    func tapAppend(samples: [Float]) {
        guard liveTake != nil, !samples.isEmpty else { return }
        recordingSamples.append(contentsOf: samples)
        peakTail.append(contentsOf: samples)
        let binSize = TakePeaks.binSize(sampleRate: recordingSampleRate)
        let whole = peakTail.count / binSize * binSize
        var newBins: [Float] = []
        if whole > 0 {
            newBins = TakePeaks.peaks(of: peakTail[0..<whole], binSize: binSize)
            peakTail.removeFirst(whole)
        }
        liveTake?.peaks.append(contentsOf: newBins)
        liveTake?.duration = Double(recordingSamples.count) / Double(recordingSampleRate)
    }

    func tapFinishStreaming() {
        liveTake?.isGenerationFinished = true
        finalizeRecording(complete: true)
    }

    func tapStop() {
        finalizeRecording(complete: false)
    }

    func tapPlaybackFinished() {
        readAlong.playbackFinished()
        overlayController.utteranceEnded()
    }

    func tapSegment(text: String, base: TimeInterval) {
        let firstWord = recordingSegments.last.map { $0.firstWord + $0.timeline.words.count } ?? 0
        let segment = ReadAlongMap.Segment(
            timeline: WordTimeline(text: text), base: base, end: nil, firstWord: firstWord)
        recordingSegments.append(segment)
        readAlong.appendSegment(segment)
    }

    func tapScheduledDuration(_ cumulative: TimeInterval) {
        if !recordingSegments.isEmpty {
            recordingSegments[recordingSegments.count - 1].end = cumulative
        }
        readAlong.setScheduledEnd(cumulative)
    }

    func tapGenerationComplete() {
        readAlong.markGenerationComplete()
    }

    func tapDismiss() {
        readAlong.endLive()
        overlayController.utteranceEnded()
    }

    /// The source of the utterance being spoken, for the overlay's scope.
    var liveSource: SpeechTake.Source? { liveTake?.source }

    private func consumeTag() -> Tag? {
        defer { pendingTag = nil }
        guard let tag = pendingTag, Date.now.timeIntervalSince(tag.date) < 120 else { return nil }
        return tag
    }

    private func snapshot() -> VoiceSnapshot {
        VoiceSnapshot(
            description: settings?.ttsVoiceDescription ?? "",
            language: settings?.ttsLanguage ?? "English",
            parameters: settings?.ttsParameters ?? TTSParameters(),
            seed: settings?.ttsSeed ?? 0)
    }

    private func finalizeRecording(complete: Bool) {
        guard let live = liveTake else { return }
        liveTake = nil
        let samples = recordingSamples
        recordingSamples = []
        let segments = recordingSegments
        recordingSegments = []
        let snapshot = recordingSnapshot ?? self.snapshot()
        // Under a quarter second is a false start, not a take.
        guard samples.count > recordingSampleRate / 4 else { return }
        let spoken = segments.map { $0.timeline.words.map(\.text).joined(separator: " ") }
            .joined(separator: " ")
        let text = spoken.isEmpty ? (live.requestText ?? "") : spoken
        // The coordinator stores the session's Reference Take after every
        // segment, so by the end of the stream the file holds this take's.
        // A stopped retake keeps the old one, so it gets none.
        let pinned: PinnedVoice? =
            complete || live.source != .voiceTake
            ? PinnedVoiceStore().voice(
                description: snapshot.description.isEmpty ? nil : snapshot.description,
                language: snapshot.language, model: ModelDefinition.textToSpeechModelSpec)
            : nil
        var take = SpeechTake(
            id: live.id, createdAt: live.startedAt, text: text, source: live.source,
            blockID: live.blockID, voiceDescription: snapshot.description,
            language: snapshot.language, parameters: snapshot.parameters, seed: snapshot.seed,
            audio: TakeAudio(samples: samples, sampleRate: recordingSampleRate),
            segments: segments, wordOffset: live.wordOffset, isComplete: complete)
        take.pinnedVoice = pinned
        add(take)
        if live.source == .voiceTake || live.source == .page { rememberVoice(snapshot.description) }
    }

    // MARK: - Offline render (faster than real time, not played)

    /// Renders `text` in the current voice straight to a take, without the
    /// speakers — the engine's eager pacing, ~3–4× faster than real time.
    /// Stops any live speech first: the engine speaks one thing at a time.
    @discardableResult
    func renderOffline(_ text: String, blockID: UUID? = nil, seed: Int? = nil) async -> SpeechTake?
    {
        guard let settings, let enginePresenter,
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, render == nil
        else { return nil }
        coordinator?.stop()
        player.stop()
        let job = RenderState(id: UUID(), blockID: blockID, progress: 0)
        render = job
        defer { render = nil }

        let snapshot = self.snapshot()
        let description = snapshot.description.isEmpty ? nil : snapshot.description
        let spec = ModelDefinition.textToSpeechModelSpec
        do {
            if !enginePresenter.isModelLoaded {
                enginePresenter.noteLoading("Loading voice model…")
            }
            let key = "\(description ?? "")|\(snapshot.language)"
            let session: SpeechSession
            if let cached = renderSession, cached.key == key {
                session = cached.session
            } else {
                let pinned = PinnedVoiceStore().voice(
                    description: description, language: snapshot.language, model: spec)
                let voice: Voice =
                    pinned.map { .pinned($0) }
                    ?? description.map { .designed(description: $0, language: snapshot.language) }
                    ?? .standard(language: snapshot.language)
                session = try await enginePresenter.engine.session(
                    SessionProfile(reference: .pinned, pacing: .eager), voice: voice)
                renderSession = (key, session)
            }
            enginePresenter.noteReady()

            let seedValue = seed ?? snapshot.seed
            let utterance = try await session.speak(
                text,
                options: SpeechOptions(
                    seed: .fixed(UInt64(clamping: seedValue)), parameters: snapshot.parameters))
            var samples: [Float] = []
            var segments: [ReadAlongMap.Segment] = []
            let total = max(utterance.segmentCount, 1)
            for try await event in utterance.events {
                switch event {
                case .segment(let script):
                    let firstWord =
                        segments.last.map { $0.firstWord + $0.timeline.words.count } ?? 0
                    segments.append(
                        ReadAlongMap.Segment(
                            timeline: WordTimeline(text: script.text),
                            base: Double(samples.count) / Double(utterance.sampleRate), end: nil,
                            firstWord: firstWord))
                case .audio(let chunk):
                    samples.append(contentsOf: chunk.samples)
                case .segmentDone(let index):
                    if !segments.isEmpty {
                        segments[segments.count - 1].end =
                            Double(samples.count) / Double(utterance.sampleRate)
                    }
                    render?.progress = Double(index + 1) / Double(total)
                case .finished:
                    break
                }
            }
            guard !samples.isEmpty else { return nil }
            let take = SpeechTake(
                id: UUID(), createdAt: .now, text: text, source: .render, blockID: blockID,
                voiceDescription: snapshot.description, language: snapshot.language,
                parameters: snapshot.parameters, seed: seedValue,
                audio: TakeAudio(samples: samples, sampleRate: utterance.sampleRate),
                segments: segments, wordOffset: 0, isComplete: true)
            add(take)
            return take
        } catch {
            enginePresenter.noteFailed()
            Log.speech.error("[SpeechLab] render failed: \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: - Sample text

    static let sampleText = """
        The lighthouse keeper climbed the ninety-nine steps every night, a lantern in one hand and a book in the other. From the top, the town looked like a scatter of embers along the shore.

        He read aloud to the empty sea, slowly, the way his mother used to read to him. And the sea, as it always did, listened.

        Tonight the wind carried a different sound. Somewhere past the rocks, a small boat was ringing its bell, again and again, as if asking the light a question.
        """
}
