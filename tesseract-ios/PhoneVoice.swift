//
//  PhoneVoice.swift
//  tesseract-ios
//
//  The phone's neural voice (#515, ADR-0084): its download, its Voice
//  Preparation and Speed Check, and which voice reads each segment. Until it
//  is ready, and wherever it can't run, the System Voice reads, and `state`
//  says why.
//

import Combine
import CoreML
import Foundation
import Observation
import TesseractSpeech

@Observable @MainActor
final class PhoneVoice {
    enum State: Equatable {
        /// Not on the phone; the System Voice reads.
        case notDownloaded
        case downloading(received: Int64, total: Int64)
        /// **Voice Preparation**: the downloaded checkpoint becoming the
        /// Neural Engine graphs this phone runs.
        case preparing(VoicePreparationPhase)
        /// The neural voice reads, at the rates its Speed Check allows.
        case ready(SpeedCheck)
        /// The System Voice reads, for this reason.
        case unavailable(String)
    }

    private(set) var state: State = .notDownloaded

    let downloads: ModelDownloadManager
    let fetching: PhoneModelFetching
    let synthesizer: Qwen3Synthesizer
    @ObservationIgnored private let trimming: TrimmingModelFetching<PhoneModelFetching>
    @ObservationIgnored private let settings: PhoneSettings
    @ObservationIgnored private var statusWatch: AnyCancellable?
    @ObservationIgnored private var preparation: Task<Void, Never>?
    @ObservationIgnored private var progressTicker: Task<Void, Never>?
    @ObservationIgnored private var started = false
    /// The voice failed while reading: it prepares again when the app is
    /// next in front.
    @ObservationIgnored private var stopped = false

    private let model = ModelDefinition.phoneVoice
    @ObservationIgnored private var record: Record

    /// What the voice remembers across launches.
    struct Record: Codable, Equatable {
        /// The owner asked for the voice: a relaunch carries the download on.
        var requested = false

        static func load(from url: URL) -> Record {
            (try? JSONDecoder().decode(Record.self, from: Data(contentsOf: url))) ?? Record()
        }

        func save(to url: URL) {
            try? JSONEncoder().encode(self).write(to: url)
        }
    }

    init(settings: PhoneSettings) {
        self.settings = settings
        let fetching = PhoneModelFetching(allowsCellular: { settings.allowsCellularDownloads })
        let trimming = TrimmingModelFetching(base: fetching, trims: ModelDefinition.phoneVoiceTrims)
        let downloads = ModelDownloadManager(
            fetching: trimming, definitions: [ModelDefinition.phoneVoice])
        let checkpoint = downloads.modelPath(for: ModelDefinition.phoneVoice.id)!
        self.fetching = fetching
        self.trimming = trimming
        self.downloads = downloads
        synthesizer = Qwen3Synthesizer(
            checkpointDirectory: { _ in checkpoint }, neuralEngineCache: Self.graphsDirectory,
            neuralVoice: true)
        record = Record.load(from: Self.recordURL)
        state =
            !Self.hasNeuralEngine
            ? .unavailable(Self.noNeuralEngine)
            : downloads.isDownloaded(model.id) ? .preparing(.loading) : .notDownloaded
    }

    /// Whether Core ML can use a Neural Engine here: never in the simulator,
    /// where MLX can't run either.
    static let hasNeuralEngine = MLModel.availableComputeDevices.contains {
        if case .neuralEngine = $0 { return true }
        return false
    }

    private static let noNeuralEngine =
        "This device has no Neural Engine for the neural voice, so the system voice reads."

    /// The compiled graphs: kept with the models, out of backups (they come
    /// back from the checkpoint), where the system won't purge them as it
    /// does Caches; a purge would cost a whole Voice Preparation.
    static let graphsDirectory: URL = {
        let url = URL.applicationSupportDirectory.appendingPathComponent(
            "neural-voice", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        var mutable = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? mutable.setResourceValues(values)
        return url
    }()

    private static var recordURL: URL { graphsDirectory.appendingPathComponent("voice.json") }

    // MARK: - Which voice reads

    /// The voice for the next segment: the neural one when it is ready and
    /// the Thermal Policy lets it read.
    var choice: VoiceHandover.Choice {
        guard case .ready = state else { return .fallback }
        return ThermalPolicy.decision(for: ProcessInfo.processInfo.thermalState) == .neuralVoice
            ? .primary : .fallback
    }

    /// Whether the voice's files are on the phone.
    var isDownloaded: Bool { downloads.isDownloaded(model.id) }

    /// Whether the neural voice is the one reading.
    var isReading: Bool {
        if case .ready = state { return true }
        return false
    }

    /// The speed menu's rates: those the neural voice keeps up with, once it
    /// reads.
    var playableRates: [Double] {
        if case .ready(let check) = state { return check.playableRates() }
        return PlaybackRate.menu
    }

    /// What Settings and the Library say about the voice.
    var summary: String {
        switch state {
        case .notDownloaded:
            "The system voice reads until the neural voice (\(model.sizeDescription)) is on this iPhone."
        case .downloading(let received, let total):
            total > 0
                ? "Downloading the neural voice: \(Self.bytes(received)) of \(Self.bytes(total))."
                : "Downloading the neural voice."
        case .preparing(let phase):
            "Preparing the neural voice: \(Self.describe(phase)). The system voice reads meanwhile."
        case .ready(let check):
            check.fastestRate().map {
                "The neural voice reads, up to \($0.formatted(.number.precision(.fractionLength(0 ... 2))))×."
            } ?? "The neural voice reads."
        case .unavailable(let reason):
            reason
        }
    }

    // MARK: - Lifecycle

    /// Each time the app is in front. The first time, a voice on disk is
    /// prepared (in seconds after the first time), and a download the owner
    /// started goes on; after the voice failed, it prepares again.
    func start() {
        guard Self.hasNeuralEngine else { return }
        guard !started else {
            if stopped, isDownloaded {
                stopped = false
                prepare(reloading: true)
            }
            return
        }
        started = true
        statusWatch = downloads.$statuses.sink { [weak self] _ in
            Task { @MainActor in self?.statusChanged() }
        }
        if downloads.isDownloaded(model.id) {
            prepare()
        } else if record.requested {
            download()
        }
    }

    func download() {
        guard Self.hasNeuralEngine else { return }
        setRequested(true)
        downloads.download(modelID: model.id)
    }

    func cancelDownload() {
        setRequested(false)
        downloads.cancelDownload(modelID: model.id)
    }

    /// The neural voice failed on a segment before any of its audio, and the
    /// System Voice read the segment (`VoiceHandover`). The System Voice
    /// reads on until the app is next in front.
    func failed(_ error: Error) {
        guard case .ready = state else { return }
        stopped = true
        state = .unavailable(
            "The neural voice stopped, so the system voice reads until you next open Tesseract. "
                + error.localizedDescription)
    }

    private func setRequested(_ requested: Bool) {
        record.requested = requested
        record.save(to: Self.recordURL)
    }

    /// Removes the voice and its graphs; the System Voice reads again.
    func delete() async {
        preparation?.cancel()
        await synthesizer.unload()
        stopped = false
        setRequested(false)
        downloads.deleteModel(modelID: model.id)
        fetching.discardLeftovers()
        for item
            in (try? FileManager.default.contentsOfDirectory(
                at: Self.graphsDirectory, includingPropertiesForKeys: nil)) ?? []
        where item.lastPathComponent != Self.recordURL.lastPathComponent {
            try? FileManager.default.removeItem(at: item)
        }
        state = Self.hasNeuralEngine ? .notDownloaded : .unavailable(Self.noNeuralEngine)
    }

    private func statusChanged() {
        switch downloads.status(for: model.id) {
        case .downloading, .verifying:
            if progressTicker == nil { startProgressTicker() }
        case .downloaded:
            stopProgressTicker()
            if case .ready = state { return }
            prepare()
        case .error(let message):
            stopProgressTicker()
            state = .unavailable("The neural voice's download stopped: \(message)")
        case .notDownloaded:
            stopProgressTicker()
            if case .downloading = state { state = .notDownloaded }
        }
    }

    /// The download's bytes, twice a second: the files finished on disk and
    /// the one moving now.
    private func startProgressTicker() {
        progressTicker = Task { [weak self] in
            while !Task.isCancelled {
                self?.updateProgress()
                try? await Task.sleep(for: .milliseconds(500))
            }
        }
    }

    private func stopProgressTicker() {
        progressTicker?.cancel()
        progressTicker = nil
    }

    private func updateProgress() {
        guard let directory = downloads.modelPath(for: model.id), let repo = model.repoID else {
            return
        }
        let finished = Self.size(of: directory)
        let total = Int64(trimming.listedBytes[repo] ?? 0)
        state = .downloading(received: finished + fetching.activity.received, total: total)
    }

    // MARK: - Voice Preparation

    /// `reloading`: from the checkpoint again, as after a failure (a
    /// prepared voice keeps only its Neural Engine copy).
    private func prepare(reloading: Bool = false) {
        guard preparation == nil else { return }
        state = .preparing(.loading)
        preparation = Task { [weak self] in
            guard let self else { return }
            defer { preparation = nil }
            do {
                if reloading { await synthesizer.unload() }
                _ = try await synthesizer.prepareNeuralVoice(.customVoice06B) { [weak self] phase in
                    Task { @MainActor in
                        guard let self, case .preparing = self.state else { return }
                        self.state = .preparing(phase)
                    }
                }
                let check = try await speedCheck()
                if check.keepsUpAtNormalSpeed {
                    if let fastest = check.fastestRate(), settings.ttsPlaybackRate > fastest {
                        settings.ttsPlaybackRate = fastest
                    }
                    state = .ready(check)
                } else {
                    state = .unavailable(check.verdict ?? "")
                }
            } catch is CancellationError {
            } catch {
                // A recovery tries again the next time; a first preparation
                // that fails would only fail again.
                if reloading { stopped = true }
                state = .unavailable(
                    "The neural voice couldn't be prepared, so the system voice reads. "
                        + error.localizedDescription)
            }
        }
    }

    /// The **Speed Check**: a short render, timed. It also warms the voice,
    /// so the first reading starts fast.
    private func speedCheck() async throws -> SpeedCheck {
        let timing = try await synthesizer.timeRender(
            "Here is how quickly the neural voice reads on this iPhone.",
            speaker: settings.phoneVoice, language: "English")
        return SpeedCheck(realTimeFactor: timing.realTimeFactor, firstAudio: timing.firstAudio)
    }

    // MARK: - Formatting

    private static func describe(_ phase: VoicePreparationPhase) -> String {
        switch phase {
        case .loading: "loading it"
        case .measuring: "measuring it (the first time only)"
        case .building(let part): "building its \(part)"
        case .checking: "checking it (the first time only)"
        }
    }

    private static func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }

    private static func size(of directory: URL) -> Int64 {
        let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey])
        var total: Int64 = 0
        while let file = enumerator?.nextObject() as? URL {
            let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey])
            if values?.isDirectory == false { total += Int64(values?.fileSize ?? 0) }
        }
        return total
    }
}
