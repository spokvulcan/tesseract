//
//  OnboardingTourController.swift
//  tesseract
//
//  The Onboarding Tour's one seam (PRD #171, ADR-0021, `CONTEXT.md` →
//  Onboarding tour): every behaviour of the tour that isn't pixels — Chapter
//  navigation, close-=-skip completion, the hardware-aware model pick, the
//  speech→voice→agent setup download queue, and ready-state derivation.
//  Chapter views render this controller's state and stay dumb.
//
//  The queue is strictly sequential (speech first, voice second, agent last)
//  so the small models land early enough for mid-tour Try-its; a settled
//  failure advances the queue rather than stalling it, and completion never
//  cancels an in-flight download (downloads survive skip; cancel lives on the
//  Models page).
//

import Combine
import Foundation
import Observation

@Observable @MainActor
final class OnboardingTourController {

    /// The six Chapters, in narrative order. `ready` is the honest finish.
    enum Chapter: Int, CaseIterable, Identifiable {
        case welcome
        case agent
        case dictation
        case voice
        case server
        case ready

        var id: Int { rawValue }
    }

    // MARK: - Observable state

    private(set) var chapter: Chapter = .welcome

    /// Set exactly once, by finishing *or* skipping (closing the Welcome
    /// Window) — the two are equivalent by design.
    private(set) var didComplete = false

    private(set) var hasBegunSetup = false

    /// The agent model the tour will install. Starts as the already-installed
    /// selection when there is one (a relaunch on a configured machine never
    /// re-picks), otherwise as the RAM recommendation.
    private(set) var chosenAgentModelID: String

    /// Mirror of the download manager's statuses so chapter views observing
    /// only this controller re-render as bytes land.
    private(set) var statuses: [String: ModelStatus] = [:]

    let recommendedAgentModelID: String
    let speechToTextModelID: String
    let voiceModelID: String

    /// The RAM the pick was made for — display-only ("chosen for your 64 GB Mac").
    let physicalMemoryBytes: UInt64

    // MARK: - Dependencies

    @ObservationIgnored private let settings: SettingsManager
    @ObservationIgnored private let downloadManager: ModelDownloadManager
    @ObservationIgnored private var statusCancellable: AnyCancellable?

    /// Model ids awaiting their turn; the head of the line is
    /// `activeDownloadID` once started.
    @ObservationIgnored private var pendingQueue: [String] = []
    @ObservationIgnored private var activeDownloadID: String?

    // MARK: - Init

    init(
        settings: SettingsManager,
        downloadManager: ModelDownloadManager,
        speechToTextModelID: String,
        voiceModelID: String,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) {
        self.settings = settings
        self.downloadManager = downloadManager
        self.speechToTextModelID = speechToTextModelID
        self.voiceModelID = voiceModelID
        self.physicalMemoryBytes = physicalMemoryBytes

        let recommended = OnboardingModelPick.recommendedAgentModelID(
            physicalMemoryBytes: physicalMemoryBytes)
        self.recommendedAgentModelID = recommended
        self.chosenAgentModelID =
            downloadManager.isDownloaded(settings.selectedAgentModelID)
            ? settings.selectedAgentModelID : recommended
        self.statuses = downloadManager.statuses

        self.statusCancellable = downloadManager.$statuses.sink { [weak self] statuses in
            self?.statuses = statuses
            self?.advanceQueueIfNeeded()
        }
    }

    // MARK: - Navigation

    var canGoBack: Bool { chapter != .welcome }
    var isLastChapter: Bool { chapter == .ready }

    func advance() {
        go(to: Chapter(rawValue: chapter.rawValue + 1) ?? .ready)
    }

    func goBack() {
        go(to: Chapter(rawValue: chapter.rawValue - 1) ?? .welcome)
    }

    func go(to target: Chapter) {
        chapter = target
    }

    // MARK: - Setup

    /// Kick off first-run setup: persist the chosen agent model and start the
    /// sequential download queue, skipping whatever is already on disk.
    /// Idempotent — the Welcome chapter calls it on every appearance.
    func beginSetupIfNeeded() {
        guard !hasBegunSetup else { return }
        hasBegunSetup = true

        settings.selectedAgentModelID = chosenAgentModelID
        pendingQueue = [speechToTextModelID, voiceModelID, chosenAgentModelID]
            .filter { !downloadManager.isDownloaded($0) }
        startNextIfIdle()
    }

    /// The "Change" link on the Welcome chapter. Mid-flight, the old agent
    /// download is cancelled and the new choice takes its queue slot.
    func selectAgentModel(_ id: String) {
        guard id != chosenAgentModelID else { return }
        let previous = chosenAgentModelID
        chosenAgentModelID = id

        guard hasBegunSetup else { return }
        settings.selectedAgentModelID = id

        pendingQueue.removeAll { $0 == previous || $0 == id }
        if !downloadManager.isDownloaded(id) {
            pendingQueue.append(id)
        }
        if activeDownloadID == previous {
            activeDownloadID = nil
            downloadManager.cancelDownload(modelID: previous)
        }
        startNextIfIdle()
    }

    // MARK: - Readiness

    var speechModelReady: Bool { isOnDisk(speechToTextModelID) }
    var voiceModelReady: Bool { isOnDisk(voiceModelID) }
    var agentModelReady: Bool { isOnDisk(chosenAgentModelID) }

    var isSetupComplete: Bool {
        speechModelReady && voiceModelReady && agentModelReady
    }

    /// Mean completion across the three setup targets, for the ambient
    /// indicator: downloaded counts 1, an in-flight download its fraction.
    var setupProgress: Double {
        let targets = [speechToTextModelID, voiceModelID, chosenAgentModelID]
        let total = targets.reduce(0.0) { sum, id in sum + completionFraction(of: id) }
        return total / Double(targets.count)
    }

    func status(for id: String) -> ModelStatus {
        statuses[id] ?? .notDownloaded
    }

    // MARK: - Completion

    /// Finish and skip are the same act (ADR-0021): mark onboarding completed,
    /// once. Never touches the download queue — downloads survive skip.
    func complete() {
        guard !didComplete else { return }
        didComplete = true
        settings.hasCompletedOnboarding = true
    }

    // MARK: - Queue mechanics

    private func startNextIfIdle() {
        while activeDownloadID == nil, !pendingQueue.isEmpty {
            let next = pendingQueue.removeFirst()
            downloadManager.download(modelID: next)
            if isSettled(downloadManager.status(for: next)) {
                // A no-op download (already on disk, unknown id): keep going.
                continue
            }
            activeDownloadID = next
        }
    }

    private func advanceQueueIfNeeded() {
        // Read the mirrored `statuses`, not the manager: `$statuses` delivers
        // on willSet, so at sink time the manager still holds the old value.
        guard let active = activeDownloadID,
            isSettled(status(for: active))
        else { return }
        activeDownloadID = nil
        startNextIfIdle()
    }

    private func isSettled(_ status: ModelStatus) -> Bool {
        switch status {
        case .downloading, .verifying: false
        case .notDownloaded, .downloaded, .error: true
        }
    }

    private func isOnDisk(_ id: String) -> Bool {
        ModelCatalog.isDownloaded(id, statuses: statuses)
    }

    private func completionFraction(of id: String) -> Double {
        switch status(for: id) {
        case .downloaded: 1
        case .downloading(let progress): progress
        case .verifying: 1
        case .notDownloaded, .error: 0
        }
    }
}
