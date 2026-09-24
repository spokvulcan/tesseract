//
//  RetiredCheckpoints.swift
//  tesseract
//

import Foundation
import TesseractSpeech

/// Checkpoints an earlier catalog entry downloaded into the model store that no
/// entry lists and nothing loads anymore. Today that is the Voice Engine's bf16
/// checkpoint: the entry downloaded it while `SpeechEngine` loaded the q6 repo
/// (ADR-0037), so every onboarding left ~4.5 GB on disk that the app never
/// read. The Models page only shows catalog entries, so without this the owner
/// has no in-app way to get that space back.
///
/// Removed once, at launch, off the main thread. Only the exact directories
/// named here are touched, and never one the catalog lists again. A failure is
/// logged and retried next launch; it never takes launch down.
nonisolated enum RetiredCheckpoints {

    /// Repos a previous catalog entry downloaded.
    static let repos = [TTSModelSpec.voiceDesign17B(.bf16).repo]

    /// Set once removal finishes (or finds nothing to remove), so later
    /// launches leave the store alone. That matters on a dev machine, where
    /// `v2-listen --precision bf16` puts the bf16 reference checkpoint back on
    /// purpose. Stored in standard defaults, like
    /// `SandboxMigration.completionDefaultsKey`.
    static let completionDefaultsKey = "retiredCheckpointsRemoved"

    /// Launch entry: skipped under tests and headless harness launches (which
    /// must not mutate user data), otherwise removes in the background. Call
    /// after `SandboxMigration`, so a checkpoint still in the old container has
    /// already moved to the store.
    @MainActor
    static func scheduleRemoval() {
        guard !ProcessEnvironment.isRunningTests, !TesseractApp.isHarnessLaunch else { return }
        let storageRoot = ModelDownloadManager.modelStorageURL
        let listedRepos = Set(ModelDefinition.all.compactMap(\.repoID))
        Task.detached(priority: .utility) {
            removeIfNeeded(from: storageRoot, listedRepos: listedRepos)
        }
    }

    /// Removes each retired repo's directory under `storageRoot`, skipping any
    /// repo in `listedRepos`. Sets the completion flag unless a removal
    /// failed. Returns the directories it removed.
    @discardableResult
    static func removeIfNeeded(
        from storageRoot: URL,
        listedRepos: Set<String>,
        fileManager: FileManager = .default,
        defaults: UserDefaults = .standard
    ) -> [URL] {
        guard !defaults.bool(forKey: completionDefaultsKey) else { return [] }

        var removed: [URL] = []
        var failed = false
        for repo in repos where !listedRepos.contains(repo) {
            let directory = storageRoot.appendingPathComponent(
                ModelDefinition.storageSubdirectory(forRepo: repo))
            guard fileManager.fileExists(atPath: directory.path) else { continue }
            do {
                try fileManager.removeItem(at: directory)
                removed.append(directory)
                Log.general.info("RetiredCheckpoints: removed \(directory.path)")
            } catch {
                failed = true
                Log.general.error(
                    "RetiredCheckpoints: could not remove \(directory.path), retrying next launch: "
                        + error.localizedDescription)
            }
        }

        if !failed { defaults.set(true, forKey: completionDefaultsKey) }
        return removed
    }
}
