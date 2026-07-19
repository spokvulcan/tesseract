//
//  SandboxMigration.swift
//  tesseract
//

import Foundation
import MLXAudioCore

/// One-time, models-only migration for the un-sandbox cutover (#381, ADR-0047).
///
/// When the agent left the App Sandbox, `applicationSupportDirectory` stopped
/// resolving into the per-app container and now points at
/// `~/Library/Application Support`. Everything the sandboxed build wrote —
/// memory, conversations, settings, and the ~160 GB of downloaded models —
/// still sits in the old container. The owner's call (grilled 2026-07-19):
/// carry **only** the models across so the un-sandboxed build boots as a clean
/// install with models already present; memory, conversations, and settings
/// start fresh on purpose.
///
/// The move is a same-volume rename — instant, no 160 GB copy. It is
/// idempotent (no-ops once the models live at the new path), never merges into
/// a populated destination, and never touches the old container beyond lifting
/// the one directory out of it. A failure degrades to a fresh install (the app
/// re-downloads) rather than taking launch down.
enum SandboxMigration {

    /// The shared root every model family (agent, TTS, ASR) stores under —
    /// sourced from the vendored constant so it can't drift from the real path.
    private static let modelsDirName = ModelUtils.storageDirectoryName

    /// The models directory inside the retired sandbox container, expressed
    /// relative to the real home. Non-sandboxed, `NSHomeDirectory()` is the
    /// real home, so this resolves to the old container path.
    private static let containerRelativeModels =
        "Library/Containers/app.tesseract.agent/Data/Library/Application Support/\(modelsDirName)"

    /// Carry the downloaded models across, once. Must run before anything reads
    /// the model storage path (called from `applicationDidFinishLaunching`,
    /// ahead of `container.setup`).
    ///
    /// Merge, not replace: an empty destination takes one atomic directory
    /// rename; a destination that already holds some models (e.g. a
    /// pre-existing non-sandboxed `models` dir) is filled in per model, and a
    /// model already present at the destination is always kept — the migration
    /// never clobbers or deletes anything the new path already has. Idempotent:
    /// a second run finds every model present and moves nothing.
    static func migrateModelsIfNeeded(fileManager: FileManager = .default) {
        // Never move the owner's models during a test run — test hosts share
        // the real container (issue #360), and a non-sandboxed test host would
        // otherwise relocate live data mid-suite.
        guard !ProcessEnvironment.isRunningTests else { return }

        let newModels = URL.applicationSupportDirectory.appendingPathComponent(modelsDirName)
        let oldModels = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(containerRelativeModels)

        // Same location (still sandboxed, or the paths coincide): nothing to do.
        guard oldModels.standardizedFileURL != newModels.standardizedFileURL else { return }

        // The source must exist and hold something.
        guard hasContents(oldModels, fileManager) else { return }

        do {
            // Clean destination: one atomic, same-volume directory rename.
            if !fileManager.fileExists(atPath: newModels.path) {
                try fileManager.createDirectory(
                    at: newModels.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                try fileManager.moveItem(at: oldModels, to: newModels)
                Log.general.info(
                    "SandboxMigration: moved models dir → \(newModels.path)")
                return
            }

            // Destination already holds models — merge per model, keeping every
            // model the new path already has (it may hold ones the container
            // lacks). Each move is a same-volume rename.
            var moved = 0
            var kept = 0
            for entry in try fileManager.contentsOfDirectory(atPath: oldModels.path) {
                let destination = newModels.appendingPathComponent(entry)
                guard !fileManager.fileExists(atPath: destination.path) else {
                    kept += 1
                    continue
                }
                try fileManager.moveItem(
                    at: oldModels.appendingPathComponent(entry), to: destination)
                moved += 1
            }
            Log.general.info(
                "SandboxMigration: merged models — moved \(moved), kept \(kept) already present"
                    + " → \(newModels.path)")
        } catch {
            Log.general.error(
                "SandboxMigration: models migration failed, starting fresh: "
                    + error.localizedDescription)
        }
    }

    /// True iff `url` is a directory holding at least one entry.
    private static func hasContents(_ url: URL, _ fileManager: FileManager) -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else { return false }
        let entries = (try? fileManager.contentsOfDirectory(atPath: url.path)) ?? []
        return !entries.isEmpty
    }
}
