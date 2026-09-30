//
//  RetiredCompanionData.swift
//  tesseract
//
//  The one-time clean start for Companion v2. The old living memory and the
//  Mission Control conversation were test data, and the owner approved
//  deleting them, so Jarvis starts clean: the memory store is deleted, and so
//  is the standing conversation. `tasks.md` is not deleted but moved out of
//  the agent's folder, where the model can no longer see it: tasks live in
//  Reminders now, and the file may hold something the owner wants back.
//
//  Idempotent by construction: it acts only on what still exists, so running
//  it on every launch costs a few `stat`s and does nothing after the first.
//

import Foundation

nonisolated enum RetiredCompanionData {

    struct Locations: Sendable {
        /// The agent's sandbox root (holds `memory/` and `tasks.md`).
        var agentRoot: URL
        /// The conversation store's directory.
        var conversationsDirectory: URL
        /// Where retired files are kept, outside the agent's reach.
        var retiredDirectory: URL

        static var production: Locations {
            let support = StorageEnvironment.applicationSupport
                .appendingPathComponent("Tesseract Agent", isDirectory: true)
            return Locations(
                agentRoot: PathSandbox.defaultRoot,
                conversationsDirectory: support.appendingPathComponent(
                    "agent/conversations", isDirectory: true),
                retiredDirectory: support.appendingPathComponent("retired", isDirectory: true))
        }
    }

    struct Report: Equatable, Sendable {
        var deletedMemoryStore = false
        var deletedMissionControl = false
        var retiredTasksFile = false

        var didAnything: Bool { deletedMemoryStore || deletedMissionControl || retiredTasksFile }
    }

    /// Delete the retired memory store and Mission Control conversation, and
    /// move `tasks.md` aside. Returns what it did.
    static func clean(_ locations: Locations, fileManager: FileManager = .default) -> Report {
        var report = Report()

        let memory = locations.agentRoot.appendingPathComponent("memory", isDirectory: true)
        if fileManager.fileExists(atPath: memory.path) {
            report.deletedMemoryStore = (try? fileManager.removeItem(at: memory)) != nil
        }

        let missionControl = locations.conversationsDirectory.appendingPathComponent(
            "\(AgentConversation.retiredMissionControlID.uuidString).json")
        if fileManager.fileExists(atPath: missionControl.path) {
            report.deletedMissionControl = (try? fileManager.removeItem(at: missionControl)) != nil
        }

        let tasks = locations.agentRoot.appendingPathComponent("tasks.md")
        if fileManager.fileExists(atPath: tasks.path) {
            try? fileManager.createDirectory(
                at: locations.retiredDirectory, withIntermediateDirectories: true)
            var destination = locations.retiredDirectory.appendingPathComponent("tasks.md")
            if fileManager.fileExists(atPath: destination.path) {
                destination = locations.retiredDirectory.appendingPathComponent(
                    "tasks-\(Int(Date().timeIntervalSince1970)).md")
            }
            report.retiredTasksFile = (try? fileManager.moveItem(at: tasks, to: destination)) != nil
        }
        return report
    }

    /// The launch step: clean the owner's data and record what happened.
    static func cleanAtLaunch(trace: CompanionTrace) {
        let report = clean(.production)
        guard report.didAnything else { return }
        trace.record(
            .migrationWiped,
            fields: [
                "memoryStore": .bool(report.deletedMemoryStore),
                "missionControl": .bool(report.deletedMissionControl),
                "tasksFileRetired": .bool(report.retiredTasksFile),
            ])
        trace.flushForTesting()
        Log.companion.info(
            "Companion v2 clean start: memory \(report.deletedMemoryStore), Mission Control \(report.deletedMissionControl), tasks.md retired \(report.retiredTasksFile)"
        )
    }
}
