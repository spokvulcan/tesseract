//
//  RetiredCompanionDataTests.swift
//  tesseractTests
//
//  Companion v2's clean start, over scratch directories only (ADR-0073): the
//  retired memory store and Mission Control conversation are deleted,
//  tasks.md moves out of the agent's folder, other conversations are left
//  alone, and a second run does nothing.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct RetiredCompanionDataTests {

    private func scratchLocations() -> RetiredCompanionData.Locations {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("retired-companion-\(UUID().uuidString)", isDirectory: true)
        return RetiredCompanionData.Locations(
            agentRoot: root.appendingPathComponent("agent", isDirectory: true),
            conversationsDirectory: root.appendingPathComponent(
                "agent/conversations", isDirectory: true),
            retiredDirectory: root.appendingPathComponent("retired", isDirectory: true))
    }

    private func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    @Test func deletesTheOldStoresAndRetiresTasksOnce() throws {
        let locations = scratchLocations()
        let fm = FileManager.default
        let memoryDB = locations.agentRoot.appendingPathComponent("memory/memory.sqlite")
        let missionControl = locations.conversationsDirectory.appendingPathComponent(
            "\(AgentConversation.retiredMissionControlID.uuidString).json")
        let ordinaryChat = locations.conversationsDirectory.appendingPathComponent(
            "\(UUID().uuidString).json")
        let tasks = locations.agentRoot.appendingPathComponent("tasks.md")
        let note = locations.agentRoot.appendingPathComponent("notes/keep.md")
        try write("sqlite", to: memoryDB)
        try write("{}", to: missionControl)
        try write("{}", to: ordinaryChat)
        try write("- [ ] call the dentist", to: tasks)
        try write("# keep", to: note)

        let first = RetiredCompanionData.clean(locations)
        #expect(
            first
                == .init(
                    deletedMemoryStore: true, deletedMissionControl: true,
                    retiredTasksFile: true))
        #expect(!fm.fileExists(atPath: memoryDB.deletingLastPathComponent().path))
        #expect(!fm.fileExists(atPath: missionControl.path))
        #expect(!fm.fileExists(atPath: tasks.path))
        let retired = locations.retiredDirectory.appendingPathComponent("tasks.md")
        #expect(try String(contentsOf: retired, encoding: .utf8) == "- [ ] call the dentist")
        // Nothing else is touched.
        #expect(fm.fileExists(atPath: ordinaryChat.path))
        #expect(fm.fileExists(atPath: note.path))

        // Idempotent: a second run finds nothing to do.
        let second = RetiredCompanionData.clean(locations)
        #expect(!second.didAnything)
    }

    /// An earlier retired copy is never overwritten.
    @Test func neverOverwritesAnEarlierRetiredTasksFile() throws {
        let locations = scratchLocations()
        try write("old", to: locations.retiredDirectory.appendingPathComponent("tasks.md"))
        try write("new", to: locations.agentRoot.appendingPathComponent("tasks.md"))

        #expect(RetiredCompanionData.clean(locations).retiredTasksFile)
        let names = try FileManager.default.contentsOfDirectory(
            atPath: locations.retiredDirectory.path)
        #expect(names.count == 2)
        #expect(
            try String(
                contentsOf: locations.retiredDirectory.appendingPathComponent("tasks.md"),
                encoding: .utf8) == "old")
    }
}
