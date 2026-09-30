//
//  ClaudeCodeHooksTests.swift
//  tesseractTests
//
//  The Claude Code Config Merge over fixture settings: Tesseract's hooks go
//  in and come out, everything else is kept, a second run changes nothing,
//  and an unreadable file is never overwritten. Scratch files only.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct ClaudeCodeHooksTests {

    private func object(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func installsBothHooksIntoAnEmptyFile() throws {
        let merged = try ClaudeCodeHooks.merge(existing: nil, port: 8321)
        #expect(ClaudeCodeHooks.isInstalled(merged))
        let text = String(decoding: merged, as: UTF8.self)
        #expect(text.contains("http://127.0.0.1:8321/companion/agent-signal/waiting"))
        #expect(text.contains("http://127.0.0.1:8321/companion/agent-signal/finished"))
        let hooks = try #require(try object(merged)["hooks"] as? [String: Any])
        #expect(Set(hooks.keys) == ["Notification", "Stop"])
    }

    @Test func keepsTheOwnersSettingsAndHooks() throws {
        let existing = """
            {"model": "opus", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "say done"}]}],
             "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "audit"}]}]}}
            """
        let merged = try ClaudeCodeHooks.merge(existing: Data(existing.utf8), port: 8321)
        let root = try object(merged)
        #expect(root["model"] as? String == "opus")
        let hooks = try #require(root["hooks"] as? [String: Any])
        #expect(hooks["PreToolUse"] != nil)
        let stop = try #require(hooks["Stop"] as? [[String: Any]])
        #expect(stop.count == 2)
        #expect(String(decoding: merged, as: UTF8.self).contains("say done"))
    }

    @Test func runningItTwiceChangesNothing() throws {
        let once = try ClaudeCodeHooks.merge(existing: nil, port: 8321)
        let twice = try ClaudeCodeHooks.merge(existing: once, port: 8321)
        #expect(once == twice)
        // A new port replaces the old hooks rather than adding more.
        let moved = try ClaudeCodeHooks.merge(existing: once, port: 9000)
        #expect(!String(decoding: moved, as: UTF8.self).contains("8321"))
    }

    @Test func removingLeavesOnlyTheOwnersHooks() throws {
        let existing =
            #"{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "say done"}]}]}}"#
        let merged = try ClaudeCodeHooks.merge(existing: Data(existing.utf8), port: 8321)
        let removed = try ClaudeCodeHooks.remove(existing: merged)
        #expect(!ClaudeCodeHooks.isInstalled(removed))
        let hooks = try #require(try object(removed)["hooks"] as? [String: Any])
        #expect(Set(hooks.keys) == ["Stop"])
        let bare = try ClaudeCodeHooks.remove(
            existing: try ClaudeCodeHooks.merge(existing: nil, port: 1))
        #expect(try object(bare)["hooks"] == nil)
    }

    @Test func anUnreadableFileIsNeverOverwritten() {
        #expect(throws: ClaudeCodeHooks.MergeError.unreadable) {
            _ = try ClaudeCodeHooks.merge(existing: Data("{ not json".utf8), port: 8321)
        }
    }

    @Test func installAndUninstallKeepABackup() throws {
        let url = makeTempDir("claude-settings").appendingPathComponent("settings.json")
        try Data(#"{"model": "opus"}"#.utf8).write(to: url)
        try ClaudeCodeHooks.install(port: 8321, at: url)
        #expect(ClaudeCodeHooks.installed(at: url))
        let backup = url.appendingPathExtension("tesseract-backup")
        #expect(try String(contentsOf: backup, encoding: .utf8) == #"{"model": "opus"}"#)
        try ClaudeCodeHooks.uninstall(at: url)
        #expect(!ClaudeCodeHooks.installed(at: url))
    }

    @Test func aHookBodyBecomesASignal() {
        let body =
            #"{"session_id": "abc", "cwd": "/Users/me/src/tesseract", "hook_event_name": "Notification", "message": "Claude needs your permission to use Bash"}"#
        let signal = AgentSignal.parse(
            hookBody: Data(body.utf8), kind: .waiting, at: Date(timeIntervalSince1970: 0))
        #expect(signal.id == "abc")
        #expect(signal.project == "tesseract")
        #expect(signal.title == "Claude Code in tesseract")
        #expect(signal.message == "Claude needs your permission to use Bash")
        let bare = AgentSignal.parse(
            hookBody: nil, kind: .finished, at: Date(timeIntervalSince1970: 0))
        #expect(bare.message == "Finished")
    }

    @Test func theOneLinerRunsTheScriptFromTheLocalServer() {
        #expect(
            ClaudeCodeHooks.oneLiner(port: 8321)
                == "curl -fsSL http://127.0.0.1:8321/integrations/claude-code/setup.sh | sh")
        #expect(
            ClaudeCodeHooks.setupScript(port: 8321).contains(
                "http://127.0.0.1:8321/integrations/claude-code/merge"))
    }
}
