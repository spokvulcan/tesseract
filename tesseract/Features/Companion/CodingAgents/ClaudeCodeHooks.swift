//
//  ClaudeCodeHooks.swift
//  tesseract
//
//  The Claude Code integration: Tesseract's hook block for the
//  `Notification` (the agent waits on the owner) and `Stop` (it finished)
//  events, merged into the owner's Claude Code user settings. Each hook
//  forwards the event's JSON to a localhost-only route on Tesseract's server.
//
//  The Config Merge is pure and idempotent: Tesseract's own hooks (found by
//  their route) are replaced, everything else in the file is kept, and a
//  file that isn't valid JSON is never overwritten. The Setup One-liner
//  fetches a script from the local server that runs the same merge, and the
//  Companion settings pane runs it in-process.
//

import Foundation

nonisolated enum ClaudeCodeHooks {

    /// The route prefix; also how Tesseract recognises its own hooks.
    static let routePrefix = "/companion/agent-signal/"
    static let setupScriptPath = "/integrations/claude-code/setup.sh"
    static let mergePath = "/integrations/claude-code/merge"

    static let events: [(name: String, kind: AgentSignal.Kind)] = [
        ("Notification", .waiting),
        ("Stop", .finished),
    ]

    /// The hook command: forward the event's stdin to Tesseract, quietly, and
    /// never fail the agent if Tesseract isn't running.
    static func command(port: Int, kind: AgentSignal.Kind) -> String {
        "curl -s -m 2 -X POST -H 'Content-Type: application/json' --data-binary @- "
            + "http://127.0.0.1:\(port)\(routePrefix)\(kind.rawValue) >/dev/null 2>&1 || true"
    }

    /// The owner's Claude Code user settings.
    static var settingsURL: URL {
        StorageEnvironment.home
            .appendingPathComponent(".claude", isDirectory: true)
            .appendingPathComponent("settings.json")
    }

    // MARK: Config Merge

    enum MergeError: Error, Equatable {
        /// The existing file isn't a JSON object; it is left alone.
        case unreadable
    }

    /// The settings with Tesseract's hooks installed for `port`.
    static func merge(existing: Data?, port: Int) throws -> Data {
        var root = try parse(existing)
        var hooks = root["hooks"] as? [String: Any] ?? [:]
        for (event, kind) in events {
            var groups = withoutTesseract(hooks[event] as? [[String: Any]] ?? [])
            var group: [String: Any] = [
                "hooks": [["type": "command", "command": command(port: port, kind: kind)]]
            ]
            if event == "Notification" { group["matcher"] = "" }
            groups.append(group)
            hooks[event] = groups
        }
        root["hooks"] = hooks
        return try serialize(root)
    }

    /// The settings with Tesseract's hooks removed, everything else kept.
    static func remove(existing: Data?) throws -> Data {
        var root = try parse(existing)
        var hooks = root["hooks"] as? [String: Any] ?? [:]
        for (event, _) in events {
            let groups = withoutTesseract(hooks[event] as? [[String: Any]] ?? [])
            hooks[event] = groups.isEmpty ? nil : groups
        }
        root["hooks"] = hooks.isEmpty ? nil : hooks
        return try serialize(root)
    }

    static func isInstalled(_ data: Data?) -> Bool {
        guard let root = try? parse(data), let hooks = root["hooks"] as? [String: Any] else {
            return false
        }
        return events.allSatisfy { event, _ in
            let groups = hooks[event] as? [[String: Any]] ?? []
            return groups.contains { group in
                (group["hooks"] as? [[String: Any]] ?? []).contains(where: isTesseract)
            }
        }
    }

    private static func parse(_ data: Data?) throws -> [String: Any] {
        guard let data, !data.isEmpty,
            !String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty
        else { return [:] }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MergeError.unreadable
        }
        return object
    }

    private static func serialize(_ root: [String: Any]) throws -> Data {
        var data = try JSONSerialization.data(
            withJSONObject: root, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        data.append(0x0A)
        return data
    }

    private static func isTesseract(_ hook: [String: Any]) -> Bool {
        (hook["command"] as? String)?.contains(routePrefix) == true
    }

    /// Groups with Tesseract's hooks taken out; groups left empty go.
    private static func withoutTesseract(_ groups: [[String: Any]]) -> [[String: Any]] {
        groups.compactMap { group in
            var group = group
            let hooks = (group["hooks"] as? [[String: Any]] ?? []).filter { !isTesseract($0) }
            guard !hooks.isEmpty else { return nil }
            group["hooks"] = hooks
            return group
        }
    }

    // MARK: Installing (the settings pane)

    /// Install (or refresh) the hooks in the owner's settings file, keeping a
    /// backup of the previous file beside it.
    static func install(port: Int, at url: URL = settingsURL) throws {
        let existing = try? Data(contentsOf: url)
        let merged = try merge(existing: existing, port: port)
        try write(merged, replacing: existing, at: url)
    }

    static func uninstall(at url: URL = settingsURL) throws {
        guard let existing = try? Data(contentsOf: url) else { return }
        try write(try remove(existing: existing), replacing: existing, at: url)
    }

    static func installed(at url: URL = settingsURL) -> Bool {
        isInstalled(try? Data(contentsOf: url))
    }

    private static func write(_ data: Data, replacing existing: Data?, at url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let existing {
            try existing.write(to: url.appendingPathExtension("tesseract-backup"), options: .atomic)
        }
        try data.write(to: url, options: .atomic)
    }

    // MARK: Setup One-liner

    static func oneLiner(port: Int) -> String {
        "curl -fsSL http://127.0.0.1:\(port)\(setupScriptPath) | sh"
    }

    /// The script the one-liner runs: back up, merge through the local
    /// server, write the result into place.
    static func setupScript(port: Int) -> String {
        """
        #!/bin/sh
        # Tesseract: connect Claude Code's hooks to the Companion.
        set -e
        dir="$HOME/.claude"
        file="$dir/settings.json"
        mkdir -p "$dir"
        tmp="$(mktemp)"
        if [ -f "$file" ]; then
          cp "$file" "$file.tesseract-backup"
          body="$file"
        else
          body=/dev/null
        fi
        if curl -fsS -X POST --data-binary @"$body" http://127.0.0.1:\(port)\(mergePath) -o "$tmp"; then
          mv "$tmp" "$file"
          echo "Claude Code now reports to Tesseract. A backup is at $file.tesseract-backup."
        else
          rm -f "$tmp"
          echo "Tesseract couldn't merge $file — it was left unchanged." >&2
          exit 1
        fi

        """
    }
}
