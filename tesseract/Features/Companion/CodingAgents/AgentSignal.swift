//
//  AgentSignal.swift
//  tesseract
//
//  A coding agent reporting in. Claude Code's hooks post a small JSON signal
//  to a localhost-only route when an agent is waiting on the owner
//  (`Notification`) or has finished (`Stop`). Waiting agents show in Waiting
//  on you; the Day Engine decides the rest without a model.
//

import Foundation

nonisolated struct AgentSignal: Sendable, Equatable, Hashable, Codable, Identifiable {
    enum Kind: String, Sendable, Equatable, Codable {
        /// The agent needs the owner (a permission prompt, a question, idle input).
        case waiting
        /// The agent finished its turn.
        case finished
    }

    /// The agent session: one item per session, the latest signal wins.
    let id: String
    var kind: Kind
    var agent: String
    /// The project's folder name.
    var project: String
    var directory: String
    var message: String
    var at: Date

    /// "Claude Code in tesseract".
    var title: String { project.isEmpty ? agent : "\(agent) in \(project)" }

    /// Parse a Claude Code hook's stdin, which the hook command forwards as
    /// the request body. Unknown or missing fields degrade to empty text; a
    /// body that isn't JSON still produces a signal, so a hook never fails
    /// silently.
    static func parse(hookBody: Data?, kind: Kind, at: Date) -> AgentSignal {
        let object =
            hookBody.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            ?? [:]
        let directory = object["cwd"] as? String ?? ""
        let session = object["session_id"] as? String ?? "claude-\(directory)"
        var message = object["message"] as? String ?? ""
        if message.isEmpty {
            message = kind == .waiting ? "Waiting for you" : "Finished"
        }
        return AgentSignal(
            id: session, kind: kind, agent: "Claude Code",
            project: directory.isEmpty ? "" : URL(fileURLWithPath: directory).lastPathComponent,
            directory: directory, message: message, at: at)
    }
}

/// Apps where a coding agent's terminal lives: in front means the owner can
/// already see the agent.
nonisolated enum TerminalApps {
    static let bundleIDs: Set<String> = [
        "com.apple.Terminal", "com.googlecode.iterm2", "dev.warp.Warp-Stable",
        "com.mitchellh.ghostty", "net.kovidgoyal.kitty", "com.github.wez.wezterm",
        "org.alacritty", "co.zeit.hyper", "com.microsoft.VSCode", "com.todesktop.230313mzl4w4u92",
        "dev.zed.Zed", "com.anthropic.claudefordesktop",
    ]

    static func contains(_ bundleID: String?) -> Bool {
        bundleID.map(bundleIDs.contains) ?? false
    }
}
