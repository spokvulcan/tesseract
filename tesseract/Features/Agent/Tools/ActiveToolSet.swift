//
//  ActiveToolSet.swift
//  tesseract
//
//  The **Active Tool Set**: the one home for "which tools does the agent see
//  right now". The registry answers what tools *exist* (identity + precedence
//  order); this module answers what tools are *live* for one consumer under
//  the current gates — and, from that same resolved set, which prompt
//  orientation sections belong in the system prompt. Deriving both answers
//  from one resolve is the point: before this module, the callable set was
//  filtered per turn while the prompt was assembled from the unfiltered
//  registry, so the two could (and did) diverge — a prompt instructing
//  browser tools the loop had stripped, or omitting them because the MCP
//  connection raced agent bootstrap (ADR-0048).
//

import Foundation

// MARK: - ToolGating

/// The gating context one resolve runs under: the state of the Web Access
/// switch. Every conversation and every Companion moment resolves the same
/// way, so they all share one cached system-and-tools prefix.
nonisolated struct ToolGating: Sendable, Equatable {
    /// The **Web Access** switch (`webAccessEnabled`).
    var webAccessEnabled: Bool
}

// MARK: - PromptToolFacts

/// The prompt-facing facts of a *resolved* tool set — the membership answers
/// `SystemPromptAssembler` conditions its orientation sections on. Derived
/// only via ``ActiveToolSet/promptFacts(for:)`` so the prompt can never be
/// built from a different tool universe than the callable set.
nonisolated struct PromptToolFacts: Sendable, Equatable {
    /// The `use_skill` tool is live, so the skills listing belongs in the prompt.
    var hasSkillTool: Bool
    /// Browser tools are live, so the web-orientation block belongs (ADR-0028).
    var carriesBrowserTools: Bool
}

// MARK: - ActiveToolSet

/// Pure resolution: registry-ordered tools + gating in, the live set and its
/// prompt facts out. No state, no effects — the interface is the test surface.
nonisolated enum ActiveToolSet {

    /// Tool names the **Web Access** switch governs: the built-in Browser
    /// server's MCP tools — the sole web surface now that search and fetch
    /// live under `browser.*` (ADR-0028). Sourced from
    /// ``MCPServerConfig/browserToolNames`` — the same namespace the live
    /// tools are built with — so the gated set and the materialized tools
    /// can't drift.
    static let webGatedToolNames: Set<String> = MCPServerConfig.browserToolNames

    /// Resolve the live tool set. Preserves the input order (registry order
    /// is the loop's dispatch precedence).
    static func resolve(
        from all: [AgentToolDefinition], gating: ToolGating
    ) -> [AgentToolDefinition] {
        var tools = all
        if !gating.webAccessEnabled {
            tools.removeAll { webGatedToolNames.contains($0.name) }
        }
        return tools
    }

    /// The prompt facts of a resolved set. Always feed this the *resolved*
    /// tools, never the raw registry — that invariant is what keeps the
    /// system prompt and the callable set from diverging.
    static func promptFacts(for resolved: [AgentToolDefinition]) -> PromptToolFacts {
        PromptToolFacts(
            hasSkillTool: resolved.contains { $0.name == skillToolName },
            carriesBrowserTools: resolved.contains { webGatedToolNames.contains($0.name) }
        )
    }
}
