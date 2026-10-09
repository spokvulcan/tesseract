//
//  NotificationRuleTool.swift
//  tesseract
//
//  The owner tunes the notification filter by talking: "never tell me about
//  CI passing" becomes a rule — app, sender, keywords → ignore, hold or
//  raise — that applies before the model ever sees a banner, and sticks.
//

import Foundation

/// Reads and writes the owner's rules (a JSON setting).
@MainActor
final class TriageRuleStore {
    private let settings: SettingsManager

    init(settings: SettingsManager) {
        self.settings = settings
    }

    var rules: [TriageRule] { TriageRules.decode(settings.companionTriageRulesJSON) }

    func add(_ rule: TriageRule) {
        settings.companionTriageRulesJSON = TriageRules.encode([rule] + rules)
    }

    @discardableResult
    func remove(id: String) -> TriageRule? {
        var all = rules
        guard let index = all.firstIndex(where: { $0.id == id }) else { return nil }
        let removed = all.remove(at: index)
        settings.companionTriageRulesJSON = TriageRules.encode(all)
        return removed
    }
}

@MainActor
func createNotificationRuleTool(store: TriageRuleStore) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "notification_rule",
        label: "notification_rule",
        description: """
            Set, list or remove the owner's rules for other apps' notifications. When they say \
            something like "never tell me about CI passing" or "always tell me when Anna writes", \
            add a rule: an app, a sender and/or keywords, and what to do — ignore (never show), \
            hold (wait for their next break) or raise (tell them at once).
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "action": PropertySchema(
                    type: "string", description: "add, list or remove.",
                    enumValues: ["add", "list", "remove"]),
                "app": PropertySchema(
                    type: "string", description: "The app's name, e.g. Slack or GitHub."),
                "sender": PropertySchema(
                    type: "string",
                    description:
                        "Who it's from: a name, matched where the banner shows its sender (the title, or a chat app's sender line)."
                ),
                "keywords": PropertySchema(
                    type: "string", description: "Comma-separated words that must all appear."),
                "then": PropertySchema(
                    type: "string", description: "ignore, hold or raise.",
                    enumValues: ["ignore", "hold", "raise"]),
                "phrase": PropertySchema(
                    type: "string", description: "The owner's own words for the rule."),
                "id": PropertySchema(
                    type: "string", description: "The rule to remove (from list)."),
            ],
            required: ["action"]),
        execute: { _, args, _, _ in
            let action = ToolArgExtractor.string(args, key: "action") ?? "list"
            let app = ToolArgExtractor.string(args, key: "app")
            let sender = ToolArgExtractor.string(args, key: "sender")
            let keywords = (ToolArgExtractor.string(args, key: "keywords") ?? "")
                .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            let then = ToolArgExtractor.string(args, key: "then")
            let phrase = ToolArgExtractor.string(args, key: "phrase")
            let id = ToolArgExtractor.string(args, key: "id")
            return try await MainActor.run {
                switch action {
                case "add":
                    guard let then, let verdict = TriageRule.Action(rawValue: then) else {
                        throw AgendaError.invalid("A rule needs then: ignore, hold or raise.")
                    }
                    let rule = TriageRule(
                        app: app, sender: sender, keywords: keywords, action: verdict,
                        phrase: phrase
                            ?? [app, sender, keywords.joined(separator: " ")]
                            .compactMap { $0 }.joined(separator: " "))
                    guard rule.isSpecific else {
                        throw AgendaError.invalid("A rule needs an app, a sender or keywords.")
                    }
                    store.add(rule)
                    return .text("Rule saved: \(describe(rule)). (id: \(rule.id))")
                case "remove":
                    guard let id, let removed = store.remove(id: id) else {
                        throw AgendaError.notFound("rule \(id ?? "")")
                    }
                    return .text("Removed the rule: \(describe(removed)).")
                default:
                    let rules = store.rules
                    guard !rules.isEmpty else { return .text("No notification rules yet.") }
                    return .text(
                        rules.map { "- \(describe($0)) [id \($0.id)]" }.joined(separator: "\n"))
                }
            }
        })
}

nonisolated private func describe(_ rule: TriageRule) -> String {
    var what: [String] = []
    if let app = rule.app { what.append("from \(app)") }
    if let sender = rule.sender { what.append("by \(sender)") }
    if !rule.keywords.isEmpty {
        what.append("mentioning \(rule.keywords.joined(separator: " + "))")
    }
    let verb =
        switch rule.action {
        case .ignore: "never show"
        case .hold: "hold for the next break"
        case .raise: "tell at once"
        }
    return "\(verb) notifications \(what.joined(separator: ", "))"
}
