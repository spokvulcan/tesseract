//
//  CronListTool.swift
//  tesseract
//

import Foundation
import MLXLMCommon

private nonisolated(unsafe) let cronListDateFormatter: ISO8601DateFormatter = {
    let fmt = ISO8601DateFormatter()
    fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return fmt
}()

// MARK: - CronListTool Factory

nonisolated func createCronListTool(schedulingService: SchedulingService) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "cron_list",
        label: "cron_list",
        description: "List scheduled tasks. Filter by status or creator.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "filter": PropertySchema(
                    type: "string",
                    description: "Filter tasks: 'all' (default), 'active' (enabled only), 'paused' (disabled only), 'mine' (agent-created only)",
                    enumValues: ["all", "active", "paused", "mine"]
                ),
            ],
            required: []
        ),
        execute: { _, argsJSON, _, _ in
            let filter = ToolArgExtractor.string(argsJSON, key: "filter") ?? "all"

            let allTasks = await MainActor.run {
                schedulingService.tasks
            }

            let filtered: [ScheduledTask]
            switch filter {
            case "active":
                filtered = allTasks.filter(\.enabled)
            case "paused":
                filtered = allTasks.filter { !$0.enabled }
            case "mine":
                filtered = allTasks.filter { $0.createdBy.isAgent }
            default:
                filtered = allTasks
            }

            if filtered.isEmpty {
                return .text("No scheduled tasks found.")
            }

            var lines: [String] = ["\(filtered.count) scheduled task(s) (filter: \(filter)):"]
            for task in filtered {
                let status = task.enabled ? "active" : "paused"
                let nextRun = task.nextRunAt.map { cronListDateFormatter.string(from: $0) } ?? "none"
                let lastRun = task.lastRunAt.map { cronListDateFormatter.string(from: $0) } ?? "never"
                let lastResult = task.lastRunResult?.displaySummary ?? "n/a"

                lines.append("")
                lines.append("- ID: \(task.id)")
                lines.append("  Name: \(task.name)")
                lines.append("  Schedule: \(task.cronExpression) (\(task.humanReadableSchedule))")
                lines.append("  Status: \(status)")
                lines.append("  Next run: \(nextRun)")
                lines.append("  Last run: \(lastRun) (\(lastResult))")
                lines.append("  Run count: \(task.runCount)")
                lines.append("  Created by: \(task.createdBy.displayString)")
            }

            return .text(lines.joined(separator: "\n"))
        }
    )
}
