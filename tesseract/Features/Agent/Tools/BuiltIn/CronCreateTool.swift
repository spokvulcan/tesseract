//
//  CronCreateTool.swift
//  tesseract
//

import Foundation
import MLXLMCommon

// MARK: - Constants

private nonisolated enum CronToolConstants {
    static let maxActiveTasks = 50
    static let minimumIntervalSeconds: TimeInterval = 300
}

private nonisolated(unsafe) let iso8601Formatter: ISO8601DateFormatter = {
    let fmt = ISO8601DateFormatter()
    fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return fmt
}()

// MARK: - CronCreateTool Factory

nonisolated func createCronCreateTool(schedulingService: SchedulingService) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "cron_create",
        label: "cron_create",
        description: "Create a new scheduled task that runs on a cron schedule. The task will execute a prompt at the specified times. Minimum interval is 5 minutes.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "name": PropertySchema(
                    type: "string",
                    description: "Short display name for the task"
                ),
                "cron": PropertySchema(
                    type: "string",
                    description: "5-field cron expression (minute hour dayOfMonth month dayOfWeek). Example: '0 9 * * 1-5' for weekdays at 9am"
                ),
                "prompt": PropertySchema(
                    type: "string",
                    description: "Instruction to execute on each run"
                ),
                "description": PropertySchema(
                    type: "string",
                    description: "What the task does"
                ),
                "speak": PropertySchema(
                    type: "boolean",
                    description: "Speak the result via TTS (default: false)"
                ),
                "max_runs": PropertySchema(
                    type: "integer",
                    description: "Maximum number of runs before auto-disable"
                ),
            ],
            required: ["name", "cron", "prompt"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let name = ToolArgExtractor.string(argsJSON, key: "name") else {
                return .error("Missing required parameter: name")
            }
            guard let cronExpr = ToolArgExtractor.string(argsJSON, key: "cron") else {
                return .error("Missing required parameter: cron")
            }
            guard let prompt = ToolArgExtractor.string(argsJSON, key: "prompt") else {
                return .error("Missing required parameter: prompt")
            }

            let description = ToolArgExtractor.string(argsJSON, key: "description") ?? ""
            let speak = ToolArgExtractor.bool(argsJSON, key: "speak") ?? false
            let maxRuns = ToolArgExtractor.int(argsJSON, key: "max_runs")

            if let maxRuns, maxRuns < 1 {
                return .error("max_runs must be at least 1 (got \(maxRuns)).")
            }

            // Parse cron expression
            let parsed: CronExpression
            do {
                parsed = try CronExpression(parsing: cronExpr)
            } catch {
                return .error("Invalid cron expression '\(cronExpr)': \(error.localizedDescription)")
            }

            // Min-interval check: compute two consecutive occurrences and verify gap >= 5 minutes
            let epoch = Date(timeIntervalSince1970: 0)
            if let first = parsed.nextOccurrence(after: epoch),
               let second = parsed.nextOccurrence(after: first) {
                let gap = second.timeIntervalSince(first)
                if gap < CronToolConstants.minimumIntervalSeconds {
                    return .error("Schedule interval too short (\(Int(gap))s). Minimum interval is 5 minutes (300s).")
                }
            }

            // Max-tasks check
            let activeCount = await MainActor.run {
                schedulingService.tasks.filter(\.enabled).count
            }
            if activeCount >= CronToolConstants.maxActiveTasks {
                return .error("Maximum active tasks limit reached (\(CronToolConstants.maxActiveTasks)). Disable or delete existing tasks first.")
            }

            // Create and save the task
            let task: ScheduledTask
            do {
                task = try ScheduledTask.create(
                    name: name,
                    cronExpression: cronExpr,
                    prompt: prompt,
                    description: description,
                    createdBy: .agent(reason: "Created via cron_create tool"),
                    maxRuns: maxRuns,
                    notifyUser: true,  // Forced true for agent-created tasks
                    speakResult: speak
                )
            } catch {
                return .error("Failed to create task: \(error.localizedDescription)")
            }

            await MainActor.run {
                schedulingService.createTask(task)
            }

            let nextRunStr = task.nextRunAt.map { iso8601Formatter.string(from: $0) } ?? "unknown"
            return .text(
                "Created scheduled task '\(task.name)' (\(task.id))\n"
                + "Schedule: \(cronExpr) (\(parsed.humanReadable))\n"
                + "Next run: \(nextRunStr)"
            )
        }
    )
}
