//
//  CronDeleteTool.swift
//  tesseract
//

import Foundation
import MLXLMCommon

// MARK: - CronDeleteTool Factory

nonisolated func createCronDeleteTool(schedulingService: SchedulingService) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "cron_delete",
        label: "cron_delete",
        description: "Delete a scheduled task by its ID.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "task_id": PropertySchema(
                    type: "string",
                    description: "UUID of the task to delete"
                ),
                "reason": PropertySchema(
                    type: "string",
                    description: "Why the task is being deleted"
                ),
            ],
            required: ["task_id"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let taskIdStr = ToolArgExtractor.string(argsJSON, key: "task_id") else {
                return .error("Missing required parameter: task_id")
            }
            guard let taskId = UUID(uuidString: taskIdStr) else {
                return .error("Invalid UUID format: '\(taskIdStr)'")
            }

            let reason = ToolArgExtractor.string(argsJSON, key: "reason")

            let taskName: String? = await MainActor.run {
                let name = schedulingService.tasks.first(where: { $0.id == taskId })?.name
                if name != nil {
                    schedulingService.deleteTask(id: taskId)
                }
                return name
            }

            guard let name = taskName else {
                return .error("Task not found: \(taskId)")
            }

            let reasonSuffix = reason.map { ": \($0)" } ?? ""
            Log.agent.info("Deleted scheduled task '\(name)' (\(taskId))\(reasonSuffix)")

            return .text("Deleted task '\(name)' (\(taskId))")
        }
    )
}
