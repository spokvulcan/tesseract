import Foundation
import MLXLMCommon
import UserNotifications

// MARK: - Model

struct Reminder: Codable, Sendable {
    let id: UUID
    let message: String
    let triggerDate: Date
    let createdAt: Date
    var delivered: Bool

    init(message: String, triggerDate: Date) {
        self.id = UUID()
        self.message = message
        self.triggerDate = triggerDate
        self.createdAt = Date()
        self.delivered = false
    }
}

// MARK: - Set Reminder Tool

struct ReminderSetTool: AgentTool {
    let name = "reminder_set"
    let description = "Set a reminder that will show as a system notification at the specified time"
    let parameters: [ToolParameter] = [
        .required("message", type: .string, description: "Reminder message"),
        .required("time", type: .string, description: "When to remind (e.g. 'in 30 minutes', 'at 3pm', 'tomorrow at 9am', ISO8601)"),
    ]

    let store: AgentDataStore

    func execute(arguments: [String: JSONValue]) async throws -> String {
        guard let message = arguments.string(for: "message") else {
            throw AgentToolError.missingArgument("message")
        }
        guard let timeStr = arguments.string(for: "time") else {
            throw AgentToolError.missingArgument("time")
        }
        guard let triggerDate = DateParsingUtility.parse(timeStr) else {
            throw AgentToolError.invalidArgument("time", "could not parse \"\(timeStr)\". Try: 'in 30 minutes', 'at 3pm', 'tomorrow at 9am', or ISO8601")
        }
        guard triggerDate > Date() else {
            throw AgentToolError.invalidArgument("time", "reminder time must be in the future")
        }

        // Request notification permission if needed
        let center = UNUserNotificationCenter.current()
        let settings = await center.notificationSettings()
        if settings.authorizationStatus == .notDetermined {
            let granted = try await center.requestAuthorization(options: [.alert, .sound])
            if !granted {
                return "Notification permission denied. Enable notifications in System Settings to use reminders."
            }
        } else if settings.authorizationStatus == .denied {
            return "Notifications are disabled. Enable them in System Settings > Notifications > Tesseract."
        }

        // Schedule the notification
        let reminder = Reminder(message: message, triggerDate: triggerDate)

        let content = UNMutableNotificationContent()
        content.title = "Tesse Reminder"
        content.body = message
        content.sound = .default

        let interval = triggerDate.timeIntervalSinceNow
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: max(interval, 1), repeats: false)
        let request = UNNotificationRequest(identifier: reminder.id.uuidString, content: content, trigger: trigger)
        try await center.add(request)

        // Persist
        await store.append(reminder, to: "reminders.json")

        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return "Done. Reminder set: \"\(message)\" at \(formatter.string(from: triggerDate))."
    }
}
