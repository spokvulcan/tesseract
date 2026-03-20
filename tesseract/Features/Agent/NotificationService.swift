//
//  NotificationService.swift
//  tesseract
//

import Foundation
import UserNotifications

// MARK: - TaskCompletionInfo

nonisolated struct TaskCompletionInfo: Sendable {
    let taskId: UUID
    let taskName: String
    let sessionId: UUID
    let runId: UUID
    let result: TaskRunResult
    let notifyUser: Bool
    let isHeartbeat: Bool

    var summary: String { result.displaySummary }
}

// MARK: - NotificationService

@MainActor
final class NotificationService {

    // MARK: - Constants

    nonisolated static let scheduledTaskCategoryId = "scheduledTask"
    nonisolated static let viewSessionActionId = "viewSession"
    nonisolated static let sessionIdKey = "sessionId"
    nonisolated static let taskIdKey = "taskId"

    // MARK: - Dependencies

    private let settings: SettingsManager

    // MARK: - State

    private(set) var isAuthorized = false

    // MARK: - Init

    init(settings: SettingsManager) {
        self.settings = settings
        registerCategories()
    }

    // MARK: - Authorization

    func requestAuthorization() async {
        let center = UNUserNotificationCenter.current()

        do {
            let granted = try await center.requestAuthorization(options: [.alert, .sound, .badge])
            isAuthorized = granted
            Log.agent.info("NotificationService: authorization \(granted ? "granted" : "denied")")
        } catch {
            isAuthorized = false
            Log.agent.error("NotificationService: authorization request failed — \(error)")
        }
    }

    // MARK: - Category Registration

    private func registerCategories() {
        let viewAction = UNNotificationAction(
            identifier: Self.viewSessionActionId,
            title: "View Session",
            options: .foreground
        )

        let category = UNNotificationCategory(
            identifier: Self.scheduledTaskCategoryId,
            actions: [viewAction],
            intentIdentifiers: []
        )

        UNUserNotificationCenter.current().setNotificationCategories([category])
    }

    // MARK: - Posting

    /// Posts a system notification if all gates pass. Returns whether a notification was posted.
    @discardableResult
    func postIfNeeded(for info: TaskCompletionInfo) -> Bool {
        // Gate 1: Global kill switch
        guard settings.showNotifications else { return false }

        // Gate 2: Runtime permission
        guard isAuthorized else { return false }

        // Gate 3: Per-task opt-in
        guard info.notifyUser else { return false }

        // Gate 4: Heartbeat filter — only .success is interesting
        if info.isHeartbeat {
            guard case .success = info.result else { return false }
        }

        // Gate 5: Skip non-actionable results
        switch info.result {
        case .interrupted, .missed, .noActionNeeded:
            return false
        case .success, .error:
            break
        }

        let content = UNMutableNotificationContent()
        content.title = info.taskName
        content.body = info.summary
        content.categoryIdentifier = Self.scheduledTaskCategoryId
        content.userInfo = [
            Self.sessionIdKey: info.sessionId.uuidString,
            Self.taskIdKey: info.taskId.uuidString,
        ]

        if settings.playSounds {
            content.sound = .default
        }

        let request = UNNotificationRequest(
            identifier: info.runId.uuidString,
            content: content,
            trigger: nil
        )

        UNUserNotificationCenter.current().add(request) { error in
            if let error {
                Log.agent.error("NotificationService: failed to post notification — \(error)")
            }
        }

        Log.agent.info("NotificationService: posted notification for '\(info.taskName)'")
        return true
    }
}
