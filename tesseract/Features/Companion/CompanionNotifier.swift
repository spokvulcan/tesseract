//
//  CompanionNotifier.swift
//  tesseract
//

import Foundation
import UserNotifications

// MARK: - CompanionPingOutcome

/// How a heartbeat ping ended, per the anchor decision (#302): engage or
/// recorded dismissal — never a silent give-up. (`expired` is stamped by the
/// heartbeat itself when the next beat fires over an unanswered ping.)
nonisolated enum CompanionPingOutcome: String, Sendable {
    /// Clicked through to the app.
    case engaged
    /// Answered inline from the notification — the zero-friction feedback hook.
    case replied
    /// Explicitly waved off ("Not now" or closing the banner).
    case dismissed
}

// MARK: - CompanionNotifier

/// PROTOTYPE — the walking skeleton's notification surface (map #301, #303).
///
/// Wraps `UNUserNotificationCenter`: one category with a Reply text action and
/// a "Not now" action, `customDismissAction` so even closing the banner is a
/// recorded dismissal. First `UserNotifications` use in the app — the TODO at
/// `PersonalAssistantPackage.swift` notwithstanding, nothing else owns the
/// center's delegate today.
final class CompanionNotifier {

    nonisolated static let categoryID = "companion.ping"
    nonisolated static let replyActionID = "companion.reply"
    nonisolated static let laterActionID = "companion.later"
    nonisolated static let beatUserInfoKey = "beat"

    var onOutcome: ((UUID, String, CompanionPingOutcome, String?) -> Void)?

    private let delegate = CompanionNotificationDelegate()
    private var isArmed = false

    /// Installs the delegate + category (once) and requests authorization.
    /// Returns whether notifications are authorized.
    func activate() async -> Bool {
        let center = UNUserNotificationCenter.current()
        if !isArmed {
            delegate.onOutcome = { [weak self] pingID, beatID, outcome, note in
                self?.onOutcome?(pingID, beatID, outcome, note)
            }
            center.delegate = delegate
            let reply = UNTextInputNotificationAction(
                identifier: Self.replyActionID, title: "Reply", options: [],
                textInputButtonTitle: "Send",
                textInputPlaceholder: "An answer — or how this ping felt")
            let later = UNNotificationAction(
                identifier: Self.laterActionID, title: "Not now", options: [])
            let category = UNNotificationCategory(
                identifier: Self.categoryID, actions: [reply, later],
                intentIdentifiers: [], options: [.customDismissAction])
            center.setNotificationCategories([category])
            isArmed = true
        }
        do {
            let granted = try await center.requestAuthorization(options: [.alert, .sound])
            Log.companion.info("Notification authorization granted: \(granted)")
            return granted
        } catch {
            Log.companion.error("Notification authorization request failed: \(error)")
            return false
        }
    }

    /// Posts one ping immediately; the ping's UUID is the request identifier,
    /// so outcomes route back to the exact `fired` line in the log.
    func post(pingID: UUID, beatID: String, title: String, body: String) async {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        content.categoryIdentifier = Self.categoryID
        content.userInfo = [Self.beatUserInfoKey: beatID]
        let request = UNNotificationRequest(
            identifier: pingID.uuidString, content: content, trigger: nil)
        do {
            try await UNUserNotificationCenter.current().add(request)
        } catch {
            Log.companion.error("Posting ping \(pingID.uuidString) failed: \(error)")
        }
    }
}

// MARK: - CompanionNotificationDelegate

/// Nonisolated because notification-center callbacks arrive off the main
/// thread; every callback hops to the MainActor notifier.
nonisolated final class CompanionNotificationDelegate: NSObject,
    UNUserNotificationCenterDelegate
{
    /// Written exactly once on the MainActor before the delegate is installed
    /// (`center.delegate =` publishes it); read-only afterwards.
    nonisolated(unsafe) var onOutcome:
        (@MainActor @Sendable (UUID, String, CompanionPingOutcome, String?) -> Void)?

    func userNotificationCenter(
        _ center: UNUserNotificationCenter, willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        // The ping must land even while Tesseract is frontmost.
        [.banner, .sound]
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse
    ) async {
        let request = response.notification.request
        guard request.content.categoryIdentifier == CompanionNotifier.categoryID,
            let pingID = UUID(uuidString: request.identifier)
        else { return }
        let beatID =
            request.content.userInfo[CompanionNotifier.beatUserInfoKey] as? String ?? "unknown"

        let outcome: CompanionPingOutcome
        var note: String?
        switch response.actionIdentifier {
        case CompanionNotifier.replyActionID:
            outcome = .replied
            note = (response as? UNTextInputNotificationResponse)?.userText
        case CompanionNotifier.laterActionID, UNNotificationDismissActionIdentifier:
            outcome = .dismissed
        default:
            // UNNotificationDefaultActionIdentifier — clicked through.
            outcome = .engaged
        }

        await onOutcome?(pingID, beatID, outcome, note)
    }
}
