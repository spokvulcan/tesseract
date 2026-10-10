//
//  CompanionNotifier.swift
//  tesseract
//
//  Tesseract's Notification Center surface, the banner rung of the Delivery
//  Ladder: Jarvis's own banners when the owner is away or the screen is
//  locked, and the event nudges scheduled with the OS. It owns the
//  notification-center delegate, so clicks and deliveries route back to the
//  Companion. A scratch launch shares the installed app's notification center
//  (one bundle id), so there it keeps its nudges in memory and posts nothing:
//  reconciling against its empty Agenda would cancel the owner's real nudges.
//

import AppKit
import Foundation
import UserNotifications

@MainActor
final class CompanionNotifier {

    nonisolated static let cardCategory = "companion.card"
    nonisolated static let nudgeCategory = "companion.nudge"
    /// A call's nudge, with a Join button that opens its link.
    nonisolated static let callCategory = "companion.nudge.call"
    nonisolated static let joinAction = "companion.join"
    nonisolated static let cardUserInfoKey = "card"
    nonisolated static let eventUserInfoKey = "event"
    nonisolated static let linkUserInfoKey = "link"

    /// A banner or nudge was clicked: open Today.
    var onOpen: (() -> Void)?
    /// A nudge was delivered with Tesseract in front (other deliveries are
    /// read back with `deliveredNudges()`).
    var onNudgeDelivered: ((String) -> Void)?

    private let delegate = CompanionNotificationDelegate()
    private var isArmed = false
    /// False for a scratch launch: nothing reaches the OS notification center.
    private let usesOS: Bool
    /// The nudges a scratch launch scheduled, in memory only.
    private var memoryNudges: Set<String> = []

    init(usesOS: Bool = true) {
        self.usesOS = usesOS
    }

    /// Install the delegate and categories (once) and ask for permission.
    /// Returns whether banners are allowed.
    @discardableResult
    func activate() async -> Bool {
        guard usesOS else { return false }
        let center = UNUserNotificationCenter.current()
        if !isArmed {
            delegate.onResponse = { [weak self] action, link in
                // Join opens the call; any other click opens Today.
                if let url = Self.joinURL(action: action, link: link) {
                    NSWorkspace.shared.open(url)
                } else {
                    self?.onOpen?()
                }
            }
            delegate.onPresent = { [weak self] id, category in
                if category == Self.nudgeCategory || category == Self.callCategory {
                    self?.onNudgeDelivered?(id)
                }
            }
            center.delegate = delegate
            center.setNotificationCategories([
                UNNotificationCategory(
                    identifier: Self.cardCategory, actions: [], intentIdentifiers: [], options: []),
                UNNotificationCategory(
                    identifier: Self.nudgeCategory, actions: [], intentIdentifiers: [], options: []),
                UNNotificationCategory(
                    identifier: Self.callCategory,
                    actions: [
                        UNNotificationAction(
                            identifier: Self.joinAction, title: "Join", options: [.foreground])
                    ], intentIdentifiers: [], options: []),
            ])
            isArmed = true
        }
        do {
            return try await center.requestAuthorization(options: [.alert, .sound])
        } catch {
            Log.companion.error("Notification authorization failed: \(error)")
            return false
        }
    }

    /// Post one banner now.
    func post(title: String, body: String, cardID: String? = nil) async {
        guard usesOS else {
            Log.companion.info("Scratch launch: banner not posted — \(title)")
            return
        }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        content.categoryIdentifier = Self.cardCategory
        if let cardID { content.userInfo = [Self.cardUserInfoKey: cardID] }
        let request = UNNotificationRequest(
            identifier: "card.\(cardID ?? UUID().uuidString)", content: content, trigger: nil)
        do { try await UNUserNotificationCenter.current().add(request) } catch {
            Log.companion.error("Posting a banner failed: \(error)")
        }
    }

    // MARK: Nudges

    /// The ids of the nudges scheduled with the OS right now.
    func scheduledNudgeIDs() async -> Set<String> {
        guard usesOS else { return memoryNudges }
        let pending = await UNUserNotificationCenter.current().pendingNotificationRequests()
        return Set(
            pending.map(\.identifier).filter { $0.hasPrefix(NudgePlanner.familyPrefix) })
    }

    /// The nudges macOS has delivered and still keeps in Notification Center,
    /// whether or not Tesseract was in front when they fired.
    func deliveredNudges() async -> [DeliveredNudge] {
        guard usesOS else { return [] }
        let delivered = await UNUserNotificationCenter.current().deliveredNotifications()
        return delivered.compactMap { notification in
            let request = notification.request
            guard request.identifier.hasPrefix(NudgePlanner.familyPrefix) else { return nil }
            return DeliveredNudge(
                id: request.identifier, title: request.content.title, at: notification.date)
        }
    }

    /// The link a response opens: Join on a call's nudge.
    nonisolated static func joinURL(action: String, link: String?) -> URL? {
        guard action == joinAction, let link else { return nil }
        return URL(string: link)
    }

    func schedule(_ nudge: Nudge) async {
        guard usesOS else {
            memoryNudges.insert(nudge.id)
            return
        }
        let content = UNMutableNotificationContent()
        content.title = nudge.title
        content.body = nudge.body
        content.sound = .default
        // A call's nudge joins it in one click, ten minutes ahead or late.
        content.categoryIdentifier = nudge.link == nil ? Self.nudgeCategory : Self.callCategory
        var info: [String: String] = [Self.eventUserInfoKey: nudge.eventID]
        if let link = nudge.link { info[Self.linkUserInfoKey] = link.absoluteString }
        content.userInfo = info
        content.interruptionLevel = .timeSensitive
        let request = UNNotificationRequest(
            identifier: nudge.id, content: content,
            trigger: UNCalendarNotificationTrigger(
                dateMatching: Self.triggerComponents(for: nudge.fireAt), repeats: false))
        do { try await UNUserNotificationCenter.current().add(request) } catch {
            Log.companion.error("Scheduling a nudge failed: \(error)")
        }
    }

    /// The moment a nudge fires, with its time zone attached: without one a
    /// calendar trigger matches the clock time wherever the Mac is, and after
    /// a flight the nudge for a 16:00 Berlin meeting fired at 15:50 London
    /// time, fifty minutes after it began. Its id doesn't change with the
    /// zone, so it was never rescheduled.
    nonisolated static func triggerComponents(for date: Date, calendar: Calendar = .current)
        -> DateComponents
    {
        var parts = calendar.dateComponents(
            [.year, .month, .day, .hour, .minute, .second], from: date)
        parts.calendar = calendar
        parts.timeZone = calendar.timeZone
        return parts
    }

    func cancel(nudgeIDs: [String]) {
        guard !nudgeIDs.isEmpty else { return }
        guard usesOS else {
            memoryNudges.subtract(nudgeIDs)
            return
        }
        UNUserNotificationCenter.current().removePendingNotificationRequests(
            withIdentifiers: nudgeIDs)
    }
}

/// Nonisolated because notification-center callbacks arrive off the main
/// thread; every callback hops to the main-actor notifier.
nonisolated final class CompanionNotificationDelegate: NSObject, UNUserNotificationCenterDelegate {

    /// Written once on the main actor before the delegate is installed. The
    /// action chosen, and the call's link if the notification carries one.
    nonisolated(unsafe) var onResponse: (@MainActor @Sendable (String, String?) -> Void)?
    nonisolated(unsafe) var onPresent: (@MainActor @Sendable (String, String) -> Void)?

    func userNotificationCenter(
        _ center: UNUserNotificationCenter, willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        let request = notification.request
        await onPresent?(request.identifier, request.content.categoryIdentifier)
        // A nudge or banner must land even while Tesseract is frontmost.
        return [.banner, .list, .sound]
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse
    ) async {
        let link =
            response.notification.request.content.userInfo[CompanionNotifier.linkUserInfoKey]
            as? String
        await onResponse?(response.actionIdentifier, link)
    }
}

/// Brings an app to the front by its display name: a card item's "Open".
@MainActor
enum AppOpener {
    static func open(named name: String) {
        if let running = NSWorkspace.shared.runningApplications.first(where: {
            $0.localizedName == name
        }) {
            running.activate()
            return
        }
        for folder in ["/Applications", "/System/Applications", "/System/Applications/Utilities"] {
            let url = URL(fileURLWithPath: folder).appendingPathComponent("\(name).app")
            if FileManager.default.fileExists(atPath: url.path) {
                NSWorkspace.shared.openApplication(at: url, configuration: .init())
                return
            }
        }
        Log.companion.info("No app named \(name) to open")
    }
}
