//
//  CompanionNotifier.swift
//  tesseract
//
//  Tesseract's Notification Center surface, the banner rung of the Delivery
//  Ladder: Jarvis's own banners when the owner is away or the screen is
//  locked, and the event nudges scheduled with the OS. It owns the
//  notification-center delegate, so clicks and deliveries route back to the
//  Companion.
//

import AppKit
import Foundation
import UserNotifications

@MainActor
final class CompanionNotifier {

    nonisolated static let cardCategory = "companion.card"
    nonisolated static let nudgeCategory = "companion.nudge"
    nonisolated static let cardUserInfoKey = "card"
    nonisolated static let eventUserInfoKey = "event"

    /// A banner or nudge was clicked: open Today.
    var onOpen: (() -> Void)?
    /// A nudge was delivered with Tesseract in front (other deliveries are
    /// read back with `deliveredNudges()`).
    var onNudgeDelivered: ((String) -> Void)?

    private let delegate = CompanionNotificationDelegate()
    private var isArmed = false

    /// Install the delegate and categories (once) and ask for permission.
    /// Returns whether banners are allowed.
    @discardableResult
    func activate() async -> Bool {
        let center = UNUserNotificationCenter.current()
        if !isArmed {
            delegate.onResponse = { [weak self] _ in self?.onOpen?() }
            delegate.onPresent = { [weak self] id, category in
                if category == Self.nudgeCategory { self?.onNudgeDelivered?(id) }
            }
            center.delegate = delegate
            center.setNotificationCategories([
                UNNotificationCategory(
                    identifier: Self.cardCategory, actions: [], intentIdentifiers: [], options: []),
                UNNotificationCategory(
                    identifier: Self.nudgeCategory, actions: [], intentIdentifiers: [], options: []),
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
        let pending = await UNUserNotificationCenter.current().pendingNotificationRequests()
        return Set(pending.map(\.identifier).filter { $0.hasPrefix(NudgePlanner.idPrefix) })
    }

    /// The nudges macOS has delivered and still keeps in Notification Center,
    /// whether or not Tesseract was in front when they fired.
    func deliveredNudges() async -> [DeliveredNudge] {
        let delivered = await UNUserNotificationCenter.current().deliveredNotifications()
        return delivered.compactMap { notification in
            let request = notification.request
            guard request.identifier.hasPrefix(NudgePlanner.idPrefix) else { return nil }
            return DeliveredNudge(
                id: request.identifier, title: request.content.title, at: notification.date)
        }
    }

    func schedule(_ nudge: Nudge) async {
        let content = UNMutableNotificationContent()
        content.title = nudge.title
        content.body = nudge.body
        content.sound = .default
        content.categoryIdentifier = Self.nudgeCategory
        content.userInfo = [Self.eventUserInfoKey: nudge.eventID]
        content.interruptionLevel = .timeSensitive
        let parts = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute, .second], from: nudge.fireAt)
        let request = UNNotificationRequest(
            identifier: nudge.id, content: content,
            trigger: UNCalendarNotificationTrigger(dateMatching: parts, repeats: false))
        do { try await UNUserNotificationCenter.current().add(request) } catch {
            Log.companion.error("Scheduling a nudge failed: \(error)")
        }
    }

    func cancel(nudgeIDs: [String]) {
        guard !nudgeIDs.isEmpty else { return }
        UNUserNotificationCenter.current().removePendingNotificationRequests(
            withIdentifiers: nudgeIDs)
    }
}

/// Nonisolated because notification-center callbacks arrive off the main
/// thread; every callback hops to the main-actor notifier.
nonisolated final class CompanionNotificationDelegate: NSObject, UNUserNotificationCenterDelegate {

    /// Written once on the main actor before the delegate is installed.
    nonisolated(unsafe) var onResponse: (@MainActor @Sendable (String) -> Void)?
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
        await onResponse?(response.notification.request.identifier)
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
