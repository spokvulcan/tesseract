//
//  NotificationCenterWatcher.swift
//  tesseract
//
//  The Notification Hub's perception (#378, PRD #376, ADR-0047): an
//  `AXObserver` on the `NotificationCenter` process that turns every displayed
//  banner into a `CapturedNotification` for `CompanionPerception`'s admission
//  door. This is the only offline, no-private-entitlement way to read *other*
//  apps' notifications — verified live on macOS 26.5 (research 2026-07-18); it
//  works only because the agent left the App Sandbox (#381), which forbids
//  cross-process AX reads.
//
//  The AX tree is unofficial UI, not API: a banner is an `AXGroup` with subrole
//  `AXNotificationCenterBanner`, carrying an `AXIdentifier` UUID, an
//  `AXDescription` of "app, title, subtitle, body", and `AXStaticText`
//  children. Read defensively — structured children first, the flattened
//  description as the fallback — because Apple can restructure it any release.
//  The pure event-shaping (self-exclusion, body cap, deterministic id) lives in
//  `CompanionEvent.notification`; this file is only the live AX plumbing, which
//  no test host runs (no live NC, and the #360 container-sharing rule).
//

import AppKit
import ApplicationServices
import Foundation

@MainActor
final class NotificationCenterWatcher {

    /// The banner renderer's bundle id (verified: pid running as
    /// `/System/Library/CoreServices/NotificationCenter.app`). `nonisolated`
    /// so the `@Sendable`-block match helper can read it.
    nonisolated static let notificationCenterBundleID = "com.apple.notificationcenterui"

    /// The banner subroles in the NC tree: the newest live banner, and the
    /// stacked older ones a panel-open exposes.
    private nonisolated static let bannerSubroles: Set<String> = [
        "AXNotificationCenterBanner", "AXNotificationCenterBannerStack",
    ]

    /// A bound on the tree walk — banners sit a few levels under the window;
    /// this only guards against a pathological tree, never a real one.
    private nonisolated static let maxWalkDepth = 8

    private let onNotification: (CapturedNotification) -> Void
    /// The Companion toggle. Gates the *read*, not just admission: while the
    /// Companion is off, a delivered banner's content is never even inspected.
    private let isEnabled: () -> Bool

    private var observer: AXObserver?
    private var appElement: AXUIElement?
    private var watchedPID: pid_t?
    private var launchObserver: NSObjectProtocol?
    private var terminateObserver: NSObjectProtocol?

    init(
        isEnabled: @escaping () -> Bool,
        onNotification: @escaping (CapturedNotification) -> Void
    ) {
        self.isEnabled = isEnabled
        self.onNotification = onNotification
    }

    func start() {
        // Survive a NotificationCenter restart (pid change): a crash/relaunch of
        // the renderer must not silence the Hub. Terminate detaches the stale
        // observer; launch re-attaches to the fresh pid.
        launchObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didLaunchApplicationNotification, object: nil, queue: .main
        ) { [weak self] note in
            guard Self.isNotificationCenter(note) else { return }
            MainActor.assumeIsolated { self?.attach() }
        }
        terminateObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didTerminateApplicationNotification, object: nil, queue: .main
        ) { [weak self] note in
            guard Self.isNotificationCenter(note) else { return }
            MainActor.assumeIsolated { self?.detachObserver() }
        }
        attach()
    }

    func stop() {
        for observer in [launchObserver, terminateObserver].compactMap({ $0 }) {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
        }
        launchObserver = nil
        terminateObserver = nil
        detachObserver()
    }

    // MARK: - Attach / detach

    private func attach() {
        guard let pid = Self.runningNotificationCenterPID() else { return }
        // Already bound to the live pid — nothing to do. Bound to a *stale* pid
        // (NC relaunched and `didLaunch` beat `didTerminate`, or arrived without
        // it) — detach the dead observer first, so a restart never leaves the
        // Hub watching a pid that no longer renders banners.
        if observer != nil {
            guard watchedPID != pid else { return }
            detachObserver()
        }
        let app = AXUIElementCreateApplication(pid)

        var created: AXObserver?
        let status = AXObserverCreate(pid, notificationWatcherAXCallback, &created)
        guard status == .success, let created else {
            Log.companion.error(
                "NotificationCenterWatcher: AXObserverCreate failed (\(status.rawValue))")
            return
        }
        let refcon = Unmanaged.passUnretained(self).toOpaque()
        // Window-created is the clean live-banner signal, walked for the banner
        // under it. Created-element also catches a banner group built into an
        // existing window a beat after — but it fires per element as NC builds
        // its UI, so that path is a leaf check (read only when the element is
        // itself the banner), never a walk per element. A banner seen twice
        // collapses at admission on its UUID.
        AXObserverAddNotification(
            created, app, kAXWindowCreatedNotification as CFString, refcon)
        AXObserverAddNotification(created, app, kAXCreatedNotification as CFString, refcon)
        CFRunLoopAddSource(
            CFRunLoopGetCurrent(), AXObserverGetRunLoopSource(created), .defaultMode)

        appElement = app
        observer = created
        watchedPID = pid
        Log.companion.info(
            "NotificationCenterWatcher: attached to NotificationCenter pid \(pid)")
    }

    private func detachObserver() {
        if let observer {
            CFRunLoopRemoveSource(
                CFRunLoopGetCurrent(), AXObserverGetRunLoopSource(observer), .defaultMode)
        }
        observer = nil
        appElement = nil
        watchedPID = nil
    }

    /// Called from the AX callback in its own (nonisolated) domain, on the main
    /// run loop. The AX reads happen here, so the non-Sendable `element` never
    /// crosses onto the main actor — only the toggle check and the Sendable
    /// results do. Off the toggle it reads nothing: disabled means its eyes are
    /// closed, not merely that admissions are dropped downstream.
    ///
    /// `deep` follows the notification: a new *window* is walked for the banner
    /// under it; a bare *element*-created fires per element as NC builds its UI
    /// (a firehose), so there it's a leaf check — read only when the created
    /// element is itself the banner group, never a subtree walk per element.
    fileprivate nonisolated func handleCreated(_ element: AXUIElement, deep: Bool) {
        guard MainActor.assumeIsolated({ isEnabled() }) else { return }
        let banners =
            deep
            ? Self.bannerElements(in: element)
            : (Self.isBanner(element) ? [element] : [])
        let captured = banners.compactMap(Self.read)
        guard !captured.isEmpty else { return }
        MainActor.assumeIsolated {
            for notification in captured { onNotification(notification) }
        }
    }

    // MARK: - Process lookup

    private nonisolated static func runningNotificationCenterPID() -> pid_t? {
        NSRunningApplication
            .runningApplications(withBundleIdentifier: notificationCenterBundleID)
            .first?.processIdentifier
    }

    // Called from the workspace observers' `@Sendable` blocks, so it must not
    // hop the Notification (non-Sendable) onto the main actor: `nonisolated`
    // keeps the read in the block's own domain.
    private nonisolated static func isNotificationCenter(_ note: Notification) -> Bool {
        (note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication)?
            .bundleIdentifier == notificationCenterBundleID
    }

    // MARK: - Tree walk + field reading

    /// Whether `element` is itself a banner group (a leaf check, one AX read).
    private nonisolated static func isBanner(_ element: AXUIElement) -> Bool {
        guard let subrole = string(element, kAXSubroleAttribute) else { return false }
        return bannerSubroles.contains(subrole)
    }

    /// Every banner element at or under `root`. A banner is a leaf for this
    /// walk — its own `AXStaticText` children are content, not more banners.
    private nonisolated static func bannerElements(in root: AXUIElement, depth: Int = 0)
        -> [AXUIElement]
    {
        if isBanner(root) { return [root] }
        guard depth < maxWalkDepth else { return [] }
        return children(root).flatMap { bannerElements(in: $0, depth: depth + 1) }
    }

    /// Read one banner group into the raw perception, defensively: prefer the
    /// structured `AXStaticText` children, fall back to splitting the flattened
    /// `AXDescription` ("app, title, subtitle, body").
    private nonisolated static func read(_ banner: AXUIElement) -> CapturedNotification? {
        let uuid = string(banner, kAXIdentifierAttribute)
        let description = string(banner, kAXDescriptionAttribute) ?? ""
        let descParts = description.components(separatedBy: ", ")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        // The app display name is the first token of the description — the tree
        // exposes no bundle id, so this is the only identity there is.
        guard let app = descParts.first, !app.isEmpty else { return nil }

        let texts = staticTexts(banner)
        let title: String
        let subtitle: String
        let body: String
        switch texts.count {
        case let n where n >= 3:
            title = texts[0]
            subtitle = texts[1]
            body = texts[2...].joined(separator: " ")
        case 2:
            title = texts[0]
            subtitle = ""
            body = texts[1]
        case 1:
            title = texts[0]
            subtitle = ""
            body = ""
        default:
            // No structured children — reconstruct from the description tail.
            let tail = Array(descParts.dropFirst())
            title = tail.first ?? ""
            subtitle = ""
            body = tail.dropFirst().joined(separator: " — ")
        }

        return CapturedNotification(
            app: app, title: title, subtitle: subtitle, body: body, uuid: uuid)
    }

    /// Every `AXStaticText` value at or under `root`, in tree order.
    private nonisolated static func staticTexts(_ root: AXUIElement, depth: Int = 0) -> [String] {
        var out: [String] = []
        if string(root, kAXRoleAttribute) == "AXStaticText",
            let value = string(root, kAXValueAttribute),
            !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            out.append(value.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard depth < maxWalkDepth else { return out }
        for child in children(root) {
            out.append(contentsOf: staticTexts(child, depth: depth + 1))
        }
        return out
    }

    // MARK: - AX attribute reads

    private nonisolated static func copyValue(_ element: AXUIElement, _ attribute: String)
        -> AnyObject?
    {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success
        else { return nil }
        return value
    }

    private nonisolated static func string(_ element: AXUIElement, _ attribute: String) -> String? {
        copyValue(element, attribute) as? String
    }

    private nonisolated static func children(_ element: AXUIElement) -> [AXUIElement] {
        (copyValue(element, kAXChildrenAttribute) as? [AXUIElement]) ?? []
    }
}

/// The `AXObserver` C callback — a non-capturing free function so it coerces to
/// `@convention(c)`. `nonisolated` is load-bearing: under the project's
/// MainActor default isolation a global func is otherwise actor-isolated, and
/// an isolated func cannot form a C function pointer. It runs on the main run
/// loop (the source was added there), so hopping onto the main actor is a sound
/// assertion, not a dispatch.
private nonisolated func notificationWatcherAXCallback(
    _ observer: AXObserver, _ element: AXUIElement, _ axNotification: CFString,
    _ refcon: UnsafeMutableRawPointer?
) {
    guard let refcon else { return }
    // A new window is walked for the banner under it; a bare created element is
    // read only if it's itself the banner (see `handleCreated(_:deep:)`).
    let deep =
        CFStringCompare(
            axNotification, kAXWindowCreatedNotification as CFString, []) == .compareEqualTo
    // The watcher rebuilds from refcon in this same nonisolated domain, so
    // `handleCreated` (also nonisolated) reads the element here without either
    // one crossing an isolation boundary; only its Sendable results hop to the
    // main actor.
    Unmanaged<NotificationCenterWatcher>.fromOpaque(refcon)
        .takeUnretainedValue()
        .handleCreated(element, deep: deep)
}
