//
//  PermissionsManager.swift
//  tesseract
//

import Foundation
import Combine
import AVFoundation
import AppKit
import UserNotifications

enum PermissionState: Sendable {
    case unknown
    case requesting
    case granted
    case denied
    case restricted
}

@MainActor
final class PermissionsManager: ObservableObject {
    @Published private(set) var microphonePermission: PermissionState = .unknown
    @Published private(set) var accessibilityPermission: PermissionState = .unknown
    @Published private(set) var notificationPermission: PermissionState = .unknown

    /// Called after notification authorization changes so NotificationService can sync its gate.
    var onNotificationAuthorizationChanged: ((Bool) -> Void)?

    init() {
        checkMicrophonePermission()
        checkAccessibilityPermission()
        // Notification permission is checked in DependencyContainer.setup() after authorization request
    }

    // MARK: - Microphone Permission

    func checkMicrophonePermission() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .notDetermined:
            microphonePermission = .unknown
        case .restricted:
            microphonePermission = .restricted
        case .denied:
            microphonePermission = .denied
        case .authorized:
            microphonePermission = .granted
        @unknown default:
            microphonePermission = .unknown
        }
    }

    func requestMicrophonePermission() async -> Bool {
        microphonePermission = .requesting

        let granted = await AVCaptureDevice.requestAccess(for: .audio)
        microphonePermission = granted ? .granted : .denied
        return granted
    }

    // MARK: - Accessibility Permission

    func checkAccessibilityPermission() {
        let trusted = AXIsProcessTrusted()
        accessibilityPermission = trusted ? .granted : .denied
    }

    func requestAccessibilityPermission() {
        // Open System Settings directly to the Accessibility panel
        // AXIsProcessTrustedWithOptions with prompt only shows a dialog once,
        // so opening System Settings is more reliable
        openSystemPreferences(for: "accessibility")

        // Poll for permission change since there's no callback
        Task {
            for _ in 0..<60 {
                try? await Task.sleep(for: .seconds(1))
                checkAccessibilityPermission()
                if accessibilityPermission == .granted {
                    break
                }
            }
        }
    }

    // MARK: - Notification Permission

    /// Queries system notification authorization and updates `notificationPermission`.
    /// Returns the resolved `PermissionState` and whether authorization is granted (for callers
    /// that need to sync `NotificationService.isAuthorized`).
    @discardableResult
    func checkNotificationPermission() async -> Bool {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        let newState: PermissionState
        let authorized: Bool
        switch settings.authorizationStatus {
        case .notDetermined:
            newState = .unknown
            authorized = false
        case .denied:
            newState = .denied
            authorized = false
        case .authorized, .provisional:
            newState = .granted
            authorized = true
        @unknown default:
            newState = .unknown
            authorized = false
        }
        if notificationPermission != newState {
            notificationPermission = newState
        }
        onNotificationAuthorizationChanged?(authorized)
        return authorized
    }

    func requestNotificationPermission() async -> Bool {
        notificationPermission = .requesting
        do {
            let granted = try await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .sound, .badge])
            let newState: PermissionState = granted ? .granted : .denied
            if notificationPermission != newState {
                notificationPermission = newState
            }
            onNotificationAuthorizationChanged?(granted)
            return granted
        } catch {
            if notificationPermission != .denied {
                notificationPermission = .denied
            }
            onNotificationAuthorizationChanged?(false)
            return false
        }
    }

    // MARK: - System Settings

    func openSystemPreferences(for permission: String) {
        let urlString: String
        switch permission {
        case "microphone":
            urlString = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        case "accessibility":
            urlString = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        case "notifications":
            urlString = "x-apple.systempreferences:com.apple.preference.notifications"
        default:
            urlString = "x-apple.systempreferences:com.apple.preference.security"
        }

        if let url = URL(string: urlString) {
            NSWorkspace.shared.open(url)
        }
    }

    var allPermissionsGranted: Bool {
        microphonePermission == .granted && accessibilityPermission == .granted
    }

    /// Returns true if the minimum required permissions (microphone) are granted.
    /// Accessibility is recommended but not strictly required.
    var minimumPermissionsGranted: Bool {
        microphonePermission == .granted
    }
}
