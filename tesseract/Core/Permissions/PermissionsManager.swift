//
//  PermissionsManager.swift
//  tesseract
//

import Foundation
import Combine
import AVFoundation
import AppKit

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
    @Published private(set) var screenRecordingPermission: PermissionState = .unknown

    init() {
        checkMicrophonePermission()
        checkAccessibilityPermission()
        checkScreenRecordingPermission()
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

        pollForGrant(check: { [weak self] in
            self?.checkAccessibilityPermission()
            return self?.accessibilityPermission == .granted
        })
    }

    /// Poll a TCC grant the system gives no callback for: re-check once a
    /// second for a minute, stopping early once `check` reports granted.
    private func pollForGrant(check: @escaping @MainActor () -> Bool) {
        Task {
            for _ in 0..<60 {
                try? await Task.sleep(for: .seconds(1))
                if check() {
                    break
                }
            }
        }
    }

    // MARK: - Screen Recording Permission (Appshots)

    func checkScreenRecordingPermission() {
        screenRecordingPermission = CGPreflightScreenCaptureAccess() ? .granted : .denied
    }

    /// Requested lazily — only the Appshot flow ever asks, so users who never
    /// take one never see macOS's Screen Recording re-nags. Triggers the
    /// one-time system prompt, opens the Privacy pane, and polls: the grant
    /// lands in System Settings and (per macOS) may only apply after relaunch.
    func requestScreenRecordingPermission() {
        CGRequestScreenCaptureAccess()
        openSystemPreferences(for: "screenRecording")

        pollForGrant(check: { [weak self] in
            self?.checkScreenRecordingPermission()
            return self?.screenRecordingPermission == .granted
        })
    }

    // MARK: - System Settings

    func openSystemPreferences(for permission: String) {
        let urlString: String
        switch permission {
        case "microphone":
            urlString = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        case "accessibility":
            urlString =
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        case "screenRecording":
            urlString =
                "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
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
