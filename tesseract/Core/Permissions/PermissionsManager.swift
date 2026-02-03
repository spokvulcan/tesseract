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

    init() {
        checkMicrophonePermission()
        checkAccessibilityPermission()
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

    func openSystemPreferences(for permission: String) {
        let urlString: String
        switch permission {
        case "microphone":
            urlString = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        case "accessibility":
            urlString = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
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
