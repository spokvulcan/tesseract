//
//  PermissionsManager.swift
//  whisper-on-device
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
        // Request accessibility permission with prompt
        // Use the raw string key to avoid concurrency issues with the global constant
        let promptKey = "AXTrustedCheckOptionPrompt" as CFString
        let options: CFDictionary = [promptKey: kCFBooleanTrue as Any] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)

        // Poll for permission change since there's no callback
        Task {
            for _ in 0..<30 {
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
        microphonePermission == .granted
        // Note: Accessibility is optional for clipboard-based injection
    }
}
