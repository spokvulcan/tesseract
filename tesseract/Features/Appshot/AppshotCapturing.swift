//
//  AppshotCapturing.swift
//  tesseract
//

import Foundation

/// One captured Appshot: the frontmost window's image, already encoded under
/// the ingest byte cap, plus the metadata that labels it for the user and the
/// model.
struct AppshotCapture: Equatable, Sendable {
    let pngData: Data
    let appName: String?
    let windowTitle: String?
}

/// Why an Appshot could not be captured.
enum AppshotCaptureError: Error, Equatable {
    /// Screen Recording permission is missing — route to the explainer.
    case noPermission
    /// Nothing capturable is frontmost (no on-screen window to shoot).
    case noCapturableWindow
    /// The capture or encoding itself failed.
    case captureFailed
}

/// The single Appshot seam: everything impure about taking one — window
/// enumeration, ScreenCaptureKit, the permission check — lives behind this
/// protocol, so the flow above it is driven with a fake in tests.
protocol AppshotCapturing {
    func capture() async -> Result<AppshotCapture, AppshotCaptureError>
}
