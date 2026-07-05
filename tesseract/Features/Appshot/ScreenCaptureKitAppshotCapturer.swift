//
//  ScreenCaptureKitAppshotCapturer.swift
//  tesseract
//

import AppKit
import CoreGraphics
import ScreenCaptureKit

/// The impure side of the `AppshotCapturing` seam: resolves the frontmost
/// window (literally whatever is frontmost — Tesseract's own window included),
/// captures its full content via ScreenCaptureKit regardless of occlusion, and
/// encodes it under the ingest byte cap.
@MainActor
final class ScreenCaptureKitAppshotCapturer: AppshotCapturing {

    func capture() async -> Result<AppshotCapture, AppshotCaptureError> {
        guard CGPreflightScreenCaptureAccess() else {
            return .failure(.noPermission)
        }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
            guard let window = Self.frontmostWindow(in: content.windows) else {
                return .failure(.noCapturableWindow)
            }

            let filter = SCContentFilter(desktopIndependentWindow: window)
            let configuration = SCStreamConfiguration()
            configuration.width = Int(filter.contentRect.width * CGFloat(filter.pointPixelScale))
            configuration.height = Int(filter.contentRect.height * CGFloat(filter.pointPixelScale))
            configuration.captureResolution = .best
            configuration.showsCursor = false

            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter, configuration: configuration)
            guard let png = AppshotEncoder.pngDataFittingCap(image, cap: ImageIngest.maxBytes)
            else {
                Log.image.error("Appshot: PNG encoding failed for the captured window")
                return .failure(.captureFailed)
            }
            return .success(
                AppshotCapture(
                    pngData: png,
                    appName: window.owningApplication?.applicationName,
                    windowTitle: window.title
                ))
        } catch {
            Log.image.error("Appshot capture failed: \(error.localizedDescription)")
            return .failure(.captureFailed)
        }
    }

    /// The frontmost app's foremost normal window. SCShareableContent does not
    /// guarantee z-order, so ordering comes from `CGWindowListCopyWindowInfo`
    /// (documented front-to-back) and the two are joined by window ID.
    private static func frontmostWindow(in windows: [SCWindow]) -> SCWindow? {
        guard let frontApp = NSWorkspace.shared.frontmostApplication else { return nil }
        let pid = frontApp.processIdentifier

        let candidates = windows.filter { window in
            window.owningApplication?.processID == pid
                && window.windowLayer == 0
                && window.isOnScreen
        }
        guard !candidates.isEmpty else { return nil }
        guard candidates.count > 1 else { return candidates[0] }

        let info =
            CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID)
            as? [[String: Any]] ?? []
        let frontToBack = info.compactMap { $0[kCGWindowNumber as String] as? UInt32 }
        return candidates.min { lhs, rhs in
            let lhsIndex = frontToBack.firstIndex(of: lhs.windowID) ?? .max
            let rhsIndex = frontToBack.firstIndex(of: rhs.windowID) ?? .max
            return lhsIndex < rhsIndex
        }
    }
}
