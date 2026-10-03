//
//  TargetApp.swift
//  tesseract
//
//  Where a take is going: the app in front when the take started (PRD
//  #612). Learned Words are left alone per app, the Lens names the app, and
//  a fix is put back only while that app is still in front.
//

import AppKit

nonisolated struct TargetApp: Equatable, Sendable {
    let bundleID: String?
    let name: String
    let pid: pid_t

    /// The frontmost app now. A non-activating panel being key (the Lens,
    /// the capture panel) does not change it.
    @MainActor
    static func frontmost() -> TargetApp? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        return TargetApp(
            bundleID: app.bundleIdentifier, name: app.localizedName ?? "the app",
            pid: app.processIdentifier)
    }
}
