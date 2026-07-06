//
//  AppshotController.swift
//  tesseract
//
//  The **Appshot** flow (PRD #170): the frontmost-window capture invoked by the
//  double-Command hotkey, staged as a pending composer image labeled with its
//  source app and window title. Everything above the `AppshotCapturing` seam:
//  ingest, staging through the Composer Draft (whose vision-hint, cap, and notice
//  rules apply unchanged), the window-label prefill, failure routing, and
//  the window summon.
//

import Foundation
import Observation

@Observable @MainActor
final class AppshotController {

    // MARK: - Observable State

    /// Drives the Screen Recording explainer (lazy permission UX): set when a
    /// capture fails on the missing permission, cleared by the banner.
    var showPermissionExplainer = false

    // MARK: - Dependencies

    /// Summons the main window onto the Agent view. Wired by the app delegate,
    /// like the menu bar's activation callbacks.
    @ObservationIgnored var onSummon: (() -> Void)?

    @ObservationIgnored private let capturer: any AppshotCapturing
    @ObservationIgnored private let composerDraft: ComposerDraftController

    init(capturer: any AppshotCapturing, composerDraft: ComposerDraftController) {
        self.capturer = capturer
        self.composerDraft = composerDraft
    }

    // MARK: - Flow

    /// Take one Appshot: capture, stage, summon. Every outcome summons the
    /// window — the staged image, the switch hint, the notice, or the
    /// permission explainer is the feedback, and all of them live there.
    func takeAppshot() async {
        switch await capturer.capture() {
        case .success(let capture):
            stage(capture)
        case .failure(.noPermission):
            showPermissionExplainer = true
        case .failure(.noCapturableWindow), .failure(.captureFailed):
            composerDraft.showNotice("Couldn't capture the frontmost window.")
        }
        onSummon?()
    }

    private func stage(_ capture: AppshotCapture) {
        let ingested = ImageIngest.ingest(
            data: capture.pngData,
            typeIdentifier: "image/png",
            filename: Self.filename(appName: capture.appName, windowTitle: capture.windowTitle)
        )
        switch ingested {
        case .success(let attachment):
            let added = composerDraft.handleGesture(ImageGesturePayload(attachments: [attachment]))
            // Offer the window label only when the shot actually staged (a switch
            // hint or a full queue already speaks for itself) and the composer is
            // empty — the user's own draft always wins. Written straight to the
            // draft the composer owns, so it lands even if the view isn't mounted
            // when the Appshot fires.
            if !added.isEmpty, composerDraft.text.isEmpty {
                composerDraft.text = Self.prefill(
                    appName: capture.appName, windowTitle: capture.windowTitle)
            }
        case .failure(let rejection):
            composerDraft.handleGesture(ImageGesturePayload(rejections: [rejection]))
        }
    }

    // MARK: - Labels

    /// The attachment filename shown on the composer chip: "App — Title.png".
    static func filename(appName: String?, windowTitle: String?) -> String {
        let parts = [appName, windowTitle].compactMap(\.self).filter { !$0.isEmpty }
        guard !parts.isEmpty else { return "Appshot.png" }
        let name = parts.joined(separator: " — ")
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
        return "\(name.prefix(120)).png"
    }

    /// The editable composer line telling the model what it's looking at.
    static func prefill(appName: String?, windowTitle: String?) -> String {
        switch (appName, windowTitle) {
        case (let app?, let title?) where !app.isEmpty && !title.isEmpty:
            return "Appshot of \(app) — “\(title)”."
        case (let app?, _) where !app.isEmpty:
            return "Appshot of \(app)."
        default:
            return "Appshot of the frontmost window."
        }
    }
}
