//
//  ImageDraftController.swift
//  tesseract
//
//  The **Image Draft** module: the composer's pending-image queue, the full-window
//  image drop, and the Quick Look preview viewer — carved out of `AgentCoordinator`
//  as a publisher-agnostic leaf. Owns its own state and dependencies, testable
//  through its interface without an Agent, command registry, or conversation store.
//
//  Analogous to ``AgentVoiceInputController`` (the voice-to-text leaf): errors are
//  local, the surface is minimal, and the coordinator retains only a reference.
//

import Foundation
import Observation
import UniformTypeIdentifiers

@Observable @MainActor
final class ImageDraftController {

    // MARK: - Observable State

    /// Images queued in the composer awaiting send. Settable so the composer can
    /// remove individual images directly (ForEach with `removeAll`).
    var pendingImages: [ImageAttachment] = []

    /// Whether the selected model can serve images, synced from the composer
    /// (which owns the capability probe). The window-level drop reads this to
    /// decide between attaching and surfacing the switch hint.
    var imageInputAvailable = false

    /// Drives the composer's "switch to a vision model" hint — set when an image is
    /// pasted or dropped onto a model that can't see images.
    var showImageSwitchHint = false

    /// True while an image-bearing drag hovers the window or the composer —
    /// drives the "Drop image to attach" overlay. One source for both the
    /// SwiftUI window drop and the composer text view's AppKit drag tracking.
    var isDropTargeted = false

    /// The transient composer notice for an Image Gesture that couldn't fully
    /// attach (cap trim, oversize, unreadable). Auto-dismisses; the composer's
    /// ✕ clears it directly. An Image Gesture never falls back to pasting text,
    /// so its failures must speak here (issue #167).
    var attachmentNotice: String?

    /// Invalidates a scheduled auto-dismiss when a newer notice replaces it.
    @ObservationIgnored private var noticeGeneration = 0

    /// The pending Quick Look open request. Set by `openQuickLook`, presented by
    /// the `QuickLookContainer` host, cleared when the panel closes.
    private(set) var quickLookRequest: QuickLookRequest?

    // MARK: - Constants

    /// Max images queued in the composer at once (PRD #112 cap).
    static let maxPendingImages = 8

    // MARK: - Dependencies

    /// Digest-keyed temp-file cache backing the Quick Look viewer.
    @ObservationIgnored private let imagePreviewCache: ImagePreviewFileCache

    /// Every committed image in the conversation, in message order — user
    /// attachments and tool-result images — the navigable set the preview-set
    /// projection pages through. Injected as a closure so the module stays free of
    /// any `Agent` reference.
    @ObservationIgnored private let conversationImages: @MainActor () -> [ImageAttachment]

    // MARK: - Init

    init(
        conversationImages: @MainActor @escaping () -> [ImageAttachment],
        imagePreviewCache: ImagePreviewFileCache = ImagePreviewFileCache()
    ) {
        self.conversationImages = conversationImages
        self.imagePreviewCache = imagePreviewCache
    }

    // MARK: - Image Draft

    /// Append ingested images, capped to the room remaining under `maxPendingImages`.
    /// Returns the slice actually appended.
    @discardableResult
    func attachImages(_ attachments: [ImageAttachment]) -> [ImageAttachment] {
        let added = ImageIngest.capBatch(
            attachments, alreadyQueued: pendingImages.count, limit: Self.maxPendingImages
        )
        pendingImages.append(contentsOf: added)
        return added
    }

    /// Resolve one Image Gesture payload (issue #167): when the selected model
    /// can't see images, surface the switch hint; otherwise attach what fits
    /// under the cap and voice everything that didn't — trimmed, oversize, or
    /// unreadable — through the transient composer notice. The gesture already
    /// suppressed any text fallthrough, so this feedback is the only signal.
    /// Returns the slice actually attached (empty when the hint fired instead).
    @discardableResult
    func handleGesture(_ payload: ImageGesturePayload) -> [ImageAttachment] {
        guard !payload.isEmpty else { return [] }
        guard imageInputAvailable else {
            showImageSwitchHint = true
            return []
        }
        let added = attachImages(payload.attachments)
        showNotice(
            Self.gestureNotice(
                requested: payload.attachments.count,
                attached: added.count,
                rejections: payload.rejections
            ))
        return added
    }

    /// Handle image item providers dropped anywhere on the window (slice #117).
    /// When the selected model can't see images, surface the switch hint instead
    /// of attaching. Providers load asynchronously into one payload, then resolve
    /// through `handleGesture`. Returns whether the drop was accepted.
    @discardableResult
    func handleWindowImageDrop(_ providers: [NSItemProvider]) -> Bool {
        // `handleGesture` re-checks availability; this early gate exists only to
        // answer `.onDrop` synchronously and to skip the async provider load.
        guard imageInputAvailable else {
            showImageSwitchHint = true
            return false
        }
        guard !providers.isEmpty else { return false }
        Task { [weak self] in
            let payload = await ImageItemProviderReader.load(providers)
            self?.handleGesture(payload)
        }
        return true
    }

    /// The one-line composer notice for a partially or fully failed gesture,
    /// nil when everything attached. Pure and testable.
    static func gestureNotice(
        requested: Int, attached: Int, rejections: [ImageIngest.Rejection],
        limit: Int = maxPendingImages
    ) -> String? {
        var parts: [String] = []
        if attached < requested {
            parts.append(
                attached == 0
                    ? "Attachment limit reached — up to \(limit) images per message."
                    : "Attached \(attached) of \(requested) — up to \(limit) images per message.")
        }
        let oversize = rejections.filter {
            if case .oversize = $0 { return true } else { return false }
        }.count
        if oversize > 0 {
            let capMB = ImageIngest.maxBytes / (1024 * 1024)
            parts.append("Images over \(capMB) MB can't be attached.")
        }
        let unreadable = rejections.count - oversize
        if unreadable > 0 {
            parts.append(
                unreadable == 1
                    ? "One item couldn't be read as an image."
                    : "\(unreadable) items couldn't be read as images.")
        }
        return parts.isEmpty ? nil : parts.joined(separator: " ")
    }

    /// Show a transient notice, auto-dismissing after a few seconds unless a
    /// newer notice (or a manual ✕) superseded it. Internal so the Appshot
    /// flow's capture failures speak through the same composer surface.
    func showNotice(_ notice: String?) {
        guard let notice else { return }
        attachmentNotice = notice
        noticeGeneration += 1
        let generation = noticeGeneration
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(5))
            guard let self, self.noticeGeneration == generation else { return }
            self.attachmentNotice = nil
        }
    }

    /// Return and clear the pending images — consumed by the send path.
    func drainImages() -> [ImageAttachment] {
        let images = pendingImages
        pendingImages = []
        return images
    }

    /// Restore images into the composer for Edit & resend. When the selected model
    /// can see images, the images become the live draft; otherwise they're discarded
    /// and the switch hint surfaces if the edited message had images.
    func restoreImages(_ images: [ImageAttachment]) {
        if imageInputAvailable {
            pendingImages = images
        } else {
            pendingImages = []
            if !images.isEmpty { showImageSwitchHint = true }
        }
    }

    // MARK: - Quick Look Image Viewer (slice #114)

    /// Open the full-size Quick Look viewer on the clicked image, navigable
    /// across every image in the conversation in order. Builds the preview-set
    /// projection, materializes digest-keyed temp files (reused when pre-warmed),
    /// and hands the host an ordered URL set + start index. A no-op if the clicked
    /// image is no longer in the conversation or nothing materialized.
    func openQuickLook(clicked id: UUID, includingPending pending: [ImageAttachment] = []) {
        let all = conversationImages() + pending
        guard let set = ImagePreviewSet.project(all: all, clicked: id) else { return }
        let materialized = set.attachments.enumerated().compactMap { index, attachment in
            (try? imagePreviewCache.url(for: attachment)).map { (index, $0) }
        }
        guard let startIndex = materialized.firstIndex(where: { $0.0 == set.startIndex }) else {
            return
        }
        quickLookRequest = QuickLookRequest(urls: materialized.map(\.1), startIndex: startIndex)
    }

    /// Pre-warm one image's temp file (called as the transcript decodes it) so a
    /// later click opens near-instantly.
    func prewarmImagePreview(_ attachment: ImageAttachment) {
        imagePreviewCache.prewarm(attachment)
    }

    /// Clear the pending request when the panel closes so the same image can be
    /// reopened later.
    func dismissQuickLook() {
        quickLookRequest = nil
    }

    // MARK: - Lifecycle

    /// Reset all image state and drop the previous conversation's temp previews.
    func reset() {
        quickLookRequest = nil
        imagePreviewCache.clear()
        pendingImages = []
        showImageSwitchHint = false
        attachmentNotice = nil
        isDropTargeted = false
    }
}
