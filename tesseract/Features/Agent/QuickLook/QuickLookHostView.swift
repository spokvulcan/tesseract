//
//  QuickLookHostView.swift
//  tesseract
//

import AppKit
import Quartz
import SwiftUI

// MARK: - QuickLookRequest

/// One request to open the Quick Look panel: the ordered file URLs (already
/// materialized by `ImagePreviewFileCache`) and the index to open on. The `id`
/// makes repeated SwiftUI `updateNSView` passes idempotent — only a *new*
/// request id re-presents.
nonisolated struct QuickLookRequest: Equatable {
    let id: UUID
    let urls: [URL]
    let startIndex: Int

    init(id: UUID = UUID(), urls: [URL], startIndex: Int) {
        self.id = id
        self.urls = urls
        self.startIndex = startIndex
    }
}

// MARK: - Preview item

/// A single Quick Look item — a file URL plus a human title (the filename).
/// `NSURL` already conforms to `QLPreviewItem`, but a wrapper lets us supply a
/// nicer title and keeps the data source's type explicit.
final class QuickLookPreviewItem: NSObject, QLPreviewItem {
    let previewItemURL: URL?
    let previewItemTitle: String?

    init(url: URL) {
        self.previewItemURL = url
        self.previewItemTitle = url.lastPathComponent
    }
}

// MARK: - Host view (responder-chain shim)

/// The AppKit edge for slice #114. `QLPreviewPanel` finds its controller by
/// walking the key window's responder chain for the first responder that
/// returns `true` from `acceptsPreviewPanelControl(_:)`. SwiftUI puts no such
/// responder in the chain, so this view makes *itself* first responder on
/// present, then vends the panel's data source/delegate. Untested edge (UI +
/// responder chain); the navigable set + file URLs it consumes are unit-tested
/// upstream (`ImagePreviewSet`, `ImagePreviewFileCache`).
final class QuickLookHostView: NSView, QLPreviewPanelDataSource, QLPreviewPanelDelegate {

    private var items: [QuickLookPreviewItem] = []
    private var startIndex = 0
    private var presentedRequestID: UUID?

    /// Invoked when the panel relinquishes control (Esc / close) so the owner can
    /// clear its request and allow the same image to be reopened later.
    var onClose: (() -> Void)?

    /// Open (or re-target) the Quick Look panel for `request`. Idempotent for a
    /// request already presented — guards the repeated `updateNSView` passes.
    func present(_ request: QuickLookRequest) {
        guard request.id != presentedRequestID else { return }
        presentedRequestID = request.id
        items = request.urls.map(QuickLookPreviewItem.init(url:))
        guard !items.isEmpty else { return }
        startIndex = min(max(0, request.startIndex), items.count - 1)

        guard let panel = QLPreviewPanel.shared() else { return }
        window?.makeFirstResponder(self)
        if QLPreviewPanel.sharedPreviewPanelExists(), panel.isVisible {
            panel.reloadData()
            panel.currentPreviewItemIndex = startIndex
        } else {
            panel.makeKeyAndOrderFront(nil)
        }
    }

    override var acceptsFirstResponder: Bool { true }

    // MARK: QLPreviewPanelController (NSResponder informal protocol)
    //
    // These category methods are imported `nonisolated`, so under the build's
    // MainActor-default isolation the overrides must be `nonisolated` too. AppKit
    // always invokes them on the main thread, so the bodies assume MainActor to
    // touch the panel and this view's state.

    override nonisolated func acceptsPreviewPanelControl(_ panel: QLPreviewPanel!) -> Bool { true }

    override nonisolated func beginPreviewPanelControl(_ panel: QLPreviewPanel!) {
        MainActor.assumeIsolated {
            panel.dataSource = self
            panel.delegate = self
            panel.reloadData()
            panel.currentPreviewItemIndex = startIndex
        }
    }

    override nonisolated func endPreviewPanelControl(_ panel: QLPreviewPanel!) {
        MainActor.assumeIsolated {
            presentedRequestID = nil
            onClose?()
        }
    }

    // MARK: QLPreviewPanelDataSource

    func numberOfPreviewItems(in panel: QLPreviewPanel!) -> Int { items.count }

    func previewPanel(_ panel: QLPreviewPanel!, previewItemAt index: Int) -> QLPreviewItem! {
        items.indices.contains(index) ? items[index] : nil
    }
}

// MARK: - SwiftUI bridge

/// Hosts an invisible `QuickLookHostView` in the chat window and drives it from
/// the coordinator's observable `quickLookRequest`. Attach as a `.background`
/// of the chat root so it lives in the window's view (and responder) hierarchy.
struct QuickLookContainer: NSViewRepresentable {
    let request: QuickLookRequest?
    let onClose: () -> Void

    func makeNSView(context: Context) -> QuickLookHostView {
        let view = QuickLookHostView()
        view.onClose = onClose
        return view
    }

    func updateNSView(_ nsView: QuickLookHostView, context: Context) {
        nsView.onClose = onClose
        if let request {
            nsView.present(request)
        }
    }
}
