//
//  ImagePreviewSet.swift
//  tesseract
//

import Foundation

/// The **preview-set projection** (PRD #112, slice #114): a pure mapping from
/// *(the conversation's ordered image set, the clicked image)* to the ordered
/// set Quick Look should page through plus the index to open on.
///
/// Pure and `nonisolated` so it is unit-tested without any view, coordinator, or
/// file I/O: feed scripted attachments + a clicked id, assert order + start
/// index. The caller (the coordinator) assembles `all` in conversation order;
/// who contributes to that set widens in #116 (tool-result + pending images),
/// but this projection never changes.
nonisolated struct ImagePreviewSet: Equatable {
    /// Every navigable image, in conversation order.
    let attachments: [ImageAttachment]
    /// Index of the clicked image within `attachments` — where Quick Look opens.
    let startIndex: Int

    /// Project the clicked image onto the full ordered set. Returns `nil` when
    /// the clicked id isn't in `all` (a stale row tapped after a conversation
    /// reset) — the caller then opens nothing rather than guessing.
    static func project(all: [ImageAttachment], clicked id: UUID) -> ImagePreviewSet? {
        guard let index = all.firstIndex(where: { $0.id == id }) else { return nil }
        return ImagePreviewSet(attachments: all, startIndex: index)
    }
}
