//
//  ImagePreviewSetTests.swift
//  tesseractTests
//
//  Pins the preview-set projection (PRD #112, slice #114): clicked image →
//  ordered navigable set + start index. Pure — no view, coordinator, or file
//  I/O. `ImageAttachment` equality is id-based, so a divergent-bytes fixture is
//  irrelevant; only id order and the start index matter.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct ImagePreviewSetTests {

    private func makeAttachment() -> ImageAttachment {
        ImageAttachment(data: ImageTestFixtures.tinyPNGData, mimeType: "image/png")
    }

    @Test
    func clickedImageBecomesStartIndexAndOrderIsPreserved() {
        let a = makeAttachment(), b = makeAttachment(), c = makeAttachment()
        let all = [a, b, c]

        let set = ImagePreviewSet.project(all: all, clicked: b.id)

        #expect(set == ImagePreviewSet(attachments: all, startIndex: 1))
    }

    @Test
    func firstAndLastClicksMapToTheirIndices() {
        let a = makeAttachment(), b = makeAttachment(), c = makeAttachment()
        let all = [a, b, c]

        #expect(ImagePreviewSet.project(all: all, clicked: a.id)?.startIndex == 0)
        #expect(ImagePreviewSet.project(all: all, clicked: c.id)?.startIndex == 2)
    }

    @Test
    func unknownClickedIDProjectsToNil() {
        let a = makeAttachment()
        #expect(ImagePreviewSet.project(all: [a], clicked: UUID()) == nil)
    }

    @Test
    func emptyConversationProjectsToNil() {
        #expect(ImagePreviewSet.project(all: [], clicked: UUID()) == nil)
    }

    /// Slice #116 contract: the coordinator merges pending composer images after
    /// the committed conversation images, then projects. Clicking a pending image
    /// opens at its merged index and the whole set stays navigable.
    @Test
    func pendingImagesMergeAfterConversationImages() {
        let conversation = [makeAttachment(), makeAttachment()]
        let pending = [makeAttachment()]
        let merged = conversation + pending  // the coordinator's merge order

        let set = ImagePreviewSet.project(all: merged, clicked: pending[0].id)

        #expect(set?.startIndex == 2)
        #expect(set?.attachments.count == 3)
    }
}
