//
//  ImageDraftControllerTests.swift
//  tesseractTests
//
//  Tests the **Image Draft** leaf directly: pending-image queue state, model
//  capability gating, and Quick Look request construction. No `Agent`, command
//  registry, conversation store, SwiftUI view, or real filesystem is required.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ImageDraftControllerTests {

    private let root = URL(fileURLWithPath: "/image-draft-previews", isDirectory: true)

    private func makeAttachment(byte: UInt8) -> ImageAttachment {
        ImageAttachment(
            data: ImageTestFixtures.tinyPNGData + Data([byte]),
            mimeType: "image/png",
            filename: "image-\(byte).png"
        )
    }

    @Test
    func attachImagesCapsAtLimitAndDrainClearsTheDraft() {
        let controller = ImageDraftController(conversationImages: { [] })
        let images = (0..<10).map { makeAttachment(byte: UInt8($0)) }

        controller.attachImages(images)

        let expected = Array(images.prefix(ImageDraftController.maxPendingImages))
        #expect(controller.pendingImages.map(\.id) == expected.map(\.id))

        let drained = controller.drainImages()
        #expect(drained.map(\.id) == expected.map(\.id))
        #expect(controller.pendingImages.isEmpty)
    }

    @Test
    func restoreImagesKeepsAttachmentsOnlyWhenImageInputIsAvailable() {
        let controller = ImageDraftController(conversationImages: { [] })
        let images = [makeAttachment(byte: 1), makeAttachment(byte: 2)]

        controller.imageInputAvailable = true
        controller.restoreImages(images)

        #expect(controller.pendingImages.map(\.id) == images.map(\.id))
        #expect(controller.showImageSwitchHint == false)

        controller.imageInputAvailable = false
        controller.restoreImages(images)

        #expect(controller.pendingImages.isEmpty)
        #expect(controller.showImageSwitchHint == true)
    }

    @Test
    func openQuickLookMergesConversationAndPendingImagesInOrder() throws {
        let fs = InMemoryImagePreviewFileSystem()
        let cache = ImagePreviewFileCache(root: root, fileSystem: fs)
        let conversation = [makeAttachment(byte: 1), makeAttachment(byte: 2)]
        let pending = [makeAttachment(byte: 3)]
        let controller = ImageDraftController(
            conversationImages: { conversation },
            imagePreviewCache: cache
        )

        controller.openQuickLook(clicked: pending[0].id, includingPending: pending)

        let request = try #require(controller.quickLookRequest)
        #expect(request.urls.count == 3)
        #expect(request.startIndex == 2)
        #expect(fs.writes.count == 3)
        #expect(Set(request.urls) == Set(fs.storage.keys))
    }

    @Test
    func quickLookDismissAndResetClearObservableAndCachedState() throws {
        let fs = InMemoryImagePreviewFileSystem()
        let cache = ImagePreviewFileCache(root: root, fileSystem: fs)
        let image = makeAttachment(byte: 9)
        let controller = ImageDraftController(
            conversationImages: { [image] },
            imagePreviewCache: cache
        )

        controller.pendingImages = [image]
        controller.showImageSwitchHint = true
        controller.openQuickLook(clicked: image.id)
        #expect(controller.quickLookRequest != nil)
        #expect(fs.storage.isEmpty == false)

        controller.dismissQuickLook()
        #expect(controller.quickLookRequest == nil)

        controller.openQuickLook(clicked: image.id)
        controller.reset()

        #expect(controller.quickLookRequest == nil)
        #expect(controller.pendingImages.isEmpty)
        #expect(controller.showImageSwitchHint == false)
        #expect(fs.storage.isEmpty)
    }

    @Test
    func textOnlyDropSurfacesSwitchHintWithoutAcceptingTheDrop() {
        let controller = ImageDraftController(conversationImages: { [] })

        let accepted = controller.handleWindowImageDrop([])

        #expect(accepted == false)
        #expect(controller.showImageSwitchHint == true)
        #expect(controller.pendingImages.isEmpty)
    }

    // MARK: - Image Gesture resolution (issue #167)

    @Test
    func gestureOnUnavailableModelSurfacesHintAndAttachesNothing() {
        let controller = ImageDraftController(conversationImages: { [] })
        controller.imageInputAvailable = false

        controller.handleGesture(ImageGesturePayload(attachments: [makeAttachment(byte: 1)]))

        #expect(controller.pendingImages.isEmpty)
        #expect(controller.showImageSwitchHint == true)
        #expect(controller.attachmentNotice == nil)
    }

    @Test
    func gestureAttachesUnderCapWithoutANotice() {
        let controller = ImageDraftController(conversationImages: { [] })
        controller.imageInputAvailable = true
        let images = [makeAttachment(byte: 1), makeAttachment(byte: 2)]

        controller.handleGesture(ImageGesturePayload(attachments: images))

        #expect(controller.pendingImages.map(\.id) == images.map(\.id))
        #expect(controller.attachmentNotice == nil)
        #expect(controller.showImageSwitchHint == false)
    }

    @Test
    func gestureOverCapAttachesWhatFitsAndReportsTheTrim() {
        let controller = ImageDraftController(conversationImages: { [] })
        controller.imageInputAvailable = true
        let images = (0..<10).map { makeAttachment(byte: UInt8($0)) }

        controller.handleGesture(ImageGesturePayload(attachments: images))

        #expect(controller.pendingImages.count == ImageDraftController.maxPendingImages)
        let notice = controller.attachmentNotice
        #expect(notice?.contains("8 of 10") == true)
    }

    @Test
    func gestureWithRejectionsReportsThemInTheNotice() {
        let controller = ImageDraftController(conversationImages: { [] })
        controller.imageInputAvailable = true

        controller.handleGesture(
            ImageGesturePayload(
                attachments: [makeAttachment(byte: 1)],
                rejections: [.oversize(bytes: 20_000_000), .notAnImage]
            ))

        #expect(controller.pendingImages.count == 1)
        let notice = controller.attachmentNotice
        #expect(notice?.contains("10 MB") == true)
        #expect(notice?.contains("couldn't be read") == true)
    }

    @Test
    func emptyGesturePayloadIsANoOp() {
        let controller = ImageDraftController(conversationImages: { [] })
        controller.imageInputAvailable = false

        controller.handleGesture(ImageGesturePayload())

        #expect(controller.showImageSwitchHint == false)
        #expect(controller.attachmentNotice == nil)
    }

    @Test
    func gestureNoticeIsNilWhenEverythingAttached() {
        #expect(
            ImageDraftController.gestureNotice(requested: 3, attached: 3, rejections: []) == nil)
    }

    @Test
    func gestureNoticeReportsFullCapWhenNothingFit() {
        let notice = ImageDraftController.gestureNotice(
            requested: 2, attached: 0, rejections: [])
        #expect(notice?.contains("limit reached") == true)
    }

    @Test
    func resetClearsGestureFeedbackState() {
        let controller = ImageDraftController(conversationImages: { [] })
        controller.imageInputAvailable = true
        controller.isDropTargeted = true
        controller.handleGesture(ImageGesturePayload(rejections: [.notAnImage]))
        #expect(controller.attachmentNotice != nil)

        controller.reset()

        #expect(controller.attachmentNotice == nil)
        #expect(controller.isDropTargeted == false)
    }
}
