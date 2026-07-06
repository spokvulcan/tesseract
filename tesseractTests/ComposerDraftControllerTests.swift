//
//  ComposerDraftControllerTests.swift
//  tesseractTests
//
//  Tests the **Composer Draft** leaf directly: unsent text + pending-image queue,
//  capability gating, and Quick Look request construction. No `Agent`, command
//  registry, conversation store, SwiftUI view, or real filesystem is required.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ComposerDraftControllerTests {

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
        let controller = ComposerDraftController(conversationImages: { [] })
        let images = (0..<10).map { makeAttachment(byte: UInt8($0)) }

        controller.attachImages(images)

        let expected = Array(images.prefix(ComposerDraftController.maxPendingImages))
        #expect(controller.pendingImages.map(\.id) == expected.map(\.id))

        let drained = controller.drainImages()
        #expect(drained.map(\.id) == expected.map(\.id))
        #expect(controller.pendingImages.isEmpty)
    }

    @Test
    func restoreImagesKeepsAttachmentsOnlyWhenImageInputIsAvailable() {
        let controller = ComposerDraftController(conversationImages: { [] })
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
        let controller = ComposerDraftController(
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
        let controller = ComposerDraftController(
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
        controller.resetEphemeral()
        controller.clearDraft()

        #expect(controller.quickLookRequest == nil)
        #expect(controller.pendingImages.isEmpty)
        #expect(controller.showImageSwitchHint == false)
        #expect(fs.storage.isEmpty)
    }

    @Test
    func textOnlyDropSurfacesSwitchHintWithoutAcceptingTheDrop() {
        let controller = ComposerDraftController(conversationImages: { [] })

        let accepted = controller.handleWindowImageDrop([])

        #expect(accepted == false)
        #expect(controller.showImageSwitchHint == true)
        #expect(controller.pendingImages.isEmpty)
    }

    // MARK: - Image Gesture resolution (issue #167)

    @Test
    func gestureOnUnavailableModelSurfacesHintAndAttachesNothing() {
        let controller = ComposerDraftController(conversationImages: { [] })
        controller.imageInputAvailable = false

        controller.handleGesture(ImageGesturePayload(attachments: [makeAttachment(byte: 1)]))

        #expect(controller.pendingImages.isEmpty)
        #expect(controller.showImageSwitchHint == true)
        #expect(controller.attachmentNotice == nil)
    }

    @Test
    func gestureAttachesUnderCapWithoutANotice() {
        let controller = ComposerDraftController(conversationImages: { [] })
        controller.imageInputAvailable = true
        let images = [makeAttachment(byte: 1), makeAttachment(byte: 2)]

        controller.handleGesture(ImageGesturePayload(attachments: images))

        #expect(controller.pendingImages.map(\.id) == images.map(\.id))
        #expect(controller.attachmentNotice == nil)
        #expect(controller.showImageSwitchHint == false)
    }

    @Test
    func gestureOverCapAttachesWhatFitsAndReportsTheTrim() {
        let controller = ComposerDraftController(conversationImages: { [] })
        controller.imageInputAvailable = true
        let images = (0..<10).map { makeAttachment(byte: UInt8($0)) }

        controller.handleGesture(ImageGesturePayload(attachments: images))

        #expect(controller.pendingImages.count == ComposerDraftController.maxPendingImages)
        let notice = controller.attachmentNotice
        #expect(notice?.contains("8 of 10") == true)
    }

    @Test
    func gestureWithRejectionsReportsThemInTheNotice() {
        let controller = ComposerDraftController(conversationImages: { [] })
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
        let controller = ComposerDraftController(conversationImages: { [] })
        controller.imageInputAvailable = false

        controller.handleGesture(ImageGesturePayload())

        #expect(controller.showImageSwitchHint == false)
        #expect(controller.attachmentNotice == nil)
    }

    @Test
    func gestureNoticeIsNilWhenEverythingAttached() {
        #expect(
            ComposerDraftController.gestureNotice(requested: 3, attached: 3, rejections: []) == nil)
    }

    @Test
    func gestureNoticeReportsFullCapWhenNothingFit() {
        let notice = ComposerDraftController.gestureNotice(
            requested: 2, attached: 0, rejections: [])
        #expect(notice?.contains("limit reached") == true)
    }

    @Test
    func resetClearsGestureFeedbackState() {
        let controller = ComposerDraftController(conversationImages: { [] })
        controller.imageInputAvailable = true
        controller.isDropTargeted = true
        controller.handleGesture(ImageGesturePayload(rejections: [.notAnImage]))
        #expect(controller.attachmentNotice != nil)

        controller.resetEphemeral()

        #expect(controller.attachmentNotice == nil)
        #expect(controller.isDropTargeted == false)
    }

    // MARK: - Composer Draft lifetime (text + images as one unit)

    @Test
    func resetEphemeralPreservesTheDraftButClearsTransientState() {
        let controller = ComposerDraftController(conversationImages: { [] })
        controller.imageInputAvailable = true
        controller.text = "unsent thought"
        controller.attachImages([makeAttachment(byte: 1)])
        controller.showImageSwitchHint = true
        controller.isDropTargeted = true

        controller.resetEphemeral()

        // The unsent draft survives a conversation switch...
        #expect(controller.text == "unsent thought")
        #expect(controller.pendingImages.count == 1)
        // ...while the ephemeral view state is cleared.
        #expect(controller.showImageSwitchHint == false)
        #expect(controller.isDropTargeted == false)
    }

    @Test
    func clearDraftDiscardsTextAndPendingImagesTogether() {
        let controller = ComposerDraftController(conversationImages: { [] })
        controller.imageInputAvailable = true
        controller.text = "unsent thought"
        controller.attachImages([makeAttachment(byte: 1), makeAttachment(byte: 2)])

        controller.clearDraft()

        #expect(controller.text == "")
        #expect(controller.pendingImages.isEmpty)
    }
}
