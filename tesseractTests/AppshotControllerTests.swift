//
//  AppshotControllerTests.swift
//  tesseractTests
//
//  Tests the Appshot flow above the `AppshotCapturing` seam with a fake
//  capturer and a real `ComposerDraftController`: what gets staged, how it's
//  labeled, what the composer is offered, and how failures route. No
//  ScreenCaptureKit, no event tap, no window.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AppshotControllerTests {

    private final class FakeCapturer: AppshotCapturing {
        var result: Result<AppshotCapture, AppshotCaptureError>

        init(_ result: Result<AppshotCapture, AppshotCaptureError>) {
            self.result = result
        }

        func capture() async -> Result<AppshotCapture, AppshotCaptureError> {
            result
        }
    }

    private struct Harness {
        let controller: AppshotController
        let composerDraft: ComposerDraftController
        let summons: () -> Int
    }

    private func makeHarness(
        _ result: Result<AppshotCapture, AppshotCaptureError>,
        imageInputAvailable: Bool = true
    ) -> Harness {
        let composerDraft = ComposerDraftController(conversationImages: { [] })
        composerDraft.imageInputAvailable = imageInputAvailable
        let controller = AppshotController(
            capturer: FakeCapturer(result), composerDraft: composerDraft)
        let counter = SummonCounter()
        controller.onSummon = { counter.count += 1 }
        return Harness(
            controller: controller, composerDraft: composerDraft, summons: { counter.count })
    }

    private final class SummonCounter {
        var count = 0
    }

    private func makeCapture(
        appName: String? = "Safari", windowTitle: String? = "PR #123"
    ) -> AppshotCapture {
        AppshotCapture(
            pngData: ImageTestFixtures.tinyPNGData, appName: appName, windowTitle: windowTitle)
    }

    // MARK: - Success

    @Test
    func successStagesLabeledAttachmentPrefillsComposerAndSummons() async {
        let harness = makeHarness(.success(makeCapture()))
        await harness.controller.takeAppshot()

        #expect(harness.composerDraft.pendingImages.count == 1)
        #expect(harness.composerDraft.pendingImages.first?.filename == "Safari — PR #123.png")
        #expect(harness.composerDraft.text == "Appshot of Safari — “PR #123”.")
        #expect(harness.summons() == 1)
        #expect(!harness.controller.showPermissionExplainer)
    }

    @Test
    func labelsFallBackWhenMetadataIsMissing() {
        #expect(AppshotController.filename(appName: "Safari", windowTitle: nil) == "Safari.png")
        #expect(AppshotController.filename(appName: nil, windowTitle: nil) == "Appshot.png")
        #expect(
            AppshotController.prefill(appName: "Safari", windowTitle: nil) == "Appshot of Safari.")
        #expect(
            AppshotController.prefill(appName: nil, windowTitle: nil)
                == "Appshot of the frontmost window.")
        // Path separators can't leak into a filename shown as a file.
        #expect(
            AppshotController.filename(appName: "Finder", windowTitle: "a/b:c")
                == "Finder — a-b-c.png")
    }

    // MARK: - Failure routing

    @Test
    func missingPermissionShowsExplainerAndSummons() async {
        let harness = makeHarness(.failure(.noPermission))
        await harness.controller.takeAppshot()

        #expect(harness.controller.showPermissionExplainer)
        #expect(harness.composerDraft.pendingImages.isEmpty)
        #expect(harness.composerDraft.text == "")
        #expect(harness.summons() == 1)
    }

    @Test
    func captureFailureVoicesANoticeAndSummons() async {
        let harness = makeHarness(.failure(.captureFailed))
        await harness.controller.takeAppshot()

        #expect(harness.composerDraft.attachmentNotice != nil)
        #expect(harness.composerDraft.pendingImages.isEmpty)
        #expect(harness.composerDraft.text == "")
        #expect(harness.summons() == 1)
    }

    // MARK: - Existing composer rules apply

    @Test
    func textOnlyModelSurfacesTheSwitchHintInsteadOfStaging() async {
        let harness = makeHarness(.success(makeCapture()), imageInputAvailable: false)
        await harness.controller.takeAppshot()

        #expect(harness.composerDraft.showImageSwitchHint)
        #expect(harness.composerDraft.pendingImages.isEmpty)
        #expect(harness.composerDraft.text == "")
        #expect(harness.summons() == 1)
    }

    @Test
    func pendingCapIsRespectedWithoutPrefill() async {
        let harness = makeHarness(.success(makeCapture()))
        let filler = (0..<ComposerDraftController.maxPendingImages).map { index in
            ImageAttachment(
                data: ImageTestFixtures.tinyPNGData, mimeType: "image/png",
                filename: "filler-\(index).png")
        }
        harness.composerDraft.attachImages(filler)

        await harness.controller.takeAppshot()

        #expect(
            harness.composerDraft.pendingImages.count == ComposerDraftController.maxPendingImages)
        #expect(harness.composerDraft.text == "")
        #expect(harness.composerDraft.attachmentNotice != nil)
        #expect(harness.summons() == 1)
    }
}
