//
//  VisionAvailabilityControllerTests.swift
//  tesseractTests
//
//  The **Vision Availability** leaf at its own seam: hermetic settings
//  (`InMemorySettingsStore`), a real Composer Draft, and closure-injected
//  catalog reads. Covers the refresh lifecycle (probe → verdict → draft
//  reconciliation) and the switch-hint remedy ladder — behavior that
//  previously lived as view orchestration nothing could drive.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct VisionAvailabilityControllerTests {

    private static func makeModel(id: String, name: String) -> ModelDefinition {
        ModelDefinition(
            id: id, displayName: name, description: "",
            category: .agent,
            source: .huggingFace(repo: "test/\(id)", requiredExtension: "safetensors"),
            sizeDescription: "", dependencies: [])
    }

    private func makeController(
        visionCapableIDs: Set<String>,
        downloaded: [ModelDefinition] = [],
        settings: SettingsManager
    ) -> (VisionAvailabilityController, ComposerDraftController) {
        let draft = ComposerDraftController(conversationImages: { [] })
        let controller = VisionAvailabilityController(
            settings: settings,
            draft: draft,
            isVisionCapable: { visionCapableIDs.contains($0) },
            downloadedAgentModels: { downloaded }
        )
        return (controller, draft)
    }

    // MARK: - Refresh

    @Test func refreshDerivesTheVerdictAndMirrorsItToTheDraft() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "vlm"
        settings.useVisionWhenAvailable = true
        let (controller, draft) = makeController(
            visionCapableIDs: ["vlm"], settings: settings)

        controller.refresh()

        #expect(controller.selectedModelIsVisionCapable)
        #expect(controller.imageInputAvailable)
        #expect(draft.imageInputAvailable)
    }

    @Test func availabilityGainLowersTheSwitchHint() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "text-only"
        settings.useVisionWhenAvailable = true
        let (controller, draft) = makeController(
            visionCapableIDs: ["vlm"], settings: settings)
        controller.refresh()
        draft.showImageSwitchHint = true

        settings.selectedAgentModelID = "vlm"
        controller.refresh()

        #expect(controller.imageInputAvailable)
        #expect(!draft.showImageSwitchHint)
    }

    @Test func availabilityLossClearsThePendingImagesTheContainerWouldDrop() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "vlm"
        settings.useVisionWhenAvailable = true
        let (controller, draft) = makeController(
            visionCapableIDs: ["vlm"], settings: settings)
        controller.refresh()
        draft.pendingImages = [
            ImageAttachment(data: ImageTestFixtures.tinyPNGData, mimeType: "image/png")
        ]

        settings.useVisionWhenAvailable = false
        controller.refresh()

        #expect(!controller.imageInputAvailable)
        #expect(!draft.imageInputAvailable)
        #expect(draft.pendingImages.isEmpty)
    }

    /// A refresh that doesn't flip the verdict must not touch the queue —
    /// re-probing on every status change would otherwise eat drafts.
    @Test func unchangedVerdictLeavesThePendingImagesAlone() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "vlm"
        settings.useVisionWhenAvailable = true
        let (controller, draft) = makeController(
            visionCapableIDs: ["vlm"], settings: settings)
        controller.refresh()
        draft.pendingImages = [
            ImageAttachment(data: ImageTestFixtures.tinyPNGData, mimeType: "image/png")
        ]

        controller.refresh()

        #expect(draft.pendingImages.count == 1)
    }

    // MARK: - Remedy ladder

    @Test func capableModelWithVisionOffOffersTheSettingToggle() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "vlm"
        settings.useVisionWhenAvailable = false
        let (controller, _) = makeController(visionCapableIDs: ["vlm"], settings: settings)
        controller.refresh()

        #expect(controller.remedy == .turnOnSetting)
        #expect(controller.remedy.actionTitle == "Turn On")
    }

    @Test func textOnlyModelOffersTheFirstDownloadedVisionModel() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "text-only"
        settings.useVisionWhenAvailable = true
        let (controller, _) = makeController(
            visionCapableIDs: ["vlm"],
            downloaded: [
                Self.makeModel(id: "text-only", name: "Text Only"),
                Self.makeModel(id: "vlm", name: "Vision Model"),
            ],
            settings: settings)
        controller.refresh()

        #expect(controller.remedy == .switchModel(id: "vlm", name: "Vision Model"))
    }

    @Test func noDownloadedVisionModelHasNoAction() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "text-only"
        settings.useVisionWhenAvailable = true
        let (controller, _) = makeController(
            visionCapableIDs: [],
            downloaded: [Self.makeModel(id: "text-only", name: "Text Only")],
            settings: settings)
        controller.refresh()

        #expect(controller.remedy == .noVisionModel)
        #expect(controller.remedy.actionTitle == nil)
    }

    @Test func applyRemedySwitchesModelTurnsVisionOnAndLowersTheHint() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "text-only"
        settings.useVisionWhenAvailable = false
        let (controller, draft) = makeController(
            visionCapableIDs: ["vlm"],
            downloaded: [Self.makeModel(id: "vlm", name: "Vision Model")],
            settings: settings)
        controller.refresh()
        draft.showImageSwitchHint = true

        controller.applyRemedy()

        #expect(settings.selectedAgentModelID == "vlm")
        #expect(settings.useVisionWhenAvailable)
        #expect(!draft.showImageSwitchHint)
    }
}
