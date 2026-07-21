//
//  ModelSelectionHealingTests.swift
//  tesseractTests
//
//  The availability-follows-selection rules as a decision table (#406):
//  the dictation-model heal and the Companion agent-default adoption
//  (ADR-0040 §9). An available selection is never overridden; nothing
//  downloaded means nothing decided.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ModelSelectionHealingTests {

    private let sttModels = ModelDefinition.all.filter { $0.category == .speechToText }

    private func statuses(downloaded: [String]) -> [String: ModelStatus] {
        var statuses: [String: ModelStatus] = [:]
        for id in downloaded { statuses[id] = .downloaded(sizeOnDisk: 0) }
        return statuses
    }

    // MARK: - Dictation heal

    /// An available selection is never overridden — even with other
    /// variants on disk.
    @Test func availableSelectionIsNeverOverridden() {
        let selected = sttModels[0].id
        let other = sttModels.count > 1 ? sttModels[1].id : selected
        let decided = ModelSelectionHealing.healedSpeechToTextSelection(
            selectedID: selected,
            definitions: ModelDefinition.all,
            statuses: statuses(downloaded: [selected, other]))
        #expect(decided == nil)
    }

    /// Fresh-install-compact-only and deletion-of-selected: the selected
    /// variant is missing, another is on disk — heal to the downloaded one.
    @Test func missingSelectionHealsToTheDownloadedVariant() throws {
        try #require(sttModels.count > 1)
        let selected = sttModels[0].id
        let downloaded = sttModels[1].id
        let decided = ModelSelectionHealing.healedSpeechToTextSelection(
            selectedID: selected,
            definitions: ModelDefinition.all,
            statuses: statuses(downloaded: [downloaded]))
        #expect(decided == downloaded)
    }

    /// Nothing on disk means nothing to heal to — the selection stands
    /// (dictation stays dead until a download, never a random flip).
    @Test func nothingDownloadedLeavesTheSelectionAlone() {
        let decided = ModelSelectionHealing.healedSpeechToTextSelection(
            selectedID: sttModels[0].id,
            definitions: ModelDefinition.all,
            statuses: [:])
        #expect(decided == nil)
    }

    // MARK: - Companion agent default (ADR-0040 §9)

    /// The enabled Companion adopts its downloaded model as the agent
    /// default.
    @Test func enabledCompanionAdoptsItsDownloadedModel() {
        let decided = ModelSelectionHealing.adoptedCompanionAgentDefault(
            companionEnabled: true,
            companionModelID: "jarvis-model",
            selectedAgentModelID: "other-model",
            isDownloaded: { _ in true })
        #expect(decided == "jarvis-model")
    }

    /// An undownloaded model must never become the interactive default.
    @Test func undownloadedCompanionModelIsNeverAdopted() {
        let decided = ModelSelectionHealing.adoptedCompanionAgentDefault(
            companionEnabled: true,
            companionModelID: "jarvis-model",
            selectedAgentModelID: "other-model",
            isDownloaded: { _ in false })
        #expect(decided == nil)
    }

    /// A disabled Companion, an empty companion model, or an already
    /// matching selection all decide nothing.
    @Test func disabledEmptyOrMatchingDecidesNothing() {
        #expect(
            ModelSelectionHealing.adoptedCompanionAgentDefault(
                companionEnabled: false, companionModelID: "jarvis-model",
                selectedAgentModelID: "other", isDownloaded: { _ in true })
                == nil)
        #expect(
            ModelSelectionHealing.adoptedCompanionAgentDefault(
                companionEnabled: true, companionModelID: "",
                selectedAgentModelID: "other", isDownloaded: { _ in true })
                == nil)
        #expect(
            ModelSelectionHealing.adoptedCompanionAgentDefault(
                companionEnabled: true, companionModelID: "jarvis-model",
                selectedAgentModelID: "jarvis-model", isDownloaded: { _ in true })
                == nil)
    }
}
