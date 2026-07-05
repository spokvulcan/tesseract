//
//  SkillPillControllerTests.swift
//  tesseractTests
//
//  Tests the **Skill Pill** module at its own seam (PRD #174) — fixture skill
//  metadata through the injected discovery closure, persistence through the
//  in-memory Settings Store. Covers pill list derivation (membership × usage
//  ranking × curated order), ranking stability between refreshes, usage
//  recording, and invocation argument assembly (the translate default-target
//  wiring). No `Agent`, no SwiftUI.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SkillPillControllerTests {

    // MARK: - Fixtures

    private func skill(_ name: String, pill: Bool = true, label: String? = nil) -> SkillMetadata {
        SkillMetadata(
            name: name,
            description: "\(name) description",
            filePath: "/skills/\(name)/SKILL.md",
            disableModelInvocation: false,
            composerPill: pill,
            label: label
        )
    }

    /// The six shipped essentials, in (shuffled) discovery order.
    private var essentials: [SkillMetadata] {
        [
            skill("summarize"), skill("translate"), skill("proofread"),
            skill("explain"), skill("proofread-tweet"), skill("reply"),
        ]
    }

    private func makeController(
        skills: [SkillMetadata],
        settings: SettingsManager? = nil
    ) -> SkillPillController {
        SkillPillController(discoverSkills: { skills }, settings: settings)
    }

    private func makeSettings() -> SettingsManager {
        SettingsManager(store: InMemorySettingsStore())
    }

    // MARK: - Membership

    @Test func onlyComposerPillSkillsBecomePills() {
        let controller = makeController(skills: [
            skill("proofread", pill: true),
            skill("memory", pill: false),
            skill("notes", pill: false),
        ])
        #expect(controller.pills.map(\.name) == ["proofread"])
    }

    @Test func pillCarriesLabelDescriptionAndPath() {
        let controller = makeController(skills: [skill("proofread-tweet", label: "Tweet")])
        let pill = controller.pills[0]
        #expect(pill.label == "Tweet")
        #expect(pill.description == "proofread-tweet description")
        #expect(pill.filePath == "/skills/proofread-tweet/SKILL.md")
    }

    @Test func derivedLabelTitleCasesKebabName() {
        let controller = makeController(skills: [skill("proofread-tweet")])
        #expect(controller.pills[0].label == "Proofread Tweet")
    }

    // MARK: - Curated default order (fresh install)

    @Test func zeroCountsFollowCuratedOrder() {
        let controller = makeController(skills: essentials, settings: makeSettings())
        #expect(
            controller.pills.map(\.name) == [
                "proofread", "proofread-tweet", "translate", "reply", "summarize", "explain",
            ])
    }

    @Test func nonCuratedSkillsSortAfterCuratedAlphabetically() {
        let controller = makeController(
            skills: [skill("zeta-custom"), skill("alpha-custom"), skill("proofread")],
            settings: makeSettings())
        #expect(controller.pills.map(\.name) == ["proofread", "alpha-custom", "zeta-custom"])
    }

    // MARK: - Usage ranking

    @Test func usageCountsRankMostUsedLeftmost() {
        let settings = makeSettings()
        let controller = makeController(skills: essentials, settings: settings)

        controller.recordUserInvocation(skillName: "summarize")
        controller.recordUserInvocation(skillName: "summarize")
        controller.recordUserInvocation(skillName: "explain")
        controller.refreshPills()

        #expect(controller.pills.map(\.name).prefix(2) == ["summarize", "explain"])
        // The zero-count tail keeps the curated order.
        #expect(
            controller.pills.map(\.name).dropFirst(2) == [
                "proofread", "proofread-tweet", "translate", "reply",
            ])
    }

    @Test func rankingHoldsStableUntilRefresh() {
        let settings = makeSettings()
        let controller = makeController(skills: essentials, settings: settings)
        let before = controller.pills.map(\.name)

        controller.recordUserInvocation(skillName: "explain")
        controller.recordUserInvocation(skillName: "explain")

        // No live re-sort mid-conversation…
        #expect(controller.pills.map(\.name) == before)

        // …the recorded usage lands at the next conversation start.
        controller.refreshPills()
        #expect(controller.pills.first?.name == "explain")
    }

    @Test func usagePersistsThroughTheSettingsStore() {
        let settings = makeSettings()
        let first = makeController(skills: essentials, settings: settings)
        first.recordUserInvocation(skillName: "translate")

        // A fresh controller over the same store (a new session) sees the count.
        let second = makeController(skills: essentials, settings: settings)
        #expect(second.pills.first?.name == "translate")
    }

    // MARK: - Row visibility

    @Test func rowHiddenWhenSettingOff() {
        let settings = makeSettings()
        let controller = makeController(skills: essentials, settings: settings)
        #expect(controller.isRowVisible == true)

        settings.showSkillPills = false
        #expect(controller.isRowVisible == false)
    }

    @Test func rowHiddenWhenNoPillSkillsExist() {
        let controller = makeController(
            skills: [skill("memory", pill: false)], settings: makeSettings())
        #expect(controller.isRowVisible == false)
    }

    // MARK: - Invocation argument assembly

    @Test func argumentsPassComposerTextTrimmed() {
        let controller = makeController(skills: essentials, settings: makeSettings())
        let arguments = controller.assembleArguments(
            skillName: "proofread", userText: "  fix this text  \n")
        #expect(arguments == "fix this text")
    }

    @Test func bareInvocationYieldsEmptyArguments() {
        let controller = makeController(skills: essentials, settings: makeSettings())
        #expect(controller.assembleArguments(skillName: "proofread", userText: "").isEmpty)
    }

    @Test func translateAppendsConfiguredDefaultTarget() {
        let settings = makeSettings()
        settings.translateTargetLanguage = "Ukrainian"
        let controller = makeController(skills: essentials, settings: settings)

        let withText = controller.assembleArguments(skillName: "translate", userText: "hello")
        #expect(withText == "hello\n\nDefault target language: Ukrainian")

        let bare = controller.assembleArguments(skillName: "translate", userText: "")
        #expect(bare == "Default target language: Ukrainian")
    }

    @Test func nonTranslateSkillsNeverGetTheTargetLine() {
        let settings = makeSettings()
        settings.translateTargetLanguage = "Ukrainian"
        let controller = makeController(skills: essentials, settings: settings)
        #expect(
            controller.assembleArguments(skillName: "summarize", userText: "text") == "text")
    }
}
