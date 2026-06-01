//
//  SlashCommandPaletteControllerTests.swift
//  tesseractTests
//
//  Tests the **Command Palette** at its own seam — no `Agent`. With no extension
//  host or package registry wired, the registry holds the always-present built-in
//  commands (`/compact`, `/new`, `/clear`); the popup filter / selection /
//  autocomplete interaction is driven directly.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SlashCommandPaletteControllerTests {

    private func makePalette() -> SlashCommandPaletteController {
        SlashCommandPaletteController()
    }

    // MARK: - Registry

    @Test func registryContainsBuiltInCommands() {
        let palette = makePalette()
        let names = Set(palette.commandRegistry.commands.map(\.name))
        #expect(names.isSuperset(of: ["compact", "new", "clear"]))
    }

    // MARK: - Popup show/filter

    @Test func typingCommandPrefixShowsFilteredPopup() {
        let palette = makePalette()
        palette.updateCommandPopup(for: "/comp")

        #expect(palette.showCommandPopup == true)
        #expect(palette.commandFilteredResults.contains { $0.name == "compact" })
        #expect(palette.commandFilteredResults.allSatisfy { $0.name != "unrelated" })
    }

    @Test func pastCommandNameHidesPopup() {
        let palette = makePalette()
        palette.updateCommandPopup(for: "/comp")
        #expect(palette.showCommandPopup == true)

        // Typing an argument (a space + text) means the user is past the
        // command name — the popup hides.
        palette.updateCommandPopup(for: "/compact now")
        #expect(palette.showCommandPopup == false)
    }

    @Test func nonCommandInputKeepsPopupHidden() {
        let palette = makePalette()
        palette.updateCommandPopup(for: "hello there")
        #expect(palette.showCommandPopup == false)
    }

    // MARK: - Selection reset

    @Test func updatingPopupResetsSelectionIndex() {
        let palette = makePalette()
        palette.commandSelectedIndex = 3
        palette.updateCommandPopup(for: "/n")
        #expect(palette.commandSelectedIndex == 0)
    }

    // MARK: - Dismiss / autocomplete

    @Test func dismissClearsPopupState() {
        let palette = makePalette()
        palette.updateCommandPopup(for: "/comp")
        palette.commandSelectedIndex = 1

        palette.dismissCommandPopup()
        #expect(palette.showCommandPopup == false)
        #expect(palette.commandFilteredResults.isEmpty)
        #expect(palette.commandSelectedIndex == 0)
    }

    @Test func autocompleteReturnsCommandTextAndDismisses() {
        let palette = makePalette()
        palette.updateCommandPopup(for: "/comp")
        let compact = palette.commandRegistry.commands.first { $0.name == "compact" }!

        let completed = palette.autocompleteCommand(compact)
        #expect(completed == "/compact ")
        #expect(palette.showCommandPopup == false)
    }
}
