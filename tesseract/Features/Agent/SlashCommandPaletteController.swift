//
//  SlashCommandPaletteController.swift
//  tesseract
//
//  The **Command Palette** module: the slash-command popup *presentation*,
//  carved out of `AgentCoordinator`. It owns the registry rebuild (discovered
//  skills + extensions) and the filter / selection / autocomplete interaction
//  over the already-pure `SlashCommandParser` / `SlashCommandRegistry`.
//
//  Command *execution* deliberately stays on the coordinator spine (it routes
//  `/compact`, `/new`, `/clear`, skills back into three spine concerns) — this
//  module is presentation only.
//

import Foundation
import Observation

@Observable @MainActor
final class SlashCommandPaletteController {

    // MARK: - Observable State

    private(set) var commandRegistry = SlashCommandRegistry()
    var showCommandPopup: Bool = false
    var commandSelectedIndex: Int = 0
    var commandFilteredResults: [SlashCommand] = []

    // MARK: - Dependencies

    private let extensionHost: ExtensionHost?
    private let packageRegistry: PackageRegistry?

    // MARK: - Init

    init(
        extensionHost: ExtensionHost? = nil,
        packageRegistry: PackageRegistry? = nil
    ) {
        self.extensionHost = extensionHost
        self.packageRegistry = packageRegistry
        rebuildRegistry()
    }

    // MARK: - Registry

    /// Rebuild the command registry from current skills and extensions.
    func rebuildRegistry() {
        let agentRoot = PathSandbox.defaultRoot
        let skillsDir = agentRoot.appendingPathComponent("skills")
        let packageSkillFiles: [URL]
        if let packageRegistry {
            packageSkillFiles = PackageBootstrap.cachedSkillPaths(
                from: packageRegistry, agentRoot: agentRoot)
        } else {
            packageSkillFiles = []
        }
        let skills = SkillRegistry.discover(
            locations: [skillsDir], packageSkillFiles: packageSkillFiles)
        commandRegistry.rebuild(skills: skills, extensionHost: extensionHost)
    }

    // MARK: - Popup

    /// Update popup state based on current input text.
    func updateCommandPopup(for inputText: String) {
        if let prefix = SlashCommandParser.autocompletePrefix(inputText) {
            if !showCommandPopup { showCommandPopup = true }
            commandFilteredResults = commandRegistry.filter(prefix: prefix)
        } else {
            if showCommandPopup { showCommandPopup = false }
        }
        if commandSelectedIndex != 0 {
            commandSelectedIndex = 0
        }
    }

    func dismissCommandPopup() {
        showCommandPopup = false
        commandFilteredResults = []
        commandSelectedIndex = 0
    }

    /// Autocomplete a command into the input text (used by both keyboard and click).
    func autocompleteCommand(_ command: SlashCommand) -> String {
        dismissCommandPopup()
        return "/\(command.name) "
    }
}
