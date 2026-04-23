import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SettingsManagerModelSelectionTests {

    private func clearModelSelectionDefaults() {
        UserDefaults.standard.removeObject(forKey: "selectedAgentModelID")
    }

    @Test
    func removedPersistedAgentModelFallsBackToDefaultAndPersists() {
        clearModelSelectionDefaults()
        defer { clearModelSelectionDefaults() }

        UserDefaults.standard.set("qwen3.6-35b-a3b-ud-3bit", forKey: "selectedAgentModelID")

        let settings = SettingsManager()
        #expect(settings.selectedAgentModelID == ModelDefinition.defaultAgentModelID)
        #expect(
            UserDefaults.standard.string(forKey: "selectedAgentModelID")
            == ModelDefinition.defaultAgentModelID
        )
    }
}
