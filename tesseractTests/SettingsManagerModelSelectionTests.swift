import Foundation
import Testing

@testable import Tesseract_Agent

/// Focused regression for the stale-agent-model migration, pinned against a
/// realistic removed model id. Runs through the in-memory adapter — hermetic,
/// no `UserDefaults.standard` pollution. The general normalize-on-launch
/// behaviour and its write boundary are also pinned in `SettingsManagerTests`.
@MainActor
struct SettingsManagerModelSelectionTests {

    @Test
    func removedPersistedAgentModelFallsBackToDefaultAndPersists() {
        let store = InMemorySettingsStore()
        store.set("qwen3.6-35b-a3b-ud-3bit", for: "selectedAgentModelID")

        let settings = SettingsManager(store: store)
        #expect(settings.selectedAgentModelID == ModelDefinition.defaultAgentModelID)
        #expect(
            store.string(for: "selectedAgentModelID", default: "")
                == ModelDefinition.defaultAgentModelID
        )
    }

    @Test
    func removedPersistedSpeechToTextModelFallsBackToDefaultAndPersists() {
        let store = InMemorySettingsStore()
        store.set("whisper-large-v2-deprecated", for: "selectedSpeechToTextModelID")

        let settings = SettingsManager(store: store)
        #expect(settings.selectedSpeechToTextModelID == ModelDefinition.defaultSpeechToTextModelID)
        #expect(
            store.string(for: "selectedSpeechToTextModelID", default: "")
                == ModelDefinition.defaultSpeechToTextModelID
        )
    }
}
