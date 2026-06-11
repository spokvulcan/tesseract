import Foundation
import Testing

@testable import Tesseract_Agent

/// Behaviour of the **Config Merge** for the OpenCode Integration: the
/// `provider.tesseract` block is generated output replaced wholesale from the
/// snapshot; everything else in the file is the user's and survives
/// untouched. The interface is the test surface — bytes in, bytes out,
/// asserted through plain JSON parsing.
struct OpenCodeConfigMergeTests {

    // MARK: - Fresh config

    @Test func freshConfigWritesSchemaProviderAndDefaultModel() throws {
        let output = OpenCodeConfigMerge.merge(existingConfig: nil, snapshot: snapshot())

        let root = try parse(output.configData)
        #expect(output.replacedCorruptInput == false)
        #expect(root["$schema"] as? String == "https://opencode.ai/config.json")
        #expect(root["model"] as? String == "tesseract/qwen3.5-27b-paro")
        let tesseract = try providerBlock(root)
        #expect(tesseract["npm"] as? String == "@ai-sdk/openai-compatible")
        #expect(tesseract["name"] as? String == "Tesseract")
        let options = tesseract["options"] as? [String: Any]
        #expect(options?["baseURL"] as? String == "http://127.0.0.1:8321/v1")
    }

    @Test func emptyDataIsFreshNotCorrupt() throws {
        let output = OpenCodeConfigMerge.merge(existingConfig: Data(), snapshot: snapshot())

        #expect(output.replacedCorruptInput == false)
        let root = try parse(output.configData)
        #expect(root["$schema"] as? String == "https://opencode.ai/config.json")
    }

    // MARK: - Preservation & wholesale replacement

    @Test func foreignKeysAndProvidersSurviveUntouched() throws {
        let existing = Data(#"""
        {
          "provider": {
            "omlx": { "npm": "@ai-sdk/openai-compatible", "options": { "apiKey": "secret-k" } }
          },
          "mcp": { "pencil": { "enabled": true } },
          "keybinds": { "leader": "ctrl+x" }
        }
        """#.utf8)

        let output = OpenCodeConfigMerge.merge(existingConfig: existing, snapshot: snapshot())

        let root = try parse(output.configData)
        let provider = root["provider"] as? [String: Any]
        let omlx = provider?["omlx"] as? [String: Any]
        #expect((omlx?["options"] as? [String: Any])?["apiKey"] as? String == "secret-k")
        #expect((root["mcp"] as? [String: Any])?.keys.contains("pencil") == true)
        #expect((root["keybinds"] as? [String: Any])?["leader"] as? String == "ctrl+x")
    }

    @Test func tesseractBlockIsReplacedWholesale() throws {
        let existing = Data(#"""
        {
          "provider": {
            "tesseract": {
              "handTunedKey": true,
              "models": { "deleted-model": { "name": "stale" } }
            }
          }
        }
        """#.utf8)

        let output = OpenCodeConfigMerge.merge(existingConfig: existing, snapshot: snapshot())

        let tesseract = try providerBlock(try parse(output.configData))
        #expect(tesseract["handTunedKey"] == nil)
        let models = tesseract["models"] as? [String: Any]
        #expect(models?["deleted-model"] == nil)
        #expect(models?["qwen3.5-27b-paro"] != nil)
    }

    @Test func mergeIsIdempotent() throws {
        let first = OpenCodeConfigMerge.merge(existingConfig: nil, snapshot: snapshot())
        let second = OpenCodeConfigMerge.merge(
            existingConfig: first.configData,
            snapshot: snapshot()
        )

        #expect(second.configData == first.configData)
        #expect(second.replacedCorruptInput == false)
    }

    // MARK: - JSONC input (OpenCode parses every config file as JSONC)

    @Test func commentsAndTrailingCommasAreLegalInput() throws {
        let existing = Data(#"""
        {
          // line comment
          "mcp": {
            /* block
               comment */
            "pencil": { "enabled": true },
          },
          "model": "omlx/some-model",
        }
        """#.utf8)

        let output = OpenCodeConfigMerge.merge(existingConfig: existing, snapshot: snapshot())

        #expect(output.replacedCorruptInput == false)
        let root = try parse(output.configData)
        #expect((root["mcp"] as? [String: Any])?.keys.contains("pencil") == true)
        #expect(root["model"] as? String == "tesseract/qwen3.5-27b-paro")
    }

    @Test func commentMarkersInsideStringsAreNotComments() throws {
        let existing = Data(#"""
        {
          "provider": {
            "omlx": {
              "options": { "baseURL": "http://127.0.0.1:8000/v1" },
              "note": "a /* literal */ value, trailing"
            }
          }
        }
        """#.utf8)

        let output = OpenCodeConfigMerge.merge(existingConfig: existing, snapshot: snapshot())

        #expect(output.replacedCorruptInput == false)
        let omlx = try #require(
            (try parse(output.configData)["provider"] as? [String: Any])?["omlx"]
                as? [String: Any]
        )
        #expect((omlx["options"] as? [String: Any])?["baseURL"] as? String
            == "http://127.0.0.1:8000/v1")
        #expect(omlx["note"] as? String == "a /* literal */ value, trailing")
    }

    // MARK: - Corrupt input

    @Test func unparseableInputIsFlaggedAndReplacedFresh() throws {
        let output = OpenCodeConfigMerge.merge(
            existingConfig: Data("{ not json".utf8),
            snapshot: snapshot()
        )

        #expect(output.replacedCorruptInput == true)
        let root = try parse(output.configData)
        #expect(root["$schema"] as? String == "https://opencode.ai/config.json")
        #expect(try providerBlock(root)["name"] as? String == "Tesseract")
    }

    @Test func nonObjectRootIsCorrupt() throws {
        let output = OpenCodeConfigMerge.merge(
            existingConfig: Data("[1, 2, 3]".utf8),
            snapshot: snapshot()
        )

        #expect(output.replacedCorruptInput == true)
    }

    // MARK: - Model entries

    @Test func visionModelAdvertisesImageInputAndAttachment() throws {
        let output = OpenCodeConfigMerge.merge(existingConfig: nil, snapshot: snapshot())

        let entry = try modelEntry(output.configData, id: "qwen3.5-27b-paro")
        #expect(entry["attachment"] as? Bool == true)
        let modalities = entry["modalities"] as? [String: Any]
        #expect(modalities?["input"] as? [String] == ["text", "image"])
        #expect(modalities?["output"] as? [String] == ["text"])
        #expect(entry["name"] as? String == "Qwen3.5-27B PARO (Tesseract)")
    }

    @Test func textModelIsTextOnlyWithoutAttachment() throws {
        let output = OpenCodeConfigMerge.merge(existingConfig: nil, snapshot: snapshot())

        let entry = try modelEntry(output.configData, id: "qwen3.6-27b")
        #expect(entry["attachment"] == nil)
        let modalities = entry["modalities"] as? [String: Any]
        #expect(modalities?["input"] as? [String] == ["text"])
    }

    @Test func limitsCarryTheContextLength() throws {
        let output = OpenCodeConfigMerge.merge(existingConfig: nil, snapshot: snapshot())

        let entry = try modelEntry(output.configData, id: "qwen3.5-27b-paro")
        let limit = entry["limit"] as? [String: Any]
        #expect(limit?["context"] as? Int == 262_144)
        #expect(limit?["output"] as? Int == 262_144)
    }

    // MARK: - Default model

    @Test func defaultModelOverwritesExistingDefault() throws {
        let existing = Data(#"{ "model": "omlx/Qwen3.5-9B-MLX-4bit" }"#.utf8)

        let output = OpenCodeConfigMerge.merge(existingConfig: existing, snapshot: snapshot())

        let root = try parse(output.configData)
        #expect(root["model"] as? String == "tesseract/qwen3.5-27b-paro")
    }

    @Test func emptySnapshotPreservesExistingDefault() throws {
        let existing = Data(#"{ "model": "omlx/Qwen3.5-9B-MLX-4bit" }"#.utf8)
        let empty = IntegrationSnapshot(port: 8321, models: [], defaultModelID: nil)

        let output = OpenCodeConfigMerge.merge(existingConfig: existing, snapshot: empty)

        let root = try parse(output.configData)
        #expect(root["model"] as? String == "omlx/Qwen3.5-9B-MLX-4bit")
    }

    // MARK: - Fixtures

    private func snapshot(port: Int = 8321) -> IntegrationSnapshot {
        IntegrationSnapshot(
            port: port,
            models: [
                IntegrationSnapshot.Model(
                    id: "qwen3.5-27b-paro",
                    displayName: "Qwen3.5-27B PARO",
                    visionCapable: true,
                    contextLength: 262_144
                ),
                IntegrationSnapshot.Model(
                    id: "qwen3.6-27b",
                    displayName: "Qwen3.6-27B (MLX 4bit)",
                    visionCapable: false,
                    contextLength: 262_144
                ),
            ],
            defaultModelID: "qwen3.5-27b-paro"
        )
    }

    private func parse(_ data: Data) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func providerBlock(_ root: [String: Any]) throws -> [String: Any] {
        try #require((root["provider"] as? [String: Any])?["tesseract"] as? [String: Any])
    }

    private func modelEntry(_ data: Data, id: String) throws -> [String: Any] {
        let tesseract = try providerBlock(try parse(data))
        return try #require((tesseract["models"] as? [String: Any])?[id] as? [String: Any])
    }
}
