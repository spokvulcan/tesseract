import Foundation
import Testing

@testable import Tesseract_Agent

/// Behaviour of the **Setup One-liner** payload: the rendered script is dumb
/// plumbing whose every variable part comes from the snapshot — port, merge
/// URL, summary lines. Tests assert the user-visible contract (what the
/// script does and reports), not shell internals.
struct OpenCodeSetupScriptTests {

    @Test func oneLinerFetchesTheScriptFromTheLivePort() {
        let command = OpenCodeSetupScript.oneLiner(port: 9999)

        #expect(command == "curl -fsSL http://127.0.0.1:9999/integrations/opencode/setup.sh | sh")
    }

    @Test func scriptTargetsTheSnapshotPortAndMergeRoute() {
        let script = OpenCodeSetupScript.render(snapshot: snapshot(port: 9999))

        #expect(script.contains(#"BASE="http://127.0.0.1:9999""#))
        #expect(script.contains(#"MERGE_URL="$BASE/integrations/opencode/merge""#))
    }

    @Test func scriptBacksUpThenPostsTheExistingConfig() {
        let script = OpenCodeSetupScript.render(snapshot: snapshot())

        #expect(script.contains(#"cp "$config" "$backup""#))
        #expect(script.contains(#"--data-binary @"$config""#))
        // Write is atomic: merge result lands in a temp file, then replaces
        // the config — never a partial write of the live file.
        #expect(script.contains(#"mv "$tmp" "$config""#))
    }

    /// OpenCode merges opencode.jsonc over opencode.json, so the setup must
    /// land in the winning file — and refresh both when both exist, or a
    /// stale tesseract block in the lower-precedence file would leak through
    /// OpenCode's deep merge.
    @Test func scriptTargetsJSONCWhenPresentAndRefreshesBothFiles() throws {
        let script = OpenCodeSetupScript.render(snapshot: snapshot())

        #expect(script.contains(#"if [ -f "$CONFIG_DIR/opencode.jsonc" ]; then"#))
        // The trailing quote disambiguates .json from .jsonc in the search.
        let jsoncApply = try #require(script.range(of: #"apply "$CONFIG_DIR/opencode.jsonc""#))
        let jsonApply = try #require(script.range(of: #"apply "$CONFIG_DIR/opencode.json""#))
        // Both-files branch: jsonc (the winning file) first, json refreshed after.
        #expect(jsoncApply.lowerBound < jsonApply.lowerBound)
        #expect(script.contains("comments in opencode.jsonc are not preserved"))
    }

    @Test func scriptSurfacesTheCorruptConfigWarningHeader() {
        let script = OpenCodeSetupScript.render(snapshot: snapshot())

        #expect(script.contains("X-Tesseract-Config-Warning"))
    }

    @Test func summaryListsModelsWithModalityAndDefault() {
        let script = OpenCodeSetupScript.render(snapshot: snapshot())

        #expect(script.contains("qwen3.6-27b-paro"))
        #expect(script.contains("vision"))
        #expect(script.contains("262k"))
        #expect(script.contains("Default model -> tesseract/qwen3.6-27b-paro"))
    }

    @Test func emptySnapshotHintsAtDownloadingInsteadOfDefault() {
        let script = OpenCodeSetupScript.render(
            snapshot: IntegrationSnapshot(port: 8321, models: [], defaultModelID: nil)
        )

        #expect(script.contains("No models downloaded yet"))
        #expect(!script.contains("Default model ->"))
    }

    @Test func scriptPointsAtTheOpenCodeInstallerWhenMissing() {
        let script = OpenCodeSetupScript.render(snapshot: snapshot())

        #expect(script.contains("command -v opencode"))
        #expect(script.contains("curl -fsSL https://opencode.ai/install | bash"))
    }

    // MARK: - Fixtures

    private func snapshot(port: Int = 8321) -> IntegrationSnapshot {
        IntegrationSnapshot(
            port: port,
            models: [
                IntegrationSnapshot.Model(
                    id: "qwen3.6-27b-paro",
                    displayName: "Qwen3.6-27B PARO",
                    visionCapable: true,
                    contextLength: 262_144
                )
            ],
            defaultModelID: "qwen3.6-27b-paro"
        )
    }
}
