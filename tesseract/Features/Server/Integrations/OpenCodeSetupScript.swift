//
//  OpenCodeSetupScript.swift
//  tesseract
//

import Foundation

/// Renders the payload of the Setup One-liner: the shell script served at
/// `GET /integrations/opencode/setup.sh`. Deliberately dumb plumbing — back
/// up, POST the existing config to the Config Merge, write the result
/// atomically, report. Every decision lives server-side in
/// `OpenCodeConfigMerge`; the script embeds no model data beyond the printed
/// summary, and the one-liner re-fetches it on every run, so a copied command
/// stays correct as models and port change.
nonisolated enum OpenCodeSetupScript {

    /// Response header signalling the existing config was unparseable and got
    /// replaced. The script surfaces it; the backup already preserves the
    /// original.
    static let configWarningHeader = "X-Tesseract-Config-Warning"

    /// The terminal command users copy from Settings.
    static func oneLiner(port: Int) -> String {
        "curl -fsSL http://127.0.0.1:\(port)\(IntegrationRoutes.openCodeSetupScript) | sh"
    }

    static func render(snapshot: IntegrationSnapshot) -> String {
        """
        #!/bin/sh
        # Tesseract — OpenCode setup. Served by the local Tesseract server.
        # Re-run the one-liner any time to refresh (models, port, defaults).
        set -eu

        BASE="http://127.0.0.1:\(snapshot.port)"
        MERGE_URL="$BASE\(IntegrationRoutes.openCodeMerge)"
        CONFIG_DIR="$HOME/.config/opencode"

        mkdir -p "$CONFIG_DIR"
        tmp=""
        hdrs=""
        trap 'rm -f "$tmp" "$hdrs"' EXIT

        apply() {
          config="$1"
          tmp="$(mktemp)"
          hdrs="$(mktemp)"
          if [ -f "$config" ]; then
            backup="$config.$(date +%s).$$.bak"
            cp "$config" "$backup"
            echo "+ Backed up $(basename "$config") -> $backup"
            curl -fsS -X POST --data-binary @"$config" -D "$hdrs" -o "$tmp" "$MERGE_URL"
          else
            curl -fsS -X POST -D "$hdrs" -o "$tmp" "$MERGE_URL"
          fi
          if grep -qi '^\(Self.configWarningHeader)' "$hdrs"; then
            echo "! $(basename "$config") could not be parsed — replaced it (backup kept)."
          fi
          mv "$tmp" "$config"
          echo "+ Wrote provider \\"tesseract\\" -> $config"
        }

        # OpenCode merges global configs in the order config.json ->
        # opencode.json -> opencode.jsonc, later files winning per key. So the
        # merge must land in opencode.jsonc when it exists — and when both
        # opencode files exist, both are refreshed so no stale tesseract block
        # leaks through the deep merge from the lower-precedence one.
        if [ -f "$CONFIG_DIR/opencode.jsonc" ]; then
          apply "$CONFIG_DIR/opencode.jsonc"
          echo "  note: comments in opencode.jsonc are not preserved (backup kept)"
          if [ -f "$CONFIG_DIR/opencode.json" ]; then
            apply "$CONFIG_DIR/opencode.json"
          fi
        else
          apply "$CONFIG_DIR/opencode.json"
        fi
        \(modelSummaryLines(snapshot: snapshot))

        if command -v opencode >/dev/null 2>&1; then
          echo 'Run `opencode` to start.'
        else
          echo 'OpenCode is not installed. Install it with:'
          echo '  curl -fsSL https://opencode.ai/install | bash'
        fi
        """
    }

    // MARK: - Private

    private static func modelSummaryLines(snapshot: IntegrationSnapshot) -> String {
        guard !snapshot.models.isEmpty else {
            return """
                echo "! No models downloaded yet — download one in Settings -> Models, then re-run."
                """
        }
        let idWidth = snapshot.models.map(\.id.count).max() ?? 0
        var lines = snapshot.models.map { model in
            let paddedID = model.id.padding(toLength: idWidth, withPad: " ", startingAt: 0)
            let modality =
                switch (model.visionCapable, model.audioCapable) {
                case (true, true): "v+a   "
                case (true, false): "vision"
                case (false, true): "audio "
                case (false, false): "text  "
                }
            let context = "\(model.contextLength / 1000)k"
            return "echo \"    \(paddedID)  \(modality)  \(context)\""
        }
        if let defaultModelID = snapshot.defaultModelID {
            lines.append("echo \"+ Default model -> tesseract/\(defaultModelID)\"")
        }
        return lines.joined(separator: "\n")
    }
}
