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
        CONFIG="$CONFIG_DIR/opencode.json"

        mkdir -p "$CONFIG_DIR"
        TMP="$(mktemp)"
        HDRS="$(mktemp)"
        trap 'rm -f "$TMP" "$HDRS"' EXIT

        if [ -f "$CONFIG" ]; then
          BACKUP="$CONFIG.$(date +%s).$$.bak"
          cp "$CONFIG" "$BACKUP"
          echo "+ Backed up opencode.json -> $BACKUP"
          curl -fsS -X POST --data-binary @"$CONFIG" -D "$HDRS" -o "$TMP" "$MERGE_URL"
        else
          curl -fsS -X POST -D "$HDRS" -o "$TMP" "$MERGE_URL"
        fi

        if grep -qi '^\(Self.configWarningHeader)' "$HDRS"; then
          echo "! Existing config could not be parsed — replaced it (backup kept)."
        fi

        mv "$TMP" "$CONFIG"
        echo "+ Wrote provider \\"tesseract\\" -> $CONFIG"
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
            let modality = model.visionCapable ? "vision" : "text  "
            let context = "\(model.contextLength / 1000)k"
            return "echo \"    \(paddedID)  \(modality)  \(context)\""
        }
        if let defaultModelID = snapshot.defaultModelID {
            lines.append("echo \"+ Default model -> tesseract/\(defaultModelID)\"")
        }
        return lines.joined(separator: "\n")
    }
}
