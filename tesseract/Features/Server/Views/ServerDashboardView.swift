//
//  ServerDashboardView.swift
//  tesseract
//

import SwiftUI

struct ServerDashboardView: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(HTTPServer.self) private var httpServer
    @Environment(InferenceArbiter.self) private var arbiter

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
            statusHeader
            ServerActivityPanel()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(Theme.Spacing.lg)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .navigationTitle("Dashboard")
    }

    private var runState: ServerRunState {
        ServerRunState(
            enabled: settings.isServerEnabled,
            isRunning: httpServer.isRunning,
            lastStartError: httpServer.lastStartError
        )
    }

    private var statusHeader: some View {
        HStack(spacing: Theme.Spacing.md) {
            runStatePill
            endpointDisplay
            Spacer()
            triAttentionChip
        }
    }

    private var runStatePill: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(runState.dotColor)
                .frame(width: 6, height: 6)
            Text(runState.displayLabel)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.primary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .glassEffect(
            runState == .stopped ? .regular : .regular.tint(runState.dotColor),
            in: Capsule()
        )
        .help(runState.failureMessage ?? runState.displayLabel)
    }

    @ViewBuilder
    private var endpointDisplay: some View {
        HStack(spacing: Theme.Spacing.xs) {
            Text(serverEndpointURL(port: settings.serverPort))
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(runState == .running ? .primary : .secondary)
                .textSelection(.enabled)

            Button {
                copyServerEndpointToPasteboard(port: settings.serverPort)
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .buttonStyle(.glass)
            .help("Copy Endpoint")
        }
    }

    @ViewBuilder
    private var triAttentionChip: some View {
        HStack(spacing: 6) {
            Image(systemName: "cpu")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(triAttentionModeDescription(for: arbiter))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .glassEffect(.regular, in: Capsule())
    }
}
