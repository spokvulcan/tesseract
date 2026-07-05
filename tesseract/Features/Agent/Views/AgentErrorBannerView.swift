//
//  AgentErrorBannerView.swift
//  tesseract
//

import SwiftUI

struct AgentInputStatusStrip: View {
    @Environment(AgentEngine.self) private var agentEngine
    @Environment(AgentCoordinator.self) private var coordinator
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    private var currentStatus: AgentInputStatus? {
        let selectedID = settings.selectedAgentModelID
        let voiceMessage: String?
        if case .error(let message) = coordinator.voiceInput.voiceState {
            voiceMessage = message
        } else {
            voiceMessage = nil
        }
        return AgentInputStatus.derive(
            error: coordinator.error,
            voiceErrorMessage: voiceMessage,
            isEngineLoading: agentEngine.isLoading,
            loadingStatus: agentEngine.loadingStatus,
            isModelLoaded: agentEngine.isModelLoaded,
            selectedModelDisplayName: ModelDefinition.withID(selectedID)?.displayName
                ?? selectedID,
            selectedModelStatus: downloadManager.status(for: selectedID)
        )
    }

    var body: some View {
        if let status = currentStatus {
            HStack(spacing: Theme.Spacing.sm) {
                statusIcon(status)
                statusLabel(status)
                Spacer(minLength: 0)
                if case .error = status {
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                            coordinator.error = nil
                        }
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 18, height: 18)
                            .background(.quaternary, in: Circle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
            .background {
                let shape = RoundedRectangle(cornerRadius: Theme.Radius.medium, style: .continuous)
                shape.fill(.ultraThinMaterial)
                    .overlay {
                        shape.fill(status.tint.opacity(0.06))
                    }
                    .overlay {
                        shape.strokeBorder(.quaternary, lineWidth: 0.5)
                    }
            }
            .shadow(color: .black.opacity(0.08), radius: 6, y: 2)
            .transition(.blurReplace.combined(with: .opacity))
            .animation(.spring(response: 0.35, dampingFraction: 0.85), value: currentStatus)
        }
    }

    @ViewBuilder
    private func statusIcon(_ status: AgentInputStatus) -> some View {
        switch status {
        case .error:
            Image(systemName: "exclamationmark.circle.fill")
                .font(.system(size: 13))
                .foregroundStyle(.red)
                .symbolRenderingMode(.hierarchical)
        case .voiceError:
            Image(systemName: "mic.slash.fill")
                .font(.system(size: 12))
                .foregroundStyle(.orange)
                .symbolRenderingMode(.hierarchical)
        case .loading:
            ProgressView()
                .controlSize(.mini)
                .tint(.blue)
        case .downloadingModel:
            Image(systemName: "arrow.down.circle.fill")
                .font(.system(size: 13))
                .foregroundStyle(.blue)
                .symbolRenderingMode(.hierarchical)
                .symbolEffect(.pulse, options: .repeating)
        case .notDownloaded:
            Image(systemName: "arrow.down.circle.fill")
                .font(.system(size: 13))
                .foregroundStyle(.yellow)
                .symbolRenderingMode(.hierarchical)
        }
    }

    private func statusLabel(_ status: AgentInputStatus) -> some View {
        let (text, style): (String, AnyShapeStyle) =
            switch status {
            case .error(let m), .voiceError(let m): (m, AnyShapeStyle(.primary.opacity(0.7)))
            case .loading(let t): (t, AnyShapeStyle(.secondary))
            case .downloadingModel(let name, let progress):
                (
                    "\(name) is on its way — \(progress.formatted(.wholePercent))",
                    AnyShapeStyle(.secondary)
                )
            case .notDownloaded:
                ("Download an agent model to get started", AnyShapeStyle(.secondary))
            }
        // Errors carry actionable guidance (e.g. "Reduce the number or size of
        // the attached images") that must stay readable; transient status lines
        // stay a single line.
        let lineLimit: Int =
            switch status {
            case .error, .voiceError: 3
            default: 1
            }
        return Text(text)
            .font(.caption)
            .foregroundStyle(style)
            .lineLimit(lineLimit)
    }
}

extension AgentInputStatus {
    fileprivate var tint: Color {
        switch self {
        case .error: .red
        case .voiceError: .orange
        case .loading, .downloadingModel: .blue
        case .notDownloaded: .yellow
        }
    }
}

struct AgentSpeechIndicatorBar: View {
    let onStop: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "speaker.wave.2.fill")
                .foregroundStyle(.tint)
                .symbolEffect(.variableColor.iterative, options: .repeating)
            Text("Speaking\u{2026}")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                onStop()
            } label: {
                Image(systemName: "stop.circle.fill")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.tint.opacity(0.08))
    }
}
