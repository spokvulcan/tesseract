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

    private var isModelDownloaded: Bool {
        if case .downloaded = downloadManager.statuses[settings.selectedAgentModelID] {
            return true
        }
        return false
    }

    private enum Status: Equatable {
        case error(String)
        case voiceError(String)
        case loading(String)
        case notDownloaded

        var tint: Color {
            switch self {
            case .error: .red
            case .voiceError: .orange
            case .loading: .blue
            case .notDownloaded: .yellow
            }
        }
    }

    private var currentStatus: Status? {
        if let error = coordinator.error {
            return .error(error)
        }
        if case .error(let message) = coordinator.voiceState {
            return .voiceError(message)
        }
        if agentEngine.isLoading {
            let text = agentEngine.loadingStatus.isEmpty ? "Loading model\u{2026}" : agentEngine.loadingStatus
            return .loading(text)
        }
        if !agentEngine.isModelLoaded && !isModelDownloaded {
            return .notDownloaded
        }
        return nil
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
    private func statusIcon(_ status: Status) -> some View {
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
        case .notDownloaded:
            Image(systemName: "arrow.down.circle.fill")
                .font(.system(size: 13))
                .foregroundStyle(.yellow)
                .symbolRenderingMode(.hierarchical)
        }
    }

    private func statusLabel(_ status: Status) -> some View {
        let (text, style): (String, AnyShapeStyle) = switch status {
        case .error(let m), .voiceError(let m): (m, AnyShapeStyle(.primary.opacity(0.7)))
        case .loading(let t): (t, AnyShapeStyle(.secondary))
        case .notDownloaded: ("Download an agent model to get started", AnyShapeStyle(.secondary))
        }
        return Text(text)
            .font(.caption)
            .foregroundStyle(style)
            .lineLimit(1)
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
