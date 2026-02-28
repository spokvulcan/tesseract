//
//  AgentErrorBannerView.swift
//  tesseract
//

import SwiftUI

struct AgentModelLoadingBanner: View {
    @EnvironmentObject private var agentEngine: AgentEngine

    var body: some View {
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
            Text(agentEngine.loadingStatus.isEmpty ? "Loading model…" : agentEngine.loadingStatus)
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(.bar)
    }
}

struct AgentModelNotDownloadedBanner: View {
    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            Text("Download an agent model from the Models page to use the agent.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(.bar)
    }
}

struct AgentErrorBanner: View {
    let message: String
    let onDismiss: () -> Void

    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.callout)
                .lineLimit(2)
            Spacer()
            Button {
                onDismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, 6)
        .background(.red.opacity(0.1))
    }
}

struct AgentVoiceErrorBanner: View {
    let message: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "mic.slash.fill")
                .foregroundStyle(.orange)
            Text(message)
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .background(.orange.opacity(0.1))
        .transition(.move(edge: .top).combined(with: .opacity))
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
