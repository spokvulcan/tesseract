import SwiftUI
import os

struct AgentContentView: View {
    @ObservedObject var coordinator: AgentCoordinator
    @ObservedObject var agentEngine: AgentEngine
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    @State private var inputText = ""

    private let agentModelID = "nanbeige4.1-3b"

    private var isModelDownloaded: Bool {
        if case .downloaded = downloadManager.statuses[agentModelID] {
            return true
        }
        return false
    }

    var body: some View {
        VStack(spacing: 0) {
            if agentEngine.isLoading {
                modelLoadingBanner
            } else if !agentEngine.isModelLoaded && !isModelDownloaded {
                modelNotDownloadedBanner
            }

            if let error = coordinator.error {
                errorBanner(error)
            }

            messageList

            Divider()

            inputBar
        }
        .navigationTitle("Agent")
    }

    // MARK: - Model Status

    private var modelLoadingBanner: some View {
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

    private var modelNotDownloadedBanner: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            Text("Download Nanbeige4.1-3B from the Models page to use the agent.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(.bar)
    }

    // MARK: - Error

    private func errorBanner(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.callout)
                .lineLimit(2)
            Spacer()
            Button {
                coordinator.error = nil
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.red.opacity(0.1))
    }

    // MARK: - Messages

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if coordinator.messages.isEmpty && !coordinator.isGenerating {
                        emptyState
                    }

                    ForEach(Array(coordinator.messages.enumerated()), id: \.offset) { index, message in
                        messageBubble(message, id: "msg-\(index)")
                    }

                    if coordinator.isGenerating &&
                        (!coordinator.streamingText.isEmpty || !coordinator.streamingThinking.isEmpty) {
                        streamingBubble
                    }

                    if coordinator.isGenerating
                        && coordinator.streamingText.isEmpty
                        && coordinator.streamingThinking.isEmpty {
                        HStack(spacing: 6) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Generating…")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 12)
                        .id("generating")
                    }
                }
                .padding()
            }
            .onChange(of: coordinator.streamingText) {
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("streaming", anchor: .bottom)
                }
            }
            .onChange(of: coordinator.streamingThinking) {
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("streaming", anchor: .bottom)
                }
            }
            .onChange(of: coordinator.messages.count) {
                let target = coordinator.isGenerating ? "streaming" : "msg-\(coordinator.messages.count - 1)"
                withAnimation(.easeOut(duration: 0.15)) {
                    proxy.scrollTo(target, anchor: .bottom)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 40))
                .foregroundStyle(.quaternary)
            Text("Start a conversation")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 80)
    }

    @ViewBuilder
    private func messageBubble(_ message: AgentChatMessage, id: String) -> some View {
        if message.role == .assistant {
            AssistantMessageBubble(message: message)
                .id(id)
        } else {
            HStack {
                if message.role == .user { Spacer(minLength: 60) }

                Text(message.content)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        message.role == .user
                            ? AnyShapeStyle(.tint.opacity(0.15))
                            : AnyShapeStyle(.fill.quaternary)
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                if message.role != .user { Spacer(minLength: 60) }
            }
            .id(id)
        }
    }

    private var streamingBubble: some View {
        HStack {
            VStack(alignment: .leading, spacing: 0) {
                if !coordinator.streamingThinking.isEmpty {
                    DisclosureGroup(isExpanded: .constant(true)) {
                        Text(coordinator.streamingThinking)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .padding(.top, 4)
                    } label: {
                        Label(
                            coordinator.isThinking ? "Thinking…" : "Thinking",
                            systemImage: "brain.head.profile"
                        )
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    }
                    .padding(.bottom, 6)
                }

                if !coordinator.streamingText.isEmpty {
                    Text(coordinator.streamingText)
                        .textSelection(.enabled)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.fill.quaternary)
            .clipShape(RoundedRectangle(cornerRadius: 12))

            Spacer(minLength: 60)
        }
        .id("streaming")
    }

    // MARK: - Input

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Message…", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .onSubmit { send() }

            if coordinator.isGenerating {
                Button {
                    coordinator.cancelGeneration()
                } label: {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.red)
                }
                .buttonStyle(.plain)
                .help("Cancel generation")
            } else {
                Button {
                    send()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(canSend ? AnyShapeStyle(.tint) : AnyShapeStyle(.quaternary))
                }
                .buttonStyle(.plain)
                .disabled(!canSend)
                .help("Send message")
            }
        }
        .padding(12)
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !coordinator.isGenerating
    }

    // MARK: - Actions

    private func send() {
        let text = inputText
        inputText = ""

        Task {
            await loadModelIfNeeded()
            guard agentEngine.isModelLoaded else { return }
            coordinator.sendMessage(text)
        }
    }

    private func loadModelIfNeeded() async {
        guard isModelDownloaded,
              !agentEngine.isModelLoaded,
              !agentEngine.isLoading,
              let path = downloadManager.modelPath(for: agentModelID)
        else { return }

        do {
            try await agentEngine.loadModel(from: path)
        } catch {
            coordinator.error = "Failed to load model: \(error.localizedDescription)"
        }
    }
}

// MARK: - Assistant Message Bubble

private struct AssistantMessageBubble: View {
    let message: AgentChatMessage
    @State private var isThinkingExpanded = false

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 0) {
                if let thinking = message.thinking, !thinking.isEmpty {
                    thinkingSection(thinking)
                }
                if !message.content.isEmpty {
                    Text(message.content)
                        .textSelection(.enabled)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.fill.quaternary)
            .clipShape(RoundedRectangle(cornerRadius: 12))

            Spacer(minLength: 60)
        }
    }

    private func thinkingSection(_ thinking: String) -> some View {
        DisclosureGroup(isExpanded: $isThinkingExpanded) {
            Text(thinking)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .padding(.top, 4)
        } label: {
            Label("Thinking", systemImage: "brain.head.profile")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(.bottom, 6)
    }
}
