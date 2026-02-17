import SwiftUI
import os

struct AgentContentView: View {
    @ObservedObject var coordinator: AgentCoordinator
    @ObservedObject var agentEngine: AgentEngine
    @ObservedObject var conversationStore: AgentConversationStore
    @ObservedObject var transcriptionEngine: TranscriptionEngine
    @ObservedObject var audioCapture: AudioCaptureEngine
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    @State private var inputText = ""
    @State private var showingHistory = false
    @State private var isHoldingMic = false

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

            if case .error(let message) = coordinator.voiceState {
                voiceErrorBanner(message)
            }

            messageList

            Divider()

            inputBar
        }
        .navigationTitle("Agent")
        .onExitCommand {
            if coordinator.voiceState == .recording {
                coordinator.cancelVoiceInput()
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    coordinator.newConversation()
                } label: {
                    Image(systemName: "plus.message")
                }
                .help("New conversation")
                .disabled(coordinator.isGenerating)

                Button {
                    showingHistory.toggle()
                } label: {
                    Image(systemName: "clock.arrow.circlepath")
                }
                .help("Conversation history")
                .popover(isPresented: $showingHistory) {
                    conversationHistoryPopover
                }
            }
        }
    }

    // MARK: - Conversation History

    private var conversationHistoryPopover: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                if conversationStore.conversations.isEmpty {
                    Text("No past conversations")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .padding()
                        .frame(maxWidth: .infinity)
                } else {
                    ForEach(conversationStore.conversations) { summary in
                        conversationRow(summary)
                    }
                }
            }
        }
        .frame(maxWidth: 280, maxHeight: 360)
    }

    private func conversationRow(_ summary: AgentConversationSummary) -> some View {
        let isCurrent = conversationStore.currentConversation?.id == summary.id
        return Button {
            coordinator.loadConversation(summary.id)
            showingHistory = false
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(summary.title)
                        .font(.callout)
                        .lineLimit(1)
                        .foregroundStyle(isCurrent ? AnyShapeStyle(.tint) : AnyShapeStyle(.primary))
                    Text(summary.updatedAt.formatted(.relative(presentation: .named)))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text("\(summary.messageCount)")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button(role: .destructive) {
                coordinator.deleteConversation(summary.id)
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
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

    private func voiceErrorBanner(_ message: String) -> some View {
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

    // MARK: - Messages

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if coordinator.messages.isEmpty && !coordinator.isGenerating {
                        emptyState
                    }

                    ForEach(coordinator.messages) { message in
                        messageBubble(message)
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
                if let lastID = coordinator.messages.last?.id {
                    let target: AnyHashable = coordinator.isGenerating ? "streaming" : lastID
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(target, anchor: .bottom)
                    }
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
    private func messageBubble(_ message: AgentChatMessage) -> some View {
        if message.role == .assistant {
            AssistantMessageBubble(message: message)
                .id(message.id)
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
            .id(message.id)
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
                .disabled(coordinator.voiceState == .recording || coordinator.voiceState == .transcribing)

            micButton

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

    private var micButton: some View {
        let state = coordinator.voiceState

        return micIcon(for: state)
            .font(.title2)
            .frame(width: 28, height: 28)
            .contentShape(Circle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !isHoldingMic else { return }
                        isHoldingMic = true
                        coordinator.startVoiceInput()
                    }
                    .onEnded { _ in
                        isHoldingMic = false
                        if coordinator.voiceState == .recording {
                            coordinator.stopVoiceInputAndSend()
                        }
                    }
            )
            .disabled(!canUseVoice)
            .help(voiceButtonHelp)
    }

    @ViewBuilder
    private func micIcon(for state: AgentVoiceState) -> some View {
        switch state {
        case .idle:
            Image(systemName: "mic.fill")
                .foregroundStyle(canUseVoice ? AnyShapeStyle(.secondary) : AnyShapeStyle(.quaternary))
        case .recording:
            Image(systemName: "stop.fill")
                .foregroundStyle(.red)
                .symbolEffect(.pulse, options: .repeating)
        case .transcribing:
            Image(systemName: "waveform")
                .foregroundStyle(.tint)
                .symbolEffect(.variableColor.iterative, options: .repeating)
        case .error:
            Image(systemName: "mic.slash.fill")
                .foregroundStyle(.red)
        }
    }

    private var canUseVoice: Bool {
        !coordinator.isGenerating
            && coordinator.voiceState != .transcribing
            && transcriptionEngine.isModelLoaded
    }

    private var voiceButtonHelp: String {
        if !transcriptionEngine.isModelLoaded {
            return "Download Whisper model to use voice input"
        }
        if coordinator.isGenerating {
            return "Voice input unavailable during generation"
        }
        switch coordinator.voiceState {
        case .recording: return "Release to send"
        case .transcribing: return "Transcribing…"
        case .error(let msg): return msg
        default: return "Hold to speak"
        }
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
