//
//  TodayComposer.swift
//  tesseract
//
//  Today's one field: the agent composer simplified. The same glass panel,
//  notice slot and action row, with only what the day needs. Return asks
//  Jarvis (his answer is in Chat), Add task (⌘↩) turns the words into a
//  reminder through Capture with no model, and the mic is hold-to-talk into
//  the field. The notice slot is the composer's one banner: the latest change
//  to the agenda with its undo, or why a capture, a take or a turn didn't
//  land.
//

import SwiftUI

// The agent composer's action-row metrics, so the two read as one family.
private let actionIconFont: Font = .system(size: 15, weight: .medium)
private let actionIconFrame: CGFloat = 26
private let composerRadius: CGFloat = 16

struct TodayComposer: View {
    @Environment(DayThread.self) private var thread
    @Environment(CaptureService.self) private var capture
    @Environment(AgentVoiceInputController.self) private var voice
    @Environment(TranscriptionEngine.self) private var transcriptionEngine
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var downloads: ModelDownloadManager

    /// After a question is sent: show the answer.
    let onAsk: () -> Void

    @State private var text = ""
    @State private var holdingMic = false
    /// A task is being saved: ⌘↩ again saves it once.
    @State private var adding = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ComposerNotice()
            TextField("Ask Jarvis about your day…", text: $text, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .onSubmit(ask)
                .disabled(voice.voiceState == .recording || voice.voiceState == .transcribing)
                .padding(.horizontal, 16)
                .padding(.top, 14)
                .padding(.bottom, 10)
            HStack(spacing: 10) {
                addTaskButton
                Spacer(minLength: 8)
                micButton
                sendButton
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 10)
        }
        .glassEffect(
            .regular.interactive(),
            in: RoundedRectangle(cornerRadius: composerRadius, style: .continuous)
        )
        .overlay {
            RoundedRectangle(cornerRadius: composerRadius, style: .continuous)
                .strokeBorder(.quaternary, lineWidth: 0.5)
        }
        .frame(maxWidth: 760)
        .onAppear {
            voice.onVoiceTranscription = { [text = $text] words in
                let current = text.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)
                text.wrappedValue = current.isEmpty ? words : current + " " + words
            }
        }
    }

    // MARK: - Controls

    /// The words as a task: through the capture parser into Reminders, no
    /// model ("call the dentist tomorrow at 10" lands tomorrow at ten).
    private var addTaskButton: some View {
        Button(action: addTask) {
            HStack(spacing: 7) {
                Image(systemName: "plus")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(width: 22, height: 22)
                    .background(.quinary, in: Circle())
                Text("Add task")
            }
            .foregroundStyle(.secondary)
            .frame(height: actionIconFrame)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .focusable(false)
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(trimmed.isEmpty)
        .help("Add as a task, without asking Jarvis (⌘↩)")
    }

    /// Hold to talk: the words land in the field, to ask or add.
    private var micButton: some View {
        micIcon
            .font(actionIconFont)
            .frame(width: actionIconFrame, height: actionIconFrame)
            .contentShape(Circle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !holdingMic else { return }
                        holdingMic = true
                        voice.start()
                    }
                    .onEnded { _ in
                        holdingMic = false
                        if voice.voiceState == .recording { voice.finishCapture() }
                    }
            )
            .disabled(!canUseVoice)
            .help(micHelp)
    }

    @ViewBuilder private var micIcon: some View {
        switch voice.voiceState {
        case .idle:
            Image(systemName: "mic.fill")
                .foregroundStyle(
                    canUseVoice ? AnyShapeStyle(.secondary) : AnyShapeStyle(.quaternary))
        case .recording:
            Image(systemName: "stop.fill")
                .foregroundStyle(.red)
                .symbolEffect(.pulse, options: .repeating, isActive: !reduceMotion)
        case .transcribing:
            Image(systemName: "waveform")
                .foregroundStyle(.tint)
                .symbolEffect(
                    .variableColor.iterative, options: .repeating, isActive: !reduceMotion)
        case .error:
            Image(systemName: "mic.slash.fill")
                .foregroundStyle(.red)
        }
    }

    /// Send, or stop the owner's own turn while it runs. A moment Jarvis is
    /// thinking through is not the owner's to stop; it holds the send.
    @ViewBuilder private var sendButton: some View {
        if thread.chat.isGenerating, thread.momentRunning == nil {
            Button {
                thread.chat.cancelGeneration()
            } label: {
                Image(systemName: "stop.circle.fill")
                    .font(.system(size: 22))
                    .foregroundStyle(.red)
                    .frame(width: actionIconFrame, height: actionIconFrame)
            }
            .buttonStyle(.plain)
            .focusable(false)
            .help("Stop")
        } else {
            Button(action: ask) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 22))
                    .foregroundStyle(canAsk ? AnyShapeStyle(.tint) : AnyShapeStyle(.tertiary))
                    .frame(width: actionIconFrame, height: actionIconFrame)
            }
            .buttonStyle(.plain)
            .focusable(false)
            .disabled(!canAsk)
            .help(
                thread.momentRunning == nil
                    ? "Ask Jarvis" : "Jarvis is thinking; ask when he's done")
        }
    }

    // MARK: - State

    private var trimmed: String { text.trimmingCharacters(in: .whitespacesAndNewlines) }

    private var canAsk: Bool { !trimmed.isEmpty && thread.canSend }

    private var isSpeechModelThere: Bool {
        // Any downloaded speech model counts: selection heals onto one.
        transcriptionEngine.isModelLoaded
            || !downloads.downloadedModels(in: .speechToText).isEmpty
    }

    private var canUseVoice: Bool { voice.voiceState != .transcribing && isSpeechModelThere }

    private var micHelp: String {
        if !isSpeechModelThere { return "Download a speech model to talk to Jarvis" }
        switch voice.voiceState {
        case .recording: return "Release to stop"
        case .transcribing: return "Transcribing…"
        case .error(let message): return message
        case .idle: return "Hold to talk"
        }
    }

    // MARK: - Actions

    private func ask() {
        let question = trimmed
        guard !question.isEmpty, thread.canSend else { return }
        thread.send(question)
        text = ""
        onAsk()
    }

    private func addTask() {
        let words = trimmed
        guard !words.isEmpty, !adding else { return }
        adding = true
        Task {
            let outcome = await capture.capture(words, source: "today")
            adding = false
            // Only saved words leave the field; a failure keeps them to retry.
            if case .added = outcome, trimmed == words { text = "" }
        }
    }
}

// MARK: - The notice slot

/// The composer's one banner, most pressing first: a take that failed, a
/// capture that didn't save, a turn that failed, then the latest change to
/// the agenda (for a few seconds, with its undo). Never a stack of banners.
private struct ComposerNotice: View {
    @Environment(Agenda.self) private var agenda
    @Environment(CaptureService.self) private var capture
    @Environment(DayThread.self) private var thread
    @Environment(AgentVoiceInputController.self) private var voice

    /// How long a confirmation stays up.
    private static let shownFor: Duration = .seconds(8)

    var body: some View {
        if case .error(let message) = voice.voiceState {
            banner(icon: "mic.slash", tint: .orange, message: message) { voice.cancel() }
        } else if let outcome = capture.lastOutcome, !outcome.isAdded {
            banner(icon: "exclamationmark.circle", message: outcome.line) { capture.clear() }
                .task(id: outcome.line) {
                    try? await Task.sleep(for: Self.shownFor)
                    capture.clear()
                }
        } else if let error = thread.chat.error {
            banner(icon: "exclamationmark.triangle", tint: .red, message: error) {
                thread.chat.error = nil
            }
        } else if let change = agenda.lastChange, Date().timeIntervalSince(change.at) < 15 {
            banner(
                icon: "checkmark.circle", tint: .accentColor, message: change.line,
                actionTitle: "Undo", action: { Task { await capture.undo(change) } }
            ) { agenda.clearLastChange() }
            .task(id: change.id) {
                try? await Task.sleep(for: Self.shownFor)
                if agenda.lastChange?.id == change.id { agenda.clearLastChange() }
            }
        }
    }

    /// The agent composer's banner: an icon, the message, an optional
    /// action, a dismiss.
    private func banner(
        icon: String, tint: Color? = nil, message: String, actionTitle: String? = nil,
        action: @escaping () -> Void = {}, onDismiss: @escaping () -> Void
    ) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(tint.map(AnyShapeStyle.init) ?? AnyShapeStyle(.secondary))
            Text(message)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            Spacer(minLength: 8)
            if let actionTitle {
                Button(actionTitle, action: action)
                    .buttonStyle(.borderless)
                    .fontWeight(.medium)
                    .focusable(false)
            }
            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
            .focusable(false)
            .help("Dismiss")
        }
        .font(.system(size: 12))
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 2)
    }
}

private extension CaptureOutcome {
    var isAdded: Bool {
        if case .added = self { true } else { false }
    }
}
