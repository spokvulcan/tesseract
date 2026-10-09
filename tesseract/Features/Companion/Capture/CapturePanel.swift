//
//  CapturePanel.swift
//  tesseract
//
//  The capture hotkey's panel: a glass bar near the top of the screen, over
//  any app. Tap the hotkey and type; hold it and speak. Either way the words
//  go through the capture door into Reminders, and the panel answers with
//  one line and an undo, then fades. Words that didn't save stay in the
//  field with the reason; only the owner's Escape or × throws them away.
//
//  Nothing shows until the press says what it is: a release before the hold
//  threshold is a tap, a hold past it starts listening. That lets the hotkey
//  be one modifier key — another key pressed with it cancels, and the owner
//  types ⌥-something as usual.
//

import AppKit
import Observation
import SwiftUI

@Observable @MainActor
final class CapturePanelModel {
    enum Mode: Equatable {
        case typing
        case listening
        case transcribing
    }

    var text = ""
    var mode: Mode = .typing
    /// The confirmation line, once a capture ran.
    var outcome: CaptureOutcome?
    /// Bumped to ask the view to focus its field.
    var focusRequest = 0
}

@MainActor
final class CapturePanelController {

    private let capture: CaptureService
    private let voice: AgentVoiceInputController
    private let model = CapturePanelModel()
    private var panel: GlassPanel?
    /// Waiting out the hold threshold after a press.
    private var holdTask: Task<Void, Never>?
    /// The press became a hold: the microphone is on, or was asked to be.
    private var isHolding = false
    /// The panel was up when the hold began: the owner was typing.
    private var wasUpBeforeHold = false
    /// Words typed before a hold: the spoken ones join them.
    private var typedBeforeHold = ""
    /// A save or an undo is under way: Return or Undo again does it once.
    private var isBusy = false
    private var dismissTask: Task<Void, Never>?

    private static let width: CGFloat = 560
    private static let barHeight: CGFloat = 60
    private static let withLineHeight: CGFloat = 96
    /// A press shorter than this is a tap (type); longer is a hold (speak).
    private static let holdThreshold: TimeInterval = 0.35

    init(capture: CaptureService, voice: AgentVoiceInputController) {
        self.capture = capture
        self.voice = voice
        voice.onVoiceTranscription = { [weak self] text in
            self?.voiceFinished(text)
        }
        voice.onVoiceFailure = { [weak self] message in
            self?.voiceFailed(message)
        }
        voice.onVoiceRejected = { [weak self] raw in
            self?.voiceRejected(raw)
        }
    }

    // MARK: Hotkey

    func hotkeyDown() {
        holdTask?.cancel()
        holdTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(Self.holdThreshold))
            guard !Task.isCancelled else { return }
            self?.beginHolding()
        }
    }

    func hotkeyUp() {
        holdTask?.cancel()
        holdTask = nil
        if isHolding {
            isHolding = false
            // The microphone never started (busy, no permission), or the
            // last take is still being written down: the line says so.
            guard voice.voiceState == .recording else { return }
            model.mode = .transcribing
            voice.finishCapture()
            return
        }
        // A tap: type.
        openForTyping()
    }

    /// The bar, ready to type into: a tap of the hotkey, or the menu bar's
    /// "Write a Thought Down…".
    func openForTyping() {
        show()
        model.mode = .typing
        model.focusRequest += 1
        panel?.makeKey()
    }

    /// Another key joined the one-key hotkey: the owner is typing with it.
    func hotkeyCancelled() {
        holdTask?.cancel()
        holdTask = nil
        guard isHolding else { return }
        isHolding = false
        // Only the listening this hold began: a take still being written
        // down goes on.
        guard model.mode == .listening else { return }
        voice.cancel()
        typedBeforeHold = ""
        guard wasUpBeforeHold else {
            hide()
            return
        }
        // ⌥ held for an accent while typing here: back to the words.
        model.mode = .typing
        model.focusRequest += 1
        panel?.makeKey()
    }

    /// Held past the threshold: listen.
    private func beginHolding() {
        holdTask = nil
        isHolding = true
        // One take at a time: say so, rather than listen to nothing.
        if voice.voiceState == .transcribing {
            show()
            model.outcome = .failed(
                "Still writing the last one down. Hold the key again in a moment.")
            panel?.setHeight(Self.withLineHeight)
            return
        }
        wasUpBeforeHold = panel?.isVisible == true
        typedBeforeHold =
            wasUpBeforeHold ? model.text.trimmingCharacters(in: .whitespacesAndNewlines) : ""
        show()
        model.mode = .listening
        voice.start()
    }

    // MARK: Panel

    /// Up from hidden, the bar starts empty; already up (a second tap, the
    /// menu), the words stay.
    private func show() {
        dismissTask?.cancel()
        let panel = self.panel ?? makePanel()
        self.panel = panel
        if !panel.isVisible {
            model.outcome = nil
            model.text = ""
            panel.setHeight(Self.barHeight)
            panel.placeTopCenter()
        } else if case .added = model.outcome {
            // The last confirmation has done its job.
            model.outcome = nil
            panel.setHeight(Self.barHeight)
        }
        panel.orderFrontRegardless()
    }

    private func makePanel() -> GlassPanel {
        let panel = GlassPanel(
            size: NSSize(width: Self.width, height: Self.barHeight), cornerRadius: 24,
            becomesKeyOnlyIfNeeded: false)
        panel.onCancel = { [weak self] in self?.hide() }
        panel.host(
            CapturePanelView(
                model: model,
                submit: { [weak self] in self?.submit() },
                undo: { [weak self] in self?.undo() },
                close: { [weak self] in self?.hide() }))
        return panel
    }

    /// Closed by the owner (Escape, the ×) or after a saved capture: what is
    /// in the field goes with it.
    func hide() {
        dismissTask?.cancel()
        holdTask?.cancel()
        holdTask = nil
        isHolding = false
        typedBeforeHold = ""
        voice.cancel()
        panel?.orderOut(nil)
        model.text = ""
        model.outcome = nil
        model.mode = .typing
    }

    private func voiceFinished(_ text: String) {
        model.mode = .typing
        guard !typedBeforeHold.isEmpty else {
            model.text = text
            submit()
            return
        }
        // Words were waiting in the field: the spoken ones join them, for
        // the owner to look over and save.
        model.text = typedBeforeHold + " " + text
        typedBeforeHold = ""
        model.focusRequest += 1
        panel?.makeKey()
    }

    /// The take ended without words (too short, no speech, a failed
    /// transcription): back to typing, with the reason, instead of waiting.
    private func voiceFailed(_ message: String) {
        guard panel?.isVisible == true, model.mode != .typing else { return }
        typedBeforeHold = ""
        model.mode = .typing
        model.outcome = .failed("\(message). Type it instead, or hold the key and speak again.")
        panel?.setHeight(Self.withLineHeight)
        model.focusRequest += 1
        panel?.makeKey()
    }

    /// A take the proofreader couldn't make sense of: in the field to look
    /// over, never saved as heard.
    private func voiceRejected(_ raw: String) {
        model.mode = .typing
        model.text = typedBeforeHold.isEmpty ? raw : typedBeforeHold + " " + raw
        typedBeforeHold = ""
        model.outcome = .failed("Not sure I heard that right. Fix it if needed, then press Return.")
        panel?.setHeight(Self.withLineHeight)
        model.focusRequest += 1
        panel?.makeKey()
    }

    private func submit() {
        let text = model.text
        guard !isBusy, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        isBusy = true
        typedBeforeHold = ""
        Task {
            let outcome = await capture.capture(text, source: "hotkey")
            isBusy = false
            model.outcome = outcome
            panel?.setHeight(Self.withLineHeight)
            guard case .added = outcome else {
                // Not saved: the words stay, with the reason, until the
                // owner fixes them or closes the panel.
                model.focusRequest += 1
                panel?.makeKey()
                return
            }
            // Only the words that were saved leave the field.
            if model.text == text { model.text = "" }
            scheduleDismiss(after: .seconds(5))
        }
    }

    /// Undo the capture on the line — and say "Undone." only when it was.
    private func undo() {
        guard !isBusy, case .added(let change) = model.outcome else { return }
        isBusy = true
        Task {
            let undone = await capture.undo(change)
            isBusy = false
            model.outcome =
                undone ? .failed("Undone.") : capture.lastOutcome ?? .failed("Couldn't undo.")
            scheduleDismiss(after: .seconds(undone ? 2 : 5))
        }
    }

    private func scheduleDismiss(after delay: Duration) {
        dismissTask?.cancel()
        dismissTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled, let self else { return }
            // Words typed meanwhile, the next thought: they and the bar stay.
            let typing = !self.model.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            guard !typing, self.model.mode == .typing else { return }
            self.hide()
        }
    }
}

// MARK: - View

private struct CapturePanelView: View {
    @Bindable var model: CapturePanelModel
    let submit: () -> Void
    let undo: () -> Void
    let close: () -> Void
    @FocusState private var fieldFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Image(systemName: symbol)
                    .foregroundStyle(.secondary)
                    .symbolEffect(.pulse, options: .repeating, isActive: model.mode == .listening)
                // Present from the first layout: a field that appears later
                // freezes the main thread on macOS 27.0.
                TextField(placeholder, text: $model.text)
                    .textFieldStyle(.plain)
                    .focused($fieldFocused)
                    .onSubmit(submit)
                    .disabled(model.mode != .typing)
                Button(action: close) {
                    Image(systemName: "xmark")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .focusable(false)
                .help("Close (Esc)")
            }
            .font(.title3)
            if let outcome = model.outcome {
                HStack(spacing: 8) {
                    Text(outcome.line)
                        .foregroundStyle(
                            isFailure(outcome) ? AnyShapeStyle(.secondary) : AnyShapeStyle(.primary)
                        )
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 0)
                    if case .added = outcome {
                        Button("Undo", action: undo)
                            .buttonStyle(.plain)
                            .foregroundStyle(.tint)
                            .focusable(false)
                    }
                }
                .font(.callout)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onChange(of: model.focusRequest) { fieldFocused = true }
    }

    private var symbol: String {
        switch model.mode {
        case .typing: "plus.circle"
        case .listening: "waveform"
        case .transcribing: "ellipsis"
        }
    }

    private var placeholder: String {
        switch model.mode {
        case .typing: "Remind me to… (try “call the dentist tomorrow at 10”)"
        case .listening: "Listening…"
        case .transcribing: "Writing it down…"
        }
    }

    private func isFailure(_ outcome: CaptureOutcome) -> Bool {
        if case .added = outcome { return false }
        return true
    }
}
