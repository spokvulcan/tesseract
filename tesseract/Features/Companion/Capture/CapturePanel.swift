//
//  CapturePanel.swift
//  tesseract
//
//  The capture hotkey's panel: a glass bar near the top of the screen, over
//  any app. Tap the hotkey and type; hold it and speak. Either way the words
//  go through the capture door into Reminders, and the panel answers with
//  one line and an undo, then fades.
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
    private var pressedAt: Date?
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
    }

    // MARK: Hotkey

    func hotkeyDown() {
        pressedAt = Date()
        show()
        model.mode = .listening
        voice.start()
    }

    func hotkeyUp() {
        let held = Date().timeIntervalSince(pressedAt ?? Date())
        pressedAt = nil
        if held < Self.holdThreshold {
            voice.cancel()
            model.mode = .typing
            model.focusRequest += 1
            panel?.makeKey()
        } else {
            model.mode = .transcribing
            voice.finishCapture()
        }
    }

    // MARK: Panel

    private func show() {
        dismissTask?.cancel()
        model.outcome = nil
        model.text = ""
        let panel = self.panel ?? makePanel()
        self.panel = panel
        panel.setHeight(Self.barHeight)
        panel.placeTopCenter()
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

    func hide() {
        dismissTask?.cancel()
        voice.cancel()
        panel?.orderOut(nil)
        model.text = ""
        model.outcome = nil
        model.mode = .typing
    }

    private func voiceFinished(_ text: String) {
        model.text = text
        model.mode = .typing
        submit()
    }

    private func submit() {
        let text = model.text
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        Task {
            let outcome = await capture.capture(text, source: "hotkey")
            model.outcome = outcome
            if case .added = outcome { model.text = "" }
            panel?.setHeight(Self.withLineHeight)
            scheduleDismiss(after: .seconds(5))
        }
    }

    private func undo() {
        Task {
            await capture.undoLast()
            model.outcome = .failed("Undone.")
            scheduleDismiss(after: .seconds(2))
        }
    }

    private func scheduleDismiss(after delay: Duration) {
        dismissTask?.cancel()
        dismissTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self?.hide()
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
