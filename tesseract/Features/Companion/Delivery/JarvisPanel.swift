//
//  JarvisPanel.swift
//  tesseract
//
//  The Jarvis panel: the Breakpoint card (and urgent items) in a Liquid
//  Glass panel near the top-right, in the style of the macOS 27 Siri panel —
//  a close button top-left, an expand button top-right that opens Today, the
//  card, and an "Ask Jarvis" field between + and mic buttons. A Step Cue (a
//  planned step starting now) takes a shorter panel. It never steals typing
//  from the app in front: it becomes key only when the field is clicked.
//  Replies go to the Day Thread.
//
//  Built with the prototype lab's constraints (tools/jarvis-panel-lab): a
//  borderless non-activating panel over an `NSGlassEffectView`; every button
//  non-focusable and the field present from the first layout, because a
//  focusable control appearing later freezes the main thread on macOS 27.0.
//

import AppKit
import Observation
import SwiftUI

@Observable @MainActor
final class JarvisPanelModel {
    var card: DayCard?
    /// A planned step starting now; shown instead of a card.
    var cue: StepCue?
    var showQuiet = false
    var draft = ""
    var listening = false
    /// The last exchange typed here, so the answer shows in the panel.
    var asked: String?
}

@MainActor
final class JarvisPanelController {

    private let model = JarvisPanelModel()
    private var panel: GlassPanel?
    private let thread: DayThread
    private let voice: AgentVoiceInputController
    private let onAction: (CardAction) -> Void
    private let onExpand: () -> Void
    private let onCapture: (String) -> Void

    static let size = NSSize(width: 400, height: 560)
    /// A Step Cue's panel: the step, its choices and the field.
    static let cueHeight: CGFloat = 252

    init(
        thread: DayThread, voice: AgentVoiceInputController,
        onAction: @escaping (CardAction) -> Void, onExpand: @escaping () -> Void,
        onCapture: @escaping (String) -> Void
    ) {
        self.thread = thread
        self.voice = voice
        self.onAction = onAction
        self.onExpand = onExpand
        self.onCapture = onCapture
    }

    var isShowing: Bool { panel?.isVisible == true }
    var shownCardID: String? { isShowing ? model.card?.id : nil }

    /// Show (or update in place) a card.
    func show(_ card: DayCard) {
        model.cue = nil
        model.card = card
        model.showQuiet = false
        present(height: Self.size.height)
    }

    /// Show a planned step that starts now, in place of whatever is up.
    func show(_ cue: StepCue) {
        model.card = nil
        model.cue = cue
        present(height: Self.cueHeight)
    }

    private func present(height: CGFloat) {
        let panel = self.panel ?? makePanel()
        self.panel = panel
        // A question asked in the panel keeps the room its answer needs.
        if model.asked == nil || !panel.isVisible { panel.setHeight(height) }
        if !panel.isVisible {
            model.asked = nil
            panel.placeTopRight()
            panel.alphaValue = 0
            panel.orderFrontRegardless()
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.2
                panel.animator().alphaValue = 1
            }
        }
    }

    /// Take a card down (dismissed elsewhere, or replaced).
    func retract(cardID: String) {
        guard model.card?.id == cardID else { return }
        close()
    }

    func close() {
        voice.cancel()
        model.listening = false
        panel?.orderOut(nil)
    }

    private func makePanel() -> GlassPanel {
        let panel = GlassPanel(size: Self.size, cornerRadius: 28, becomesKeyOnlyIfNeeded: true)
        panel.onCancel = { [weak self] in self?.close() }
        voice.onVoiceTranscription = { [weak self] text in
            guard let self else { return }
            self.model.listening = false
            self.model.draft = text
            self.send()
        }
        // A take with no words (too short, no speech) ends listening too.
        voice.onVoiceFailure = { [weak self] _ in self?.model.listening = false }
        panel.host(
            JarvisPanelView(
                model: model, thread: thread,
                close: { [weak self] in
                    guard let self else { return }
                    if let card = self.model.card { self.onAction(.dismiss(cardID: card.id)) }
                    if let cue = self.model.cue {
                        self.onAction(.step(reminderID: cue.reminderID, .dismiss))
                    }
                    self.close()
                },
                expand: { [weak self] in
                    self?.close()
                    self?.onExpand()
                },
                act: { [weak self] action in self?.onAction(action) },
                choose: { [weak self] choice in
                    guard let self, let cue = self.model.cue else { return }
                    self.onAction(.step(reminderID: cue.reminderID, choice))
                    self.close()
                },
                send: { [weak self] in self?.send() },
                capture: { [weak self] in self?.capture() },
                mic: { [weak self] in self?.toggleMic() }))
        return panel
    }

    private func send() {
        let text = model.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, thread.canSend else { return }
        thread.send(text)
        if model.asked == nil { panel?.setHeight(Self.size.height) }
        model.asked = text
        model.draft = ""
    }

    /// The + button: the field's words become a reminder, no model.
    private func capture() {
        let text = model.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        onCapture(text)
        model.draft = ""
    }

    private func toggleMic() {
        if model.listening {
            voice.finishCapture()
            model.listening = false
        } else {
            voice.start()
            model.listening = true
        }
    }
}

// MARK: - View

struct JarvisPanelView: View {
    @Bindable var model: JarvisPanelModel
    let thread: DayThread
    let close: () -> Void
    let expand: () -> Void
    let act: (CardAction) -> Void
    let choose: (StepChoice) -> Void
    let send: () -> Void
    let capture: () -> Void
    let mic: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                CircleButton(symbol: "xmark", help: "Dismiss", action: close)
                Spacer()
                Text("Jarvis").fontWeight(.semibold)
                Spacer()
                CircleButton(
                    symbol: "arrow.up.left.and.arrow.down.right", help: "Open Today", action: expand
                )
            }
            .padding(.horizontal, 14)
            .padding(.top, 14)
            .padding(.bottom, 6)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if let cue = model.cue {
                        StepCueContent(cue: cue, choose: choose)
                    } else if let card = model.card {
                        CardContent(card: card, showQuiet: $model.showQuiet, act: act)
                    }
                    if let asked = model.asked {
                        Exchange(asked: asked, thread: thread)
                    }
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(spacing: 10) {
                CircleButton(symbol: "plus", help: "Add as a reminder", action: capture)
                // Present from the first layout: a field that appears later
                // freezes the main thread on macOS 27.0.
                TextField("Ask Jarvis", text: $model.draft)
                    .textFieldStyle(.plain)
                    .onSubmit(send)
                CircleButton(
                    symbol: model.listening ? "waveform" : "mic", help: "Speak", action: mic)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(.quaternary.opacity(0.35), in: Capsule())
            .padding(14)
        }
        .font(.system(size: 13))
        .frame(width: JarvisPanelController.size.width)
        .frame(maxHeight: .infinity)
    }
}

/// A planned step starting now: what it is, until when and what follows,
/// and the four ways on — start, a quarter of an hour on, tomorrow, done.
private struct StepCueContent: View {
    let cue: StepCue
    let choose: (StepChoice) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Time for").fontWeight(.semibold).foregroundStyle(Color.accentColor)
                    Spacer()
                    Text("\(AgendaTime.clock(cue.start))–\(AgendaTime.clock(cue.end))")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Text(cue.title)
                    .font(.system(size: 15, weight: .semibold))
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Text(detail)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Every button non-focusable: they appear after the panel's
            // first layout (GlassPanel's macOS 27.0 focus freeze).
            HStack(spacing: 8) {
                Button("Start") { choose(.start) }
                    .buttonStyle(PanelButtonStyle(prominent: true))
                    .focusable(false)
                Button("In \(DayEngine.stepLaterMinutes) min") { choose(.later) }
                    .buttonStyle(PanelButtonStyle())
                    .focusable(false)
                    .help("Move it a quarter of an hour on; Jarvis asks again then")
                Button("Tomorrow") { choose(.tomorrow) }
                    .buttonStyle(PanelButtonStyle())
                    .focusable(false)
                Spacer(minLength: 0)
                Button("Done") { choose(.done) }
                    .buttonStyle(PanelButtonStyle())
                    .focusable(false)
            }
        }
    }

    private var detail: String {
        var parts = ["\(MomentPrompts.minutesText(cue.minutes))"]
        if cue.isMustDo { parts.append("your must-do") }
        parts.append(cue.areaName)
        var line = parts.joined(separator: " · ")
        if let next = cue.next { line += ". Then \(next)." }
        return line
    }
}

/// The panel's word buttons: capsules on the glass, the main one in the
/// accent. Drawn by hand so the main one keeps its color in a panel that is
/// never key (a system prominent button turns gray there).
private struct PanelButtonStyle: ButtonStyle {
    var prominent = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .fontWeight(prominent ? .semibold : .regular)
            .foregroundStyle(prominent ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
            .padding(.horizontal, 14)
            .frame(height: 30)
            .background(
                prominent ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.primary.opacity(0.1)),
                in: Capsule()
            )
            .opacity(configuration.isPressed ? 0.7 : 1)
            .contentShape(Capsule())
    }
}

private struct CircleButton: View {
    let symbol: String
    let help: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .frame(width: 28, height: 28)
                .background(.primary.opacity(0.1), in: Circle())
        }
        .buttonStyle(.plain)
        .focusable(false)
        .help(help)
    }
}

private struct CardContent: View {
    let card: DayCard
    @Binding var showQuiet: Bool
    let act: (CardAction) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            switch card.body {
            case .breakpoint(let breakpoint):
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("Welcome back").fontWeight(.semibold)
                    Text(awayText(breakpoint)).foregroundStyle(.secondary)
                }
                Text(breakpoint.line).fixedSize(horizontal: false, vertical: true)
                if !breakpoint.needsYou.isEmpty {
                    Group {
                        Text("Needs you").fontWeight(.semibold)
                        ForEach(breakpoint.needsYou) { item in
                            ItemRow(cardID: card.id, item: item, act: act)
                        }
                    }
                }
                if !breakpoint.next.isEmpty {
                    Text("Next").fontWeight(.semibold)
                    ForEach(breakpoint.next) { next in
                        HStack {
                            Text(next.title)
                            Spacer()
                            if let at = next.at {
                                Text(AgendaTime.clock(at)).foregroundStyle(.secondary)
                                    .monospacedDigit()
                            }
                        }
                    }
                }
                if let place = breakpoint.whereYouWere {
                    Text("You were in \(place).").foregroundStyle(.secondary)
                }
                if breakpoint.canWaitCount > 0 {
                    HStack {
                        Text(
                            "\(breakpoint.canWaitCount) other notification\(breakpoint.canWaitCount == 1 ? "" : "s") can wait"
                        )
                        .foregroundStyle(.secondary)
                        Spacer()
                        Button(showQuiet ? "Hide" : "Show") { showQuiet.toggle() }
                            .buttonStyle(.plain)
                            .foregroundStyle(Color.accentColor)
                            .focusable(false)
                    }
                    if showQuiet {
                        ForEach(breakpoint.canWait) { group in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(group.app).fontWeight(.medium)
                                ForEach(group.lines, id: \.self) { line in
                                    Text(line).foregroundStyle(.secondary).lineLimit(2)
                                }
                            }
                        }
                    }
                }
            case .triage(let triage):
                Text(triage.line).fixedSize(horizontal: false, vertical: true)
                ForEach(triage.raise) { item in
                    ItemRow(cardID: card.id, item: item, act: act)
                }
            case .morningPlan, .eveningWrapUp, .reflection:
                Text(card.kind.title).fontWeight(.semibold)
                Text(card.line).fixedSize(horizontal: false, vertical: true)
                if card.isRefining {
                    Text("Jarvis is still thinking it through; this card updates when he's done.")
                        .foregroundStyle(.secondary)
                }
                Text("Open Today to see it all.").foregroundStyle(.secondary)
            }
        }
    }

    private func awayText(_ card: BreakpointCard) -> String {
        let minutes = Int(card.awayUntil.timeIntervalSince(card.awayFrom) / 60)
        return
            "Away \(MomentPrompts.minutesText(minutes)) · \(AgendaTime.clock(card.awayFrom))–\(AgendaTime.clock(card.awayUntil))"
    }
}

private struct ItemRow: View {
    let cardID: String
    let item: WaitingItem
    let act: (CardAction) -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title).fontWeight(.medium).lineLimit(1)
                Text(item.detail).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer(minLength: 6)
            if item.app != nil {
                Button("Open") { act(.openItem(cardID: cardID, itemID: item.id)) }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.accentColor)
                    .focusable(false)
            }
            if item.kind != .agent {
                Button("Later") { act(.itemLater(cardID: cardID, itemID: item.id)) }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .focusable(false)
                    .help("Remind me in half an hour")
            }
            Button("Done") { act(.itemDone(cardID: cardID, itemID: item.id)) }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .focusable(false)
        }
    }
}

private struct Exchange: View {
    let asked: String
    let thread: DayThread

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(asked)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.accentColor.opacity(0.18), in: RoundedRectangle(cornerRadius: 10))
                .frame(maxWidth: .infinity, alignment: .trailing)
            if thread.chat.isGenerating {
                ProgressView().controlSize(.small)
            } else if let reply = lastReply {
                Text(reply).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var lastReply: String? {
        for item in thread.chat.items.reversed() {
            if case .assistant(let message) = item {
                let text = message.content.compactMap { part -> String? in
                    if case .text(let text) = part { return text.text }
                    return nil
                }.joined()
                if !text.isEmpty { return text }
            }
        }
        return nil
    }
}
