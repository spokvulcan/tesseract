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
    /// A step just marked done on its cue: said for a moment, then the
    /// panel closes.
    var done: StepCue?
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
    private let agenda: Agenda
    private let liveCard: (String) -> DayCard?
    private let onAction: (CardAction) -> Void
    private let onExpand: () -> Void
    private let onCapture: (String) -> Void

    /// The panel's width, and the tallest it grows.
    static let size = NSSize(width: 400, height: 560)
    /// The header and the field around what the panel says.
    static let chromeHeight: CGFloat = 124
    static let minimumHeight: CGFloat = 220

    /// - Parameter liveCard: the card as the day has it now, so items the
    ///   owner handles leave the panel and a refined card updates in it.
    init(
        thread: DayThread, voice: AgentVoiceInputController, agenda: Agenda,
        liveCard: @escaping (String) -> DayCard?,
        onAction: @escaping (CardAction) -> Void, onExpand: @escaping () -> Void,
        onCapture: @escaping (String) -> Void
    ) {
        self.thread = thread
        self.voice = voice
        self.agenda = agenda
        self.liveCard = liveCard
        self.onAction = onAction
        self.onExpand = onExpand
        self.onCapture = onCapture
    }

    var isShowing: Bool { panel?.isVisible == true }
    var shownCardID: String? { isShowing ? model.card?.id : nil }

    /// How long a step marked done is said before the panel closes.
    static let doneLinger: Duration = .milliseconds(2500)

    /// Show (or update in place) a card.
    func show(_ card: DayCard) {
        model.cue = nil
        model.done = nil
        model.card = card
        model.showQuiet = false
        present()
    }

    /// Show a planned step at its start or end, in place of whatever is up.
    func show(_ cue: StepCue) {
        model.card = nil
        model.done = nil
        model.cue = cue
        present()
    }

    /// Done on a cue is said for a moment — the win, and what comes next —
    /// then the panel closes, unless something else took it meanwhile.
    private func acknowledge(_ cue: StepCue) {
        model.cue = nil
        model.done = cue
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: Self.doneLinger)
            guard let self, self.model.done == cue else { return }
            self.model.done = nil
            self.close()
        }
    }

    /// As tall as what the panel says, between its least and full height.
    static func height(forContent content: CGFloat) -> CGFloat {
        min(max(content + chromeHeight, minimumHeight), size.height)
    }

    private func present() {
        let panel = self.panel ?? makePanel()
        self.panel = panel
        guard !panel.isVisible else { return }
        model.asked = nil
        panel.placeTopRight()
        panel.alphaValue = 0
        panel.orderFrontRegardless()
        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.2
            panel.animator().alphaValue = 1
        }
    }

    /// The content was laid out: the panel follows its height, keeping its
    /// top edge where it is.
    private func fit(content: CGFloat) {
        guard let panel else { return }
        let height = Self.height(forContent: content)
        if abs(panel.frame.height - height) > 0.5 { panel.setHeight(height) }
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

    /// A cue the owner leaves without choosing (Escape, Open Today) is
    /// answered as closed, so the engine knows it is off the panel.
    private func leaveCue() {
        if let cue = model.cue { onAction(.step(reminderID: cue.reminderID, .dismiss)) }
    }

    private func makePanel() -> GlassPanel {
        let panel = GlassPanel(size: Self.size, cornerRadius: 28, becomesKeyOnlyIfNeeded: true)
        panel.onCancel = { [weak self] in
            self?.leaveCue()
            self?.close()
        }
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
                model: model, thread: thread, agenda: agenda, liveCard: liveCard,
                close: { [weak self] in
                    guard let self else { return }
                    if let card = self.model.card { self.onAction(.dismiss(cardID: card.id)) }
                    if let cue = self.model.cue {
                        self.onAction(.step(reminderID: cue.reminderID, .dismiss))
                    }
                    self.close()
                },
                expand: { [weak self] in
                    self?.leaveCue()
                    self?.close()
                    self?.onExpand()
                },
                act: { [weak self] action in self?.onAction(action) },
                keep: { [weak self] in
                    guard let self, let card = self.model.card else { return }
                    self.onAction(.keep(cardID: card.id))
                    self.close()
                },
                choose: { [weak self] choice in
                    guard let self, let cue = self.model.cue else { return }
                    self.onAction(.step(reminderID: cue.reminderID, choice))
                    if choice == .done { self.acknowledge(cue) } else { self.close() }
                },
                send: { [weak self] in self?.send() },
                capture: { [weak self] in self?.capture() },
                mic: { [weak self] in self?.toggleMic() },
                onContentHeight: { [weak self] height in
                    // On the next turn: never resize the window inside its own
                    // layout pass.
                    Task { @MainActor [weak self] in self?.fit(content: height) }
                }))
        return panel
    }

    private func send() {
        let text = model.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, thread.canSend else { return }
        thread.send(text)
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
    let agenda: Agenda
    let liveCard: (String) -> DayCard?
    let close: () -> Void
    let expand: () -> Void
    let act: (CardAction) -> Void
    /// Take the card in: off the panel, still in Today.
    let keep: () -> Void
    let choose: (StepChoice) -> Void
    let send: () -> Void
    let capture: () -> Void
    let mic: () -> Void
    /// What the panel says was laid out at this height.
    var onContentHeight: (CGFloat) -> Void = { _ in }
    /// The clock the plan's steps are read against (fixed in the gallery).
    var now: () -> Date = Date.init

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
                    } else if let done = model.done {
                        StepDoneContent(cue: done)
                    } else if let shown = model.card {
                        CardContent(
                            card: liveCard(shown.id) ?? shown, agenda: agenda, now: now(),
                            showQuiet: $model.showQuiet, act: act, keep: keep, expand: expand)
                    }
                    if let asked = model.asked {
                        Exchange(asked: asked, thread: thread)
                    }
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .onGeometryChange(for: CGFloat.self) {
                    $0.size.height
                } action: {
                    onContentHeight($0)
                }
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

/// A planned step at its start — what it is, until when and what follows,
/// and the four ways on: start, a quarter of an hour on, tomorrow, done — or
/// at the end of the slot the owner started: done, longer, or tomorrow.
private struct StepCueContent: View {
    let cue: StepCue
    let choose: (StepChoice) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text(heading)
                        .fontWeight(.semibold)
                        .foregroundStyle(Color.accentColor)
                    Spacer()
                    Text("\(AgendaTime.clock(cue.start))–\(AgendaTime.clock(cue.end))")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Text(cue.title)
                    .fontWeight(.semibold)
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
                switch cue.phase {
                case .start:
                    choice("Start", .start, prominent: true)
                    choice("In \(DayEngine.stepLaterMinutes) min", .later)
                        .help("Move it a quarter of an hour on; Jarvis asks again then")
                    choice("Tomorrow", .tomorrow)
                    Spacer(minLength: 0)
                    choice("Done", .done)
                case .end:
                    choice("Done", .done, prominent: true)
                    choice("\(DayEngine.stepLaterMinutes) more min", .extend)
                        .help("Keep going a quarter of an hour; Jarvis asks again then")
                    choice("Tomorrow", .tomorrow)
                    Spacer(minLength: 0)
                }
            }
        }
    }

    private func choice(_ title: String, _ choice: StepChoice, prominent: Bool = false)
        -> some View
    {
        Button(title) { choose(choice) }
            .buttonStyle(PanelButtonStyle(prominent: prominent))
            .focusable(false)
    }

    /// A late cue (the owner was away or busy) says so, without blame.
    private var heading: String {
        switch (cue.phase, cue.late) {
        case (.start, false): "Time for"
        case (.start, true): "Still time for"
        case (.end, false): "Time's up"
        case (.end, true): "How did it go?"
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

/// The win, said: the step is done, the must-do credited, and what comes
/// next — a moment's word before the panel closes.
private struct StepDoneContent: View {
    let cue: StepCue

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("Done", systemImage: "checkmark.circle.fill")
                .fontWeight(.semibold)
                .foregroundStyle(Color.accentColor)
            Text(cue.title)
                .fontWeight(.semibold)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            if let line = cue.doneLine {
                Text(line)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

extension StepCue {
    /// What the panel says once the step is done: the must-do credited (the
    /// Now Card's word too), then, on its own line, what comes next.
    var doneLine: String? {
        let next = next.map { "Next: \($0)." }
        guard isMustDo else { return next }
        return ["That's the must-do — the rest is a bonus.", next].compactMap(\.self)
            .joined(separator: "\n")
    }
}

/// The panel's word buttons: capsules on the glass, the main one in the
/// accent. Drawn by hand so the main one keeps its color in a panel that is
/// never key (a system prominent button turns gray there).
private struct PanelButtonStyle: ButtonStyle {
    var prominent = false
    /// A row's own choices: smaller than the card's main buttons.
    var compact = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .fontWeight(prominent ? .semibold : .regular)
            .foregroundStyle(prominent ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
            .padding(.horizontal, compact ? 10 : 14)
            .frame(height: compact ? 24 : 30)
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
    let agenda: Agenda
    let now: Date
    @Binding var showQuiet: Bool
    let act: (CardAction) -> Void
    let keep: () -> Void
    let expand: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            switch card.body {
            case .breakpoint(let breakpoint):
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(card.kind.title).fontWeight(.semibold)
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
            case .morningPlan(let plan):
                MorningPlanContent(
                    card: card, plan: plan, agenda: agenda, now: now, keep: keep, expand: expand)
            case .eveningWrapUp(let wrapUp):
                WrapUpContent(card: card, wrapUp: wrapUp, act: act, keep: keep, expand: expand)
            case .reflection:
                Text(card.kind.title).fontWeight(.semibold)
                Text(card.line).fixedSize(horizontal: false, vertical: true)
                Text("Open Today to see it all.").foregroundStyle(.secondary)
            }
        }
    }

    private func awayText(_ card: BreakpointCard) -> String {
        let minutes = Int(card.awayUntil.timeIntervalSince(card.awayFrom) / 60)
        return
            "\(MomentPrompts.minutesText(minutes)) · \(AgendaTime.clock(card.awayFrom))–\(AgendaTime.clock(card.awayUntil))"
    }
}

/// The plan, whole on the panel: Jarvis's line, the steps ahead (the tasks
/// he placed among the day's events, the must-do starred), his tips, and a
/// yes — no trip to Today to see it.
private struct MorningPlanContent: View {
    /// Between the rows of a list on the panel (its sections are 12 apart).
    static let rowSpacing: CGFloat = 6

    let card: DayCard
    let plan: MorningPlanCard
    let agenda: Agenda
    let now: Date
    let keep: () -> Void
    let expand: () -> Void

    var body: some View {
        let steps = PlanStep.ahead(plan: plan, agenda: agenda, now: now)
        VStack(alignment: .leading, spacing: 12) {
            Text(card.kind.title).fontWeight(.semibold)
            Text(card.line).fixedSize(horizontal: false, vertical: true)
            if card.isRefining {
                Text("Jarvis is still thinking it through; this card updates when he's done.")
                    .foregroundStyle(.secondary)
            }
            if !steps.isEmpty {
                VStack(alignment: .leading, spacing: Self.rowSpacing) {
                    ForEach(steps) { step in
                        PlanStepRow(step: step)
                    }
                }
            }
            if !plan.suggestions.isEmpty {
                VStack(alignment: .leading, spacing: Self.rowSpacing) {
                    ForEach(plan.suggestions, id: \.self) { tip in
                        Text("· \(tip)")
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            HStack(spacing: 8) {
                Button("Looks Good", action: keep)
                    .buttonStyle(PanelButtonStyle(prominent: true))
                    .focusable(false)
                Button("Open Today", action: expand)
                    .buttonStyle(PanelButtonStyle())
                    .focusable(false)
            }
        }
    }
}

/// One step of the plan on the panel: a placed task or an event.
struct PlanStep: Identifiable, Equatable {
    enum Kind: Equatable {
        case task(isMustDo: Bool)
        case event(colorHex: String?)
    }

    let id: String
    var start: Date
    var title: String
    var minutes: Int
    var kind: Kind

    /// The most the panel lists; Today has the rest.
    static let shown = 5

    /// The day's events and the plan's tasks still ahead, in time order —
    /// read through the same Timeline as Today.
    @MainActor
    static func ahead(plan: MorningPlanCard, agenda: Agenda, now: Date) -> [PlanStep] {
        let facts = DayFacts(
            snapshot: agenda.snapshot, areas: agenda.areas, inboxListID: agenda.inbox?.id,
            now: now, mustDoID: plan.mustDoID, plan: plan.placements)
        let steps = TimelineBuilder.build(facts: facts).rows.compactMap { row -> PlanStep? in
            guard !row.isPast else { return nil }
            switch row.kind {
            case .event(let event):
                return PlanStep(
                    id: row.id, start: event.start, title: event.title,
                    minutes: Int(event.duration / 60), kind: .event(colorHex: event.colorHex))
            case .task(let task) where !task.isDone:
                return PlanStep(
                    id: row.id, start: row.start, title: task.reminder.title,
                    minutes: task.minutes, kind: .task(isMustDo: task.isMustDo))
            default:
                return nil
            }
        }
        return Array(steps.prefix(shown))
    }
}

private struct PlanStepRow: View {
    let step: PlanStep

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(AgendaTime.clock(step.start))
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 42, alignment: .leading)
            marker
                .frame(width: 14)
            Text(step.title).lineLimit(1)
            Spacer(minLength: 6)
            Text(MomentPrompts.minutesText(step.minutes))
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    @ViewBuilder private var marker: some View {
        switch step.kind {
        case .task(let isMustDo):
            Image(systemName: isMustDo ? "star.fill" : "circle")
                .imageScale(.small)
                .foregroundStyle(
                    isMustDo ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.secondary))
        case .event(let colorHex):
            RoundedRectangle(cornerRadius: 1.5)
                .fill(Color(hexString: colorHex))
                .frame(width: 3, height: 12)
        }
    }
}

/// The evening, whole on the panel: Jarvis's line, what got done, each
/// leftover with its three ways on (Jarvis's pick in the accent) or all of
/// them his way at once, and how tomorrow starts.
private struct WrapUpContent: View {
    let card: DayCard
    let wrapUp: EveningWrapUpCard
    let act: (CardAction) -> Void
    let keep: () -> Void
    let expand: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(card.kind.title).fontWeight(.semibold)
            Text(card.line).fixedSize(horizontal: false, vertical: true)
            if !wrapUp.done.isEmpty {
                VStack(alignment: .leading, spacing: MorningPlanContent.rowSpacing) {
                    Text("Done today · \(wrapUp.done.count)").fontWeight(.semibold)
                    ForEach(Array(wrapUp.done.prefix(3).enumerated()), id: \.offset) { _, title in
                        Label {
                            Text(title).lineLimit(1)
                        } icon: {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(Color.accentColor)
                        }
                    }
                    if wrapUp.done.count > 3 {
                        Text("and \(wrapUp.done.count - 3) more").foregroundStyle(.secondary)
                    }
                }
            }
            if !wrapUp.leftovers.isEmpty {
                VStack(alignment: .leading, spacing: MorningPlanContent.rowSpacing * 2) {
                    Text("Left from today").fontWeight(.semibold)
                    ForEach(wrapUp.leftovers) { leftover in
                        VStack(alignment: .leading, spacing: MorningPlanContent.rowSpacing) {
                            Text(leftover.title).lineLimit(2)
                            HStack(spacing: 6) {
                                choice("Tomorrow", .tomorrow, for: leftover)
                                choice("Later", .later, for: leftover)
                                choice("Let go", .drop, for: leftover)
                            }
                        }
                    }
                }
            }
            if wrapUp.week != nil || wrapUp.focus != nil {
                VStack(alignment: .leading, spacing: MorningPlanContent.rowSpacing) {
                    Text("This week").fontWeight(.semibold)
                    if let week = wrapUp.week {
                        Text(week).fixedSize(horizontal: false, vertical: true)
                    }
                    if let focus = wrapUp.focus {
                        Text("Next week: \(focus)").foregroundStyle(Color.accentColor)
                    }
                }
            }
            if let first = wrapUp.tomorrowFirst {
                Text("Tomorrow starts with \(first).").foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                if wrapUp.leftovers.count > 1 {
                    Button("Do What Jarvis Suggests") { act(.allLeftovers(cardID: card.id)) }
                        .buttonStyle(PanelButtonStyle(prominent: true))
                        .focusable(false)
                } else {
                    Button("Good Night", action: keep)
                        .buttonStyle(PanelButtonStyle(prominent: wrapUp.leftovers.isEmpty))
                        .focusable(false)
                }
                Button("Open Today", action: expand)
                    .buttonStyle(PanelButtonStyle())
                    .focusable(false)
            }
        }
    }

    private func choice(
        _ title: String, _ suggestion: Leftover.Suggestion, for leftover: Leftover
    ) -> some View {
        Button(title) {
            act(.leftover(cardID: card.id, reminderID: leftover.reminderID, suggestion))
        }
        .buttonStyle(PanelButtonStyle(prominent: leftover.suggestion == suggestion, compact: true))
        .focusable(false)
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
