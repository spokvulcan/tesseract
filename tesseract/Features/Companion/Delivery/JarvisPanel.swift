//
//  JarvisPanel.swift
//  tesseract
//
//  The Jarvis panel: the Breakpoint card (and urgent items) in a Liquid
//  Glass panel near the top-right, in the style of the macOS 27 Siri panel —
//  a close button top-left, an expand button top-right that opens Today, the
//  card, and an "Ask Jarvis" field between + and mic buttons. A Step Cue (a
//  planned step starting now) and a Break Cue (two hours at the Mac) take a
//  shorter panel. It never steals typing
//  from the app in front: it becomes key only when the field is clicked.
//  Replies go to the Day Thread, with any picture pasted into the field or
//  dropped on the panel (ADR-0090).
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
    /// Two hours at the Mac without a break; shown instead of a card.
    var rest: BreakCue?
    var showQuiet = false
    var draft = ""
    var listening = false
    /// The last exchange typed here, so the answer shows in the panel.
    var asked: String?
    /// The pictures that went with it.
    var askedPictures: [ImageAttachment] = []
    /// Pictures going with the next question: pasted into the field or
    /// dropped on the panel.
    var pictures: [ImageAttachment] = []
    /// What just happened to the field's words: a reminder added (with its
    /// undo), or why they didn't go — a take that wasn't heard, a capture
    /// that didn't save.
    var notice: CaptureOutcome?
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
    private let captureService: CaptureService
    /// Why Jarvis can't see pictures now (vision off, or a model that
    /// can't), or nil when he can.
    private let pictureRemedy: () -> String?
    /// A capture or its undo is under way: + or Undo again does it once.
    private var isCapturing = false
    private var noticeTask: Task<Void, Never>?
    /// The "Done" on the panel closes it when this ends.
    private var doneTask: Task<Void, Never>?
    /// What the panel says was last laid out at, to refit when pictures come
    /// and go above the field.
    private var contentHeight: CGFloat = 0

    /// The panel's width, and the tallest it grows.
    static let size = NSSize(width: 400, height: 560)
    /// The header and the field around what the panel says.
    static let chromeHeight: CGFloat = 124
    static let minimumHeight: CGFloat = 220
    /// The row of pictures waiting above the field.
    static let pictureRowHeight: CGFloat = 58

    /// - Parameter liveCard: the card as the day has it now, so items the
    ///   owner handles leave the panel and a refined card updates in it.
    init(
        thread: DayThread, voice: AgentVoiceInputController, agenda: Agenda,
        liveCard: @escaping (String) -> DayCard?,
        onAction: @escaping (CardAction) -> Void, onExpand: @escaping () -> Void,
        capture: CaptureService, pictureRemedy: @escaping () -> String? = { nil }
    ) {
        self.thread = thread
        self.voice = voice
        self.agenda = agenda
        self.liveCard = liveCard
        self.onAction = onAction
        self.onExpand = onExpand
        self.captureService = capture
        self.pictureRemedy = pictureRemedy
    }

    var isShowing: Bool { panel?.isVisible == true }
    var shownCardID: String? { isShowing ? model.card?.id : nil }

    /// How long a step marked done is said, with its Undo, before the panel
    /// closes.
    static let doneLinger: Duration = .seconds(5)

    /// Show (or update in place) a card.
    func show(_ card: DayCard) {
        model.cue = nil
        model.done = nil
        model.rest = nil
        model.card = card
        model.showQuiet = false
        present()
    }

    /// Show a planned step at its start or end, in place of whatever is up.
    func show(_ cue: StepCue) {
        model.card = nil
        model.done = nil
        model.rest = nil
        model.cue = cue
        present()
    }

    /// Suggest a break, in place of whatever is up.
    func show(_ cue: BreakCue) {
        model.card = nil
        model.cue = nil
        model.done = nil
        model.rest = cue
        present()
    }

    /// Take the Break Cue down: the owner took a break meanwhile.
    func retractBreak() {
        guard isShowing, model.rest != nil else { return }
        close()
    }

    /// Done on a cue is said for a moment — the win, what comes next, and
    /// an Undo — then the panel closes, unless something else took it
    /// meanwhile, the owner took it back, or turned to Jarvis (typing,
    /// asking, speaking).
    private func acknowledge(_ cue: StepCue) {
        model.cue = nil
        model.done = cue
        // What the field held before: only what changes meanwhile counts.
        let draft = model.draft
        let asked = model.asked
        doneTask?.cancel()
        doneTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: Self.doneLinger)
            guard !Task.isCancelled, let self, self.model.done == cue else { return }
            self.model.done = nil
            let turnedToJarvis =
                self.model.listening || self.model.asked != asked || self.model.draft != draft
            if !turnedToJarvis { self.close() }
        }
    }

    /// Not done after all: the task reopens and the cue is back, to choose
    /// again.
    private func undoDone() {
        guard let cue = model.done else { return }
        doneTask?.cancel()
        onAction(.step(reminderID: cue.reminderID, .undo))
        model.done = nil
        model.cue = cue
    }

    /// As tall as what the panel says, between its least and full height,
    /// with room for the pictures waiting above the field.
    static func height(forContent content: CGFloat, pictures: Bool = false) -> CGFloat {
        let chrome = chromeHeight + (pictures ? pictureRowHeight : 0)
        return min(max(content + chrome, minimumHeight), size.height)
    }

    private func present() {
        let panel = self.panel ?? makePanel()
        self.panel = panel
        guard !panel.isVisible else { return }
        model.asked = nil
        model.askedPictures = []
        model.notice = nil
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
        contentHeight = content
        guard let panel else { return }
        let height = Self.height(forContent: content, pictures: !model.pictures.isEmpty)
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
        if model.rest != nil { onAction(.breakCue(.dismiss)) }
    }

    private func makePanel() -> GlassPanel {
        let panel = GlassPanel(size: Self.size, cornerRadius: 28, becomesKeyOnlyIfNeeded: true)
        panel.onCancel = { [weak self] in
            self?.leaveCue()
            self?.close()
        }
        // ⌘V with a picture on the clipboard: the picture, never its name.
        panel.interceptKeyEquivalent = { [weak self] event in
            guard let self, Self.isPaste(event),
                PasteboardImageReader.containsImageContent(.general)
            else { return false }
            self.receivePictures(from: .general)
            return true
        }
        voice.onVoiceTranscription = { [weak self] text in
            guard let self else { return }
            self.model.listening = false
            let typed = self.model.draft.trimmingCharacters(in: .whitespacesAndNewlines)
            guard typed.isEmpty else {
                // Words were in the field: the spoken ones join them, for
                // the owner to send or add.
                self.model.draft = typed + " " + text
                return
            }
            self.model.draft = text
            self.send()
        }
        // Not sure what was said: in the field to look over, never sent as
        // heard.
        voice.onVoiceRejected = { [weak self] raw in
            guard let self else { return }
            self.model.listening = false
            let typed = self.model.draft.trimmingCharacters(in: .whitespacesAndNewlines)
            self.model.draft = typed.isEmpty ? raw : typed + " " + raw
            self.showNotice(.failed("Not sure I heard that right. Check the words, then send."))
        }
        // A take with no words (too short, no speech), or a microphone that
        // didn't start: listening ends, and the panel says why.
        voice.onVoiceFailure = { [weak self] message in
            self?.model.listening = false
            self?.showNotice(.failed(message))
        }
        panel.host(
            JarvisPanelView(
                model: model, thread: thread, agenda: agenda, liveCard: liveCard,
                close: { [weak self] in
                    guard let self else { return }
                    // Off the panel, not thrown away: what it holds stays in Today.
                    if let card = self.model.card { self.onAction(.close(cardID: card.id)) }
                    if let cue = self.model.cue {
                        self.onAction(.step(reminderID: cue.reminderID, .dismiss))
                    }
                    if self.model.rest != nil { self.onAction(.breakCue(.dismiss)) }
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
                undoDone: { [weak self] in self?.undoDone() },
                join: { [weak self] url in
                    // Off to the call: the panel goes with the step's Done.
                    self?.doneTask?.cancel()
                    self?.model.done = nil
                    self?.close()
                    NSWorkspace.shared.open(url)
                },
                chooseBreak: { [weak self] choice in
                    guard let self, self.model.rest != nil else { return }
                    self.onAction(.breakCue(choice))
                    self.close()
                },
                send: { [weak self] in self?.send() },
                capture: { [weak self] in self?.capture() },
                undo: { [weak self] in self?.undoCapture() },
                mic: { [weak self] in self?.toggleMic() },
                removePicture: { [weak self] id in self?.removePicture(id) },
                onContentHeight: { [weak self] height in
                    // On the next turn: never resize the window inside its own
                    // layout pass.
                    Task { @MainActor [weak self] in self?.fit(content: height) }
                }),
            drop: GlassPanel.Drop(
                types: PasteboardImageReader.dragTypes + [.fileURL],
                accepts: PasteboardImageReader.containsImageContent,
                perform: { [weak self] pasteboard in self?.receivePictures(from: pasteboard) }))
        return panel
    }

    // MARK: Pictures

    static func isPaste(_ event: NSEvent) -> Bool {
        event.type == .keyDown
            && event.modifierFlags.intersection(.deviceIndependentFlagsMask) == .command
            && event.charactersIgnoringModifiers == "v"
    }

    /// Read a pasted or dropped picture off its pasteboard (a promised file
    /// arrives later), then take it.
    private func receivePictures(from pasteboard: NSPasteboard) {
        Task { @MainActor [weak self] in
            let payload = await PasteboardImageReader.read(pasteboard)
            guard let self, !payload.isEmpty else { return }
            let taken = Self.take(payload, into: self.model.pictures, remedy: self.pictureRemedy())
            self.model.pictures = taken.pictures
            self.refit()
            if let notice = taken.notice { self.showNotice(.failed(notice)) }
        }
    }

    /// What a paste or drop does to the pictures waiting for the next
    /// question, by the composer's rules: nothing while Jarvis can't see
    /// them (`remedy` says why), up to eight, and a line for what didn't
    /// come in.
    static func take(
        _ payload: ImageGesturePayload, into pictures: [ImageAttachment], remedy: String?
    ) -> (pictures: [ImageAttachment], notice: String?) {
        if let remedy { return (pictures, remedy) }
        let added = ImageIngest.capBatch(
            payload.attachments, alreadyQueued: pictures.count,
            limit: ComposerDraftController.maxPendingImages)
        let notice = ComposerDraftController.gestureNotice(
            requested: payload.attachments.count, attached: added.count,
            rejections: payload.rejections)
        return (pictures + added, notice)
    }

    private func removePicture(_ id: UUID) {
        model.pictures.removeAll { $0.id == id }
        refit()
    }

    private func refit() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.fit(content: self.contentHeight)
        }
    }

    private func send() {
        let text = model.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let pictures = model.pictures
        guard !text.isEmpty || !pictures.isEmpty, thread.canSend else { return }
        thread.send(text, images: pictures, from: .panel)
        model.asked = DayThread.question(text, pictures: pictures.count)
        model.askedPictures = pictures
        model.pictures = []
        model.draft = ""
        if !pictures.isEmpty { refit() }
    }

    /// The + button: the field's words become a reminder, no model. The
    /// panel says what happened; the words leave the field only once saved.
    private func capture() {
        let text = model.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        // A reminder is made from words, with no model; a picture needs
        // Jarvis to read it.
        guard model.pictures.isEmpty else {
            showNotice(.failed("A picture goes to Jarvis: press Return to ask him."))
            return
        }
        guard !text.isEmpty, !isCapturing else { return }
        isCapturing = true
        Task { @MainActor [weak self] in
            guard let self else { return }
            let outcome = await self.captureService.capture(text, source: "panel")
            self.isCapturing = false
            self.showNotice(outcome)
            let still = self.model.draft.trimmingCharacters(in: .whitespacesAndNewlines)
            if case .added = outcome, still == text { self.model.draft = "" }
        }
    }

    /// Undo the reminder on the line — and say "Undone." only when it was.
    private func undoCapture() {
        guard !isCapturing, case .added(let change) = model.notice else { return }
        isCapturing = true
        Task { @MainActor [weak self] in
            guard let self else { return }
            let undone = await self.captureService.undo(change)
            self.isCapturing = false
            self.showNotice(
                undone
                    ? .failed("Undone.")
                    : self.captureService.lastOutcome ?? .failed("Couldn't undo."))
        }
    }

    /// The line stays a few seconds, longer when something didn't go.
    private func showNotice(_ notice: CaptureOutcome) {
        model.notice = notice
        noticeTask?.cancel()
        let shown: Duration = if case .added = notice { .seconds(8) } else { .seconds(12) }
        noticeTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: shown)
            guard !Task.isCancelled, let self, self.model.notice == notice else { return }
            self.model.notice = nil
        }
    }

    private func toggleMic() {
        if model.listening {
            voice.finishCapture()
            model.listening = false
        } else if voice.voiceState == .transcribing {
            showNotice(.failed("Still writing the last one down."))
        } else {
            voice.start()
            // Listening only if the microphone started: a busy or refused
            // one has said why.
            model.listening = voice.voiceState == .recording
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
    /// Take back the Done the panel is saying.
    var undoDone: () -> Void = {}
    /// Open a call's link.
    var join: (URL) -> Void = { _ in }
    var chooseBreak: (BreakChoice) -> Void = { _ in }
    let send: () -> Void
    let capture: () -> Void
    /// Undo the reminder the + button just added.
    var undo: () -> Void = {}
    let mic: () -> Void
    /// Take a waiting picture back out.
    var removePicture: (UUID) -> Void = { _ in }
    /// What the panel says was laid out at this height.
    var onContentHeight: (CGFloat) -> Void = { _ in }
    /// The clock the plan's steps are read against (fixed in the gallery).
    var now: () -> Date = Date.init

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                CircleButton(symbol: "xmark", help: "Close (it stays in Today)", action: close)
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
                    } else if let rest = model.rest {
                        BreakCueContent(cue: rest, choose: chooseBreak)
                    } else if let done = model.done {
                        StepDoneContent(cue: done, now: now(), undo: undoDone, join: join)
                    } else if let shown = model.card {
                        CardContent(
                            card: liveCard(shown.id) ?? shown, agenda: agenda, now: now(),
                            showQuiet: $model.showQuiet, act: act, keep: keep, expand: expand)
                    }
                    if let asked = model.asked {
                        Exchange(asked: asked, pictures: model.askedPictures, thread: thread)
                    }
                    if let notice = model.notice {
                        NoticeLine(notice: notice, undo: undo)
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

            if !model.pictures.isEmpty {
                PictureRow(pictures: model.pictures, remove: removePicture)
                    .padding(.horizontal, 18)
                    .padding(.top, 6)
            }

            HStack(spacing: 10) {
                CircleButton(symbol: "plus", help: "Add as a reminder", action: capture)
                // Present from the first layout: a field that appears later
                // freezes the main thread on macOS 27.0. With pictures and no
                // words, its placeholder is the question they ask.
                TextField(
                    model.pictures.isEmpty
                        ? "Ask Jarvis" : DayThread.question("", pictures: model.pictures.count),
                    text: $model.draft
                )
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
                    // The must-do wears Today's star.
                    HStack(alignment: .firstTextBaseline, spacing: 5) {
                        if cue.isMustDo {
                            Image(systemName: "star.fill").font(.system(size: 11))
                        }
                        Text(cue.heading)
                    }
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
                Text(cue.detail)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Every button non-focusable: they appear after the panel's
            // first layout (GlassPanel's macOS 27.0 focus freeze).
            HStack(spacing: 8) {
                switch cue.phase {
                case .start:
                    if cue.offersSmallStart {
                        choice(
                            "Start \(DayEngine.smallStartMinutes) min", .startSmall, prominent: true
                        )
                        .help("Just five minutes, then decide; Jarvis asks then")
                    } else {
                        choice("Start", .start, prominent: true)
                    }
                    if let resumeAt = cue.resumeAt {
                        choice("At \(AgendaTime.clock(resumeAt))", .resume)
                            .help("What comes next is first: move it to when that's over")
                    } else {
                        choice("In \(DayEngine.stepLaterMinutes) min", .later)
                            .help("Move it a quarter of an hour on; Jarvis asks again then")
                    }
                    choice("Tomorrow", .tomorrow)
                    Spacer(minLength: 0)
                    // Not "Done": alone at the row's end, where a dialog's OK
                    // sits, it read as closing the cue.
                    choice("Did it", .done)
                        .help("Done already: it's ticked off")
                case .end where cue.small:
                    if let resumeAt = cue.resumeAt {
                        choice("Go on at \(AgendaTime.clock(resumeAt))", .resume, prominent: true)
                            .help("Pick it up when what comes next is over; Jarvis asks then")
                    } else {
                        choice("Keep going", .extend, prominent: true)
                            .help("A quarter of an hour more; Jarvis asks again then")
                    }
                    choice("Done", .done)
                    choice("Tomorrow", .tomorrow)
                    Spacer(minLength: 0)
                case .end:
                    choice("Done", .done, prominent: true)
                    if let resumeAt = cue.resumeAt {
                        choice("Go on at \(AgendaTime.clock(resumeAt))", .resume)
                            .help("Pick it up when what comes next is over; Jarvis asks then")
                    } else {
                        choice("\(DayEngine.stepLaterMinutes) more min", .extend)
                            .help("Keep going a quarter of an hour; Jarvis asks again then")
                    }
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

}

/// Two hours at the Mac: how long and since when, one small thing to do —
/// a different one each time that day — the step to come back to, and
/// Taking 5 or In 30 min.
private struct BreakCueContent: View {
    let cue: BreakCue
    let choose: (BreakChoice) -> Void

    static let ideas = [
        "Stand up, stretch, and get a glass of water. It will all be here when you're back.",
        "Look out of a window for a minute, roll your shoulders, and drink some water.",
        "A short walk, even to the kitchen and back, resets your focus.",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Time for a break")
                        .fontWeight(.semibold)
                        .foregroundStyle(Color.accentColor)
                    Spacer()
                    Text("since \(AgendaTime.clock(cue.since))")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Text("\(MomentPrompts.minutesText(cue.minutes)) at the Mac")
                    .fontWeight(.semibold)
                Text(Self.ideas[max(cue.number - 1, 0) % Self.ideas.count])
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let step = cue.step, let end = cue.stepEnd {
                    Text("Then back to \(step) until \(AgendaTime.clock(end)).")
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            // Non-focusable, like every button here (GlassPanel's macOS 27.0
            // focus freeze).
            HStack(spacing: 8) {
                Button("Taking 5") { choose(.taking) }
                    .buttonStyle(PanelButtonStyle(prominent: true))
                    .focusable(false)
                    .help("Step away for a few minutes; the two hours start again")
                Button("In \(DayEngine.breakLaterMinutes) min") { choose(.later) }
                    .buttonStyle(PanelButtonStyle())
                    .focusable(false)
                    .help("Jarvis asks again in half an hour")
                Spacer(minLength: 0)
            }
        }
    }
}

/// The win, said: the step is done, the must-do credited, and what comes
/// next — a moment's word before the panel closes, with an Undo for a Done
/// that wasn't meant.
private struct StepDoneContent: View {
    let cue: StepCue
    let now: Date
    let undo: () -> Void
    var join: (URL) -> Void = { _ in }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Label(cue.doneHeading(now: now), systemImage: "checkmark.circle.fill")
                    .fontWeight(.semibold)
                    .foregroundStyle(Color.accentColor)
                Spacer()
                Button("Undo", action: undo)
                    .buttonStyle(PanelButtonStyle(compact: true))
                    .focusable(false)
                    .help("Not done after all: it reopens, and you can choose again")
            }
            Text(cue.title)
                .fontWeight(.semibold)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            if let line = cue.doneLine(now: now) {
                Text(line)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let link = cue.joinLink(now: now) {
                Button("Join") { join(link) }
                    .buttonStyle(PanelButtonStyle(prominent: true))
                    .focusable(false)
                    .help("Open the call")
                    .padding(.top, 8)
            }
        }
    }
}

extension StepCue {
    /// The cue's heading. A late cue (the owner was away or busy) says so,
    /// without blame; a step put off twice is offered small, and a small
    /// start asks to go on. The must-do is named at its start: the day's
    /// one goal, not one more step.
    var heading: String {
        if offersSmallStart { return "Just five minutes?" }
        let mustDo = isMustDo ? " your must-do" : ""
        switch (phase, late) {
        case (.start, false): return "Time for" + mustDo
        case (.start, true): return "Still time for" + mustDo
        case (.end, false): return small ? "Five minutes in" : "Time's up"
        case (.end, true): return "How did it go?"
        }
    }

    /// How long, the Area and what follows; the must-do here once its
    /// heading no longer names it.
    var detail: String {
        if offersSmallStart { return "Starting is the hard part: five minutes, then decide." }
        if phase == .end, small, !late {
            return "Keep going, or leave it there — five minutes counts."
        }
        var parts = [MomentPrompts.minutesText(minutes)]
        if isMustDo, phase == .end { parts.append("your must-do") }
        parts.append(areaName)
        var line = parts.joined(separator: " · ")
        if let next { line += ". Then \(next)." }
        return line
    }

    /// What the panel says once the step is done: the must-do credited (the
    /// Now Card's word too), then, on its own line, what comes next — unless
    /// that has started since the cue went up (it sat on the panel).
    func doneLine(now: Date) -> String? {
        let next = nextAt.map({ $0 < now }) == true ? nil : next.map { "Next: \($0)." }
        guard isMustDo else { return next }
        return ["That's the must-do — the rest is a bonus.", next].compactMap(\.self)
            .joined(separator: "\n")
    }

    /// "Done in 26 min": how long a step the owner started took, said as it
    /// ends — time made visible, never weighed against the plan. Only when
    /// answered at its end; later, how long it took is unknown.
    func doneHeading(now: Date) -> String {
        guard phase == .end, now > start,
            now.timeIntervalSince(end) <= DayEngine.stepCueLate
        else { return "Done" }
        let minutes = max(1, Int(now.timeIntervalSince(start) / 60))
        return "Done in \(MomentPrompts.minutesText(minutes))"
    }

    /// What comes next is a call starting within a quarter hour (or a few
    /// minutes under way): its link, so the step done, joining is one click.
    func joinLink(now: Date) -> URL? {
        guard let nextLink, let nextAt,
            nextAt.timeIntervalSince(now) <= 15 * 60, now.timeIntervalSince(nextAt) <= 10 * 60
        else { return nil }
        return nextLink
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
                    card: card, plan: plan, agenda: agenda, now: now, act: act, keep: keep,
                    expand: expand)
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
    let act: (CardAction) -> Void
    let keep: () -> Void
    let expand: () -> Void

    var body: some View {
        let steps = PlanStep.ahead(plan: plan, agenda: agenda, now: now)
        // Jarvis's own plan, not the card code put up while he thinks.
        let first = card.isRefining ? nil : PlanStep.startable(steps, now: now)
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
                // The first step is due: one click from the plan to doing it.
                if let first, let reminderID = first.reminderID {
                    Button("Start Now") {
                        act(.step(reminderID: reminderID, .start))
                        keep()
                    }
                    .buttonStyle(PanelButtonStyle(prominent: true))
                    .focusable(false)
                    .help("Start \(first.title) now; the plan stays in Today")
                }
                Button("Looks Good", action: keep)
                    .buttonStyle(PanelButtonStyle(prominent: first == nil))
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
    /// The reminder of a task the plan placed (nil for an event, or a task
    /// only due at a time).
    var reminderID: String? = nil

    /// The most the panel lists; Today has the rest.
    static let shown = 5

    /// The plan's first step when it is a task due within ten minutes: the
    /// panel offers to start it.
    static func startable(_ steps: [PlanStep], now: Date) -> PlanStep? {
        guard let first = steps.first, first.reminderID != nil,
            first.start <= now.addingTimeInterval(10 * 60)
        else { return nil }
        return first
    }

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
                // Only a step the plan placed can be started from it.
                return PlanStep(
                    id: row.id, start: row.start, title: task.reminder.title,
                    minutes: task.minutes, kind: .task(isMustDo: task.isMustDo),
                    reminderID: task.isPlanned ? task.id : nil)
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
                    Text(wrapUp.leftoversHeading).fontWeight(.semibold)
                    ForEach(wrapUp.leftovers) { leftover in
                        VStack(alignment: .leading, spacing: MorningPlanContent.rowSpacing) {
                            Text(leftover.title).lineLimit(2)
                            if let waiting = leftover.waiting {
                                Text(waiting).foregroundStyle(.secondary)
                            }
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

/// What just happened to the field's words, above it: the reminder the +
/// button added, with its undo, or why the words didn't go (they stay in
/// the field).
private struct NoticeLine: View {
    let notice: CaptureOutcome
    let undo: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: isAdded ? "checkmark.circle.fill" : "info.circle")
                .foregroundStyle(isAdded ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
            Text(notice.line)
                .foregroundStyle(isAdded ? .primary : .secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            if isAdded {
                Button("Undo", action: undo)
                    .buttonStyle(.plain)
                    .foregroundStyle(.tint)
                    .fontWeight(.medium)
                    .focusable(false)
            }
        }
    }

    private var isAdded: Bool {
        if case .added = notice { true } else { false }
    }
}

/// The pictures waiting to go with the next question, each with its ✕.
private struct PictureRow: View {
    let pictures: [ImageAttachment]
    let remove: (UUID) -> Void

    var body: some View {
        HStack(spacing: 8) {
            ForEach(pictures) { picture in
                ImageThumbnailView(
                    attachment: picture, side: 44, onRemove: { remove(picture.id) })
            }
            Spacer(minLength: 0)
        }
        .frame(height: 52)
    }
}

private struct Exchange: View {
    let asked: String
    /// The pictures that went with the question.
    var pictures: [ImageAttachment] = []
    let thread: DayThread

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !pictures.isEmpty {
                HStack(spacing: 6) {
                    ForEach(pictures) { ImageThumbnailView(attachment: $0, side: 48) }
                }
                .frame(maxWidth: .infinity, alignment: .trailing)
            }
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
