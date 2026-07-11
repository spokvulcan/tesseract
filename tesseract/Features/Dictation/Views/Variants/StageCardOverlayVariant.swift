//
//  StageCardOverlayVariant.swift
//  tesseract
//

import AppKit
import SwiftUI

// MARK: - Metrics

/// The Stage Card's sizes — the informative pole of the overlay explorations
/// (map #283): one fixed card frame for every stage, so crossfades between
/// stages are pure content swaps with zero layout drift.
///
/// `nonisolated` so it escapes the build's MainActor default isolation — that's
/// what lets ``OverlayPlacement/stageCard`` and its tests run off the main actor.
nonisolated enum StageCardMetrics {
    /// The one card size. Every stage renders in this frame; density varies
    /// (recording is airy, committed-with-edits fills it), the frame never.
    static let cardSize = CGSize(width: 300, height: 76)

    /// The fixed panel canvas: the card plus spring-overshoot and antialiasing
    /// headroom on every side. The card is bottom-anchored inside it, so the
    /// visual bottom inset is constant across stages.
    static let canvasSize = CGSize(width: 340, height: 96)

    static let cornerRadius: CGFloat = 18
    /// The leading glyph column — fixed so the text stack's origin is
    /// identical in every stage and crossfades never shift the copy.
    static let glyphColumnWidth: CGFloat = 24
}

// MARK: - Placement

nonisolated extension OverlayPlacement {
    /// The Stage Card's canvas: centred horizontally in the visible frame,
    /// its bottom edge at a fixed inset above the visible frame's bottom —
    /// slightly lower than the pill (the card is taller; it should not creep
    /// toward the screen's center).
    static let stageCard = OverlayPlacement(
        frame: { geometry in
            let size = StageCardMetrics.canvasSize
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.midX - size.width / 2,
                y: visible.minY + stageCardBottomInset,
                width: size.width,
                height: size.height
            )
        }
    )

    /// The card canvas floats this far above the bottom of the visible frame.
    private static let stageCardBottomInset: CGFloat = 52
}

// MARK: - View

/// The **Stage Card** — an **Overlay Variant** (map #283) at the opposite pole
/// from the classic pill's minimalism: a fixed glass card that always says in
/// words *and* data exactly what the pipeline is doing. One 13 pt text size,
/// hierarchy by weight and color; one grid (glyph column, two-line stack,
/// trailing live element) shared by every stage.
///
/// Same glass discipline as the pill: one pure untinted `.glassEffect` rect,
/// state read through vibrant content (red mic recording, green check on
/// commit, orange on trouble), never through a tint. Same motion discipline:
/// the panel canvas is fixed; one spring (0.22/0.8) drives entrance/exit and
/// the id-keyed content crossfade between stages — Reduce Motion turns every
/// spring into a plain fade.
struct StageCardOverlayView: View {
    /// The Overlay Feed — the only pipeline surface this view sees.
    var feed: DictationFeed
    /// The overlay action surface (ticket #289) — what the lingering beat's
    /// click affordances invoke. Clicks only; the overlay stays keyboard-free.
    var actions: OverlayActions = .none

    /// The phase the card currently renders — updated inside `withAnimation`,
    /// so mount/unmount and stage crossfades all ride one spring.
    @State private var shownPhase: DictationFeed.Phase = .idle
    /// A lingering terminal beat (map #283): committed and rejected takes hold
    /// the card for ``DictationFeed/affordanceLinger`` after the phase returned
    /// to idle — flag/edit on a commit, insert-raw on a rejection.
    @State private var shownBeat: DictationFeed.Beat?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// What the card narrates right now — a live phase always outranks a
    /// lingering beat; `nil` (idle, no beat) renders nothing.
    private enum Stage: Equatable {
        case phase(DictationFeed.Phase)
        case beat(DictationFeed.Beat)

        /// Content identity for the stage crossfade. Stages swap by key —
        /// beats key on their id so two commits in a row read as two beats —
        /// while text changes *within* a stage (a growing timer, a different
        /// error) re-diff in place with no fade.
        var key: String {
            switch self {
            case .phase(.recording): return "recording"
            case .phase(.processing): return "processing"
            case .phase(.proofreading): return "proofreading"
            case .phase(.error): return "error"
            case .phase(.idle): return "idle"
            case .beat(let beat): return "beat-\(beat.id)"
            }
        }
    }

    private var stage: Stage? {
        if shownPhase != .idle { return .phase(shownPhase) }
        if let shownBeat { return .beat(shownBeat) }
        return nil
    }

    var body: some View {
        GlassEffectContainer {
            ZStack(alignment: .bottom) {
                if let stage {
                    card(for: stage)
                        .transition(
                            reduceMotion
                                ? .opacity
                                : .scale(scale: 0.9, anchor: .bottom).combined(with: .opacity)
                        )
                }
            }
            .frame(
                width: StageCardMetrics.canvasSize.width,
                height: StageCardMetrics.canvasSize.height,
                alignment: .bottom
            )
        }
        .onChange(of: feed.phase) { _, newPhase in
            apply(newPhase)
        }
        .onChange(of: feed.beat) { _, beat in
            applyBeat(beat)
        }
        .onAppear {
            apply(feed.phase, animated: false)
        }
    }

    // MARK: - Stage transitions

    /// The one animation of the variant: a single spring for entrance, exit,
    /// and the stage crossfade; Reduce Motion swaps it for a plain fade.
    private var stageAnimation: Animation {
        reduceMotion
            ? .easeInOut(duration: 0.18)
            : .spring(response: 0.22, dampingFraction: 0.8)
    }

    private func apply(_ newPhase: DictationFeed.Phase, animated: Bool = true) {
        guard shownPhase != newPhase else { return }
        if newPhase != .idle {
            // A live phase always outranks a lingering beat.
            shownBeat = nil
        }
        guard animated else {
            shownPhase = newPhase
            return
        }
        withAnimation(stageAnimation) {
            shownPhase = newPhase
        }
    }

    private func applyBeat(_ beat: DictationFeed.Beat?) {
        guard let beat else { return }
        switch beat.outcome {
        case .committed, .rejected:
            break
        case .empty, .cancelled, .superseded:
            return
        }
        withAnimation(stageAnimation) {
            shownBeat = beat
        }
        Task {
            try? await Task.sleep(for: DictationFeed.affordanceLinger)
            dismissBeat(ifStill: beat.id)
        }
    }

    private func dismissBeat(ifStill id: UInt64? = nil) {
        if let id, shownBeat?.id != id { return }
        withAnimation(stageAnimation) {
            shownBeat = nil
        }
    }

    // MARK: - Card

    /// The fixed glass frame with the stage content crossfading inside it.
    /// Pure, untinted `.regular` glass — the material *is* the card; state
    /// color lives in the content (same owner-selected discipline as the
    /// classic pill: a tinted card over arbitrary desktops reads as a painted
    /// background, not glass).
    private func card(for stage: Stage) -> some View {
        ZStack {
            content(for: stage)
                .id(stage.key)
                .transition(.opacity)
        }
        .frame(width: StageCardMetrics.cardSize.width, height: StageCardMetrics.cardSize.height)
        .glassEffect(.regular, in: .rect(cornerRadius: StageCardMetrics.cornerRadius))
    }

    @ViewBuilder
    private func content(for stage: Stage) -> some View {
        switch stage {
        case .phase(.recording):
            recordingContent
        case .phase(.processing):
            processingContent
        case .phase(.proofreading):
            proofreadingContent
        case .phase(.error(let error)):
            errorContent(for: error)
        case .phase(.idle):
            // Unreachable — `stage` is nil at idle; keeps the switch total.
            EmptyView()
        case .beat(let beat):
            beatContent(for: beat)
        }
    }

    // MARK: - The shared stage grid

    /// The card's one grid: fixed glyph column, two-line text stack, trailing
    /// live element. Every non-beat stage fills the same slots, so crossfades
    /// never shift the layout — only the words and the live data change.
    private func stageRow(
        glyph: String,
        glyphColor: Color = .secondary,
        twinkles: Bool = false,
        title: String,
        @ViewBuilder detail: () -> some View,
        @ViewBuilder trailing: () -> some View
    ) -> some View {
        HStack(alignment: .center, spacing: 12) {
            stageGlyph(glyph, color: glyphColor, twinkles: twinkles)
            VStack(alignment: .leading, spacing: 2) {
                titleText(title)
                detail()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            trailing()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    /// The stage glyph — decorative (the title says what it shows), colored
    /// only when the state has semantic color. `twinkles` is the proofreading
    /// stage's subtle pulse; inert elsewhere and under Reduce Motion.
    private func stageGlyph(_ systemName: String, color: Color, twinkles: Bool = false) -> some View
    {
        Image(systemName: systemName)
            .font(.system(size: 16, weight: .semibold))
            .symbolEffect(.pulse, options: .repeating, isActive: twinkles && !reduceMotion)
            .foregroundStyle(color)
            .frame(width: StageCardMetrics.glyphColumnWidth)
            .accessibilityHidden(true)
    }

    private func titleText(_ string: String) -> some View {
        Text(string)
            .font(.system(size: 13, weight: .semibold))
            .lineLimit(1)
            .minimumScaleFactor(0.85)
    }

    private func detailText(_ string: String, lines: Int = 1) -> some View {
        Text(string)
            .font(.system(size: 13))
            .foregroundStyle(.secondary)
            .lineLimit(lines)
            .minimumScaleFactor(lines == 1 ? 0.9 : 1)
    }

    // MARK: - Live phases

    private var recordingContent: some View {
        stageRow(glyph: "mic.fill", glyphColor: .red, title: "Recording") {
            // The 1 Hz tick is scoped to this one Text — the card chrome and
            // glass sit outside the timeline closure. Ticks align to the
            // recording start so the counter flips exactly on the second.
            TimelineView(.periodic(from: feed.recordingStarted ?? .now, by: 1)) { timeline in
                Text(Self.elapsed(at: timeline.date, since: feed.recordingStarted))
                    .font(.system(size: 13))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        } trailing: {
            StageCardSpectrumView(feed: feed)
        }
    }

    private var processingContent: some View {
        stageRow(glyph: "waveform", title: "Transcribing") {
            detailText("Whisper is listening back…")
        } trailing: {
            processingDots
        }
    }

    private var proofreadingContent: some View {
        stageRow(
            glyph: "wand.and.sparkles", twinkles: true, title: "Polishing"
        ) {
            detailText("Fixing punctuation and slips")
        } trailing: {
            // The same dots as `.processing`: both stages derive dot phase
            // from wall-clock time, so the crossfade lands on identical dots
            // and the activity reads as continuous.
            processingDots
        }
    }

    /// The trailing activity dots. The 60 fps timeline lives here — inside
    /// the trailing slot — so only the dots re-evaluate per frame. Scaled to
    /// card proportions (the dots' natural frame is pill-sized); the frame
    /// caps the layout width so the detail line keeps its room.
    @ViewBuilder
    private var processingDots: some View {
        if reduceMotion {
            ProgressView()
                .controlSize(.small)
                .accessibilityHidden(true)
        } else {
            TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
                ProcessingDotsView(time: timeline.date.timeIntervalSinceReferenceDate)
                    .scaleEffect(0.75)
            }
            .frame(width: 57, height: 18)
            .accessibilityHidden(true)
        }
    }

    /// Both `errorDescription` and `recoverySuggestion` render — the card has
    /// the two-line room the pill lacks, and telling the user how to recover
    /// is the point of the informative pole.
    private func errorContent(for error: DictationError) -> some View {
        stageRow(
            glyph: "exclamationmark.triangle.fill",
            glyphColor: .orange,
            title: error.errorDescription ?? "Something went wrong."
        ) {
            if let suggestion = error.recoverySuggestion {
                detailText(suggestion, lines: 2)
            }
        } trailing: {
            EmptyView()
        }
        .accessibilityElement(children: .combine)
    }

    // MARK: - Lingering beats (ticket #289)

    /// The lingering terminal-beat card: the panel is clickable for exactly
    /// this window (one App Bindings rule), so these are the overlay's only
    /// interactive moments — one-click, no focus steal, and a stray click
    /// anywhere on the card dismisses it.
    @ViewBuilder
    private func beatContent(for beat: DictationFeed.Beat) -> some View {
        switch beat.outcome {
        case .committed(let text, _, let edits):
            committedContent(text: text, edits: edits)
        case .rejected(_, let reason):
            rejectedContent(reason: reason)
        case .empty, .cancelled, .superseded:
            // Filtered in `applyBeat`; keeps the switch total.
            EmptyView()
        }
    }

    /// Post-commit: the committed text itself plus the Proofread Pass's
    /// word-diff as chips — the full narration the informative pole promises.
    /// The affordances sit on the title row (notification-style) so the text
    /// and the chips get the card's full column width.
    private func committedContent(text: String, edits: [WordEdit]) -> some View {
        HStack(alignment: .center, spacing: 12) {
            stageGlyph("checkmark", color: .green)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    titleText("Inserted")
                    Spacer(minLength: 4)
                    beatButton("flag", label: "Flag as wrong") {
                        actions.flagLastTakeWrong()
                        dismissBeat()
                    }
                    beatButton("pencil", label: "Edit in history") {
                        actions.editLastTake()
                        dismissBeat()
                    }
                }
                Text(text)
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .accessibilityLabel(
                        edits.isEmpty
                            ? "Inserted: \(text)"
                            : "Inserted, proofreader polished \(edits.count) words: \(text)")
                if !edits.isEmpty {
                    editChips(for: edits)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .onTapGesture { dismissBeat() }
    }

    /// The lingering rejected-take card. Passive-first: the press is the
    /// retry; the one affordance inserts the raw take anyway (which also
    /// flags the pass as wrong).
    private func rejectedContent(reason: String) -> some View {
        stageRow(
            glyph: "arrow.uturn.backward",
            glyphColor: .orange,
            title: "Didn't catch that"
        ) {
            detailText(reason, lines: 2)
                .accessibilityLabel("Transcription rejected: \(reason) — hold the hotkey to retry")
        } trailing: {
            Button {
                actions.insertRawAnyway()
                dismissBeat()
            } label: {
                Text("Insert anyway")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(.quaternary, in: Capsule())
                    .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Insert the raw transcription anyway")
        }
        .contentShape(Rectangle())
        .onTapGesture { dismissBeat() }
    }

    /// The Proofread Pass's word-diff as mini-capsules: up to three
    /// "peace → piece" swaps, the rest collapsed into "+N". Insertions and
    /// deletions (one side of the `WordEdit` empty) render as "+ word" / "− word".
    private func editChips(for edits: [WordEdit]) -> some View {
        let shown = edits.prefix(3)
        let overflow = edits.count - shown.count
        return HStack(spacing: 4) {
            ForEach(Array(shown.enumerated()), id: \.offset) { _, edit in
                chip(Self.chipLabel(for: edit))
            }
            if overflow > 0 {
                chip("+\(overflow)")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityHidden(true)  // narrated by the committed-text label
    }

    private static func chipLabel(for edit: WordEdit) -> String {
        if edit.original.isEmpty { return "+ \(edit.replacement)" }
        if edit.replacement.isEmpty { return "− \(edit.original)" }
        return "\(edit.original) → \(edit.replacement)"
    }

    private func chip(_ label: String) -> some View {
        Text(label)
            .font(.system(size: 10, weight: .medium))
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(.quaternary, in: Capsule())
    }

    /// One small icon affordance on the lingering card: plain style (the
    /// glass card is the chrome), secondary ink, generous hit target.
    private func beatButton(
        _ systemName: String, label: String, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 22, height: 22)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }

    // MARK: - Formatting

    /// mm:ss from the feed's recording start; "00:00" before the first tick.
    private static func elapsed(at date: Date, since started: Date?) -> String {
        guard let started else { return "00:00" }
        let seconds = max(0, Int(date.timeIntervalSince(started)))
        return String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }
}

// MARK: - Spectrum bars

/// The recording card's trailing live element: the feed's eight log-spaced
/// spectrum bands as slim capsules growing from the vertical center. Reads
/// the feed directly (not a value snapshot) so the ~47 Hz meter cadence
/// invalidates only this subtree — the card chrome and glass never re-diff
/// during steady recording.
private struct StageCardSpectrumView: View {
    var feed: DictationFeed

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let barWidth: CGFloat = 3
    private let barSpacing: CGFloat = 2
    private let barHeight: CGFloat = 26

    var body: some View {
        HStack(alignment: .center, spacing: barSpacing) {
            ForEach(0..<MeterFrame.bandCount, id: \.self) { band in
                Capsule()
                    .fill(.red.opacity(0.55 + 0.45 * value(for: band)))
                    .frame(width: barWidth, height: height(for: band))
            }
        }
        .frame(height: barHeight)
        // A short linear blend smooths the ~47 Hz frames into continuous
        // motion; the meter is live data, but under Reduce Motion the bars
        // snap instead of easing.
        .animation(reduceMotion ? nil : .linear(duration: 0.08), value: feed.spectrum)
        .accessibilityHidden(true)  // the elapsed counter carries the signal
    }

    private func value(for band: Int) -> Double {
        band < feed.spectrum.count ? Double(feed.spectrum[band]) : 0
    }

    private func height(for band: Int) -> CGFloat {
        // A visible stub even at silence, so the meter reads as armed.
        max(2, barHeight * (0.1 + 0.9 * CGFloat(value(for: band))))
    }
}

// MARK: - Registry entry

extension OverlayVariants {
    /// The Stage Card's registry entry (map #283) — defined here beside the
    /// view; the `all` roster in `OverlayVariants.swift` decides what ships.
    static let stageCard = OverlayVariant(
        id: "card",
        displayName: "Stage Card",
        placement: .stageCard
    ) { feed, actions in
        AnyView(StageCardOverlayView(feed: feed, actions: actions))
    }
}

// MARK: - Previews

#Preview("Recording") {
    let feed = DictationFeed()
    feed.setPhase(.recording)
    feed.apply(MeterFrame(level: 0.5, bands: [0.2, 0.55, 0.9, 0.7, 0.5, 0.65, 0.35, 0.15]))
    return StageCardOverlayView(feed: feed)
        .task {
            // Synthetic meter frames so the spectrum lives in the canvas.
            while !Task.isCancelled {
                let bands = (0..<MeterFrame.bandCount).map { _ in Float.random(in: 0.05...0.95) }
                feed.apply(MeterFrame(level: Float.random(in: 0.3...0.9), bands: bands))
                try? await Task.sleep(for: .milliseconds(80))
            }
        }
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Processing") {
    let feed = DictationFeed()
    feed.setPhase(.processing)
    return StageCardOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Proofreading") {
    let feed = DictationFeed()
    feed.setPhase(.proofreading)
    return StageCardOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Committed — with edits") {
    let feed = DictationFeed()
    return StageCardOverlayView(feed: feed)
        .task {
            // Re-emit past the linger so the card keeps returning in canvas.
            while !Task.isCancelled {
                feed.emit(
                    .committed(
                        text: "Let's find a quiet piece of the afternoon for the review.",
                        duration: 6.4,
                        edits: [
                            WordEdit(original: "peace", replacement: "piece"),
                            WordEdit(original: "too", replacement: "of"),
                            WordEdit(original: "reviews", replacement: "review"),
                            WordEdit(original: "", replacement: "the"),
                        ]))
                try? await Task.sleep(for: .seconds(4))
            }
        }
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Rejected") {
    let feed = DictationFeed()
    return StageCardOverlayView(feed: feed)
        .task {
            while !Task.isCancelled {
                feed.emit(
                    .rejected(
                        raw: "uh the the meeting um", reason: "Unintelligible transcription"))
                try? await Task.sleep(for: .seconds(4))
            }
        }
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Error") {
    let feed = DictationFeed()
    feed.setPhase(.error(.microphonePermissionDenied))
    return StageCardOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}
