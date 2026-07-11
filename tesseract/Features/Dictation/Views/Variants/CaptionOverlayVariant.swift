//
//  CaptionOverlayVariant.swift
//  tesseract
//

import AppKit
import SwiftUI

// MARK: - Metrics

/// The caption bar's sizes — the captions-first pole of the overlay
/// explorations (map #283), and the first variant built around the feed's
/// **Live Partial** signal (ticket #291): a cinema-subtitle glass bar,
/// bottom-centre, that shows the user's words as they speak.
///
/// The stability contract lives here: the text block reserves its two-line
/// height from the first caption on, and none of the frame math ever reads
/// the words — partials rewrite themselves wholesale, so a layout that
/// depended on text length would bounce on every revision.
///
/// `nonisolated` so it escapes the build's MainActor default isolation —
/// that's what lets ``OverlayPlacement/caption`` and its tests run off the
/// main actor. Everything in here is a plain literal; the font-measuring
/// tail fitter lives in `CaptionTailFitter` (main-actor, view-side).
nonisolated enum CaptionMetrics {
    /// The fixed panel canvas. The bar is bottom-anchored inside it with
    /// ``riseTravel`` of margin below (the entrance rises through it without
    /// window clipping) and spring/antialiasing headroom above and beside.
    static let canvasSize = CGSize(width: 560, height: 86)
    /// The bar's *visual* rest inset above the visible frame's bottom. The
    /// canvas bottom sits ``riseTravel`` lower so the rise stays in-window.
    static let bottomInset: CGFloat = 64
    /// How far the entrance rises (and the exit sinks) — also the bar's
    /// bottom margin inside the canvas, so the two can't drift apart.
    static let riseTravel: CGFloat = 14
    /// One radius for every stage: at the strips' 36 pt height it renders as
    /// a capsule, at the bar's height as a rounded rect — same shape value,
    /// so the strip→bar growth is a pure frame morph, never a shape swap.
    static let cornerRadius: CGFloat = 18

    /// The slim listening strip — recording before the first decode lands
    /// (or with streaming unavailable): dot + shimmer line, no fake words.
    static let stripSize = CGSize(width: 210, height: 36)
    /// The thinking strip — processing/proofreading when no caption was
    /// ever held this take: the classic dots treatment (plus the wand).
    static let thinkingSize = CGSize(width: 136, height: 36)

    /// The caption bar. Height = ``textBlockHeight`` + 2 × 13 pt padding.
    static let barSize = CGSize(width: 540, height: 64)
    static let barHorizontalPadding: CGFloat = 18
    static let barVerticalPadding: CGFloat = 13
    /// The fixed leading gutter column — present (even empty) in every
    /// caption stage, so the words' origin never moves when the glyph
    /// changes (red dot → wand → checkmark).
    static let gutterWidth: CGFloat = 22
    static let gutterSpacing: CGFloat = 10
    /// The gutter glyph slot, top-aligned with the first caption line — a
    /// glyph centred in it sits on the line's optical centre.
    static let gutterGlyphHeight: CGFloat = 20

    /// Caption type: one 15 pt medium voice; hierarchy by vibrancy only.
    static let captionFontSize: CGFloat = 15
    static let captionLineSpacing: CGFloat = 2
    /// Two caption lines at 15 pt medium + 2 pt line spacing — 18 + 2 + 18,
    /// measured against SF Pro on macOS 26. Reserved from the first caption
    /// on; if a future system font runs taller, glyphs bleed a point into
    /// the bottom padding — never a layout shift (the tail fitter measures
    /// its own two-line capacity live, so trimming stays correct).
    static let textBlockHeight: CGFloat = 38

    /// The caption block's layout width: bar minus padding, gutter, and the
    /// gutter gap. The tail fitter measures against exactly this width.
    static let captionTextWidth: CGFloat =
        barSize.width - 2 * barHorizontalPadding - gutterWidth - gutterSpacing
}

// MARK: - Tail fitting

/// Fits a Live Partial into the caption block, tail first (ticket #291):
/// captions follow the *most recent* words, so overflow truncates at the
/// HEAD — "… the last words you said" — never with a mid-string ellipsis.
///
/// Measurement, not heuristics: a binary search over head-drop counts, each
/// candidate measured with the caption's real font at the caption's real
/// width (AppKit string measurement has been thread-safe since 10.4, but
/// this stays on the main actor — the view is its only caller and the
/// attribute dictionary is not Sendable).
private enum CaptionTailFitter {
    private static let attributes: [NSAttributedString.Key: Any] = {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = CaptionMetrics.captionLineSpacing
        return [
            .font: NSFont.systemFont(ofSize: CaptionMetrics.captionFontSize, weight: .medium),
            .paragraphStyle: style,
        ]
    }()

    /// The real two-line capacity, measured once with the same attributes
    /// `fits` uses — self-consistent even if the system font's metrics ever
    /// drift from the hardcoded ``CaptionMetrics/textBlockHeight``.
    private static let twoLineHeight: CGFloat = height(of: "Ag\nAg", width: 1_000)

    /// The largest word-boundary tail of `text` that fits two caption
    /// lines, "… "-prefixed when anything was dropped. Whitespace runs are
    /// flattened — a caption is one flow of speech.
    static func tailFit(_ text: String) -> String {
        let words = text.split(whereSeparator: \.isWhitespace)
        guard !words.isEmpty else { return "" }
        let flattened = words.joined(separator: " ")
        if fits(flattened) { return flattened }
        // A single unbreakable monster word: hand it to the Text backstop
        // (`lineLimit(2)` + head truncation keeps the tail visible).
        guard words.count > 1 else { return flattened }
        func candidate(_ drop: Int) -> String {
            "… " + words[drop...].joined(separator: " ")
        }
        // Binary search the smallest drop that fits — dropping more words
        // only shrinks the text, so `fits` is monotone over `drop`.
        var low = 1
        var high = words.count - 1
        guard fits(candidate(high)) else { return candidate(high) }
        while low < high {
            let mid = (low + high) / 2
            if fits(candidate(mid)) { high = mid } else { low = mid + 1 }
        }
        return candidate(low)
    }

    private static func fits(_ text: String) -> Bool {
        height(of: text, width: CaptionMetrics.captionTextWidth) <= twoLineHeight + 0.5
    }

    private static func height(of text: String, width: CGFloat) -> CGFloat {
        (text as NSString).boundingRect(
            with: NSSize(width: width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: attributes
        ).height
    }
}

// MARK: - Placement

nonisolated extension OverlayPlacement {
    /// The caption bar's canvas: bottom-centre, like a cinema subtitle. The
    /// canvas origin sits `riseTravel` below the bar's visual rest inset —
    /// the bar is bottom-anchored `riseTravel` above the canvas floor, so
    /// its resting inset is exactly ``CaptionMetrics/bottomInset`` and the
    /// entrance can rise from below without leaving the fixed window.
    static let caption = OverlayPlacement(
        frame: { geometry in
            let size = CaptionMetrics.canvasSize
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.midX - size.width / 2,
                y: visible.minY + CaptionMetrics.bottomInset - CaptionMetrics.riseTravel,
                width: size.width,
                height: size.height
            )
        }
    )
}

// MARK: - View

/// The **caption** Overlay Variant (map #283, ticket #291): live captions of
/// what ASR hears, in a subtitle bar bottom-centre. The words are the whole
/// show — everything else is quiet chrome around them: a red dot while the
/// mic is hot, a wand while the Proofread Pass polishes, a green check when
/// the sentence settles into its committed form.
///
/// Same glass discipline as the classic pill: one pure untinted `.regular`
/// glass surface whose *frame* morphs between the strip and the bar; state
/// reads through vibrant content, never through a tint. Same motion
/// discipline: the panel canvas is fixed; entrance rises + fades on one
/// spring, partial revisions crossfade in place (~120 ms), and the layout
/// never moves with the words — the two-line block is reserved from the
/// first caption on. Reduce Motion turns every spring into a plain fade and
/// makes revisions swap instantly.
struct CaptionOverlayView: View {
    /// The Overlay Feed — the only pipeline surface this view sees.
    var feed: DictationFeed
    /// The overlay action surface (ticket #289) — invoked only from the
    /// lingering-beat affordances, the panel's sole clickable window.
    var actions: OverlayActions = .none

    /// Phase/beat mirrors of the feed, mutated inside `withAnimation` so
    /// frame morphs, content crossfades, and mount/unmount ride one spring.
    @State private var shownPhase: DictationFeed.Phase = .idle
    @State private var shownBeat: DictationFeed.Beat?
    /// The **held caption**: the last tail-fitted Live Partial. Mirrored
    /// into `@State` because it must outlive the signal — the feed clears
    /// `partial` when recording ends, but the words stay on screen (dimmed)
    /// through processing and proofreading until the commit resolves them.
    /// Take-scoped: reset on every `.recording` entry.
    @State private var heldCaption: String?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GlassEffectContainer {
            ZStack(alignment: .bottom) {
                if let stage {
                    surface(for: stage)
                        .padding(.bottom, CaptionMetrics.riseTravel)
                        .transition(entranceTransition)
                }
            }
            .frame(
                width: CaptionMetrics.canvasSize.width,
                height: CaptionMetrics.canvasSize.height,
                alignment: .bottom
            )
        }
        .onChange(of: feed.phase) { _, newPhase in
            apply(newPhase)
        }
        .onChange(of: feed.beat) { _, beat in
            applyBeat(beat)
        }
        .onChange(of: feed.partial) { _, partial in
            applyPartial(partial)
        }
        .onAppear {
            apply(feed.phase, animated: false)
            applyPartial(feed.partial, animated: false)
        }
    }

    // MARK: - Stage machine

    /// What the bar renders right now — a live phase always outranks a
    /// lingering beat; `nil` (idle, no beat) renders nothing. `captioning`
    /// is one case for both recording and the held stages so the caption
    /// grid keeps one view identity across the recording→processing hop —
    /// the words must dim in place, never re-fade against themselves.
    private enum Stage: Equatable {
        case listening
        case thinking(proofreading: Bool)
        case captioning(text: String, mode: CaptionMode)
        case committed(text: String, edits: [WordEdit], beatID: UInt64)
        case rejected(raw: String, reason: String, beatID: UInt64)
        case error(DictationError)
    }

    /// How the caption grid narrates around the words: `live` while the mic
    /// is hot, `held` while the pipeline works on the finished take.
    private enum CaptionMode: Equatable {
        case live
        case held(proofreading: Bool)
    }

    private var stage: Stage? {
        switch shownPhase {
        case .recording:
            guard let heldCaption else { return .listening }
            return .captioning(text: heldCaption, mode: .live)
        case .processing:
            guard let heldCaption else { return .thinking(proofreading: false) }
            return .captioning(text: heldCaption, mode: .held(proofreading: false))
        case .proofreading:
            guard let heldCaption else { return .thinking(proofreading: true) }
            return .captioning(text: heldCaption, mode: .held(proofreading: true))
        case .error(let error):
            return .error(error)
        case .idle:
            guard let shownBeat else { return nil }
            switch shownBeat.outcome {
            case .committed(let text, _, let edits):
                return .committed(text: text, edits: edits, beatID: shownBeat.id)
            case .rejected(let raw, let reason):
                return .rejected(raw: raw, reason: reason, beatID: shownBeat.id)
            case .empty, .cancelled, .superseded:
                return nil
            }
        }
    }

    // MARK: - Motion

    /// Entrance rides a snappy spring (captions must read as "on" within
    /// ~100 ms of the press); exits and dismissals a slightly softer one.
    /// Reduce Motion swaps every spring for a plain fade.
    private func transitionAnimation(showing: Bool) -> Animation {
        if reduceMotion { return .easeInOut(duration: 0.18) }
        return showing
            ? .spring(response: 0.2, dampingFraction: 0.75)
            : .spring(response: 0.25, dampingFraction: 0.8)
    }

    /// The strip→bar growth when the first caption of a take lands.
    private var morphAnimation: Animation {
        reduceMotion
            ? .easeInOut(duration: 0.18)
            : .spring(response: 0.25, dampingFraction: 0.8)
    }

    /// Rise + fade from the bottom; the exit reverses it (sink + fade). The
    /// rise distance doubles as the bar's canvas margin, so the entrance
    /// starts exactly on the canvas floor — never clipped by the window.
    private var entranceTransition: AnyTransition {
        reduceMotion
            ? .opacity
            : .opacity.combined(with: .offset(y: CaptionMetrics.riseTravel))
    }

    private func apply(_ newPhase: DictationFeed.Phase, animated: Bool = true) {
        guard shownPhase != newPhase else { return }
        let animation = animated ? transitionAnimation(showing: newPhase != .idle) : nil
        withAnimation(animation) {
            // A live phase always outranks a lingering beat.
            if newPhase != .idle { shownBeat = nil }
            // Captions are take-scoped: a fresh recording starts from the
            // listening strip, never from a stale caption.
            if newPhase == .recording { heldCaption = nil }
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
        withAnimation(transitionAnimation(showing: true)) {
            shownBeat = beat
        }
        Task {
            try? await Task.sleep(for: DictationFeed.affordanceLinger)
            dismissBeat(ifStill: beat.id)
        }
    }

    private func dismissBeat(ifStill id: UInt64? = nil) {
        if let id, shownBeat?.id != id { return }
        withAnimation(transitionAnimation(showing: false)) {
            shownBeat = nil
        }
    }

    /// Publishes a Live Partial revision into the held caption. Revisions
    /// crossfade in place (~120 ms; Reduce Motion swaps instantly); the
    /// FIRST caption of a take instead rides the strip→bar morph spring. A
    /// `nil` mid-take (decoder busy) *holds* the last caption — dropping
    /// back to the strip would bounce the bar for nothing, and the next
    /// revision replaces the words wholesale anyway.
    private func applyPartial(_ partial: String?, animated: Bool = true) {
        // Guard on the feed's phase, not the mirrored one: a partial can
        // land in the same update batch as the recording flip.
        guard feed.phase == .recording, let partial else { return }
        let fitted = CaptionTailFitter.tailFit(partial)
        guard !fitted.isEmpty, fitted != heldCaption else { return }
        guard animated else {
            heldCaption = fitted
            return
        }
        if heldCaption == nil {
            withAnimation(morphAnimation) { heldCaption = fitted }
        } else if reduceMotion {
            heldCaption = fitted
        } else {
            withAnimation(.easeInOut(duration: 0.12)) { heldCaption = fitted }
        }
    }

    // MARK: - The surface

    /// The one glass surface. Its *frame* morphs between the strip and bar
    /// sizes (the panel is fixed, so this is pure content layout); the
    /// ZStack overlays outgoing and incoming stage content, so stage hops
    /// read as a crossfade inside the stretching glass. Pure untinted
    /// `.regular` — the material *is* the bar; state color lives in the
    /// content (same owner-selected discipline as the classic pill).
    private func surface(for stage: Stage) -> some View {
        let size = surfaceSize(for: stage)
        return ZStack {
            content(for: stage)
        }
        .frame(width: size.width, height: size.height)
        .glassEffect(.regular, in: .rect(cornerRadius: CaptionMetrics.cornerRadius))
    }

    private func surfaceSize(for stage: Stage) -> CGSize {
        switch stage {
        case .listening:
            return CaptionMetrics.stripSize
        case .thinking:
            return CaptionMetrics.thinkingSize
        case .captioning, .committed, .rejected, .error:
            return CaptionMetrics.barSize
        }
    }

    @ViewBuilder
    private func content(for stage: Stage) -> some View {
        switch stage {
        case .listening:
            listeningStrip
        case .thinking(let proofreading):
            thinkingStrip(proofreading: proofreading)
        case .captioning(let text, let mode):
            captionGrid(text: text, mode: mode)
        case .committed(let text, let edits, let beatID):
            // Keyed on the beat so two commits in a row read as two beats.
            committedGrid(text: text, edits: edits)
                .id(beatID)
        case .rejected(let raw, let reason, let beatID):
            rejectedGrid(raw: raw, reason: reason)
                .id(beatID)
        case .error(let error):
            errorGrid(for: error)
        }
    }

    // MARK: - Strips (no caption)

    /// Recording before the first decode lands (or with streaming off): a
    /// slim strip that reads "listening" without pretending to caption —
    /// the pulsing tell plus a shimmering band line, no fake words.
    private var listeningStrip: some View {
        HStack(spacing: 10) {
            CaptionPulseDot(feed: feed)
            CaptionShimmerLine(feed: feed)
        }
        .padding(.horizontal, 16)
        .accessibilityLabel("Recording")
    }

    /// Processing when no caption was ever held this take: the strip morphs
    /// into the classic dots treatment; proofreading adds the twinkling
    /// wand — one visual family with the classic pill. The 60 fps timeline
    /// lives inside, so only the dots re-evaluate per frame.
    private func thinkingStrip(proofreading: Bool) -> some View {
        HStack(spacing: 8) {
            if proofreading {
                Image(systemName: "wand.and.sparkles")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .symbolEffect(.variableColor.iterative, isActive: !reduceMotion)
                    .accessibilityHidden(true)
            }
            if reduceMotion {
                ProgressView()
                    .controlSize(.small)
            } else {
                TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
                    ProcessingDotsView(time: timeline.date.timeIntervalSinceReferenceDate)
                        .frame(height: 12)
                }
            }
        }
        .accessibilityLabel(proofreading ? "Proofreading" : "Transcribing")
    }

    // MARK: - The caption grid

    /// The live-caption grid: fixed gutter, fixed two-line text block. None
    /// of the frame math reads the words — a revision can only ever change
    /// glyphs, so the layout cannot bounce (the hard part of ticket #291).
    /// The gutter narrates around the words: red dot while the mic is hot,
    /// nothing while transcription finishes (the dot's exit *is* the "mic
    /// closed" tell), the twinkling wand while the Proofread Pass polishes.
    private func captionGrid(text: String, mode: CaptionMode) -> some View {
        HStack(alignment: .top, spacing: CaptionMetrics.gutterSpacing) {
            gutter {
                switch mode {
                case .live:
                    CaptionPulseDot(feed: feed)
                case .held(let proofreading):
                    if proofreading {
                        Image(systemName: "wand.and.sparkles")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .symbolEffect(.variableColor.iterative, isActive: !reduceMotion)
                    }
                }
            }
            captionText(text, mode: mode)
                .frame(
                    width: CaptionMetrics.captionTextWidth,
                    height: CaptionMetrics.textBlockHeight,
                    alignment: .topLeading)
        }
        .padding(.horizontal, CaptionMetrics.barHorizontalPadding)
        .padding(.vertical, CaptionMetrics.barVerticalPadding)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel(text: text, mode: mode))
    }

    /// The caption itself. `contentTransition(.opacity)` is what blends the
    /// wholesale partial revisions (the 120 ms transaction comes from
    /// `applyPartial`); held stages dim it to secondary and run the shimmer
    /// sweep over the glyphs while the pipeline works.
    private func captionText(_ text: String, mode: CaptionMode) -> some View {
        let dimmed = mode != .live
        return captionTextBase(text)
            .foregroundStyle(dimmed ? AnyShapeStyle(.secondary) : AnyShapeStyle(.primary))
            .contentTransition(.opacity)
            .overlay(alignment: .topLeading) {
                if dimmed && !reduceMotion {
                    CaptionShimmerSweep()
                        .mask { captionTextBase(text) }
                        .allowsHitTesting(false)
                }
            }
    }

    /// The bare caption glyphs — built twice (display + shimmer mask), so
    /// both layouts are guaranteed identical. Head truncation is only a
    /// backstop: `CaptionTailFitter` already fitted the tail, so on the
    /// rare point where SwiftUI wraps a hair tighter than the measurement,
    /// the visible loss is still head-side — the newest words always win.
    private func captionTextBase(
        _ text: String, truncation: Text.TruncationMode = .head
    ) -> some View {
        Text(text)
            .font(.system(size: CaptionMetrics.captionFontSize, weight: .medium))
            .lineSpacing(CaptionMetrics.captionLineSpacing)
            .lineLimit(2)
            .truncationMode(truncation)
            .multilineTextAlignment(.leading)
    }

    /// The fixed leading gutter slot — rendered in every caption stage
    /// (even empty), so the words' origin never moves as the glyph changes.
    private func gutter(@ViewBuilder _ glyph: () -> some View) -> some View {
        ZStack {
            glyph()
        }
        .frame(width: CaptionMetrics.gutterWidth, height: CaptionMetrics.gutterGlyphHeight)
        .accessibilityHidden(true)
    }

    private func accessibilityLabel(text: String, mode: CaptionMode) -> String {
        switch mode {
        case .live:
            return "Recording: \(text)"
        case .held(let proofreading):
            return proofreading ? "Proofreading: \(text)" : "Transcribing: \(text)"
        }
    }

    // MARK: - Lingering beats (ticket #289)

    /// The moment of truth: the held caption resolves into the committed
    /// text — a crossfade inside the same fixed bar, green check where the
    /// dot lived, the polish chip when the Proofread Pass edited, flywheel
    /// affordances trailing. The panel is clickable for exactly this
    /// linger; a stray click anywhere on the bar dismisses it.
    private func committedGrid(text: String, edits: [WordEdit]) -> some View {
        HStack(alignment: .top, spacing: CaptionMetrics.gutterSpacing) {
            gutter {
                Image(systemName: "checkmark")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.green)
            }
            // Tail truncation here, not head: a committed sentence reads
            // from its start — tail-follow is a *live* affordance.
            captionTextBase(text, truncation: .tail)
                .frame(height: CaptionMetrics.textBlockHeight, alignment: .topLeading)
                .accessibilityLabel(
                    edits.isEmpty
                        ? "Inserted: \(text)"
                        : "Inserted, proofreader polished \(edits.count) words: \(text)")
            Spacer(minLength: 8)
            HStack(spacing: 4) {
                if !edits.isEmpty {
                    polishChip(count: edits.count)
                }
                beatButton("flag", label: "Flag as wrong") {
                    actions.flagLastTakeWrong()
                    dismissBeat()
                }
                beatButton("pencil", label: "Edit in history") {
                    actions.editLastTake()
                    dismissBeat()
                }
            }
            .frame(height: CaptionMetrics.textBlockHeight)
        }
        .padding(.horizontal, CaptionMetrics.barHorizontalPadding)
        .padding(.vertical, CaptionMetrics.barVerticalPadding)
        .contentShape(Rectangle())
        .onTapGesture { dismissBeat() }
    }

    /// The lingering rejected bar. Passive-first: the press is the retry;
    /// the raw take shows in tertiary so the user can judge the rejection,
    /// and the one affordance inserts it anyway (which also flags the pass).
    private func rejectedGrid(raw: String, reason: String) -> some View {
        HStack(alignment: .top, spacing: CaptionMetrics.gutterSpacing) {
            gutter {
                Image(systemName: "arrow.uturn.backward")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.orange)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Didn't catch that")
                    .font(.system(size: CaptionMetrics.captionFontSize, weight: .medium))
                    .lineLimit(1)
                Text(raw)
                    .font(.system(size: 13))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
            .frame(height: CaptionMetrics.textBlockHeight, alignment: .topLeading)
            .accessibilityLabel(
                "Transcription rejected: \(reason) — hold the hotkey to retry")
            Spacer(minLength: 8)
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
            .frame(height: CaptionMetrics.textBlockHeight)
        }
        .padding(.horizontal, CaptionMetrics.barHorizontalPadding)
        .padding(.vertical, CaptionMetrics.barVerticalPadding)
        .contentShape(Rectangle())
        .onTapGesture { dismissBeat() }
    }

    /// The polish tell on a committed beat: the wand plus how many words
    /// the Proofread Pass swapped. Count, not diff chips — the full diff
    /// lives in the informative variants; a caption bar keeps chrome quiet.
    private func polishChip(count: Int) -> some View {
        HStack(spacing: 3) {
            Image(systemName: "wand.and.sparkles")
                .font(.system(size: 9, weight: .semibold))
            Text("\(count)")
                .font(.system(size: 10, weight: .semibold))
                .monospacedDigit()
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(.quaternary, in: Capsule())
        .accessibilityHidden(true)  // narrated by the committed-text label
    }

    /// One small icon affordance on a lingering beat: plain style (the
    /// glass bar is the chrome), secondary ink, generous hit target.
    private func beatButton(
        _ systemName: String, label: String, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 24, height: 24)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }

    // MARK: - Error

    /// Errors take the caption bar too: description in the caption's voice,
    /// orange accent in the gutter (content color, never a glass tint), the
    /// recovery line beneath when the error knows one.
    private func errorGrid(for error: DictationError) -> some View {
        let message = error.errorDescription ?? "Something went wrong."
        let suggestion = error.recoverySuggestion
        return HStack(alignment: .top, spacing: CaptionMetrics.gutterSpacing) {
            gutter {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.orange)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(message)
                    .font(.system(size: CaptionMetrics.captionFontSize, weight: .medium))
                    .lineLimit(suggestion == nil ? 2 : 1)
                    .minimumScaleFactor(0.9)
                if let suggestion {
                    Text(suggestion)
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .minimumScaleFactor(0.9)
                }
            }
            .multilineTextAlignment(.leading)
            .frame(height: CaptionMetrics.textBlockHeight, alignment: .topLeading)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, CaptionMetrics.barHorizontalPadding)
        .padding(.vertical, CaptionMetrics.barVerticalPadding)
        .accessibilityLabel(suggestion.map { "\(message) \($0)" } ?? message)
    }
}

// MARK: - Meter leaves

/// The recording tell — the one place raw `level` moves geometry. Reads the
/// feed itself so the ~47 Hz meter cadence invalidates only this leaf; the
/// caption grid and the glass never re-diff on a meter tick.
private struct CaptionPulseDot: View {
    var feed: DictationFeed
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let level = CGFloat(min(max(feed.level, 0), 1))
        Circle()
            .fill(.red)
            .frame(width: 8, height: 8)
            .scaleEffect(reduceMotion ? 1 : 1 + level * 0.35)
            .animation(reduceMotion ? nil : .linear(duration: 0.05), value: level)
            .accessibilityHidden(true)
    }
}

/// The listening strip's band line: the eight log-spaced spectrum bands as
/// a flat row of segments that brighten with energy — a shimmer, not a
/// meter; the heights never move, so the strip holds perfectly still while
/// it listens. Reads the feed itself (meter cadence stays in this leaf).
/// Data display, so the values keep updating under Reduce Motion — only the
/// interpolation between frames is dropped.
private struct CaptionShimmerLine: View {
    var feed: DictationFeed
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let segmentWidth: CGFloat = 16
    private let segmentHeight: CGFloat = 3
    private let segmentSpacing: CGFloat = 3

    var body: some View {
        let bands = feed.spectrum
        HStack(spacing: segmentSpacing) {
            ForEach(0..<MeterFrame.bandCount, id: \.self) { index in
                let value = index < bands.count ? Double(min(max(bands[index], 0), 1)) : 0
                Capsule()
                    .fill(.red.opacity(0.3 + 0.6 * value))
                    .frame(width: segmentWidth, height: segmentHeight)
            }
        }
        .animation(reduceMotion ? nil : .linear(duration: 0.08), value: bands)
        .accessibilityHidden(true)
    }
}

// MARK: - Shimmer sweep

/// The held caption's "still working" tell: a soft specular band sweeping
/// across the glyphs (the caller masks it with the caption text) while
/// transcription finishes. The 40 fps timeline lives inside this leaf, so
/// per-frame work is one gradient offset — the text, grid, and glass sit
/// outside the closure. Never mounted under Reduce Motion.
private struct CaptionShimmerSweep: View {
    private static let bandWidth: CGFloat = 120
    /// One sweep every `period` seconds: travelling for `travelTime`, then
    /// resting off-glyph — activity without nagging.
    private static let period: Double = 2.6
    private static let travelTime: Double = 1.5

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 40.0)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [.clear, .white.opacity(0.45), .clear],
                        startPoint: .leading,
                        endPoint: .trailing)
                )
                .frame(width: Self.bandWidth)
                .offset(x: Self.offset(at: time))
        }
    }

    /// Band origin for a wall-clock time — all-Double math, converted to
    /// CGFloat once at the end (the Release type-checker discipline).
    private static func offset(at time: Double) -> CGFloat {
        let phase: Double = time.truncatingRemainder(dividingBy: period)
        let progress: Double = min(1, phase / travelTime)
        let travel: Double = Double(CaptionMetrics.captionTextWidth) + 2 * Double(bandWidth)
        let origin: Double = -Double(bandWidth) + progress * travel
        return CGFloat(origin)
    }
}

// MARK: - Registration

extension OverlayVariants {
    /// The live-captions subtitle bar (ticket #291) — the first variant
    /// built around the Live Partial signal; `usesLivePartials` is what
    /// turns the partial pump on while this variant is live. Defined here
    /// beside the view; the `all` roster in `OverlayVariants.swift` decides
    /// what ships.
    static let caption = OverlayVariant(
        id: "caption",
        displayName: "Caption",
        placement: .caption,
        usesLivePartials: true
    ) { feed, actions in
        AnyView(CaptionOverlayView(feed: feed, actions: actions))
    }
}

// MARK: - Previews

private extension View {
    /// Busy backdrop for judging glass legibility in previews.
    func captionPreviewBackdrop() -> some View {
        padding(30)
            .background(
                LinearGradient(
                    colors: [.blue, .purple, .orange],
                    startPoint: .topLeading, endPoint: .bottomTrailing
                ))
    }
}

/// Synthesized speech-shaped meter frames so preview dots and strips move.
private func pumpPreviewMeters(into feed: DictationFeed) async {
    var tick = 0.0
    while !Task.isCancelled {
        tick += 1.0 / 30.0
        let level: Double = 0.45 + 0.3 * sin(tick * 2.2) + 0.2 * sin(tick * 5.7)
        let bands = (0..<MeterFrame.bandCount).map { band -> Float in
            let phase = Double(band)
            let wave: Double = 0.5 + 0.45 * sin(tick * (1.3 + phase * 0.6) + phase)
            return Float(max(0, wave))
        }
        feed.apply(MeterFrame(level: Float(min(max(level, 0), 1)), bands: bands))
        try? await Task.sleep(for: .milliseconds(33))
    }
}

#Preview("Recording — live captions") {
    let feed = DictationFeed()
    return CaptionOverlayView(feed: feed)
        .captionPreviewBackdrop()
        .task { await pumpPreviewMeters(into: feed) }
        .task {
            // A scripted take: the partial grows, REVISES an earlier word
            // ("nine" becomes "five"), then overflows into tail-follow.
            let snapshots: [String] = [
                "Let's",
                "Let's meet at nine",
                "Let's meet at nine to walk through",
                "Let's meet at five to walk through the launch plan",
                "Let's meet at five to walk through the launch plan and the open questions",
                "Let's meet at five to walk through the launch plan and the open questions "
                    + "before the demo",
                "Let's meet at five to walk through the launch plan and the open questions "
                    + "before the demo tomorrow morning",
                "Let's meet at five to walk through the launch plan and the open questions "
                    + "before the demo tomorrow morning so we can lock the agenda",
                "Let's meet at five to walk through the launch plan and the open questions "
                    + "before the demo tomorrow morning so we can lock the agenda and split "
                    + "the follow-ups",
            ]
            while !Task.isCancelled {
                feed.setPhase(.recording)
                for snapshot in snapshots {
                    feed.setPartial(snapshot)
                    try? await Task.sleep(for: .milliseconds(700))
                }
                feed.setPhase(.idle)
                try? await Task.sleep(for: .seconds(1))
            }
        }
}

#Preview("Recording — no partials") {
    let feed = DictationFeed()
    feed.setPhase(.recording)
    return CaptionOverlayView(feed: feed)
        .captionPreviewBackdrop()
        .task { await pumpPreviewMeters(into: feed) }
}

#Preview("Processing — held caption") {
    let feed = DictationFeed()
    return CaptionOverlayView(feed: feed)
        .captionPreviewBackdrop()
        .task { await pumpPreviewMeters(into: feed) }
        .task {
            // Record → caption → dimmed shimmer hold → wand — the arc that
            // shows the caption surviving the feed's partial clear.
            while !Task.isCancelled {
                feed.setPhase(.recording)
                feed.setPartial("Draft a quick note to the team about")
                try? await Task.sleep(for: .milliseconds(900))
                feed.setPartial("Draft a quick note to the team about the beta timeline")
                try? await Task.sleep(for: .milliseconds(900))
                feed.setPhase(.processing)
                try? await Task.sleep(for: .seconds(2.2))
                feed.setPhase(.proofreading)
                try? await Task.sleep(for: .seconds(2.2))
                feed.setPhase(.idle)
                try? await Task.sleep(for: .seconds(1))
            }
        }
}

#Preview("Committed — with edits") {
    let feed = DictationFeed()
    return CaptionOverlayView(feed: feed)
        .captionPreviewBackdrop()
        .task { await pumpPreviewMeters(into: feed) }
        .task {
            // The full arc, so the held caption visibly resolves into the
            // polished commit and the linger replays while designing.
            while !Task.isCancelled {
                feed.setPhase(.recording)
                feed.setPartial("Let's grab a peace of the market")
                try? await Task.sleep(for: .milliseconds(800))
                feed.setPartial("Let's grab a peace of the market before there launch")
                try? await Task.sleep(for: .milliseconds(800))
                feed.setPhase(.processing)
                try? await Task.sleep(for: .seconds(1.2))
                feed.setPhase(.proofreading)
                try? await Task.sleep(for: .seconds(1.2))
                feed.setPhase(.idle)
                feed.emit(
                    .committed(
                        text: "Let's grab a piece of the market before their launch.",
                        duration: 4.2,
                        edits: [
                            WordEdit(original: "peace", replacement: "piece"),
                            WordEdit(original: "there", replacement: "their"),
                        ]))
                try? await Task.sleep(for: .seconds(4))
            }
        }
}

#Preview("Rejected") {
    let feed = DictationFeed()
    return CaptionOverlayView(feed: feed)
        .captionPreviewBackdrop()
        .task {
            while !Task.isCancelled {
                feed.emit(
                    .rejected(
                        raw: "uh so um the thing with the stuff",
                        reason: "Unintelligible transcription"))
                try? await Task.sleep(for: .seconds(4))
            }
        }
}

#Preview("Error") {
    let feed = DictationFeed()
    feed.setPhase(.error(.microphonePermissionDenied))
    return CaptionOverlayView(feed: feed)
        .captionPreviewBackdrop()
}
