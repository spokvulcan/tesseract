//
//  OrbOverlayVariant.swift
//  tesseract
//

import AppKit
import SwiftUI

// MARK: - Metrics

/// The single source of truth for the orb variant's sizes.
///
/// The panel's window is fixed at ``canvasSize`` and never resizes (map #283)
/// — the orb, its breathing headroom, and the widest toast all fit inside the
/// canvas, so every state change is pure content layout.
///
/// `nonisolated` so it escapes the build's MainActor default isolation — that's
/// what lets ``OverlayPlacement/orb`` and its tests run off the main actor.
nonisolated enum OrbMetrics {
    /// The resting orb diameter; recording breathes it up to
    /// ``breathingMaxScale`` (a render transform — layout never moves).
    static let orbDiameter: CGFloat = 44
    /// The breath ceiling: the orb's scale at `level == 1`.
    static let breathingMaxScale: CGFloat = 1.15

    /// Every toast (commit affordances, rejection, error) shares one size so
    /// the unfurl reads identically across outcomes. Sized to the widest
    /// content (a two-line error) within the canvas budget.
    static let toastSize = CGSize(width: 248, height: 40)
    /// Gap between the toast's trailing edge and the orb — small enough that
    /// the two glass shapes blend into one liquid form while both are up.
    static let toastGap: CGFloat = 8

    /// Padding around the content row inside the canvas — the headroom the
    /// breath (+15 % scale), the rejection shake (±5 pt), and entrance
    /// overshoot render into, since none of them may leave the fixed canvas.
    static let contentPadding: CGFloat = 8

    /// The fixed panel canvas: padding + toast + gap + orb + padding wide,
    /// with vertical headroom above the bottom-anchored content row.
    static let canvasSize = CGSize(width: 320, height: 110)

    /// The canvas's inset from the visible frame's trailing/bottom edges.
    /// Together with ``contentPadding`` it parks the orb itself 24 pt from
    /// both screen edges.
    static let canvasScreenInset: CGFloat = 16

    // The recording face's radial spectrum: spokes grow outward from a quiet
    // inner ring; the maximum reach (inner + max length + round cap) stays
    // well inside the orb's glass rim.
    static let spokeFieldSize: CGFloat = 36
    static let spokeInnerRadius: CGFloat = 5
    static let spokeMinLength: CGFloat = 3
    static let spokeMaxLength: CGFloat = 10
    static let spokeWidth: CGFloat = 3
}

// MARK: - Placement

nonisolated extension OverlayPlacement {
    /// The orb's canvas: parked in the bottom-right corner of the visible
    /// frame, a quiet companion presence out of the typing sightline. The orb
    /// anchors at the canvas's bottom-trailing corner, so the toast unfurls
    /// leftward *into* the canvas and the orb's visual inset never moves.
    static let orb = OverlayPlacement(
        frame: { geometry in
            let size = OrbMetrics.canvasSize
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.maxX - size.width - OrbMetrics.canvasScreenInset,
                y: visible.minY + OrbMetrics.canvasScreenInset,
                width: size.width,
                height: size.height
            )
        }
    )
}

// MARK: - View

/// The corner presence orb — an **Overlay Variant** (map #283): a small
/// Liquid Glass sphere in the bottom-right corner, Siri-orb energy with macOS
/// restraint. One pure (untinted) `.glassEffect` circle whose *content*
/// carries the state — red spectrum spokes while recording, a slow dot orbit
/// while processing, a twinkling sparkle at the orbit's head while
/// proofreading — plus a glass toast that unfurls leftward from the orb for
/// terminal beats and errors, then folds back in. Glass never tints for state.
///
/// The variant owns *all* motion (map #283): entrance/exit, the breath, the
/// unfurl, and the rejection shake are SwiftUI inside the panel's fixed
/// canvas — the panel never fades or resizes.
struct OrbOverlayView: View {
    /// The Overlay Feed — the only pipeline surface this view sees.
    var feed: DictationFeed
    /// The overlay action surface (ticket #289) — what the lingering toast's
    /// click affordances invoke. Clicks only; the overlay stays keyboard-free.
    var actions: OverlayActions = .none

    /// The phase the orb currently renders — updated inside `withAnimation`,
    /// so mount/unmount and face swaps all ride one spring.
    @State private var shownPhase: DictationFeed.Phase = .idle
    /// A lingering terminal beat (map #283): committed and rejected takes
    /// hold the affordance toast for ``DictationFeed/affordanceLinger`` after
    /// the phase returned to idle — flag/edit on a commit, insert-raw on a
    /// rejection. Passive: it folds on its own; the press is the retry.
    @State private var shownBeat: DictationFeed.Beat?
    /// Bumped once per rejected beat to fire the orb's soft "no" shake; never
    /// bumped under Reduce Motion, so the phase animator stays at rest.
    @State private var shakeToken = 0
    /// One glass namespace for the orb and the toast, so the appearing toast
    /// morphs out of the orb's glass — and folds back into it — instead of
    /// fading in beside it as an unrelated pane.
    @Namespace private var glassSpace
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GlassEffectContainer(spacing: 12) {
            HStack(alignment: .center, spacing: OrbMetrics.toastGap) {
                Spacer(minLength: 0)
                if case .error(let error) = shownPhase {
                    errorToast(for: error)
                        .transition(toastTransition)
                } else if shownPhase == .idle, let shownBeat {
                    beatToast(for: shownBeat)
                        .transition(toastTransition)
                }
                if orbShown {
                    orb
                        .transition(
                            reduceMotion
                                ? .opacity
                                : .scale(scale: 0.5).combined(with: .opacity)
                        )
                }
            }
            .padding(OrbMetrics.contentPadding)
            .frame(
                width: OrbMetrics.canvasSize.width,
                height: OrbMetrics.canvasSize.height,
                alignment: .bottomTrailing
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

    /// The orb stays up through every live phase *and* while a beat's toast
    /// lingers — the toast folds back into the orb before the orb itself exits.
    private var orbShown: Bool {
        shownPhase != .idle || shownBeat != nil
    }

    /// The toast's unfurl: horizontal growth anchored at the orb's side, so
    /// it reads as unrolling out of the orb; the `glassEffectID` morph carries
    /// the glass itself between the circle and the capsule.
    private var toastTransition: AnyTransition {
        reduceMotion
            ? .opacity
            : .scale(scale: 0.4, anchor: .trailing).combined(with: .opacity)
    }

    private func apply(_ newPhase: DictationFeed.Phase, animated: Bool = true) {
        guard shownPhase != newPhase else { return }
        if reduceMotion || !animated {
            if newPhase != .idle { shownBeat = nil }
            shownPhase = newPhase
            return
        }
        // Snappy in (the orb must read as "on" within ~100 ms of the press or
        // dictation feels laggy), slightly softer out.
        let showing = newPhase != .idle
        withAnimation(
            .spring(response: showing ? 0.2 : 0.25, dampingFraction: showing ? 0.75 : 0.85)
        ) {
            // A live phase always outranks a lingering beat — cleared inside
            // the same spring so the toast retracts as the new phase lands.
            if newPhase != .idle { shownBeat = nil }
            shownPhase = newPhase
        }
    }

    private func applyBeat(_ beat: DictationFeed.Beat?) {
        guard let beat else { return }
        switch beat.outcome {
        case .rejected:
            // One soft sideways shake ("no"). The token stays put under
            // Reduce Motion, so the phase animator never leaves rest.
            if !reduceMotion { shakeToken += 1 }
        case .committed:
            break
        case .empty, .cancelled, .superseded:
            return
        }
        withAnimation(reduceMotion ? nil : .spring(response: 0.25, dampingFraction: 0.8)) {
            shownBeat = beat
        }
        Task {
            try? await Task.sleep(for: DictationFeed.affordanceLinger)
            dismissBeat(ifStill: beat.id)
        }
    }

    private func dismissBeat(ifStill id: UInt64? = nil) {
        if let id, shownBeat?.id != id { return }
        withAnimation(reduceMotion ? nil : .spring(response: 0.3, dampingFraction: 0.85)) {
            shownBeat = nil
        }
    }

    // MARK: Orb

    /// The orb: one pure `.regular` glass circle (state reads through the
    /// vibrant face content, never a glass tint) that breathes with the mic
    /// level while recording and shakes once on a rejected take.
    private var orb: some View {
        OrbBreathingView(
            feed: feed,
            breathing: shownPhase == .recording && !reduceMotion
        ) {
            ZStack {
                orbFace
            }
            .frame(width: OrbMetrics.orbDiameter, height: OrbMetrics.orbDiameter)
            .glassEffect(.regular, in: .circle)
            .glassEffectID("orb", in: glassSpace)
        }
        .phaseAnimator(
            [0.0, -5.0, 4.0, -2.0], trigger: shakeToken
        ) { view, offset in
            view.offset(x: offset)
        } animation: { _ in
            .easeInOut(duration: 0.09)
        }
    }

    /// The orb's face for the rendered phase — or, at idle, the lingering
    /// beat's glyph while the toast alongside carries words and affordances.
    @ViewBuilder
    private var orbFace: some View {
        switch shownPhase {
        case .recording:
            // The spectrum subview reads the feed itself, so the ~47 Hz meter
            // cadence invalidates only the spokes — never the glass circle.
            OrbSpectrumView(feed: feed)
                .accessibilityLabel("Recording")
        case .processing, .proofreading:
            let proofreading = shownPhase == .proofreading
            if reduceMotion {
                // Same reduced path as the classic pill: the system indicator
                // for processing, a still wand for the Proofread Pass.
                if proofreading {
                    Image(systemName: "wand.and.sparkles")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("Proofreading")
                } else {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Transcribing")
                }
            } else {
                // The TimelineView lives *inside* the orb so the glass chrome
                // sits outside the 60 fps closure — only the orbit re-evaluates
                // per frame.
                TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
                    OrbOrbitView(
                        time: timeline.date.timeIntervalSinceReferenceDate,
                        proofreading: proofreading)
                }
                .accessibilityLabel(proofreading ? "Proofreading" : "Transcribing")
            }
        case .error:
            // Static by design — the toast alongside carries (and speaks)
            // the message; the orb just wears the warning.
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.orange)
                .accessibilityHidden(true)
        case .idle:
            if let shownBeat {
                beatGlyph(for: shownBeat.outcome)
            }
        }
    }

    /// The orb's face while a beat toast lingers: the outcome distilled to
    /// one glyph (decorative for accessibility — the labeled toast speaks).
    @ViewBuilder
    private func beatGlyph(for outcome: DictationFeed.Outcome) -> some View {
        switch outcome {
        case .committed(_, _, let edits):
            Image(systemName: edits.isEmpty ? "checkmark" : "wand.and.sparkles")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.green)
                .accessibilityHidden(true)
        case .rejected:
            Image(systemName: "arrow.uturn.backward")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.orange)
                .accessibilityHidden(true)
        case .empty, .cancelled, .superseded:
            EmptyView()
        }
    }

    // MARK: Toasts

    /// The lingering terminal-beat toast (ticket #289): the panel is clickable
    /// for exactly this window (one App Bindings rule), so these are the
    /// overlay's only interactive moments — one-click, no focus steal.
    @ViewBuilder
    private func beatToast(for beat: DictationFeed.Beat) -> some View {
        switch beat.outcome {
        case .committed(_, _, let edits):
            committedToast(edits: edits)
        case .rejected:
            rejectionToast
        case .empty, .cancelled, .superseded:
            EmptyView()
        }
    }

    /// Post-commit: names what happened (polish count when the Proofread
    /// Pass edited) and carries the flywheel affordances — flag "wrong" and
    /// "edit in history".
    private func committedToast(edits: [WordEdit]) -> some View {
        toastChrome {
            HStack(alignment: .center, spacing: 8) {
                Image(systemName: edits.isEmpty ? "checkmark" : "wand.and.sparkles")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.green)
                Text(edits.isEmpty ? "Inserted" : "Polished \(edits.count)")
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(1)
                Spacer(minLength: 4)
                toastButton("flag", label: "Flag as wrong") {
                    actions.flagLastTakeWrong()
                    dismissBeat()
                }
                toastButton("pencil", label: "Edit in history") {
                    actions.editLastTake()
                    dismissBeat()
                }
            }
        }
        .accessibilityLabel(
            edits.isEmpty
                ? "Inserted"
                : "Inserted, proofreader polished \(edits.count) words")
    }

    /// The lingering rejected-take toast. Passive-first: the press is the
    /// retry; the one affordance inserts the raw take anyway (which also
    /// flags the pass as wrong).
    private var rejectionToast: some View {
        toastChrome {
            HStack(alignment: .center, spacing: 8) {
                Image(systemName: "arrow.uturn.backward")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.orange)
                Text("Didn't catch that")
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.9)
                Spacer(minLength: 4)
                Button {
                    actions.insertRawAnyway()
                    dismissBeat()
                } label: {
                    Text("Insert anyway")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Insert the raw transcription anyway")
            }
        }
        .accessibilityLabel("Transcription rejected — hold the hotkey to retry")
    }

    /// The error toast: `errorDescription` only — the orb corner is a
    /// glanceable surface, recovery details live in richer UI. Informational,
    /// no affordances: the panel is click-through outside beat lingers.
    private func errorToast(for error: DictationError) -> some View {
        let message = error.errorDescription ?? "Something went wrong."
        return toastChrome {
            HStack(alignment: .center, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.orange)
                Text(message)
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                    .minimumScaleFactor(0.9)
                Spacer(minLength: 0)
            }
        }
        .accessibilityLabel(message)
    }

    /// Shared toast chrome: one untinted glass capsule at
    /// ``OrbMetrics/toastSize``, enrolled in the glass namespace so it morphs
    /// out of — and folds back into — the orb.
    private func toastChrome(@ViewBuilder content: () -> some View) -> some View {
        content()
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .frame(width: OrbMetrics.toastSize.width, height: OrbMetrics.toastSize.height)
            .glassEffect(.regular, in: .capsule)
            .glassEffectID("toast", in: glassSpace)
    }

    /// One small icon affordance on the lingering toast: plain style (the
    /// glass capsule is the chrome), secondary ink, generous hit target.
    private func toastButton(
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
}

// MARK: - Faces

/// Scales the orb with the mic level — the recording breath. Reads
/// `feed.level` in its own body so the ~47 Hz meter cadence invalidates only
/// this wrapper's transform; the glass circle inside was built once by the
/// parent and is never re-diffed per frame. When `breathing` is false the
/// level is never read, so no meter dependency exists at all.
private struct OrbBreathingView<Content: View>: View {
    var feed: DictationFeed
    var breathing: Bool
    private let content: Content

    init(feed: DictationFeed, breathing: Bool, @ViewBuilder content: () -> Content) {
        self.feed = feed
        self.breathing = breathing
        self.content = content()
    }

    var body: some View {
        let scale =
            breathing
            ? 1 + CGFloat(feed.level) * (OrbMetrics.breathingMaxScale - 1)
            : 1
        content
            .scaleEffect(scale)
            // A lazy spring turns the jittery per-frame level into a breath
            // and eases the release back to 1 when recording ends. Inert
            // under Reduce Motion, where `scale` is pinned to 1.
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: scale)
    }
}

/// The recording face: the feed's spectrum bands as short radial spokes — a
/// miniature starburst meter. One `Canvas` pass per meter frame (eight round-
/// capped strokes); even silent bands keep a small nub, so a quiet mic still
/// reads as "listening" (a dotted ring), not "off".
private struct OrbSpectrumView: View {
    /// Read directly (not passed as a value) so meter-cadence updates
    /// invalidate only this subtree — never the orb chrome or its glass.
    var feed: DictationFeed

    var body: some View {
        // Read in `body`, not the render closure — Canvas's renderer runs
        // outside body evaluation, where Observation wouldn't track the feed.
        let bands = feed.spectrum
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            for (index, band) in bands.enumerated() {
                let angle = Double(index) / Double(bands.count) * 2 * .pi - .pi / 2
                let length =
                    OrbMetrics.spokeMinLength
                    + CGFloat(band) * (OrbMetrics.spokeMaxLength - OrbMetrics.spokeMinLength)
                var spoke = Path()
                spoke.move(to: point(at: OrbMetrics.spokeInnerRadius, angle: angle, around: center))
                spoke.addLine(
                    to: point(
                        at: OrbMetrics.spokeInnerRadius + length, angle: angle, around: center))
                context.stroke(
                    spoke,
                    with: .color(Self.spokeColor(for: band)),
                    style: StrokeStyle(lineWidth: OrbMetrics.spokeWidth, lineCap: .round))
            }
        }
        .frame(width: OrbMetrics.spokeFieldSize, height: OrbMetrics.spokeFieldSize)
    }

    private func point(at radius: CGFloat, angle: Double, around center: CGPoint) -> CGPoint {
        CGPoint(
            x: center.x + radius * CGFloat(cos(angle)),
            y: center.y + radius * CGFloat(sin(angle)))
    }

    /// The same red-orange intensity ramp as `AudioBarsView` — one meter
    /// color family across variants.
    private static func spokeColor(for band: Float) -> Color {
        let intensity = Double(min(band * 1.2, 1.0))
        return Color(
            red: 0.9 + intensity * 0.1,
            green: 0.25 + (1 - intensity) * 0.15,
            blue: 0.2)
    }
}

/// The processing/proofreading face: a slow three-dot orbit, comet-ordered —
/// bright head, dimming tail. Proofreading swaps the head for a twinkling
/// sparkle, so the Proofread Pass reads as polish, not more waiting. Derived
/// straight from the driving `time` with no @State (the ProcessingDotsView
/// lesson: mirroring time into state doubles body evaluations for nothing).
private struct OrbOrbitView: View {
    let time: TimeInterval
    let proofreading: Bool

    /// One revolution takes this long — slow enough to read as thinking.
    private static let period: TimeInterval = 1.8
    private static let orbitRadius: CGFloat = 10
    /// Each tail dot lags the head by this many radians.
    private static let trailSpacing = 0.55
    private static let dotSizes: [CGFloat] = [5, 4, 3]
    private static let dotOpacities: [Double] = [1.0, 0.6, 0.35]

    var body: some View {
        let headAngle = time * 2 * .pi / Self.period
        ZStack {
            // Tail first, so the head draws on top where the comet bunches.
            ForEach(1..<3) { index in
                let angle = headAngle - Double(index) * Self.trailSpacing
                Circle()
                    .fill(Color.primary.opacity(Self.dotOpacities[index]))
                    .frame(width: Self.dotSizes[index], height: Self.dotSizes[index])
                    .offset(
                        x: Self.orbitRadius * CGFloat(cos(angle)),
                        y: Self.orbitRadius * CGFloat(sin(angle)))
            }
            if proofreading {
                // The twinkle rides an off-period sine so it never locks to
                // the orbit — the glint keeps surprising the eye, slightly.
                let twinkle = (sin(time * 5) + 1) / 2
                Image(systemName: "sparkles")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Color.primary.opacity(0.75 + twinkle * 0.25))
                    .scaleEffect(0.8 + twinkle * 0.35)
                    .offset(
                        x: Self.orbitRadius * CGFloat(cos(headAngle)),
                        y: Self.orbitRadius * CGFloat(sin(headAngle)))
            } else {
                Circle()
                    .fill(Color.primary)
                    .frame(width: Self.dotSizes[0], height: Self.dotSizes[0])
                    .offset(
                        x: Self.orbitRadius * CGFloat(cos(headAngle)),
                        y: Self.orbitRadius * CGFloat(sin(headAngle)))
            }
        }
    }
}

// MARK: - Registration

extension OverlayVariants {
    /// The corner presence orb (map #283). Defined beside its view so the
    /// exploration ships as one file; the registry's `all` enrolls it.
    static let orb = OverlayVariant(
        id: "orb",
        displayName: "Orb",
        placement: .orb
    ) { feed, actions in
        AnyView(OrbOverlayView(feed: feed, actions: actions))
    }
}

// MARK: - Previews

#Preview("Recording") {
    let feed = DictationFeed()
    feed.setPhase(.recording)
    feed.apply(MeterFrame(level: 0.55, bands: [0.7, 0.95, 0.6, 0.4, 0.65, 0.3, 0.2, 0.35]))
    return OrbOverlayView(feed: feed)
        .padding(40)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        )
        .task {
            // A synthetic meter pump so the preview shows the breath and the
            // spokes live, not one frozen frame.
            while !Task.isCancelled {
                let t = Date().timeIntervalSinceReferenceDate
                let level = Float(0.45 + 0.35 * sin(t * 2.1))
                let bands = (0..<MeterFrame.bandCount).map { band -> Float in
                    let phase = Double(band)
                    let wave: Double = max(0, sin(t * (1.7 + phase * 0.6) + phase))
                    return Float(wave) * (0.9 - Float(band) * 0.08)
                }
                feed.apply(MeterFrame(level: max(0, level), bands: bands))
                try? await Task.sleep(for: .milliseconds(33))
            }
        }
}

#Preview("Processing") {
    let feed = DictationFeed()
    feed.setPhase(.processing)
    return OrbOverlayView(feed: feed)
        .padding(40)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Proofreading") {
    let feed = DictationFeed()
    feed.setPhase(.proofreading)
    return OrbOverlayView(feed: feed)
        .padding(40)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Committed · polished") {
    let feed = DictationFeed()
    return OrbOverlayView(feed: feed)
        .padding(40)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        )
        .task {
            // Replays the real sequence — orbit, then the toast unfurls,
            // lingers, and folds — on a loop for design review.
            while !Task.isCancelled {
                feed.setPhase(.processing)
                try? await Task.sleep(for: .seconds(1.2))
                feed.setPhase(.idle)
                feed.emit(
                    .committed(
                        text: "Ship the orb overlay variant.",
                        duration: 3.2,
                        edits: [
                            WordEdit(original: "orb", replacement: "Orb"),
                            WordEdit(original: "varient", replacement: "variant"),
                            WordEdit(original: "overlays", replacement: "overlay"),
                        ]))
                try? await Task.sleep(for: .seconds(4.5))
            }
        }
}

#Preview("Rejected") {
    let feed = DictationFeed()
    return OrbOverlayView(feed: feed)
        .padding(40)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        )
        .task {
            // Loops the rejection so the shake and the insert-anyway toast
            // replay without re-running the preview.
            while !Task.isCancelled {
                feed.setPhase(.processing)
                try? await Task.sleep(for: .seconds(1.2))
                feed.setPhase(.idle)
                feed.emit(.rejected(raw: "uh let's um", reason: "gibberish"))
                try? await Task.sleep(for: .seconds(4.5))
            }
        }
}

#Preview("Error") {
    let feed = DictationFeed()
    feed.setPhase(.error(.noSpeechDetected))
    return OrbOverlayView(feed: feed)
        .padding(40)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}
