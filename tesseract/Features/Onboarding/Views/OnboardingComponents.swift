//
//  OnboardingComponents.swift
//  tesseract
//
//  Shared pieces of the Onboarding Tour: chapter scaffolding and typography,
//  the pill chapter indicators, the per-glyph welcome reveal, and the
//  permission priming card.
//

import SwiftUI

// MARK: - Card surface

/// The one material card surface all tour cards share — standard material and
/// a hairline, nothing else (design language §1: adopt by subtraction).
struct OnboardingCardModifier: ViewModifier {
    func body(content: Content) -> some View {
        content.background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(.regularMaterial)
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(.quaternary, lineWidth: 0.5)
                }
        )
    }
}

extension View {
    func onboardingCard() -> some View {
        modifier(OnboardingCardModifier())
    }
}

// MARK: - Typography

/// Tour surface constants (design language §2: one type size and one spacing
/// rhythm per surface; hierarchy comes from weight and color). The kicker and
/// title are the tour's display roles and keep their own scale — everything
/// else sets `body`.
enum OnboardingType {
    static let bodySize: CGFloat = 13
    static let rhythm: CGFloat = 12

    static let body: Font = .system(size: bodySize)

    static let titleFont: Font = .system(size: 30, weight: .semibold)
    static let titleTracking: CGFloat = -0.4

    static func title(_ text: String) -> some View {
        Text(text)
            .font(titleFont)
            .tracking(titleTracking)
            .foregroundStyle(.primary)
            .multilineTextAlignment(.center)
    }

    static func kicker(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10.5, weight: .semibold))
            .tracking(2.6)
            .textCase(.uppercase)
            .foregroundStyle(.secondary)
    }

    static func subtitle(_ text: String) -> some View {
        Text(text)
            .font(body)
            .lineSpacing(3)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
    }
}

/// The standard chapter layout: kicker, title, subtitle, then the chapter's
/// own stage area.
struct ChapterScaffold<Stage: View>: View {
    let kicker: String
    let title: String
    let subtitle: String
    @ViewBuilder var stage: Stage

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: OnboardingType.rhythm) {
                OnboardingType.kicker(kicker)
                OnboardingType.title(title)
                OnboardingType.subtitle(subtitle)
                    .frame(maxWidth: 460)
            }
            .padding(.top, 8)

            Spacer(minLength: 16)

            stage

            Spacer(minLength: 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.horizontal, 48)
    }
}

// MARK: - Chapter indicators

struct ChapterDots: View {
    var current: OnboardingTourController.Chapter
    var onSelect: (OnboardingTourController.Chapter) -> Void

    var body: some View {
        HStack(spacing: 7) {
            ForEach(OnboardingTourController.Chapter.allCases) { chapter in
                Button {
                    onSelect(chapter)
                } label: {
                    Capsule(style: .continuous)
                        .fill(
                            chapter == current
                                ? AnyShapeStyle(.primary.opacity(0.85))
                                : AnyShapeStyle(.tertiary)
                        )
                        .frame(width: chapter == current ? 22 : 6, height: 6)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Chapter \(chapter.rawValue + 1)")
                .accessibilityAddTraits(chapter == current ? [.isSelected] : [])
            }
        }
        .animation(.spring(response: 0.45, dampingFraction: 0.85), value: current)
    }
}

// MARK: - Per-glyph reveal (welcome line only)

/// One-time per-glyph entrance for the welcome headline: each glyph fades,
/// lifts, and sharpens in with a small stagger. The tour's only TextRenderer —
/// a signature, not a tic.
struct GlyphReveal: TextRenderer, Animatable {
    var progress: Double

    var animatableData: Double {
        get { progress }
        set { progress = newValue }
    }

    func draw(layout: Text.Layout, in context: inout GraphicsContext) {
        let slices = layout.flatMap { line in line.flatMap { run in run } }
        let count = max(slices.count, 1)
        for (index, slice) in slices.enumerated() {
            let stagger = Double(index) / Double(count) * 0.55
            let t = min(1, max(0, (progress - stagger) / 0.45))
            var copy = context
            copy.opacity = t
            copy.translateBy(x: 0, y: (1 - t) * 10)
            if t < 1 {
                copy.addFilter(.blur(radius: (1 - t) * 5))
            }
            copy.draw(slice)
        }
    }
}

// MARK: - Permission priming card

/// The Screen-Studio-style priming card: benefit first, a user-pressed action,
/// live status chip, inline recovery on denial. Never gates Continue.
struct PermissionCard: View {
    let icon: String
    let title: String
    let benefit: String
    let state: PermissionState
    let grantAction: () -> Void
    let recoverAction: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: OnboardingType.rhythm) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.tint)
                    .frame(width: 22)
                Text(title)
                    .font(OnboardingType.body)
                    .fontWeight(.semibold)
                Spacer(minLength: 0)
                statusChip
            }

            Text(benefit)
                .font(OnboardingType.body)
                .lineSpacing(3)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            actionRow
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .onboardingCard()
        .animation(.spring(response: 0.4, dampingFraction: 0.85), value: state)
    }

    @ViewBuilder
    private var statusChip: some View {
        switch state {
        case .granted:
            Label("Granted", systemImage: "checkmark.circle.fill")
                .font(OnboardingType.body)
                .fontWeight(.medium)
                .foregroundStyle(.green)
                .labelStyle(.titleAndIcon)
                .transition(.scale.combined(with: .opacity))
        case .requesting:
            HStack(spacing: 5) {
                ProgressView().controlSize(.mini)
                Text("Waiting\u{2026}")
                    .font(OnboardingType.body)
                    .foregroundStyle(.secondary)
            }
        case .denied, .restricted:
            Label("Not granted", systemImage: "xmark.circle")
                .font(OnboardingType.body)
                .fontWeight(.medium)
                .foregroundStyle(.orange)
        case .unknown:
            EmptyView()
        }
    }

    @ViewBuilder
    private var actionRow: some View {
        switch state {
        case .granted:
            EmptyView()
        case .denied, .restricted:
            VStack(alignment: .leading, spacing: 6) {
                Button("Open System Settings", action: recoverAction)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                Text("You can continue — this just stays off until you grant it.")
                    .font(OnboardingType.body)
                    .foregroundStyle(.tertiary)
            }
        case .unknown, .requesting:
            Button(action: grantAction) {
                Text("Enable")
                    .frame(minWidth: 64)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(state == .requesting)
        }
    }
}

// MARK: - Locked Try-it slot

/// The honest degraded state a Try-it shows while its preconditions are
/// missing — one shape shared by every chapter, so locked slots read the same
/// everywhere. The caller may override the reason (e.g. a missing permission
/// outranks the download story).
struct TryItLockedSlot: View {
    let icon: String
    let status: ModelStatus
    let modelNoun: String
    var overrideReason: String?

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 18))
                .foregroundStyle(.tertiary)
            VStack(alignment: .leading, spacing: 2) {
                Text("Try it right here")
                    .font(OnboardingType.body)
                    .fontWeight(.semibold)
                    .foregroundStyle(.secondary)
                Text(reason)
                    .font(OnboardingType.body)
                    .foregroundStyle(.tertiary)
            }
            Spacer(minLength: 0)
        }
    }

    private var reason: String {
        if let overrideReason { return overrideReason }
        switch status {
        case .downloading(let progress):
            let percent = progress.formatted(.wholePercent)
            return "The \(modelNoun) model is \(percent) of the way here — a minute more."
        default:
            return "Available as soon as the \(modelNoun) model lands."
        }
    }
}

// MARK: - Stage panel

/// The material panel every chapter stages its demo in — one consistent
/// surface so the stages read as one system.
struct StagePanel<Content: View>: View {
    var maxWidth: CGFloat = 560
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(18)
            .frame(maxWidth: maxWidth)
            .onboardingCard()
    }
}
