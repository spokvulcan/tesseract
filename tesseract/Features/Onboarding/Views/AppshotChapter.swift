//
//  AppshotChapter.swift
//  tesseract
//
//  Chapter 3 — Appshot (PRD #170): tap both Command keys in any app and its
//  frontmost window lands in the agent composer. Scripted, like the Agent
//  chapter — a real capture needs Screen Recording (asked lazily on first
//  use) and would summon the main window over the tour. The demo strings come
//  from the real `AppshotController` label builders so the vignette can't
//  drift from the product.
//

import SwiftUI

struct AppshotChapter: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var isPressed = false
    @State private var isStaged = false
    @State private var playback: Task<Void, Never>?

    private static let demoApp = "Mail"
    private static let demoTitle = "Trip itinerary"

    var body: some View {
        ChapterScaffold(
            kicker: "Chapter 3 · Appshot",
            title: "Show it what you're looking at",
            subtitle: "Tap \(settings.appshotHotkey.displayString) in any app — its frontmost "
                + "window lands in the agent composer, ready to ask about."
        ) {
            VStack(spacing: OnboardingType.rhythm) {
                StagePanel(maxWidth: 520) {
                    VStack(spacing: OnboardingType.rhythm) {
                        miniWindow
                        composerMock
                    }
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(
                        "Illustration: an Appshot of a Mail window staged in the "
                            + "agent composer as an attachment with an editable caption")
                }

                keycapHint
            }
        }
        .onAppear { startPlayback() }
        .onDisappear {
            playback?.cancel()
            playback = nil
        }
    }

    // MARK: - The captured window

    /// A miniature stand-in for whatever the user is looking at; its border
    /// flashes tint on the press beat — the capture moment.
    private var miniWindow: some View {
        VStack(spacing: 0) {
            HStack(spacing: 5) {
                ForEach(0..<3, id: \.self) { _ in
                    Circle().fill(.quaternary).frame(width: 7, height: 7)
                }
                Spacer(minLength: 0)
                Text("\(Self.demoTitle) — \(Self.demoApp)")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
                Spacer(minLength: 0)
                Color.clear.frame(width: 31, height: 7)
            }
            .padding(.horizontal, 9)
            .frame(height: 24)
            .background(.quaternary.opacity(0.4))

            VStack(alignment: .leading, spacing: 7) {
                skeletonBar(width: 180)
                skeletonBar(width: 236)
                skeletonBar(width: 148)
            }
            .padding(12)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(width: 320, height: 116)
        .background(.quaternary.opacity(0.2))
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(
                    isPressed ? AnyShapeStyle(.tint) : AnyShapeStyle(.quaternary),
                    lineWidth: isPressed ? 1.5 : 0.5)
        }
        .scaleEffect(isPressed ? 0.97 : 1)
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isPressed)
    }

    private func skeletonBar(width: CGFloat) -> some View {
        Capsule(style: .continuous)
            .fill(.quaternary.opacity(0.8))
            .frame(width: width, height: 7)
    }

    // MARK: - The composer

    /// A quiet echo of the agent composer: the shot arrives as an attachment
    /// chip, and the window label prefills the message — editable, like the
    /// real one.
    private var composerMock: some View {
        VStack(alignment: .leading, spacing: 8) {
            if isStaged {
                HStack(spacing: 5) {
                    Image(systemName: "macwindow")
                        .font(.system(size: 10))
                    Text(
                        AppshotController.filename(
                            appName: Self.demoApp, windowTitle: Self.demoTitle)
                    )
                    .font(OnboardingType.body)
                }
                .foregroundStyle(.tint)
                .padding(.horizontal, 9)
                .padding(.vertical, 5)
                .background(Capsule(style: .continuous).fill(.tint.opacity(0.14)))
                .transition(.scale(scale: 0.85).combined(with: .opacity))
            }

            Text(
                isStaged
                    ? AppshotController.prefill(
                        appName: Self.demoApp, windowTitle: Self.demoTitle)
                    : "Ask about anything\u{2026}"
            )
            .font(OnboardingType.body)
            .foregroundStyle(isStaged ? AnyShapeStyle(.primary) : AnyShapeStyle(.tertiary))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(.quaternary.opacity(0.35))
        )
        .animation(.spring(response: 0.4, dampingFraction: 0.85), value: isStaged)
    }

    // MARK: - The chord

    private var keycapHint: some View {
        HStack(spacing: 8) {
            HStack(spacing: 4) {
                if settings.appshotHotkey.isDoubleCommand {
                    keycap("⌘")
                    keycap("⌘")
                } else {
                    keycap(settings.appshotHotkey.displayString)
                }
            }
            Text(pressHintLine)
                .font(OnboardingType.body)
                .foregroundStyle(.tertiary)
        }
    }

    private var pressHintLine: String {
        let chord =
            settings.appshotHotkey.isDoubleCommand
            ? "Both Command keys, together" : "Your Appshot hotkey"
        return "\(chord) — the first shot asks for Screen Recording, once."
    }

    private func keycap(_ glyph: String) -> some View {
        Text(glyph)
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(isPressed ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
            .frame(minWidth: 26)
            .frame(height: 22)
            .background(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(
                        isPressed
                            ? AnyShapeStyle(.tint.opacity(0.15))
                            : AnyShapeStyle(.quaternary.opacity(0.5)))
            )
            .overlay {
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .strokeBorder(.quaternary, lineWidth: 0.5)
            }
            .offset(y: isPressed ? 1 : 0)
            .animation(.spring(response: 0.25, dampingFraction: 0.7), value: isPressed)
    }

    // MARK: - Playback

    /// One scripted beat per appearance: idle, the chord goes down, the shot
    /// stages. Under Reduce Motion the staged end state shows immediately.
    private func startPlayback() {
        playback?.cancel()
        if reduceMotion {
            isPressed = false
            isStaged = true
            return
        }
        isPressed = false
        isStaged = false
        playback = Task {
            try? await Task.sleep(for: .milliseconds(700))
            guard !Task.isCancelled else { return }
            withAnimation(.spring(response: 0.25, dampingFraction: 0.7)) { isPressed = true }
            try? await Task.sleep(for: .milliseconds(450))
            guard !Task.isCancelled else { return }
            withAnimation(.spring(response: 0.45, dampingFraction: 0.85)) {
                isPressed = false
                isStaged = true
            }
        }
    }
}
