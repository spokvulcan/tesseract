//
//  AgentNotchOverlayView.swift
//  tesseract
//
//  Dynamic Island overlay for agent voice interactions.
//  Reuses DynamicIslandShape from the TTS notch overlay.
//

import SwiftUI
import os

struct AgentNotchOverlayView: View {
    var state: AgentNotchState
    let menuBarHeight: CGFloat
    var frameTracker: NotchFrameTracker
    var onTap: (() -> Void)?

    @State private var expansion: CGFloat = 0
    @State private var contentVisible = false
    @State private var swipeOffset: CGFloat = 0

    private let notchHeight: CGFloat = 37
    private let notchWidth: CGFloat = 200
    private let collapsedInset: CGFloat = 8
    private let expandedInset: CGFloat = 16

    private var currentTopInset: CGFloat {
        collapsedInset + (expandedInset - collapsedInset) * expansion
    }

    private var currentBottomRadius: CGFloat {
        8 + (18 - 8) * expansion
    }

    var body: some View {
        GeometryReader { geo in
            let phase = state.phase
            let targetWidth = phase.displayWidth
            let targetHeight = menuBarHeight + phase.contentHeight
            let currentHeight = notchHeight + (targetHeight - notchHeight) * expansion
            let currentWidth = notchWidth + (targetWidth - notchWidth) * expansion

            ZStack(alignment: .top) {
                DynamicIslandShape(
                    topInset: currentTopInset,
                    bottomRadius: currentBottomRadius
                )
                .fill(.black)
                .frame(width: currentWidth, height: currentHeight)

                if contentVisible {
                    VStack(spacing: 0) {
                        Spacer().frame(height: menuBarHeight)
                        phaseContent
                    }
                    .frame(width: targetWidth, height: targetHeight)
                    .transition(.opacity)
                }
            }
            .frame(width: currentWidth, height: currentHeight, alignment: .top)
            .offset(y: swipeOffset)
            .frame(width: geo.size.width, height: geo.size.height, alignment: .top)
            .contentShape(Rectangle())
            .onTapGesture {
                onTap?()
            }
            .gesture(
                DragGesture(minimumDistance: 8)
                    .onChanged { value in
                        let dy = value.translation.height
                        swipeOffset = dy < 0 ? dy : dy * 0.15
                    }
                    .onEnded { value in
                        let dy = value.translation.height
                        let velocity = value.predictedEndTranslation.height - dy
                        if dy < -30 || velocity < -100 {
                            state.shouldDismiss = true
                        } else {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                swipeOffset = 0
                            }
                        }
                    }
            )
            .onChange(of: currentWidth) { _, w in
                frameTracker.visibleWidth = w
            }
            .onChange(of: currentHeight) { _, h in
                frameTracker.visibleHeight = h
            }
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.4)) {
                expansion = 1
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                withAnimation(.easeOut(duration: 0.25)) {
                    contentVisible = true
                }
            }
        }
        .onChange(of: state.phase) { _, newPhase in
            // Animate size changes between phases
            withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                // Trigger frame recalculation via expansion (already 1)
                expansion = 1
            }
        }
        .onChange(of: state.shouldDismiss) { _, shouldDismiss in
            if shouldDismiss {
                withAnimation(.easeIn(duration: 0.15)) {
                    contentVisible = false
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    withAnimation(.easeIn(duration: 0.3)) {
                        expansion = 0
                        swipeOffset = -60
                    }
                }
            }
        }
    }

    // MARK: - Phase Content

    @ViewBuilder
    private var phaseContent: some View {
        Group {
            switch state.phase {
            case .hidden:
                EmptyView()
            case .listening(let audioLevel):
                listeningView(level: audioLevel)
            case .transcribing(let preview):
                transcribingView(preview: preview)
            case .thinking:
                thinkingView
            case .toolCall(let name):
                toolCallView(name: name)
            case .responding(let text):
                respondingView(text: text)
            case .complete(let text):
                completeView(text: text)
            case .error(let message):
                errorView(message: message)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: state.phase)
    }

    // MARK: - Listening

    private func listeningView(level: Float) -> some View {
        HStack(spacing: 10) {
            AudioBarsView(level: CGFloat(level), phase: 0)
                .frame(width: 80, height: 20)

            Text("Listening")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white.opacity(0.9))
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Transcribing

    private func transcribingView(preview: String) -> some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate

            HStack(spacing: 8) {
                ProcessingDotsView(time: time)
                    .frame(width: 60, height: 12)

                if !preview.isEmpty {
                    Text(preview)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.7))
                        .lineLimit(1)
                        .truncationMode(.tail)
                } else {
                    Text("Transcribing...")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
            .padding(.horizontal, 16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Thinking

    private var thinkingView: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate

            HStack(spacing: 10) {
                ThinkingPulseView(time: time)
                    .frame(width: 20, height: 20)

                Text("Thinking...")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color(red: 0.616, green: 0.541, blue: 1.0)) // #9D8AFF
            }
            .padding(.horizontal, 16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Tool Call

    private func toolCallView(name: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "wrench.and.screwdriver.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Color(red: 0.392, green: 0.824, blue: 0.627)) // #64D2A0

            Text(AgentNotchPhase.toolDisplayName(name))
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Color(red: 0.392, green: 0.824, blue: 0.627))
                .lineLimit(1)
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Responding

    private func respondingView(text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(text)
                .font(.system(size: 13, weight: .regular))
                .foregroundStyle(.white.opacity(0.95))
                .lineLimit(3)
                .truncationMode(.tail)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // MARK: - Complete

    private func completeView(text: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(Color(red: 0.392, green: 0.824, blue: 0.627))

            Text(text)
                .font(.system(size: 13, weight: .regular))
                .foregroundStyle(.white.opacity(0.85))
                .lineLimit(2)
                .truncationMode(.tail)
                .multilineTextAlignment(.leading)
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }

    // MARK: - Error

    private func errorView(message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.red)

            Text(message)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white.opacity(0.9))
                .lineLimit(1)
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Thinking Pulse Animation

/// A pulsing sparkle indicator for the thinking state.
private struct ThinkingPulseView: View {
    let time: Double

    private let accentColor = Color(red: 0.616, green: 0.541, blue: 1.0) // #9D8AFF

    var body: some View {
        let pulse = (sin(time * 3.0) + 1.0) / 2.0
        let scale = 0.7 + pulse * 0.3
        let opacity = 0.6 + pulse * 0.4

        Image(systemName: "brain.filled.head.profile")
            .font(.system(size: 14, weight: .medium))
            .foregroundStyle(accentColor.opacity(opacity))
            .scaleEffect(scale)
    }
}
